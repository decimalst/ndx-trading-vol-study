"""Pre-run contracts for the combined public QQQ/NDX history.

The fixtures intentionally cover the three SEC layouts observed in the public
annual reports and the minimal Nasdaq XLSX structure.  These tests are written
before either source is downloaded by the combined-history command.
"""

from __future__ import annotations

import io
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd
import requests

from src.historical_weights import (
    _download,
    combine_disclosed_holdings,
    membership_export_url,
    parse_legacy_annual_report,
    parse_nasdaq_membership_xlsx,
    select_membership_trade_dates,
    validate_membership_snapshot,
)

NAME_FIRST_HTML = b"""
<html><body>
<h1>Schedule of Investments</h1><p>September 30, 2015</p>
<table>
<tr><th>Common Stock</th><th>Shares</th><th>Value</th></tr>
<tr><td>Apple, Inc.</td><td>40,000</td><td>$</td><td>600,000</td></tr>
<tr><td>Microsoft Corp.</td><td>20,000</td><td></td><td>400,000</td></tr>
<tr><td>Total Investments</td><td></td><td>$</td><td>1,000,000</td></tr>
</table></body></html>
"""

SHARES_FIRST_HTML = b"""
<html><body>
<h1>Schedule of Investments</h1><p>September 30, 2018</p>
<table>
<tr><th>Number of Shares</th><th>Common Stocks</th><th>Value</th></tr>
<tr><td></td><td>Computers &amp; Peripherals--60.0%</td><td></td></tr>
<tr><td>40,000</td><td>Apple, Inc.</td><td>$</td><td>600,000</td></tr>
<tr><td>20,000</td><td>Microsoft Corp.</td><td></td><td>400,000</td></tr>
<tr><td></td><td>Computers &amp; Peripherals</td><td></td><td>1,000,000</td></tr>
<tr><td></td><td>Total Investments--100.0%</td><td>$</td><td>1,000,000</td></tr>
</table></body></html>
"""

PLAIN_TEXT_REPORT = b"""
NASDAQ-100 TRUST, SERIES 1
SCHEDULE OF INVESTMENTS
SEPTEMBER 30, 2004
COMMON STOCK                                      SHARES          VALUE
Apple Computer, Inc.* .........................   40,000       $600,000
Microsoft Corporation ........................   20,000        400,000
<PAGE>
NASDAQ-100 TRUST, SERIES 1
SCHEDULE OF INVESTMENTS (CONTINUED)
SEPTEMBER 30, 2004
COMMON STOCK                                      SHARES          VALUE
NVIDIA Corporation*                                10,000        100,000
Total Investments (Cost $900,000) ............              $1,100,000
"""


def _minimal_xlsx(rows: list[list[str]]) -> bytes:
    """Build the tiny OOXML container needed to exercise our XLSX reader."""
    sheet_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for col_number, value in enumerate(row, start=1):
            col = chr(64 + col_number)
            cells.append(
                f'<c r="{col}{row_number}" t="inlineStr"><is><t>{value}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Weightings" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return output.getvalue()


class LegacyAnnualReportContract(unittest.TestCase):
    def test_name_first_html_extracts_holdings_and_reconciles_total(self):
        meta, holdings = parse_legacy_annual_report(NAME_FIRST_HTML)
        self.assertEqual(meta["report_date"], pd.Timestamp("2015-09-30"))
        self.assertEqual(meta["disclosed_total_usd"], 1_000_000.0)
        self.assertEqual(holdings["name"].tolist(), ["Apple, Inc.", "Microsoft Corp."])
        self.assertEqual(holdings["balance"].tolist(), [40_000.0, 20_000.0])
        self.assertEqual(holdings["value_usd"].tolist(), [600_000.0, 400_000.0])
        self.assertAlmostEqual(float(holdings["pct_value"].sum()), 100.0)

    def test_shares_first_html_ignores_sector_headers_and_subtotals(self):
        _, holdings = parse_legacy_annual_report(SHARES_FIRST_HTML)
        self.assertEqual(len(holdings), 2)
        self.assertEqual(holdings["name"].tolist(), ["Apple, Inc.", "Microsoft Corp."])

    def test_plain_text_layout_is_supported(self):
        meta, holdings = parse_legacy_annual_report(PLAIN_TEXT_REPORT)
        self.assertEqual(meta["report_date"], pd.Timestamp("2004-09-30"))
        self.assertEqual(len(holdings), 3)
        self.assertEqual(holdings.iloc[0]["name"], "Apple Computer, Inc.")

    def test_disclosed_total_mismatch_is_rejected(self):
        bad = NAME_FIRST_HTML.replace(b"1,000,000", b"1,100,000")
        with self.assertRaisesRegex(ValueError, "disclosed total"):
            parse_legacy_annual_report(bad)


class NasdaqMembershipContract(unittest.TestCase):
    def test_public_workbook_extracts_names_and_symbols_without_fake_weights(self):
        content = _minimal_xlsx(
            [
                ["Company Name", "Security Symbol"],
                ["Apple Inc.", "AAPL"],
                ["Microsoft Corporation", "MSFT"],
            ]
        )
        got = parse_nasdaq_membership_xlsx(content)
        self.assertEqual(got.to_dict("records"), [
            {"company_name": "Apple Inc.", "symbol": "AAPL"},
            {"company_name": "Microsoft Corporation", "symbol": "MSFT"},
        ])
        self.assertNotIn("weight", got.columns)

    def test_duplicate_security_symbol_is_rejected(self):
        frame = pd.DataFrame(
            {"company_name": ["Apple", "Apple"], "symbol": ["AAPL", "AAPL"]}
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_membership_snapshot(frame, min_members=2)

    def test_url_uses_explicit_eod_trade_date(self):
        self.assertEqual(
            membership_export_url(pd.Timestamp("2016-01-04")),
            "https://indexes.nasdaqomx.com/Index/ExportWeightings/NDX?"
            "tradeDate=2016-01-04T00%3A00%3A00.000&timeOfDay=EOD",
        )

    def test_report_dates_roll_back_to_observed_trading_sessions(self):
        report_dates = pd.to_datetime(["2023-09-29", "2023-09-30", "2023-10-01"])
        trading_dates = pd.to_datetime(["2023-09-28", "2023-09-29", "2023-10-02"])
        got = select_membership_trade_dates(report_dates, trading_dates)
        self.assertEqual(got.tolist(), [pd.Timestamp("2023-09-29")] * 3)


class CombinedHistoryContract(unittest.TestCase):
    @staticmethod
    def _row(source_form: str, report_date: str, name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "accession": [f"{source_form}-{report_date}"],
                "report_date": [pd.Timestamp(report_date)],
                "accepted_at": [pd.Timestamp(report_date, tz="UTC") + pd.Timedelta(days=60)],
                "filing_date": [pd.Timestamp(report_date) + pd.Timedelta(days=60)],
                "source_url": ["https://www.sec.gov/example"],
                "name": [name],
                "balance": [1.0],
                "units": ["NS"],
                "value_usd": [100.0],
                "pct_value": [100.0],
                "asset_category": ["EC"],
                "source_form": [source_form],
            }
        )

    def test_structured_nport_supersedes_legacy_overlap(self):
        legacy = pd.concat(
            [
                self._row("N-30B-2", "2018-09-30", "Legacy 2018"),
                self._row("N-30B-2", "2019-09-30", "Legacy overlap"),
            ],
            ignore_index=True,
        )
        nport = self._row("NPORT-P", "2019-09-30", "Structured overlap")
        got = combine_disclosed_holdings(legacy, nport)
        self.assertEqual(got["name"].tolist(), ["Legacy 2018", "Structured overlap"])
        self.assertEqual(got["source_form"].tolist(), ["N-30B-2", "NPORT-P"])


class SourceTransportContract(unittest.TestCase):
    def test_transient_sec_throttle_is_retried(self):
        blocked = requests.Response()
        blocked.status_code = 403
        blocked.url = "https://www.sec.gov/example"
        good = requests.Response()
        good.status_code = 200
        good.url = blocked.url
        good._content = b"public filing"
        with patch(
            "src.historical_weights.requests.get", side_effect=[blocked, good]
        ) as mocked:
            self.assertEqual(
                _download(blocked.url, retry_seconds=0, max_attempts=2),
                b"public filing",
            )
        self.assertEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
