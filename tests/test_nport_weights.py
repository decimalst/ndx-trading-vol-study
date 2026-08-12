"""Pre-run contract for public QQQ N-PORT weight history.

These tests were written before the parser and before the historical filing
download.  The report date is never the availability date: a snapshot becomes
usable only after SEC acceptance and relative to the 16:00 ET forecast origin.
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.nport_weights import (
    archive_url,
    assign_snapshot_asof,
    parse_nport_xml,
    validate_snapshot,
)


SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <formData>
    <genInfo>
      <regName>Invesco QQQ Trust, Series 1</regName>
      <seriesName>Invesco QQQ Trust, Series 1</seriesName>
      <repPdDate>2019-12-31</repPdDate>
    </genInfo>
    <invstOrSecs>
      <invstOrSec>
        <name>Alpha Inc.</name><title>Alpha Inc.</title><cusip>000000001</cusip>
        <identifiers><isin value="US0000000010"/></identifiers>
        <balance>10</balance><units>NS</units><valUSD>600</valUSD>
        <pctVal>60</pctVal><assetCat>EC</assetCat>
      </invstOrSec>
      <invstOrSec>
        <name>Beta Inc.</name><title>Beta Inc.</title><cusip>000000002</cusip>
        <identifiers><isin value="US0000000028"/></identifiers>
        <balance>20</balance><units>NS</units><valUSD>400</valUSD>
        <pctVal>40</pctVal><assetCat>EC</assetCat>
      </invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>
"""


class NportParserContract(unittest.TestCase):
    def test_namespace_safe_parser_extracts_weights_and_identifiers(self):
        meta, holdings = parse_nport_xml(SAMPLE_XML)
        self.assertEqual(meta["report_date"], pd.Timestamp("2019-12-31"))
        self.assertEqual(meta["registrant"], "Invesco QQQ Trust, Series 1")
        self.assertEqual(holdings["name"].tolist(), ["Alpha Inc.", "Beta Inc."])
        self.assertEqual(holdings["cusip"].tolist(), ["000000001", "000000002"])
        self.assertEqual(holdings["isin"].tolist(), ["US0000000010", "US0000000028"])
        self.assertAlmostEqual(float(holdings["pct_value"].sum()), 100.0)

    def test_parser_rejects_missing_report_date(self):
        with self.assertRaisesRegex(ValueError, "report date"):
            parse_nport_xml(SAMPLE_XML.replace(b"<repPdDate>2019-12-31</repPdDate>", b""))

    def test_snapshot_validation_rejects_incomplete_percentage_total(self):
        _, holdings = parse_nport_xml(SAMPLE_XML)
        holdings.loc[1, "pct_value"] = 20.0
        with self.assertRaisesRegex(ValueError, "percentage total"):
            validate_snapshot(holdings, min_total=98.0, max_total=102.0)

    def test_snapshot_validation_allows_small_derivative_liability(self):
        _, holdings = parse_nport_xml(SAMPLE_XML)
        derivative = holdings.iloc[0].copy()
        derivative["name"] = "N/A CME E-Mini NASDAQ 100 Index Future"
        derivative["value_usd"] = -1.0
        derivative["pct_value"] = -0.1
        derivative["asset_category"] = "DE"
        holdings = pd.concat([holdings, derivative.to_frame().T], ignore_index=True)
        validate_snapshot(holdings, min_total=98.0, max_total=102.0)

    def test_archive_url_uses_registrant_cik_not_accession_filer(self):
        self.assertEqual(
            archive_url("0001067839", "0001752724-25-211318"),
            "https://www.sec.gov/Archives/edgar/data/1067839/"
            "000175272425211318/primary_doc.xml",
        )


class NportAvailabilityContract(unittest.TestCase):
    @staticmethod
    def filings() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "accession": ["old", "before-close", "after-close"],
                "report_date": pd.to_datetime(
                    ["2019-09-30", "2019-12-31", "2020-03-31"]
                ),
                "accepted_at": pd.to_datetime(
                    [
                        "2019-11-29T15:00:00Z",
                        "2020-02-28T20:30:00Z",  # 15:30 ET: same origin
                        "2020-05-29T20:15:00Z",  # 16:15 ET: next origin
                    ],
                    utc=True,
                ),
            }
        )

    def test_report_date_never_grants_early_availability(self):
        origins = pd.to_datetime(["2019-12-31", "2020-02-27"])
        got = assign_snapshot_asof(self.filings(), origins)
        self.assertEqual(got["accession"].tolist(), ["old", "old"])

    def test_acceptance_before_close_is_same_origin_after_close_is_next(self):
        origins = pd.to_datetime(
            ["2020-02-28", "2020-05-29", "2020-06-01"]
        )
        got = assign_snapshot_asof(self.filings(), origins)
        self.assertEqual(
            got["accession"].tolist(), ["before-close", "before-close", "after-close"]
        )

    def test_future_filing_cannot_change_past_assignments(self):
        origins = pd.to_datetime(["2019-12-31", "2020-02-28"])
        base = self.filings().iloc[:2]
        future = pd.concat(
            [
                base,
                pd.DataFrame(
                    {
                        "accession": ["far-future"],
                        "report_date": pd.to_datetime(["2030-03-31"]),
                        "accepted_at": pd.to_datetime(
                            ["2030-05-30T12:00:00Z"], utc=True
                        ),
                    }
                ),
            ],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(
            assign_snapshot_asof(base, origins),
            assign_snapshot_asof(future, origins),
        )


if __name__ == "__main__":
    unittest.main()
