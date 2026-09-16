"""Generated integration contracts written before the adapted history pass."""

import copy
import hashlib
import unittest

from src.treasury_history_identity import reconcile_history_identity
from tests.test_treasury_realized_identity import ACTUAL, ANNOUNCED, fixture


def run(f):
    return reconcile_history_identity(*f["args"], **f["kwargs"])


def amended_fixture(same_date=False):
    f = fixture()
    release = "2017-01-07" if same_date else "2017-01-10"
    pretty = "January 7, 2017" if same_date else "January 10, 2017"
    f["args"][1] = (
        f["args"][1]
        .replace("TREASURY OFFERING ANNOUNCEMENT", "AMENDED ANNOUNCEMENT")
        .replace("January 7, 2017", pretty)
    )
    f["kwargs"]["amendment"] = {
        "auction_date": "2017-01-13",
        "cusip": ANNOUNCED,
        "archive_announcement_date": "2017-01-07",
        "release_date": release,
        "announcement_pdf_text_sha256": hashlib.sha256(f["args"][1].encode()).hexdigest(),
        "announcement_pdf_sha256": "a" * 64,
        "review_sha256": "b" * 64,
        "notice_url": "https://www.treasurydirect.gov/instit/annceresult/press/preanre/2017/BPD_SPL_20170110_1.pdf",
        "notice_body_sha256": "c" * 64,
        "notice_text_sha256": "d" * 64,
        "notice_release_date": "2017-01-10",
    }
    return f


class HistoryIdentityTests(unittest.TestCase):
    def test_ordinary_inputs_preserved_and_hashed(self):
        f = fixture()
        before = copy.deepcopy(f)
        out = run(f)
        self.assertEqual(f, before)
        self.assertEqual(out["actual_cusip"], ANNOUNCED)
        self.assertFalse(out["source_admitted"])
        self.assertEqual(
            out["input_hashes"]["result_xml_sha256"], hashlib.sha256(f["args"][2]).hexdigest()
        )

    def test_announcement_coupon_does_not_change_offered_context(self):
        f = fixture()
        f["args"][1] = f["args"][1].replace("2-Year Note", "2-Year 1-1/2% Note")
        out = run(f)
        self.assertEqual(out["offered_term"], "2-Year Note")
        self.assertEqual(out["announcement_term_lexeme"], "2-Year 1-1/2% Note")

    def test_coupon_cannot_hide_wrong_xml_term_or_result_term(self):
        for idx, old, new in [(0, b"2-YEAR", b"3-YEAR"), (3, "2-Year Note", "3-Year Note")]:
            f = fixture()
            f["args"][1] = f["args"][1].replace("2-Year Note", "2-Year 2% Note")
            f["args"][idx] = f["args"][idx].replace(old, new)
            with self.subTest(idx=idx), self.assertRaises(ValueError):
                run(f)

    def test_confirmed_substitution_distinguishes_current_and_original_dates(self):
        f = fixture(True)
        f["args"][2] = f["args"][2].replace(b"<DatedDate>2012-01-15", b"<DatedDate>2017-01-15")
        f["args"][3] = f["args"][3].replace(
            "Dated Date January 15, 2012", "Dated Date January 15, 2017"
        )
        out = run(f)
        self.assertTrue(out["substitution"])
        self.assertEqual(out["final_lineage"]["dated_date"], "2017-01-15")
        self.assertEqual(out["final_lineage"]["original_dated_date"], "2012-01-15")
        self.assertEqual(out["final_lineage"]["original_tenor_years"], 7)
        f["kwargs"]["confirmation"] = None
        with self.assertRaises(ValueError):
            run(f)

    def test_new_issue_february_month_end_preserves_actual_dates(self):
        f = fixture()
        for idx in (0, 2):
            f["args"][idx] = (
                f["args"][idx]
                .replace(b"2017-01-15", b"2018-02-28")
                .replace(b"2017-01-17", b"2018-03-01")
                .replace(b"2019-01-15", b"2020-02-29")
            )
        for idx in (1, 3):
            f["args"][idx] = (
                f["args"][idx]
                .replace("January 15, 2017", "February 28, 2018")
                .replace("January 17, 2017", "March 1, 2018")
                .replace("January 15, 2019", "February 29, 2020")
            )
        out = run(f)
        self.assertEqual(out["final_lineage"]["original_tenor_years"], 2)
        self.assertEqual(out["final_lineage"]["maturity_date"], "2020-02-29")
        f["args"][2] = f["args"][2].replace(b"2020-02-29", b"2020-02-28")
        with self.assertRaises(ValueError):
            run(f)

    def test_amendment_retains_archive_and_revised_dates_with_evidence(self):
        f = amended_fixture()
        out = run(f)
        self.assertEqual(out["archive_announcement_date"], "2017-01-07")
        self.assertEqual(out["announcement_release_date"], "2017-01-10")
        self.assertEqual(out["identity_available_date"], "2017-01-13")
        self.assertEqual(out["amendment"], f["kwargs"]["amendment"])
        self.assertFalse(out["amendment_provenance_verified"])

    def test_same_date_amended_title_still_requires_evidence(self):
        f = amended_fixture(True)
        self.assertEqual(run(f)["announcement_release_date"], "2017-01-07")
        del f["kwargs"]["amendment"]
        with self.assertRaises(ValueError):
            run(f)

    def test_amendment_evidence_binds_event_text_dates_and_official_notice(self):
        changes = {
            "cusip": ACTUAL,
            "auction_date": "2017-01-12",
            "archive_announcement_date": "2017-01-06",
            "release_date": "2017-01-09",
            "announcement_pdf_text_sha256": "e" * 64,
            "notice_url": "https://evil.example/notice.pdf",
            "notice_release_date": "2017-01-14",
            "review_sha256": "A" * 64,
        }
        for key, value in changes.items():
            f = amended_fixture()
            f["kwargs"]["amendment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(f)

    def test_unamended_release_mismatch_and_unsolicited_amendment_rejected(self):
        f = fixture()
        f["args"][1] = f["args"][1].replace("January 7, 2017", "January 10, 2017")
        with self.assertRaises(ValueError):
            run(f)
        f = fixture()
        f["kwargs"]["amendment"] = amended_fixture()["kwargs"]["amendment"]
        with self.assertRaises(ValueError):
            run(f)

    def test_amendment_never_overrides_xml_announcement_dates(self):
        f = amended_fixture()
        f["args"][0] = f["args"][0].replace(
            b"<AnnouncementDate>2017-01-07", b"<AnnouncementDate>2017-01-10"
        )
        with self.assertRaises(ValueError):
            run(f)

    def test_amendment_notice_filename_dates_are_bounded_and_valid(self):
        for token in ("20251021", "20170230"):
            f = amended_fixture()
            f["kwargs"]["amendment"]["notice_url"] = f["kwargs"]["amendment"][
                "notice_url"
            ].replace("20170110", token)
            with self.subTest(token=token), self.assertRaises(ValueError):
                run(f)

    def test_wrong_own_cusip_and_announced_identity_still_rejected(self):
        for idx, old, new in [
            (1, ANNOUNCED, ACTUAL),
            (2, b"<AnnouncedCUSIP>AB1234567", b"<AnnouncedCUSIP>ZZ1234567"),
        ]:
            f = fixture(True)
            f["args"][idx] = f["args"][idx].replace(old, new)
            with self.subTest(idx=idx), self.assertRaises(ValueError):
                run(f)

    def test_dated_date_attributes_and_nested_values_rejected(self):
        for old, new in [
            (b"<DatedDate>", b'<DatedDate unit="alias">'),
            (
                b"<DatedDate>2017-01-15</DatedDate>",
                b"<Wrapper><DatedDate>2017-01-15</DatedDate></Wrapper>",
            ),
        ]:
            f = fixture()
            f["args"][0] = f["args"][0].replace(old, new)
            with self.subTest(new=new), self.assertRaises(ValueError):
                run(f)


if __name__ == "__main__":
    unittest.main()
