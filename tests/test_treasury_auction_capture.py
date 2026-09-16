"""Generated receipt and request contracts; no real Treasury response values."""

import hashlib
import unittest

from src.treasury_auction_capture import annual_requests, validate_receipt


class TreasuryCaptureTests(unittest.TestCase):
    def test_exact_bounded_years_without_missing_document_filter(self):
        requests = annual_requests()
        self.assertEqual(len(requests), 16)
        self.assertEqual(requests[0][0:2], ("2010-01-01", "2010-12-31"))
        self.assertEqual(requests[-1][0:2], ("2025-01-01", "2025-10-20"))
        for start, end, url in requests:
            self.assertEqual(
                url,
                "https://www.treasurydirect.gov/TA_WS/securities/search?"
                f"startDate={start}&endDate={end}&compact=true&dateFieldName=auctionDate&format=json",
            )
            self.assertNotIn("notNull", url)

    def receipt(self, **changes):
        body = b'[{"not_a_model_value":"0.53"}]'
        d = dict(
            requested_url=annual_requests()[0][2],
            effective_url=annual_requests()[0][2],
            http_status=200,
            curl_exit=0,
            content_type="application/json;charset=UTF-8",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            retrieved_utc="2026-09-09T00:00:00+00:00",
        )
        d.update(changes)
        return body, d

    def test_valid_receipt(self):
        body, d = self.receipt()
        self.assertIsNone(validate_receipt(body, d, annual_requests()[0][2]))

    def test_rejects_drifted_body(self):
        body, d = self.receipt()
        with self.assertRaises(ValueError):
            validate_receipt(body + b" ", d, annual_requests()[0][2])

    def test_rejects_bad_transport_or_content_type(self):
        for changes in [
            dict(http_status=302),
            dict(http_status=404),
            dict(http_status=True),
            dict(curl_exit=60),
            dict(curl_exit=False),
            dict(effective_url="https://example.com/"),
            dict(content_type="text/html"),
            dict(bytes=0),
            dict(bytes=True),
            dict(sha256="f" * 64),
            dict(retrieved_utc="2010-01-01"),
            dict(retrieved_utc="2026-09-09T00:00:00"),
        ]:
            with self.subTest(changes=changes):
                body, d = self.receipt(**changes)
                with self.assertRaises(ValueError):
                    validate_receipt(body, d, annual_requests()[0][2])

    def test_requested_url_is_external_expected_identity(self):
        other = annual_requests()[1][2]
        body, d = self.receipt(requested_url=other, effective_url=other)
        with self.assertRaises(ValueError):
            validate_receipt(body, d, annual_requests()[0][2])

    def test_rejects_empty_or_not_bytes(self):
        for body in [b"", "not bytes", None]:
            with self.subTest(body=body):
                _, d = self.receipt()
                with self.assertRaises(ValueError):
                    validate_receipt(body, d, annual_requests()[0][2])

    def test_receipt_requires_exact_keys(self):
        body, d = self.receipt()
        for changed in [
            dict(d, later_overwrite="ignored"),
            {k: v for k, v in d.items() if k != "retrieved_utc"},
        ]:
            with self.assertRaises(ValueError):
                validate_receipt(body, changed, annual_requests()[0][2])


if __name__ == "__main__":
    unittest.main()
