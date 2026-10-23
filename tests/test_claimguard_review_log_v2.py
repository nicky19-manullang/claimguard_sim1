import tempfile
import unittest
from pathlib import Path

import pandas as pd

from claimguard_review_log_v2 import (
    EVIDENCE_ITEMS,
    FOLLOW_UP_ACTIONS,
    LOG_COLUMNS,
    RECOMMENDATION_FEEDBACK,
    REVIEW_OUTCOMES,
    REVIEWER_ROLES,
    read_review_log,
    record_review_action,
)


def _record(path, claim_id="KLM_001", notes="Rincian biaya telah dicocokkan."):
    return record_review_action(
        path=path,
        claim_id=claim_id,
        split_scheme="time",
        split_name="test",
        reviewer_role=REVIEWER_ROLES[0],
        evidence_reviewed=[EVIDENCE_ITEMS[0], EVIDENCE_ITEMS[1]],
        review_outcome=REVIEW_OUTCOMES[0],
        follow_up_action=FOLLOW_UP_ACTIONS[1],
        review_notes=notes,
        recommendation_feedback=RECOMMENDATION_FEEDBACK[0],
    )


class ReviewLogV2Tests(unittest.TestCase):
    def test_records_append_only_structured_demo_reviews(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review_records.csv"
            first = _record(path)
            second = _record(path, "KLM_002")
            log = read_review_log(path)

        self.assertEqual(list(log.columns), LOG_COLUMNS)
        self.assertEqual(len(log), 2)
        self.assertEqual(first["demo_only"], "true")
        self.assertEqual(first["schema_version"], "2")
        self.assertEqual(second["id_klaim"], "KLM_002")
        self.assertEqual(log["evidence_reviewed"].iloc[0], "Ringkasan pelayanan|Rincian biaya")

    def test_missing_log_returns_empty_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log = read_review_log(Path(temp_dir) / "missing.csv")

        self.assertTrue(log.empty)
        self.assertEqual(list(log.columns), LOG_COLUMNS)

    def test_rejects_invalid_claim_split_or_review_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review_records.csv"
            args = {
                "path": path,
                "claim_id": "KLM_001",
                "split_scheme": "time",
                "split_name": "test",
                "reviewer_role": REVIEWER_ROLES[0],
                "evidence_reviewed": [EVIDENCE_ITEMS[0]],
                "review_outcome": REVIEW_OUTCOMES[0],
                "follow_up_action": FOLLOW_UP_ACTIONS[0],
                "review_notes": "Bukti telah ditinjau.",
                "recommendation_feedback": RECOMMENDATION_FEEDBACK[0],
            }
            invalid_values = [
                ("claim_id", ""),
                ("split_scheme", "other"),
                ("split_name", "unknown"),
                ("reviewer_role", "Pihak tidak dikenal"),
                ("evidence_reviewed", []),
                ("evidence_reviewed", ["Dokumen tidak dikenal"]),
                ("review_outcome", "Fraud"),
                ("follow_up_action", "Tolak klaim"),
                ("review_notes", "singkat"),
                ("recommendation_feedback", "Akurat"),
            ]
            for key, value in invalid_values:
                with self.subTest(field=key, value=value):
                    with self.assertRaises(ValueError):
                        record_review_action(**{**args, key: value})
            self.assertFalse(path.exists())

    def test_rejects_legacy_log_format_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review_records.csv"
            legacy = pd.DataFrame(
                [
                    {
                        "recorded_at_utc": "now",
                        "id_klaim": "KLM_001",
                        "split_scheme": "time",
                        "split_name": "test",
                        "review_action": "Perlu klarifikasi (demo)",
                        "demo_only": "true",
                    }
                ]
            )
            legacy.to_csv(path, index=False)
            original = path.read_bytes()

            with self.assertRaises(ValueError):
                _record(path, "KLM_002")

            self.assertEqual(path.read_bytes(), original)

    def test_rejects_non_demo_rows_and_unknown_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review_records.csv"
            _record(path)
            log = pd.read_csv(path, keep_default_na=False, dtype=str)

            log.loc[0, "demo_only"] = "false"
            log.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_review_log(path)

            log.loc[0, "demo_only"] = "true"
            log.loc[0, "schema_version"] = "99"
            log.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_review_log(path)


if __name__ == "__main__":
    unittest.main()
