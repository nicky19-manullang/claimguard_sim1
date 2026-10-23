import unittest

import pandas as pd

from claimguard_review_evidence_v2 import build_claim_comparison


COST_COLUMNS = [
    "cost_akomodasi_idr",
    "cost_alkes_idr",
    "cost_konsultasi_idr",
    "cost_obat_idr",
    "cost_penunjang_idr",
    "cost_tindakan_idr",
]


def _claim(
    claim_id,
    patient,
    submitted_at,
    split_time,
    diagnosis="E11.9",
    diagnosis_codes="E11.9",
    procedure_codes="",
    item_codes="OBT_1",
    total=1000,
):
    row = {
        "id_klaim": claim_id,
        "patient_token": patient,
        "submitted_at": submitted_at,
        "service_date": submitted_at,
        "service_type": "RJTL",
        "facility_id": "FKS_1",
        "ward_class": "-",
        "severity": "-",
        "length_of_stay_days": 0,
        "claim_total_idr": total,
        "diagnosis_primary": diagnosis,
        "diagnosis_codes": diagnosis_codes,
        "procedure_codes": procedure_codes,
        "item_codes": item_codes,
        "split_time": split_time,
        "split_facility": split_time,
    }
    row.update({column: 100 for column in COST_COLUMNS})
    return row


def _fixture():
    claims = [
        _claim(f"TR_{index}", f"PAT_{index}", f"2025-01-{index + 1:02}", "train")
        for index in range(10)
    ]
    claims.extend(
        [
            _claim(
                "PRIOR_SAME_PATIENT",
                "PAT_CURRENT",
                "2025-01-18",
                "train",
                procedure_codes="PROC_1",
                total=900,
            ),
            _claim(
                "PRIOR_SIMILAR",
                "PAT_OTHER",
                "2025-01-19",
                "train",
                procedure_codes="PROC_1",
                total=1100,
            ),
            _claim(
                "FUTURE_SIMILAR",
                "PAT_FUTURE",
                "2025-01-21",
                "test",
                procedure_codes="PROC_1",
            ),
            _claim(
                "SAME_DAY_SIMILAR",
                "PAT_SAME_DAY",
                "2025-01-20",
                "test",
                procedure_codes="PROC_1",
            ),
            _claim(
                "CURRENT",
                "PAT_CURRENT",
                "2025-01-20",
                "test",
                procedure_codes="PROC_1",
                total=2500,
            ),
        ]
    )
    feature = pd.DataFrame(
        [
            {
                "id_klaim": "CURRENT",
                "split_time": "test",
                "cost_reference_level": "service_diagnosis_class",
                "expected_cost_train_median_idr": 1000,
                "cost_ratio_train_median": 2.5,
            }
        ]
    )
    return pd.DataFrame(claims), feature


class ReviewEvidenceV2Tests(unittest.TestCase):
    def test_builds_cost_and_chronological_claim_comparisons(self):
        claims, features = _fixture()

        evidence = build_claim_comparison(claims, features, "CURRENT", "split_time")

        self.assertEqual(evidence["cost"]["group_label"], "Jenis layanan, diagnosis utama, dan kelas rawat")
        self.assertEqual(evidence["cost"]["reference_count"], 12)
        self.assertEqual(evidence["cost"]["train_median_total_idr"], 1000)
        self.assertEqual(evidence["cost"]["ratio_to_train_median"], 2.5)
        self.assertEqual(evidence["similar_claim"]["claim_id"], "PRIOR_SIMILAR")
        self.assertNotIn(
            evidence["similar_claim"]["claim_id"],
            {"FUTURE_SIMILAR", "SAME_DAY_SIMILAR"},
        )
        self.assertEqual(evidence["similar_claim"]["shared_codes"], ["E11.9", "OBT_1", "PROC_1"])
        self.assertEqual(evidence["repeat"]["prior_count_30d"], 1)
        self.assertEqual(evidence["repeat"]["previous_claim"]["claim_id"], "PRIOR_SAME_PATIENT")

    def test_excludes_later_partitions_from_prior_claim_context(self):
        claims, features = _fixture()
        claims.loc[claims["id_klaim"].eq("PRIOR_SIMILAR"), "split_time"] = "val"
        claims.loc[claims["id_klaim"].eq("PRIOR_SAME_PATIENT"), "split_time"] = "val"
        claims.loc[claims["id_klaim"].eq("CURRENT"), "split_time"] = "train"
        claims.loc[claims["id_klaim"].eq("CURRENT"), "split_facility"] = "train"
        features.loc[0, "split_time"] = "train"

        evidence = build_claim_comparison(claims, features, "CURRENT", "split_time")

        self.assertNotIn(
            evidence["similar_claim"]["claim_id"],
            {"PRIOR_SIMILAR", "PRIOR_SAME_PATIENT", "FUTURE_SIMILAR"},
        )
        self.assertIsNone(evidence["repeat"]["previous_claim"])
        self.assertEqual(evidence["repeat"]["prior_count_30d"], 0)

    def test_rejects_unknown_split_and_duplicate_ids(self):
        claims, features = _fixture()
        with self.assertRaisesRegex(ValueError, "split_column"):
            build_claim_comparison(claims, features, "CURRENT", "split_random")

        duplicate = pd.concat([claims, claims.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unik"):
            build_claim_comparison(duplicate, features, "CURRENT", "split_time")


if __name__ == "__main__":
    unittest.main()
