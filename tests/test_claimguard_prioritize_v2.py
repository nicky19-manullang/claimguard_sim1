import unittest
from pathlib import Path

import pandas as pd

from claimguard_adapter_v2 import load_v2
from claimguard_prioritize_v2 import EmpiricalTail, score_claims
from claimguard_split_v2 import create_splits
from claimguard_features_v2 import create_features


ROOT = Path(__file__).resolve().parents[1]


class EmpiricalTailTests(unittest.TestCase):
    def test_larger_values_get_higher_rank_percentiles(self):
        tail = EmpiricalTail().fit([1, 2, 3, 4])
        scores = tail.transform([1, 2, 3, 4])

        self.assertEqual(list(scores), [12.5, 37.5, 62.5, 87.5])

    def test_ties_receive_midrank_not_maximum_percentile(self):
        tail = EmpiricalTail().fit([0, 0, 0, 1])
        scores = tail.transform([0, 1])

        self.assertEqual(list(scores), [37.5, 87.5])


class PrioritizeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        claims = load_v2(str(ROOT / "out")).claims
        splits, _ = create_splits(claims)
        cls.claims = claims.merge(splits, on="id_klaim", validate="one_to_one")
        cls.time_features, _ = create_features(cls.claims, "split_time")
        cls.facility_features, _ = create_features(cls.claims, "split_facility")

    def test_generates_explainable_priority_without_review_outcomes(self):
        result = score_claims(self.time_features, self.claims, "split_time", review_budget=0.05)

        self.assertEqual(len(result), len(self.claims))
        self.assertTrue(result["id_klaim"].is_unique)
        self.assertTrue(result["evidence"].str.len().gt(0).all())
        self.assertTrue(result["priority_score"].between(0, 100).all())
        self.assertFalse(
            {"hasil", "status", "perlu_klarifikasi", "is_anomaly"} & set(result.columns)
        )

    def test_review_budget_selects_expected_count_per_split(self):
        result = score_claims(self.time_features, self.claims, "split_time", review_budget=0.05)
        expected = result.groupby("split_time").size().map(lambda size: max(1, int(size * 0.05 + 0.999999)))
        actual = result.groupby("split_time")["review_recommended"].sum()

        pd.testing.assert_series_equal(actual.astype(int), expected.astype(int), check_names=False)

    def test_supports_facility_split_feature_schema(self):
        result = score_claims(
            self.facility_features, self.claims, "split_facility", review_budget=0.05
        )

        self.assertEqual(len(result), len(self.claims))
        self.assertTrue(result["review_recommended"].any())
        self.assertTrue(set(result["top_driver"]) <= {
            "Pola klaim pasien berulang",
            "Kemiripan dengan klaim pasien lain",
            "Biaya relatif tinggi terhadap referensi train",
            "Kombinasi severity dan lama rawat perlu ditinjau",
        })

    def test_rejects_invalid_budget(self):
        with self.assertRaisesRegex(ValueError, "review_budget"):
            score_claims(self.time_features, self.claims, "split_time", review_budget=0)

    def test_severity_signal_is_not_scored_for_outpatient_claims(self):
        result = score_claims(self.time_features, self.claims, "split_time", review_budget=0.05)
        outpatient_ids = set(self.claims.loc[
            self.claims["service_type"].eq("RJTL"), "id_klaim"
        ])
        scores = result.set_index("id_klaim")["severity_los_unusual_rank_percentile"]

        self.assertTrue(scores.loc[list(outpatient_ids)].eq(0).all())


if __name__ == "__main__":
    unittest.main()
