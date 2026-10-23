import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from claimguard_adapter_v2 import load_v2
from claimguard_features_v2 import create_features


ROOT = Path(__file__).resolve().parents[1]


class FeaturesV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        claims = load_v2(str(ROOT / "out")).claims
        cls.claims = claims.merge(
            pd.read_csv(
                ROOT / "out" / "prep_v2" / "splits_v2.csv",
                keep_default_na=False,
            ),
            on="id_klaim",
            validate="one_to_one",
        )

    def test_creates_feature_rows_without_ids_or_review_outcomes(self):
        features, manifest = create_features(self.claims, "split_time")

        self.assertEqual(len(features), len(self.claims))
        self.assertTrue(features["id_klaim"].is_unique)
        self.assertNotIn("patient_token", features.columns)
        self.assertNotIn("facility_id", features.columns)
        self.assertFalse(
            {"hasil", "status", "perlu_klarifikasi", "is_anomaly"} & set(features.columns)
        )
        self.assertGreater(manifest["training_reference_rows"], 0)

    def test_cost_references_are_fit_on_train_only(self):
        features, manifest = create_features(self.claims, "split_time")
        self.assertEqual(
            manifest["training_reference_rows"],
            int(self.claims["split_time"].eq("train").sum()),
        )
        numeric = features.select_dtypes(include="number")
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())
        self.assertTrue((features["split_time_days_since_patient_submission"] >= -1).all())

    def test_same_date_claims_do_not_become_each_others_history(self):
        claims = self.claims.copy()
        same_day = claims.iloc[:2].copy()
        same_day["submitted_at"] = "2025-02-01"
        same_day["split_time"] = "train"
        same_day["patient_token"] = "PAT_SHARED_TEST"
        claims = pd.concat([same_day, claims.iloc[2:]], ignore_index=True)

        features, _ = create_features(claims, "split_time")
        counts = features.set_index("id_klaim")["split_time_prior_patient_claims"]

        self.assertEqual(counts[same_day.iloc[0]["id_klaim"]], 0)
        self.assertEqual(counts[same_day.iloc[1]["id_klaim"]], 0)

    def test_rejects_unknown_split_column(self):
        with self.assertRaisesRegex(ValueError, "split_column"):
            create_features(self.claims, "split_random")


if __name__ == "__main__":
    unittest.main()
