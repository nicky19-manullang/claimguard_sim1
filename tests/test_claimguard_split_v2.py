import unittest
from pathlib import Path

import pandas as pd

from claimguard_adapter_v2 import load_v2
from claimguard_split_v2 import create_splits


ROOT = Path(__file__).resolve().parents[1]


class SplitV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.claims = load_v2(str(ROOT / "out")).claims

    def test_creates_one_time_and_facility_split_per_claim(self):
        splits, manifest = create_splits(self.claims)

        self.assertEqual(len(splits), len(self.claims))
        self.assertTrue(splits["id_klaim"].is_unique)
        self.assertEqual(set(splits["split_time"]), {"train", "val", "test"})
        self.assertEqual(set(splits["split_facility"]), {"train", "val", "test"})
        self.assertEqual(manifest["split_facility"]["facility_disjoint"], True)

    def test_time_split_is_forward_ordered_and_never_splits_same_date(self):
        splits, _ = create_splits(self.claims)
        joined = self.claims[["id_klaim", "submitted_at"]].merge(
            splits[["id_klaim", "split_time"]], on="id_klaim", validate="one_to_one"
        )
        bounds = joined.groupby("split_time")["submitted_at"].agg(["min", "max"])

        self.assertLess(bounds.loc["train", "max"], bounds.loc["val", "min"])
        self.assertLess(bounds.loc["val", "max"], bounds.loc["test", "min"])
        self.assertEqual(joined.groupby("submitted_at")["split_time"].nunique().max(), 1)

    def test_facility_split_keeps_each_facility_in_one_partition(self):
        splits, _ = create_splits(self.claims)
        joined = self.claims[["id_klaim", "facility_id"]].merge(
            splits[["id_klaim", "split_facility"]], on="id_klaim", validate="one_to_one"
        )

        self.assertEqual(joined.groupby("facility_id")["split_facility"].nunique().max(), 1)

    def test_split_is_deterministic(self):
        first, _ = create_splits(self.claims)
        second, _ = create_splits(self.claims)

        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
