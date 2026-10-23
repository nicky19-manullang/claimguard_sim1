import unittest
from pathlib import Path

import pandas as pd

from claimguard_adapter_v2 import (
    FILES,
    DataValidationError,
    load_v2,
    validate_and_normalize,
)


ROOT = Path(__file__).resolve().parents[1]


def load_tables():
    return {
        name: pd.read_csv(ROOT / "out" / "data" / filename, keep_default_na=False, dtype=str)
        for name, filename in FILES.items()
    }


class AdapterV2Tests(unittest.TestCase):
    def test_current_dataset_normalizes_one_row_per_claim(self):
        result = load_v2(str(ROOT / "out"))
        source = pd.read_csv(ROOT / "out" / "data" / "klaim.csv", keep_default_na=False)

        self.assertEqual(len(result.claims), len(source))
        self.assertTrue(result.claims["id_klaim"].is_unique)
        self.assertTrue(result.report["checks"]["claim_totals_reconcile"])
        self.assertTrue(result.report["checks"]["item_totals_reconcile"])
        self.assertFalse(
            {"hasil", "status", "perlu_klarifikasi", "is_anomaly"} & set(result.claims.columns)
        )
        self.assertIn("hasil", result.reviews.columns)

    def test_rejects_item_cost_mismatch(self):
        tables = load_tables()
        tables["items"].loc[0, "harga_satuan_idr"] = "999999999"

        with self.assertRaisesRegex(DataValidationError, "nilai item"):
            validate_and_normalize(tables)

    def test_rejects_orphan_child_claim(self):
        tables = load_tables()
        tables["diagnoses"].loc[0, "id_klaim"] = "UNKNOWN_CLAIM"

        with self.assertRaisesRegex(DataValidationError, "tidak terdapat"):
            validate_and_normalize(tables)

    def test_rejects_final_review_without_outcome(self):
        tables = load_tables()
        final_index = tables["reviews"].index[tables["reviews"]["status"].eq("final")][0]
        tables["reviews"].loc[final_index, "hasil"] = ""

        with self.assertRaisesRegex(DataValidationError, "status final"):
            validate_and_normalize(tables)


if __name__ == "__main__":
    unittest.main()
