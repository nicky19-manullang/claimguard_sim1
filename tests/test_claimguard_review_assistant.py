import unittest

from claimguard_review_assistant import build_review_assistant_content


class ReviewAssistantTests(unittest.TestCase):
    def setUp(self):
        self.claim = {"id_klaim": "KLM_100"}
        self.analysis = {
            "top_driver": "Biaya relatif tinggi terhadap referensi train",
            "evidence": "rasio biaya terhadap median train=1.50",
        }
        self.comparisons = {
            "cost": {
                "current_total_idr": 1_500_000,
                "train_median_total_idr": 1_000_000,
                "group_label": "Jenis layanan",
                "reference_count": 20,
                "ratio_to_train_median": 1.5,
            },
            "repeat": {
                "prior_count_30d": 2,
                "previous_claim": {
                    "claim_id": "KLM_090",
                    "submitted_at": "2025-03-01",
                },
            },
            "similar_claim": {
                "claim_id": "KLM_080",
                "submitted_at": "2025-02-20",
                "similarity": 0.75,
            },
            "severity": {
                "current_stay_days": 5,
                "train_median_stay_days": 3.5,
                "group_label": "Jenis layanan dan diagnosis",
            },
        }

    def test_builds_traceable_findings_and_editable_clarification_draft(self):
        content = build_review_assistant_content(self.claim, self.analysis, self.comparisons)

        self.assertEqual(len(content["findings"]), 5)
        self.assertIn("Rp 1.500.000", content["findings"][1])
        self.assertIn("KLM_090", content["findings"][2])
        self.assertIn("75%", content["findings"][3])
        self.assertEqual(
            content["suggested_evidence"],
            (
                "Ringkasan pelayanan",
                "Rincian biaya",
                "Riwayat klaim",
                "Diagnosis dan kode",
                "Tindakan dan item",
            ),
        )
        self.assertIn("bukan kesimpulan kesalahan", content["clarification_request"])
        self.assertIn("KLM_080", content["clarification_request"])
        self.assertLessEqual(len(content["clarification_request"]), 2000)

    def test_does_not_invent_prior_claim_or_similarity_findings(self):
        comparisons = {
            **self.comparisons,
            "repeat": {"prior_count_30d": 0, "previous_claim": None},
            "similar_claim": None,
            "severity": None,
        }

        content = build_review_assistant_content(self.claim, self.analysis, comparisons)

        self.assertEqual(len(content["findings"]), 2)
        self.assertEqual(content["suggested_evidence"], ("Ringkasan pelayanan", "Rincian biaya"))
        self.assertNotIn("KLM_090", content["clarification_request"])
        self.assertNotIn("KLM_080", content["clarification_request"])


if __name__ == "__main__":
    unittest.main()
