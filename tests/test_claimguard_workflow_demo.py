import tempfile
import unittest
from pathlib import Path

from claimguard_workflow_demo import (
    INITIAL_STATUS,
    EVIDENCE_ITEMS,
    REVIEWER_ROLES,
    RECOMMENDATION_FEEDBACK,
    WORKFLOW_COLUMNS,
    get_workflow_status,
    read_workflow_log,
    record_workflow_action,
)


def _record(path, source="v2", analysis_path="time", claim_id="KLM_001", split="test", actor="Petugas",
            action="Perlu klarifikasi kepada faskes", notes="Mohon lampirkan rincian yang belum tersedia."):
    staff = actor == "Petugas"
    return record_workflow_action(
        path,
        source,
        analysis_path,
        claim_id,
        split,
        actor,
        action,
        notes,
        reviewer_role=REVIEWER_ROLES[0] if staff else "",
        evidence_reviewed=[EVIDENCE_ITEMS[0], EVIDENCE_ITEMS[1]] if staff else [],
        review_outcome=(
            "Informasi belum cukup - klarifikasi diperlukan"
            if staff and action in {
                "Perlu klarifikasi kepada faskes",
                "Perlu klarifikasi tambahan kepada faskes",
            }
            else "Tidak ditemukan ketidaksesuaian pada dokumen yang ditinjau"
            if staff
            else ""
        ),
        recommendation_feedback=RECOMMENDATION_FEEDBACK[0] if staff else "",
    )


class WorkflowDemoTests(unittest.TestCase):
    def test_clarification_cycle_returns_to_staff_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            first = _record(path)
            second = _record(
                path,
                actor="Faskes (simulasi)",
                action="Melengkapi dokumen yang diminta",
                notes="Dokumen pendukung telah dilengkapi untuk ditinjau.",
            )
            third = _record(
                path,
                actor="Petugas",
                action="Tanggapan cukup; lanjut ke proses resmi",
                notes="Dokumen tambahan telah diperiksa dan dinilai cukup.",
            )
            log = read_workflow_log(path)

        self.assertEqual(first["status_after_action"], "Menunggu tanggapan faskes")
        self.assertEqual(second["status_after_action"], "Menunggu pemeriksaan ulang petugas")
        self.assertEqual(third["status_after_action"], "Diteruskan ke proses resmi")
        self.assertEqual(len(log), 3)
        self.assertEqual(
            get_workflow_status(log, "v2", "time", "KLM_001", "test"),
            "Diteruskan ke proses resmi",
        )
        self.assertEqual(list(log.columns), WORKFLOW_COLUMNS)

    def test_no_clarification_continues_official_process_without_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            event = _record(
                path,
                actor="Petugas",
                action="Tidak perlu klarifikasi; lanjut ke proses resmi",
                notes="Dokumen yang ditinjau sudah cukup untuk tahap ini.",
            )
            log = read_workflow_log(path)

        self.assertEqual(event["status_after_action"], "Diteruskan ke proses resmi")
        self.assertNotIn("Disetujui", event["status_after_action"])
        self.assertEqual(get_workflow_status(log, "v2", "time", "KLM_001", "test"),
                         "Diteruskan ke proses resmi")

    def test_rejects_actions_that_skip_the_current_workflow_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            with self.assertRaisesRegex(ValueError, "tidak sesuai"):
                _record(
                    path,
                    actor="Faskes (simulasi)",
                    action="Memberikan penjelasan",
                    notes="Penjelasan faskes untuk petugas pemeriksa.",
                )
            self.assertFalse(path.exists())

    def test_allows_additional_clarification_after_facility_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            _record(path)
            _record(
                path,
                actor="Faskes (simulasi)",
                action="Memberikan penjelasan",
                notes="Penjelasan diberikan untuk dokumen yang diminta petugas.",
            )
            event = _record(
                path,
                actor="Petugas",
                action="Perlu klarifikasi tambahan kepada faskes",
                notes="Masih diperlukan penjelasan tambahan terkait dokumen.",
            )

        self.assertEqual(event["status_after_action"], "Menunggu tanggapan faskes")

    def test_facility_can_record_a_corrected_claim_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            _record(path)
            event = _record(
                path,
                actor="Faskes (simulasi)",
                action="Mencatat klaim yang telah dikoreksi",
                notes="Faskes mencatat telah mengoreksi rincian klaim untuk diperiksa.",
            )

        self.assertEqual(event["status_after_action"], "Menunggu pemeriksaan ulang petugas")

    def test_review_outcome_and_action_must_agree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            with self.assertRaisesRegex(ValueError, "tidak sesuai"):
                record_workflow_action(
                    path,
                    "v2",
                    "time",
                    "KLM_001",
                    "test",
                    "Petugas",
                    "Perlu klarifikasi kepada faskes",
                    "Dokumen yang ditinjau belum menjawab pertanyaan.",
                    reviewer_role=REVIEWER_ROLES[0],
                    evidence_reviewed=[EVIDENCE_ITEMS[0]],
                    review_outcome="Tidak ditemukan ketidaksesuaian pada dokumen yang ditinjau",
                    recommendation_feedback=RECOMMENDATION_FEEDBACK[0],
                )
            self.assertFalse(path.exists())

    def test_reads_legacy_workflow_log_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_path = Path(temp_dir) / "workflow_records_demo.csv"
            new_path = Path(temp_dir) / "workflow_records_demo_v2.csv"
            old_path.write_text(
                "recorded_at_utc,schema_version,analysis_source,analysis_path,claim_id,"
                "split_name,actor,action,status_after_action,notes,demo_only\n"
                "2026-10-22T01:59:28+00:00,1,legacy_ml,waktu_asof,CLM-0001,test,"
                "Petugas,Minta klarifikasi kepada faskes,Menunggu tanggapan faskes,"
                "Mohon lampirkan rincian yang belum tersedia.,true\n",
                encoding="utf-8",
            )
            original = old_path.read_bytes()
            log = read_workflow_log(new_path, old_path)

            self.assertEqual(
                get_workflow_status(log, "legacy_ml", "waktu_asof", "CLM-0001", "test"),
                "Menunggu tanggapan faskes",
            )
            response = record_workflow_action(
                new_path,
                "legacy_ml",
                "waktu_asof",
                "CLM-0001",
                "test",
                "Faskes (simulasi)",
                "Melengkapi dokumen yang diminta",
                "Dokumen pendukung telah dilengkapi untuk ditinjau.",
                legacy_path=old_path,
            )
            self.assertEqual(response["status_after_action"], "Menunggu pemeriksaan ulang petugas")
            self.assertEqual(old_path.read_bytes(), original)

    def test_invalid_note_is_rejected_without_creating_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow.csv"
            with self.assertRaisesRegex(ValueError, "Catatan harus"):
                _record(path, notes="kurang")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
