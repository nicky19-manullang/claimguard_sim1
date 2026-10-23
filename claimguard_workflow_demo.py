#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store one shared, demo-only human review and clarification workflow."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


WORKFLOW_COLUMNS = [
    "recorded_at_utc",
    "schema_version",
    "analysis_source",
    "analysis_path",
    "claim_id",
    "split_name",
    "actor",
    "action",
    "reviewer_role",
    "evidence_reviewed",
    "review_outcome",
    "recommendation_feedback",
    "status_after_action",
    "notes",
    "demo_only",
]
SCHEMA_VERSION = "2"
LEGACY_SCHEMA_VERSION = "1"
MIN_NOTES_LENGTH = 10
MAX_NOTES_LENGTH = 2000
INITIAL_STATUS = "Menunggu pemeriksaan petugas"
WAITING_FOR_FACILITY_STATUS = "Menunggu tanggapan faskes"
WAITING_FOR_STAFF_STATUS = "Menunggu pemeriksaan ulang petugas"
OFFICIAL_PROCESS_STATUS = "Diteruskan ke proses resmi"

REVIEWER_ROLES = (
    "Verifikator administrasi",
    "Verifikator klinis/koding",
    "Peran lain",
    "Tidak dicatat",
)
EVIDENCE_ITEMS = (
    "Ringkasan pelayanan",
    "Rincian biaya",
    "Diagnosis dan kode",
    "Tindakan dan item",
    "Riwayat klaim",
    "Dokumen pendukung",
)
RECOMMENDATION_FEEDBACK = (
    "Membantu mengarahkan pemeriksaan",
    "Tidak membantu mengarahkan pemeriksaan",
    "Belum dapat dinilai",
)

WORKFLOW_ACTIONS = {
    INITIAL_STATUS: (
        {
            "actor": "Petugas",
            "action": "Tidak perlu klarifikasi; lanjut ke proses resmi",
            "review_outcome": "Tidak ditemukan ketidaksesuaian pada dokumen yang ditinjau",
            "status": OFFICIAL_PROCESS_STATUS,
        },
        {
            "actor": "Petugas",
            "action": "Perlu klarifikasi kepada faskes",
            "review_outcome": "Informasi belum cukup - klarifikasi diperlukan",
            "status": WAITING_FOR_FACILITY_STATUS,
        },
    ),
    WAITING_FOR_FACILITY_STATUS: (
        {
            "actor": "Faskes (simulasi)",
            "action": "Melengkapi dokumen yang diminta",
            "review_outcome": "",
            "status": WAITING_FOR_STAFF_STATUS,
        },
        {
            "actor": "Faskes (simulasi)",
            "action": "Memberikan penjelasan",
            "review_outcome": "",
            "status": WAITING_FOR_STAFF_STATUS,
        },
        {
            "actor": "Faskes (simulasi)",
            "action": "Mencatat klaim yang telah dikoreksi",
            "review_outcome": "",
            "status": WAITING_FOR_STAFF_STATUS,
        },
    ),
    WAITING_FOR_STAFF_STATUS: (
        {
            "actor": "Petugas",
            "action": "Tanggapan cukup; lanjut ke proses resmi",
            "review_outcome": "Tidak ditemukan ketidaksesuaian pada dokumen yang ditinjau",
            "status": OFFICIAL_PROCESS_STATUS,
        },
        {
            "actor": "Petugas",
            "action": "Perlu klarifikasi tambahan kepada faskes",
            "review_outcome": "Informasi belum cukup - klarifikasi diperlukan",
            "status": WAITING_FOR_FACILITY_STATUS,
        },
    ),
    OFFICIAL_PROCESS_STATUS: (),
}

LEGACY_ACTIONS = {
    "Tidak perlu klarifikasi; lanjut ke proses resmi":
        "Tidak perlu klarifikasi; lanjut ke proses resmi",
    "Minta klarifikasi kepada faskes": "Perlu klarifikasi kepada faskes",
    "Melengkapi dokumen yang diminta": "Melengkapi dokumen yang diminta",
    "Memberikan penjelasan": "Memberikan penjelasan",
    "Mencatat klaim yang telah dikoreksi": "Mencatat klaim yang telah dikoreksi",
    "Klarifikasi cukup; lanjut ke proses resmi": "Tanggapan cukup; lanjut ke proses resmi",
    "Minta klarifikasi tambahan kepada faskes": "Perlu klarifikasi tambahan kepada faskes",
}


def _workflow_key(record):
    return (
        record["analysis_source"],
        record["analysis_path"],
        record["claim_id"],
        record["split_name"],
    )


def _validate_event(
    analysis_source,
    analysis_path,
    claim_id,
    split_name,
    actor,
    action,
    reviewer_role,
    evidence_reviewed,
    review_outcome,
    recommendation_feedback,
    notes,
):
    if analysis_source not in {"legacy_ml", "v2"}:
        raise ValueError("Sumber analisis harus legacy_ml atau v2.")
    if not str(analysis_path).strip():
        raise ValueError("Jalur analisis tidak boleh kosong.")
    if not str(claim_id).strip():
        raise ValueError("Nomor klaim tidak boleh kosong.")
    if split_name not in {"train", "val", "test"}:
        raise ValueError("Kelompok data tidak dikenal.")
    if not isinstance(notes, str) or not MIN_NOTES_LENGTH <= len(notes.strip()) <= MAX_NOTES_LENGTH:
        raise ValueError(
            f"Catatan harus berisi {MIN_NOTES_LENGTH}-{MAX_NOTES_LENGTH} karakter."
        )

    evidence = list(evidence_reviewed)
    if actor == "Petugas":
        if reviewer_role not in REVIEWER_ROLES:
            raise ValueError("Peran pemeriksa tidak dikenal.")
        if not evidence or any(item not in EVIDENCE_ITEMS for item in evidence):
            raise ValueError("Pilih setidaknya satu bukti yang telah ditinjau.")
        if len(set(evidence)) != len(evidence):
            raise ValueError("Jenis bukti yang sama tidak boleh dicatat lebih dari satu kali.")
        if recommendation_feedback not in RECOMMENDATION_FEEDBACK:
            raise ValueError("Penilaian manfaat rekomendasi tidak dikenal.")
    elif actor == "Faskes (simulasi)":
        if reviewer_role or evidence or review_outcome or recommendation_feedback:
            raise ValueError("Tanggapan faskes simulasi tidak boleh mengisi hasil review petugas.")
    else:
        raise ValueError("Pihak yang mencatat langkah tidak dikenal.")
    return str(notes).strip(), evidence


def _transition(status, actor, action):
    for option in WORKFLOW_ACTIONS.get(status, ()):
        if option["actor"] == actor and option["action"] == action:
            return option["status"]
    raise ValueError("Tindakan tidak sesuai dengan tahap alur pemeriksaan saat ini.")


def _legacy_log(path):
    path = Path(path)
    if not path.exists():
        return []
    log = pd.read_csv(path, keep_default_na=False, dtype=str)
    expected = {
        "recorded_at_utc", "schema_version", "analysis_source", "analysis_path",
        "claim_id", "split_name", "actor", "action", "status_after_action",
        "notes", "demo_only",
    }
    if not expected.issubset(log.columns):
        raise ValueError("Log alur lama memiliki skema yang tidak didukung.")
    if not log["schema_version"].eq(LEGACY_SCHEMA_VERSION).all():
        raise ValueError("Log alur lama memuat versi skema yang tidak didukung.")
    if not log["demo_only"].str.casefold().eq("true").all():
        raise ValueError("Log alur lama hanya boleh berisi catatan demonstrasi.")

    records = []
    for old in log.to_dict(orient="records"):
        action = LEGACY_ACTIONS.get(old["action"])
        if action is None:
            raise ValueError("Log alur lama memuat tindakan yang tidak dikenal.")
        transition_options = [
            option
            for status_options in WORKFLOW_ACTIONS.values()
            for option in status_options
            if option["action"] == action and option["actor"] == old["actor"]
        ]
        if len(transition_options) != 1:
            raise ValueError("Langkah pada log alur lama tidak dapat dipetakan.")
        option = transition_options[0]
        if old["status_after_action"] != option["status"]:
            raise ValueError("Status pada log alur lama tidak cocok dengan langkah yang dicatat.")
        records.append(
            {
                "_legacy_imported": True,
                "recorded_at_utc": old["recorded_at_utc"],
                "schema_version": SCHEMA_VERSION,
                "analysis_source": old["analysis_source"],
                "analysis_path": old["analysis_path"],
                "claim_id": old["claim_id"],
                "split_name": old["split_name"],
                "actor": old["actor"],
                "action": action,
                "reviewer_role": "",
                "evidence_reviewed": "",
                "review_outcome": option["review_outcome"],
                "recommendation_feedback": "",
                "status_after_action": option["status"],
                "notes": old["notes"],
                "demo_only": "true",
            }
        )
    return records


def read_workflow_log(path, legacy_path=None):
    path = Path(path)
    records = []
    if legacy_path is not None and Path(legacy_path) != path:
        records.extend(_legacy_log(legacy_path))
    if path.exists():
        log = pd.read_csv(path, keep_default_na=False, dtype=str)
        missing = set(WORKFLOW_COLUMNS) - set(log.columns)
        if missing:
            raise ValueError(
                f"Log alur pemeriksaan kehilangan kolom wajib: {', '.join(sorted(missing))}"
            )
        current_records = log[WORKFLOW_COLUMNS].to_dict(orient="records")
        for record in current_records:
            record["_legacy_imported"] = False
        records.extend(current_records)
    if not records:
        return pd.DataFrame(columns=WORKFLOW_COLUMNS)

    log = pd.DataFrame(records)
    if not log["schema_version"].eq(SCHEMA_VERSION).all():
        raise ValueError("Log alur pemeriksaan memiliki versi skema yang tidak didukung.")
    if not log["demo_only"].str.casefold().eq("true").all():
        raise ValueError("Log alur pemeriksaan hanya boleh berisi catatan demonstrasi.")

    current_status = {}
    for record in log.to_dict(orient="records"):
        key = _workflow_key(record)
        status_before = current_status.get(key, INITIAL_STATUS)
        expected_status = _transition(status_before, record["actor"], record["action"])
        if record["status_after_action"] != expected_status:
            raise ValueError("Urutan status pada log alur pemeriksaan tidak konsisten.")
        notes = record["notes"].strip()
        if not MIN_NOTES_LENGTH <= len(notes) <= MAX_NOTES_LENGTH:
            raise ValueError("Log alur pemeriksaan memiliki catatan dengan panjang tidak valid.")
        if not record.get("_legacy_imported", False):
            _validate_event(
                record["analysis_source"],
                record["analysis_path"],
                record["claim_id"],
                record["split_name"],
                record["actor"],
                record["action"],
                record["reviewer_role"],
                record["evidence_reviewed"].split("|") if record["evidence_reviewed"] else [],
                record["review_outcome"],
                record["recommendation_feedback"],
                notes,
            )
        if record["actor"] == "Petugas" and record["reviewer_role"] and not record.get("_legacy_imported", False):
            action_options = [
                option for option in WORKFLOW_ACTIONS[status_before]
                if option["actor"] == "Petugas" and option["action"] == record["action"]
            ]
            if len(action_options) != 1 or record["review_outcome"] != action_options[0]["review_outcome"]:
                raise ValueError("Hasil pemeriksaan pada log tidak cocok dengan tindak lanjut.")
        if record["actor"] == "Faskes (simulasi)" and record["reviewer_role"]:
            raise ValueError("Log tanggapan faskes tidak boleh berisi peran pemeriksa.")
        current_status[key] = expected_status
    return log[WORKFLOW_COLUMNS]


def get_workflow_status(log, analysis_source, analysis_path, claim_id, split_name):
    matching = log.loc[
        log["analysis_source"].eq(analysis_source)
        & log["analysis_path"].eq(analysis_path)
        & log["claim_id"].eq(str(claim_id))
        & log["split_name"].eq(split_name)
    ]
    if matching.empty:
        return INITIAL_STATUS
    return str(matching.iloc[-1]["status_after_action"])


def record_workflow_action(
    path,
    analysis_source,
    analysis_path,
    claim_id,
    split_name,
    actor,
    action,
    notes,
    *,
    reviewer_role="",
    evidence_reviewed=(),
    review_outcome="",
    recommendation_feedback="",
    legacy_path=None,
):
    notes, evidence = _validate_event(
        analysis_source,
        analysis_path,
        claim_id,
        split_name,
        actor,
        action,
        reviewer_role,
        evidence_reviewed,
        review_outcome,
        recommendation_feedback,
        notes,
    )
    path = Path(path)
    existing = read_workflow_log(path, legacy_path)
    status_before = get_workflow_status(
        existing,
        analysis_source,
        analysis_path,
        str(claim_id),
        split_name,
    )
    status_after = _transition(status_before, actor, action)
    if actor == "Petugas":
        selected_option = next(
            option
            for option in WORKFLOW_ACTIONS[status_before]
            if option["actor"] == actor and option["action"] == action
        )
        if review_outcome != selected_option["review_outcome"]:
            raise ValueError("Hasil pemeriksaan tidak sesuai dengan tindak lanjut yang dipilih.")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "schema_version": SCHEMA_VERSION,
                "analysis_source": analysis_source,
                "analysis_path": analysis_path,
                "claim_id": str(claim_id),
                "split_name": split_name,
                "actor": actor,
                "action": action,
                "reviewer_role": reviewer_role,
                "evidence_reviewed": "|".join(evidence),
                "review_outcome": review_outcome,
                "recommendation_feedback": recommendation_feedback,
                "status_after_action": status_after,
                "notes": notes,
                "demo_only": "true",
            }
        ],
        columns=WORKFLOW_COLUMNS,
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
    return row.iloc[0].to_dict()
