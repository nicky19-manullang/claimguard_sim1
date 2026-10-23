#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store structured, demo-only human review records for synthetic claims."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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
REVIEW_OUTCOMES = (
    "Belum dapat disimpulkan",
    "Tidak ditemukan ketidaksesuaian pada dokumen yang ditinjau",
    "Ketidaksesuaian administratif teridentifikasi",
    "Ketidaksesuaian klinis/koding perlu ditinjau ahli",
    "Informasi belum cukup - klarifikasi diperlukan",
    "Pemeriksaan lanjutan diperlukan",
    "Hasil lain",
)
FOLLOW_UP_ACTIONS = (
    "Tidak ada tindak lanjut dalam simulasi",
    "Klarifikasi",
    "Tinjauan ahli",
    "Pemeriksaan lanjutan",
    "Tindak lanjut lain",
)
RECOMMENDATION_FEEDBACK = (
    "Membantu mengarahkan pemeriksaan",
    "Tidak membantu mengarahkan pemeriksaan",
    "Belum dapat dinilai",
)
LOG_COLUMNS = [
    "recorded_at_utc",
    "schema_version",
    "id_klaim",
    "split_scheme",
    "split_name",
    "reviewer_role",
    "evidence_reviewed",
    "review_outcome",
    "follow_up_action",
    "review_notes",
    "recommendation_feedback",
    "demo_only",
]
SCHEMA_VERSION = "2"
MAX_NOTES_LENGTH = 2000
MIN_NOTES_LENGTH = 10


def _validate_record(
    claim_id,
    split_scheme,
    split_name,
    reviewer_role,
    evidence_reviewed,
    review_outcome,
    follow_up_action,
    review_notes,
    recommendation_feedback,
):
    if not str(claim_id).strip():
        raise ValueError("id_klaim tidak boleh kosong.")
    if split_scheme not in {"time", "facility"}:
        raise ValueError("split_scheme harus time atau facility.")
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name harus train, val, atau test.")
    if reviewer_role not in REVIEWER_ROLES:
        raise ValueError("Peran pemeriksa tidak dikenal.")
    evidence = list(evidence_reviewed)
    if not evidence or any(item not in EVIDENCE_ITEMS for item in evidence):
        raise ValueError("Pilih setidaknya satu jenis bukti yang telah ditinjau.")
    if len(set(evidence)) != len(evidence):
        raise ValueError("Jenis bukti yang sama tidak boleh dicatat lebih dari satu kali.")
    if review_outcome not in REVIEW_OUTCOMES:
        raise ValueError("Hasil pemeriksaan tidak dikenal.")
    if follow_up_action not in FOLLOW_UP_ACTIONS:
        raise ValueError("Tindak lanjut tidak dikenal.")
    notes = str(review_notes).strip()
    if not MIN_NOTES_LENGTH <= len(notes) <= MAX_NOTES_LENGTH:
        raise ValueError(
            f"Catatan alasan harus berisi {MIN_NOTES_LENGTH}-{MAX_NOTES_LENGTH} karakter."
        )
    if recommendation_feedback not in RECOMMENDATION_FEEDBACK:
        raise ValueError("Penilaian manfaat rekomendasi tidak dikenal.")
    return evidence, notes


def read_review_log(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(path, keep_default_na=False)
    missing = set(LOG_COLUMNS) - set(log.columns)
    if missing:
        raise ValueError(
            f"Log pemeriksaan memiliki kolom wajib yang hilang: {', '.join(sorted(missing))}"
        )
    if not log["schema_version"].astype(str).eq(SCHEMA_VERSION).all():
        raise ValueError("Log pemeriksaan memuat versi skema yang tidak didukung.")
    if not log["demo_only"].astype(str).str.casefold().eq("true").all():
        raise ValueError("Log berisi baris yang tidak ditandai sebagai catatan simulasi.")
    if not log["reviewer_role"].isin(REVIEWER_ROLES).all():
        raise ValueError("Log memuat peran pemeriksa yang tidak dikenal.")
    if not log["review_outcome"].isin(REVIEW_OUTCOMES).all():
        raise ValueError("Log memuat hasil pemeriksaan yang tidak dikenal.")
    if not log["follow_up_action"].isin(FOLLOW_UP_ACTIONS).all():
        raise ValueError("Log memuat tindak lanjut yang tidak dikenal.")
    if not log["recommendation_feedback"].isin(RECOMMENDATION_FEEDBACK).all():
        raise ValueError("Log memuat penilaian rekomendasi yang tidak dikenal.")
    if (
        log["review_notes"].str.len().lt(MIN_NOTES_LENGTH)
        | log["review_notes"].str.len().gt(MAX_NOTES_LENGTH)
    ).any():
        raise ValueError("Log memuat catatan alasan di luar batas panjang yang diizinkan.")
    for evidence_text in log["evidence_reviewed"]:
        evidence = evidence_text.split("|") if evidence_text else []
        if not evidence or any(item not in EVIDENCE_ITEMS for item in evidence):
            raise ValueError("Log memuat jenis bukti yang tidak dikenal atau kosong.")
    return log[LOG_COLUMNS]


def record_review_action(
    path,
    claim_id,
    split_scheme,
    split_name,
    reviewer_role,
    evidence_reviewed,
    review_outcome,
    follow_up_action,
    review_notes,
    recommendation_feedback,
):
    evidence, notes = _validate_record(
        claim_id,
        split_scheme,
        split_name,
        reviewer_role,
        evidence_reviewed,
        review_outcome,
        follow_up_action,
        review_notes,
        recommendation_feedback,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing_columns = list(pd.read_csv(path, nrows=0).columns)
        if existing_columns != LOG_COLUMNS:
            raise ValueError(
                "Berkas log sudah memiliki format lain. Simpan format lama terpisah; "
                "jangan gabungkan otomatis."
            )

    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "schema_version": SCHEMA_VERSION,
                "id_klaim": str(claim_id),
                "split_scheme": split_scheme,
                "split_name": split_name,
                "reviewer_role": reviewer_role,
                "evidence_reviewed": "|".join(evidence),
                "review_outcome": review_outcome,
                "follow_up_action": follow_up_action,
                "review_notes": notes,
                "recommendation_feedback": recommendation_feedback,
                "demo_only": "true",
            }
        ],
        columns=LOG_COLUMNS,
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
    return row.iloc[0].to_dict()
