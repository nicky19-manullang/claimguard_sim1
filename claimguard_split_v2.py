#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create deterministic time-forward and facility-held-out splits for synthetic v2 data."""

import argparse
import json
import os

import pandas as pd

from claimguard_adapter_v2 import load_v2


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


def _time_split(claims):
    ordered = claims.sort_values(["submitted_at", "id_klaim"], kind="mergesort").copy()
    daily = ordered.groupby("submitted_at", sort=True).size()
    cumulative = daily.cumsum()
    total = len(ordered)
    train_target, val_target = total * RATIOS["train"], total * (RATIOS["train"] + RATIOS["val"])

    train_dates = cumulative.index[(cumulative - train_target).abs() == (cumulative - train_target).abs().min()]
    train_end = train_dates[-1]
    train_count = int(cumulative.loc[train_end])

    later = cumulative[cumulative.index > train_end]
    val_dates = later.index[(later - val_target).abs() == (later - val_target).abs().min()]
    if len(val_dates) == 0:
        raise ValueError("Tidak cukup tanggal unik untuk membentuk validation dan test.")
    val_end = val_dates[-1]

    split = pd.Series("test", index=ordered.index, dtype="object")
    split.loc[ordered["submitted_at"] <= train_end] = "train"
    split.loc[(ordered["submitted_at"] > train_end) & (ordered["submitted_at"] <= val_end)] = "val"
    return split.reindex(claims.index), {
        "train_through": str(train_end),
        "validation_through": str(val_end),
        "train_claims": train_count,
        "validation_claims": int(
            ((ordered["submitted_at"] > train_end) & (ordered["submitted_at"] <= val_end)).sum()
        ),
    }


def _facility_split(claims):
    counts = claims.groupby("facility_id").size().sort_values(ascending=False)
    if len(counts) < len(SPLITS):
        raise ValueError("Butuh minimal tiga faskes untuk split faskes train/val/test.")

    targets = {name: RATIOS[name] * len(claims) for name in SPLITS}
    assigned_counts = {name: 0 for name in SPLITS}
    facility_assignment = {}
    for facility_id, count in counts.items():
        selected = max(
            SPLITS,
            key=lambda name: (targets[name] - assigned_counts[name]) / targets[name],
        )
        facility_assignment[facility_id] = selected
        assigned_counts[selected] += int(count)

    split = claims["facility_id"].map(facility_assignment)
    if split.isna().any() or set(split.unique()) != set(SPLITS):
        raise ValueError("Gagal menetapkan semua faskes ke train/val/test.")
    return split, {
        "claims_by_split": {key: int(value) for key, value in split.value_counts().items()},
        "facilities_by_split": {
            name: int(sum(value == name for value in facility_assignment.values()))
            for name in SPLITS
        },
    }


def create_splits(claims):
    required = {"id_klaim", "patient_token", "facility_id", "submitted_at"}
    missing = required - set(claims.columns)
    if missing:
        raise ValueError(f"Kolom wajib untuk split hilang: {', '.join(sorted(missing))}")
    if claims["id_klaim"].duplicated().any():
        raise ValueError("id_klaim harus unik untuk membuat split.")

    dates = pd.to_datetime(claims["submitted_at"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("submitted_at harus memakai format YYYY-MM-DD yang valid.")
    if claims["facility_id"].astype(str).str.strip().eq("").any():
        raise ValueError("facility_id tidak boleh kosong.")

    time_labels, time_meta = _time_split(claims)
    facility_labels, facility_meta = _facility_split(claims)
    split_frame = pd.DataFrame(
        {
            "id_klaim": claims["id_klaim"].to_numpy(),
            "split_time": time_labels.to_numpy(),
            "split_facility": facility_labels.to_numpy(),
        }
    )

    patient_by_claim = claims.set_index("id_klaim")["patient_token"]
    split_by_claim = split_frame.set_index("id_klaim")
    patient_splits_time = pd.DataFrame(
        {"patient_token": patient_by_claim, "split": split_by_claim["split_time"]}
    ).groupby("patient_token")["split"].nunique()
    patient_splits_facility = pd.DataFrame(
        {"patient_token": patient_by_claim, "split": split_by_claim["split_facility"]}
    ).groupby("patient_token")["split"].nunique()
    time_meta["unique_submission_dates"] = int(dates.nunique())
    time_meta["claims_by_split"] = {
        key: int(value) for key, value in split_frame["split_time"].value_counts().items()
    }
    time_meta["participants_across_splits"] = int((patient_splits_time > 1).sum())
    facility_meta["participants_across_splits"] = int((patient_splits_facility > 1).sum())
    facility_meta["facility_disjoint"] = True
    facility_meta["participant_disjoint"] = False

    manifest = {
        "dataset": "synthetic_claim_tables_v2",
        "ratios_target": RATIOS,
        "split_time": time_meta,
        "split_facility": facility_meta,
        "limitations": [
            "Dataset sintetis; hasil split tidak membuktikan generalisasi ke data BPJS nyata.",
            "Split waktu mempertahankan urutan masa depan; riwayat peserta lintas periode hanya boleh dipakai sebagai sinyal historis yang tersedia pada waktu klaim dinilai.",
            "Split faskes menahan seluruh klaim suatu faskes pada satu split, tetapi peserta dapat muncul lintas split; jangan gunakan patient_token sebagai fitur model.",
            "hasil_verifikasi tidak dipakai untuk menentukan split dan bukan label anomali.",
        ],
    }
    return split_frame, manifest


def run(root, outdir=None):
    dataset = load_v2(root)
    split_frame, manifest = create_splits(dataset.claims)
    output_dir = outdir or os.path.join(root, "prep_v2")
    os.makedirs(output_dir, exist_ok=True)

    split_path = os.path.join(output_dir, "splits_v2.csv")
    merged_path = os.path.join(output_dir, "claims_v2_flat_with_splits.csv.gz")
    manifest_path = os.path.join(output_dir, "split_manifest_v2.json")
    split_frame.to_csv(split_path, index=False)
    merged = dataset.claims.merge(split_frame, on="id_klaim", how="left", validate="one_to_one")
    if merged[["split_time", "split_facility"]].isna().any().any():
        raise ValueError("Ada klaim tanpa split setelah penggabungan.")
    merged.to_csv(merged_path, index=False, compression="gzip")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return split_path, merged_path, manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="out", help="Folder output utama yang berisi subfolder data.")
    parser.add_argument("--outdir", default=None, help="Folder output; default: <root>/prep_v2.")
    args = parser.parse_args()
    split_path, merged_path, manifest_path, manifest = run(args.root, args.outdir)
    print(
        "OK: split v2 dibuat | "
        f"time={manifest['split_time']['claims_by_split']} | "
        f"facility={manifest['split_facility']['claims_by_split']} | "
        f"files={split_path}, {merged_path}, {manifest_path}"
    )


if __name__ == "__main__":
    main()
