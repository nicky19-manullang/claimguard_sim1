#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create leakage-aware, interpretable features for synthetic claim tables v2."""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

from claimguard_adapter_v2 import load_v2


SPLIT_NAMES = ("train", "val", "test")
SPLIT_RANK = {name: rank for rank, name in enumerate(SPLIT_NAMES)}
MIN_GROUP_SIZE = 8
MAD_SCALE = 1.4826
SCALE_FLOOR = 0.05
ID_COLUMNS = {"id_klaim", "patient_token", "facility_id"}
REVIEW_COLUMNS = {
    "hasil", "kode_alasan", "deskripsi_alasan", "status",
    "tgl_pemeriksaan", "perlu_klarifikasi", "is_anomaly",
}


def _robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median)) * MAD_SCALE)
    return median, max(scale, SCALE_FLOOR)


def _fit_cost_reference(claims, split_column):
    train = claims.loc[claims[split_column].eq("train")].copy()
    if train.empty:
        raise ValueError(f"Tidak ada klaim train pada {split_column}.")
    train["log_cost"] = np.log1p(train["claim_total_idr"].astype(float))
    overall_median, overall_scale = _robust_location_scale(train["log_cost"])
    overall_cost_median = float(train["claim_total_idr"].median())
    service_refs = {}
    group_refs = {}
    for service, rows in train.groupby("service_type"):
        median, scale = _robust_location_scale(rows["log_cost"])
        service_refs[service] = (median, scale, float(rows["claim_total_idr"].median()))
    key_cols = ["service_type", "diagnosis_primary", "ward_class"]
    for key, rows in train.groupby(key_cols, dropna=False):
        if len(rows) >= MIN_GROUP_SIZE:
            median, scale = _robust_location_scale(rows["log_cost"])
            group_refs[tuple(key)] = (median, scale, float(rows["claim_total_idr"].median()))
    return {
        "overall": (overall_median, overall_scale, overall_cost_median),
        "service": service_refs,
        "group": group_refs,
        "train_rows": int(len(train)),
    }


def _reference_for_claim(row, reference):
    key = (row["service_type"], row["diagnosis_primary"], row["ward_class"])
    if key in reference["group"]:
        return reference["group"][key], "service_diagnosis_class"
    if row["service_type"] in reference["service"]:
        return reference["service"][row["service_type"]], "service"
    return reference["overall"], "global"


def _codes(value):
    return {part for part in str(value).split("|") if part}


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _split_context_allows(candidate_split, current_split):
    return SPLIT_RANK[candidate_split] <= SPLIT_RANK[current_split]


def _relational_features(claims, split_column):
    ordered = claims.sort_values(["submitted_at", "id_klaim"], kind="mergesort")
    rows = ordered.to_dict("records")
    feature_rows = []
    for index, current in enumerate(rows):
        current_date = pd.Timestamp(current["submitted_at"])
        current_split = current[split_column]
        current_codes = _codes(current["diagnosis_codes"]) | _codes(current["procedure_codes"]) | _codes(current["item_codes"])
        current_items = _codes(current["item_codes"])
        prior = []
        for candidate in rows[:index]:
            if candidate["submitted_at"] >= current["submitted_at"]:
                continue
            if not _split_context_allows(candidate[split_column], current_split):
                continue
            prior.append(candidate)

        patient_prior = [row for row in prior if row["patient_token"] == current["patient_token"]]
        same_facility_prior = [
            row for row in patient_prior if row["facility_id"] == current["facility_id"]
        ]
        prior_30d = [
            row for row in patient_prior
            if (current_date - pd.Timestamp(row["submitted_at"])).days <= 30
        ]
        prior_90d = [
            row for row in patient_prior
            if (current_date - pd.Timestamp(row["submitted_at"])).days <= 90
        ]
        previous = patient_prior[-1] if patient_prior else None
        if previous:
            service_gap = (
                pd.Timestamp(current["service_date"]) - pd.Timestamp(previous["service_date"])
            ).days
            previous_items = _codes(previous["item_codes"])
            patient_item_jaccard = _jaccard(current_items, previous_items)
            previous_end = (
                pd.Timestamp(previous["service_date"])
                + pd.Timedelta(days=int(previous["length_of_stay_days"]))
            )
            overlaps_previous = int(pd.Timestamp(current["service_date"]) <= previous_end)
            previous_total = float(previous["claim_total_idr"])
            value_ratio = min(float(current["claim_total_idr"]), previous_total) / max(
                float(current["claim_total_idr"]), previous_total, 1.0
            )
        else:
            service_gap = np.nan
            patient_item_jaccard = 0.0
            overlaps_previous = 0
            value_ratio = 0.0

        other_patient_prior = [
            row for row in prior
            if row["patient_token"] != current["patient_token"]
            and row["service_type"] == current["service_type"]
        ]
        code_similarities = []
        for candidate in other_patient_prior:
            candidate_codes = (
                _codes(candidate["diagnosis_codes"])
                | _codes(candidate["procedure_codes"])
                | _codes(candidate["item_codes"])
            )
            code_similarities.append(_jaccard(current_codes, candidate_codes))
        current_primary = current["diagnosis_primary"]
        same_primary = sum(
            row["diagnosis_primary"] == current_primary for row in other_patient_prior
        )
        feature_rows.append(
            {
                "id_klaim": current["id_klaim"],
                f"{split_column}_prior_patient_claims": len(patient_prior),
                f"{split_column}_prior_patient_claims_30d": len(prior_30d),
                f"{split_column}_prior_patient_claims_90d": len(prior_90d),
                f"{split_column}_prior_same_facility_claims": len(same_facility_prior),
                f"{split_column}_patient_has_prior_claim": int(previous is not None),
                f"{split_column}_days_since_patient_submission": (
                    (current_date - pd.Timestamp(previous["submitted_at"])).days
                    if previous else -1
                ),
                f"{split_column}_patient_service_gap_days": (
                    service_gap if previous else -1
                ),
                f"{split_column}_patient_previous_item_jaccard": patient_item_jaccard,
                f"{split_column}_patient_service_overlap": overlaps_previous,
                f"{split_column}_patient_previous_value_ratio": value_ratio,
                f"{split_column}_prior_other_patient_claims": len(other_patient_prior),
                f"{split_column}_prior_same_primary_diagnosis": same_primary,
                f"{split_column}_max_prior_other_patient_code_jaccard": (
                    max(code_similarities) if code_similarities else 0.0
                ),
            }
        )
    return pd.DataFrame(feature_rows)


def _fit_code_frequencies(claims, split_column):
    train = claims.loc[claims[split_column].eq("train")]
    dx = Counter(train["diagnosis_primary"].astype(str))
    procedure = Counter(
        code
        for value in train["procedure_codes"].astype(str)
        for code in _codes(value)
    )
    item = Counter(
        code
        for value in train["item_codes"].astype(str)
        for code in _codes(value)
    )
    return {"diagnosis": dx, "procedure": procedure, "item": item}


def _code_frequency_features(claims, split_column, frequencies):
    rows = []
    for row in claims.to_dict("records"):
        proc_codes = _codes(row["procedure_codes"])
        item_codes = _codes(row["item_codes"])
        rows.append(
            {
                "id_klaim": row["id_klaim"],
                f"{split_column}_primary_dx_train_count": frequencies["diagnosis"].get(
                    str(row["diagnosis_primary"]), 0
                ),
                f"{split_column}_procedure_train_min_count": (
                    min(frequencies["procedure"].get(code, 0) for code in proc_codes)
                    if proc_codes else 0
                ),
                f"{split_column}_item_train_min_count": (
                    min(frequencies["item"].get(code, 0) for code in item_codes)
                    if item_codes else 0
                ),
                f"{split_column}_unseen_procedure_count": sum(
                    frequencies["procedure"].get(code, 0) == 0 for code in proc_codes
                ),
                f"{split_column}_unseen_item_count": sum(
                    frequencies["item"].get(code, 0) == 0 for code in item_codes
                ),
            }
        )
    return pd.DataFrame(rows)


def create_features(claims, split_column):
    if split_column not in {"split_time", "split_facility"}:
        raise ValueError("split_column harus split_time atau split_facility.")
    if split_column not in claims:
        raise ValueError(f"Kolom split tidak ditemukan: {split_column}")
    if not claims[split_column].isin(SPLIT_NAMES).all():
        raise ValueError(f"{split_column} harus berisi train/val/test.")
    if claims["id_klaim"].duplicated().any():
        raise ValueError("id_klaim harus unik.")

    output = claims[["id_klaim", split_column]].copy()
    base_numeric = [
        "claim_total_idr", "length_of_stay_days", "procedure_count", "item_count",
    ]
    cost_columns = [column for column in claims if column.startswith("cost_") and column.endswith("_idr")]
    output = output.merge(
        claims[["id_klaim"] + base_numeric + cost_columns],
        on="id_klaim",
        how="left",
        validate="one_to_one",
    )
    output["log_claim_total"] = np.log1p(output["claim_total_idr"].astype(float))
    for column in cost_columns:
        component = column.removeprefix("cost_").removesuffix("_idr")
        output[f"{component}_cost_share"] = (
            output[column].astype(float) / output["claim_total_idr"].clip(lower=1).astype(float)
        )

    references = _fit_cost_reference(claims, split_column)
    expected_log, expected_amounts, scales, levels = [], [], [], []
    for row in claims.to_dict("records"):
        (median, scale, amount_median), level = _reference_for_claim(row, references)
        expected_log.append(median)
        expected_amounts.append(amount_median)
        scales.append(scale)
        levels.append(level)
    output["expected_log_cost_train"] = expected_log
    output["expected_cost_train_median_idr"] = expected_amounts
    output["log_cost_robust_z_train"] = (
        output["log_claim_total"] - output["expected_log_cost_train"]
    ) / np.asarray(scales)
    output["cost_ratio_train_median"] = output["claim_total_idr"] / np.maximum(
        output["expected_cost_train_median_idr"], 1.0
    )
    output["cost_reference_level"] = levels

    primary_counts = _fit_code_frequencies(claims, split_column)
    output = output.merge(
        _code_frequency_features(claims, split_column, primary_counts),
        on="id_klaim",
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        _relational_features(claims, split_column),
        on="id_klaim",
        how="left",
        validate="one_to_one",
    )
    forbidden = (ID_COLUMNS - {"id_klaim"}) | REVIEW_COLUMNS
    leaked = forbidden & set(output.columns)
    if leaked:
        raise ValueError(f"Kolom identitas/hasil review tidak boleh menjadi fitur: {sorted(leaked)}")
    return output, {
        "training_reference_rows": references["train_rows"],
        "cost_reference_groups": len(references["group"]),
        "cost_reference_fallbacks": dict(Counter(levels)),
        "feature_columns": [column for column in output if column not in {"id_klaim", split_column}],
    }


def run(root, outdir=None):
    dataset = load_v2(root)
    data_dir = os.path.join(root, "prep_v2")
    split_path = os.path.join(data_dir, "claims_v2_flat_with_splits.csv.gz")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"File split belum ada: {split_path}. Jalankan claimguard_split_v2.py terlebih dahulu."
        )
    claims = pd.read_csv(split_path, keep_default_na=False)
    if set(claims["id_klaim"]) != set(dataset.claims["id_klaim"]):
        raise ValueError("File split tidak sesuai dengan klaim versi 2 yang sedang dimuat.")

    output_dir = outdir or data_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "dataset": "synthetic_claim_tables_v2",
        "feature_policy": [
            "Statistik biaya dan frekuensi kode di-fit hanya pada partisi train.",
            "Biaya dibandingkan dengan median train per layanan+diagnosis utama+kelas (minimal 8 klaim); bila tidak cukup, fallback ke median train per layanan.",
            "Riwayat dan kemiripan hanya menggunakan klaim terdahulu menurut submitted_at; tanggal pengajuan yang sama tidak dibandingkan.",
            "Nilai -1 pada fitur jarak waktu menandakan belum ada klaim sebelumnya; indikator patient_has_prior_claim disediakan.",
            "Klaim dari split berperingkat lebih akhir tidak digunakan sebagai konteks bagi split yang lebih awal.",
            "ID peserta dan faskes, hasil verifikasi, serta label tidak dimasukkan sebagai fitur.",
            "Fitur ini adalah sinyal eksplorasi pada data sintetis, bukan keputusan atau probabilitas fraud.",
        ],
        "schemes": {},
    }
    feature_paths = {}
    for split_column, suffix in (("split_time", "time"), ("split_facility", "facility")):
        features, metadata = create_features(claims, split_column)
        filename = f"features_v2_{suffix}.csv.gz"
        path = os.path.join(output_dir, filename)
        features.to_csv(path, index=False, compression="gzip")
        feature_paths[suffix] = path
        manifest["schemes"][suffix] = {
            **metadata,
            "rows": int(len(features)),
            "split_counts": {
                str(key): int(value) for key, value in features[split_column].value_counts().items()
            },
        }
    manifest_path = os.path.join(output_dir, "feature_manifest_v2.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return feature_paths, manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="out", help="Folder output utama yang berisi subfolder data.")
    parser.add_argument("--outdir", default=None, help="Folder output; default: <root>/prep_v2.")
    args = parser.parse_args()
    feature_paths, manifest_path, manifest = run(args.root, args.outdir)
    print(
        f"OK: fitur v2 dibuat | rows={manifest['schemes']['time']['rows']} | "
        f"time={feature_paths['time']} | facility={feature_paths['facility']} | "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
