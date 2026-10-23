#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank synthetic v2 claims for human review using transparent, train-referenced signals."""

import argparse
import json
import os

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
FEATURE_FILES = {
    "time": ("features_v2_time.csv.gz", "split_time"),
    "facility": ("features_v2_facility.csv.gz", "split_facility"),
}
CLAIM_FILE = "claims_v2_flat_with_splits.csv.gz"
MIN_TRAIN_REFERENCE = 5


class EmpiricalTail:
    """Train-reference upper-tail percentile; this is a rank, not a probability."""

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError("Tidak ada nilai train finite untuk membentuk referensi skor.")
        self.sorted_values = np.sort(values)
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        values = np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        below = np.searchsorted(self.sorted_values, values, side="left")
        at_or_below = np.searchsorted(self.sorted_values, values, side="right")
        midrank = below + 0.5 * (at_or_below - below)
        return 100.0 * midrank / len(self.sorted_values)


def _make_raw_signals(frame, split_column):
    prefix = split_column + "_"
    signals = pd.DataFrame(index=frame.index)
    signals["repeat"] = (
        0.35 * np.log1p(frame[f"{prefix}prior_patient_claims_30d"].astype(float))
        + 0.25 * frame[f"{prefix}patient_previous_item_jaccard"].astype(float)
        + 0.20 * frame[f"{prefix}patient_previous_value_ratio"].astype(float)
        + 0.20 * frame[f"{prefix}patient_service_overlap"].astype(float)
    )
    signals["cloning"] = frame[
        f"{prefix}max_prior_other_patient_code_jaccard"
    ].astype(float)
    signals["inflated_cost"] = np.maximum(
        frame["log_cost_robust_z_train"].astype(float),
        np.log1p(frame["cost_ratio_train_median"].astype(float).clip(lower=0)),
    )

    severity = frame["severity"].map({"-": 0.0, "I": 1.0, "II": 2.0, "III": 3.0})
    if severity.isna().any():
        raise ValueError("Severity berisi kode yang tidak dikenal.")
    short_stay = severity / (1.0 + frame["length_of_stay_days"].astype(float))
    signals["severity_los_unusual"] = np.where(
        frame["service_type"].eq("RITL"), short_stay, 0.0
    )
    signals["severity_los_unusual"] = pd.Series(
        signals["severity_los_unusual"], index=frame.index
    )
    return signals


def score_claims(features, claims, split_column, review_budget=0.05):
    if not 0 < review_budget <= 1:
        raise ValueError("review_budget harus lebih besar dari 0 dan maksimal 1.")
    if split_column not in {"split_time", "split_facility"}:
        raise ValueError("split_column harus split_time atau split_facility.")
    if split_column not in features or split_column not in claims:
        raise ValueError(f"Kolom split tidak tersedia: {split_column}")
    if features["id_klaim"].duplicated().any() or claims["id_klaim"].duplicated().any():
        raise ValueError("id_klaim harus unik di tabel fitur dan klaim.")

    merged = features.merge(
        claims,
        on=["id_klaim", split_column],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_claim"),
    )
    if len(merged) != len(features) or len(merged) != len(claims):
        raise ValueError("Fitur dan klaim tidak memiliki cakupan id_klaim/split yang sama.")
    raw = _make_raw_signals(merged, split_column)
    train_mask = merged[split_column].eq("train").to_numpy()
    if int(train_mask.sum()) < MIN_TRAIN_REFERENCE:
        raise ValueError("Jumlah train terlalu sedikit untuk membentuk referensi ranking.")

    result = merged[["id_klaim", split_column, "submitted_at", "service_type", "facility_id"]].copy()
    signal_scores = {}
    for signal in raw.columns:
        reference_mask = train_mask.copy()
        if signal == "severity_los_unusual":
            reference_mask &= merged["service_type"].eq("RITL").to_numpy()
            if not reference_mask.any():
                raise ValueError("Train tidak memiliki klaim RITL untuk referensi severity/lama rawat.")
        calibrator = EmpiricalTail().fit(raw.loc[reference_mask, signal])
        scored = calibrator.transform(raw[signal])
        if signal == "severity_los_unusual":
            scored[~merged["service_type"].eq("RITL").to_numpy()] = 0.0
        result[f"{signal}_rank_percentile"] = scored
        signal_scores[signal] = scored

    score_frame = pd.DataFrame(signal_scores, index=merged.index)
    result["priority_score"] = score_frame.max(axis=1)
    result["top_driver"] = score_frame.idxmax(axis=1).map(
        {
            "repeat": "Pola klaim pasien berulang",
            "cloning": "Kemiripan dengan klaim pasien lain",
            "inflated_cost": "Biaya relatif tinggi terhadap referensi train",
            "severity_los_unusual": "Kombinasi severity dan lama rawat perlu ditinjau",
        }
    )
    result["top_driver_rank_percentile"] = score_frame.max(axis=1)

    evidence = {
        "repeat": lambda i: (
            f"klaim terdahulu 30 hari={int(merged.iloc[i][f'{split_column}_prior_patient_claims_30d'])}; "
            f"kemiripan item sebelumnya={merged.iloc[i][f'{split_column}_patient_previous_item_jaccard']:.2f}; "
            f"layanan overlap={int(merged.iloc[i][f'{split_column}_patient_service_overlap'])}"
        ),
        "cloning": lambda i: (
            f"kemiripan kode maksimum dengan klaim terdahulu pasien lain="
            f"{merged.iloc[i][f'{split_column}_max_prior_other_patient_code_jaccard']:.2f}"
        ),
        "inflated_cost": lambda i: (
            f"rasio biaya terhadap median train={merged.iloc[i]['cost_ratio_train_median']:.2f}; "
            f"robust z biaya={merged.iloc[i]['log_cost_robust_z_train']:.2f}"
        ),
        "severity_los_unusual": lambda i: (
            f"severity={merged.iloc[i]['severity']}; "
            f"lama rawat={int(merged.iloc[i]['length_of_stay_days'])} hari"
        ),
    }
    result["evidence"] = [
        evidence[score_frame.iloc[i].idxmax()](i)
        for i in range(len(result))
    ]

    selected = pd.Series(False, index=result.index)
    for _, indices in result.groupby(split_column).groups.items():
        count = max(1, int(np.ceil(len(indices) * review_budget)))
        ordered = result.loc[indices].sort_values(
            ["priority_score", "id_klaim"], ascending=[False, True], kind="mergesort"
        )
        selected.loc[ordered.index[:count]] = True
    result["review_recommended"] = selected
    result["priority_rank"] = result.groupby(split_column)["priority_score"].rank(
        method="first", ascending=False
    ).astype(int)
    result["interpretation"] = (
        "Urutan sinyal relatif terhadap referensi train; bukan probabilitas fraud atau keputusan klaim."
    )
    result = result.sort_values([split_column, "priority_rank"], kind="mergesort").reset_index(drop=True)
    return result


def run(root, outdir=None, review_budget=0.05):
    prep_dir = os.path.join(root, "prep_v2")
    claims_path = os.path.join(prep_dir, CLAIM_FILE)
    if not os.path.isfile(claims_path):
        raise FileNotFoundError(
            f"Data klaim bersplit belum ditemukan: {claims_path}. Jalankan adapter dan split v2 dahulu."
        )
    claims = pd.read_csv(claims_path, keep_default_na=False)
    output_dir = outdir or prep_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "dataset": "synthetic_claim_tables_v2",
        "review_budget_per_split": review_budget,
        "score_definition": "maximum train-reference mid-rank percentile among four signal families",
        "scores_are_probabilities": False,
        "automatic_claim_decision": False,
        "schemes": {},
        "limitations": [
            "Score adalah persentil peringkat terhadap distribusi train, bukan probabilitas fraud.",
            "Persentil memakai mid-rank agar kelompok nilai train yang seri tidak otomatis memperoleh nilai maksimum.",
            "Priority score adalah maksimum dari empat persentil sinyal; penggabungan maksimum dapat membuat skornya cenderung tinggi dan tidak boleh dibaca sebagai tingkat keyakinan.",
            "Aturan dan sinyal adalah prototipe eksplorasi pada data sintetis, bukan detektor tervalidasi.",
            "Top budget adalah simulasi kapasitas review dan tidak menahan, menyetujui, atau menolak klaim.",
            "hasil_verifikasi digunakan hanya untuk memastikan pemisahan dari ranking; tidak dibaca sebagai target.",
        ],
    }
    output_paths = {}
    for scheme, (filename, split_column) in FEATURE_FILES.items():
        feature_path = os.path.join(prep_dir, filename)
        if not os.path.isfile(feature_path):
            raise FileNotFoundError(f"Fitur belum dibuat: {feature_path}")
        features = pd.read_csv(feature_path, keep_default_na=False)
        output = score_claims(
            features,
            claims,
            split_column,
            review_budget=review_budget,
        )
        output_path = os.path.join(output_dir, f"review_queue_v2_{scheme}.csv.gz")
        output.to_csv(output_path, index=False, compression="gzip")
        output_paths[scheme] = output_path
        manifest["schemes"][scheme] = {
            "rows": int(len(output)),
            "recommended_by_split": {
                str(key): int(value)
                for key, value in output.groupby(split_column)["review_recommended"].sum().items()
            },
            "features_used_for_ranking": [
                "repeat", "cloning", "inflated_cost", "severity_los_unusual"
            ],
            "train_reference_rows": int(features[split_column].eq("train").sum()),
        }
    manifest_path = os.path.join(output_dir, "review_queue_manifest_v2.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return output_paths, manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="out", help="Folder output utama yang berisi subfolder data.")
    parser.add_argument("--outdir", default=None, help="Folder output; default: <root>/prep_v2.")
    parser.add_argument(
        "--review-budget", type=float, default=0.05,
        help="Proporsi klaim per split yang direkomendasikan untuk demo antrean (default 0.05).",
    )
    args = parser.parse_args()
    paths, manifest_path, manifest = run(args.root, args.outdir, args.review_budget)
    print(
        f"OK: antrean demo dibuat | budget={args.review_budget:.1%} | "
        f"time={paths['time']} | facility={paths['facility']} | manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
