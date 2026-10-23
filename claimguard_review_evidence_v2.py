#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build traceable, train-only claim comparisons for synthetic review evidence."""

import pandas as pd


MIN_REFERENCE_GROUP_SIZE = 8
COST_COLUMNS = [
    "cost_akomodasi_idr",
    "cost_alkes_idr",
    "cost_konsultasi_idr",
    "cost_obat_idr",
    "cost_penunjang_idr",
    "cost_tindakan_idr",
]


def _codes(value):
    return {part for part in str(value).split("|") if part}


def _code_set(row):
    return (
        _codes(row["diagnosis_codes"])
        | _codes(row["procedure_codes"])
        | _codes(row["item_codes"])
    )


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} tidak memiliki kolom wajib: {', '.join(missing)}")


def _cost_reference_group(train, claim, reference_level):
    if reference_level == "service_diagnosis_class":
        mask = (
            train["service_type"].eq(claim["service_type"])
            & train["diagnosis_primary"].eq(claim["diagnosis_primary"])
            & train["ward_class"].eq(claim["ward_class"])
        )
        group = train.loc[mask]
        if len(group) < MIN_REFERENCE_GROUP_SIZE:
            raise ValueError("Kelompok referensi biaya tidak memenuhi ukuran minimum.")
        return group, "Jenis layanan, diagnosis utama, dan kelas rawat"
    if reference_level == "service":
        group = train.loc[train["service_type"].eq(claim["service_type"])]
        if group.empty:
            raise ValueError("Data latih tidak memiliki kelompok layanan untuk pembanding biaya.")
        return group, "Jenis layanan"
    if reference_level == "global":
        if train.empty:
            raise ValueError("Data latih tidak tersedia untuk pembanding biaya.")
        return train, "Seluruh data latih"
    raise ValueError(f"Tingkat referensi biaya tidak dikenal: {reference_level}")


def _cost_comparison(train, claim, feature_row):
    group, group_label = _cost_reference_group(
        train, claim, str(feature_row["cost_reference_level"])
    )
    components = []
    for column in COST_COLUMNS:
        current = float(claim[column])
        median = float(group[column].median())
        components.append(
            {
                "component": column,
                "current_idr": current,
                "train_median_idr": median,
                "difference_idr": current - median,
            }
        )
    return {
        "group_label": group_label,
        "reference_count": int(len(group)),
        "current_total_idr": float(claim["claim_total_idr"]),
        "train_median_total_idr": float(feature_row["expected_cost_train_median_idr"]),
        "ratio_to_train_median": float(feature_row["cost_ratio_train_median"]),
        "components": components,
    }


def _prior_claim_context(claims, claim, split_column):
    split_order = {"train": 0, "val": 1, "test": 2}
    current_split = str(claim[split_column])
    if current_split not in split_order:
        raise ValueError(f"Partisi klaim tidak dikenal: {current_split}")
    eligible_splits = {
        split_name for split_name, rank in split_order.items() if rank <= split_order[current_split]
    }
    return claims.loc[claims[split_column].isin(eligible_splits)]


def _similar_claim_comparison(prior_claims, claim):
    current_codes = _code_set(claim)
    current_date = str(claim["submitted_at"])
    candidates = prior_claims.loc[
        prior_claims["submitted_at"].astype(str).lt(current_date)
        & prior_claims["service_type"].eq(claim["service_type"])
        & prior_claims["patient_token"].ne(claim["patient_token"])
    ].copy()
    if candidates.empty:
        return None

    scored = []
    for row in candidates.to_dict("records"):
        candidate_codes = _code_set(row)
        union = current_codes | candidate_codes
        intersection = current_codes & candidate_codes
        scored.append(
            {
                "claim_id": str(row["id_klaim"]),
                "submitted_at": str(row["submitted_at"]),
                "service_type": str(row["service_type"]),
                "similarity": len(intersection) / len(union) if union else 0.0,
                "shared_codes": sorted(intersection),
                "current_only_codes": sorted(current_codes - candidate_codes),
                "comparison_only_codes": sorted(candidate_codes - current_codes),
            }
        )
    scored.sort(key=lambda row: (-row["similarity"], row["submitted_at"], row["claim_id"]))
    return scored[0] if scored[0]["similarity"] > 0 else None


def _repeat_claim_comparison(prior_claims, claim):
    current_date = pd.Timestamp(claim["submitted_at"])
    same_patient = prior_claims.loc[
        prior_claims["patient_token"].eq(claim["patient_token"])
        & prior_claims["submitted_at"].astype(str).lt(str(claim["submitted_at"]))
    ].copy()
    if same_patient.empty:
        return {"prior_count_30d": 0, "previous_claim": None}

    same_patient["_submitted_at"] = pd.to_datetime(same_patient["submitted_at"])
    same_patient = same_patient.sort_values(["_submitted_at", "id_klaim"])
    previous = same_patient.iloc[-1]
    prior_30d = int(
        (
            (current_date - same_patient["_submitted_at"]).dt.total_seconds()
            .div(86400)
            .between(0, 30, inclusive="both")
        ).sum()
    )
    return {
        "prior_count_30d": prior_30d,
        "previous_claim": {
            "claim_id": str(previous["id_klaim"]),
            "submitted_at": str(previous["submitted_at"]),
            "service_date": str(previous["service_date"]),
            "service_type": str(previous["service_type"]),
            "claim_total_idr": float(previous["claim_total_idr"]),
        },
    }


def _severity_comparison(train, claim):
    if claim["service_type"] != "RITL":
        return None
    mask = (
        train["service_type"].eq(claim["service_type"])
        & train["diagnosis_primary"].eq(claim["diagnosis_primary"])
        & train["ward_class"].eq(claim["ward_class"])
    )
    group = train.loc[mask]
    group_label = "Jenis layanan, diagnosis utama, dan kelas rawat"
    if len(group) < MIN_REFERENCE_GROUP_SIZE:
        group = train.loc[train["service_type"].eq(claim["service_type"])]
        group_label = "Jenis layanan (kelompok yang lebih luas)"
    if group.empty:
        return None
    severity_counts = group["severity"].astype(str).value_counts().to_dict()
    return {
        "group_label": group_label,
        "reference_count": int(len(group)),
        "train_median_stay_days": float(group["length_of_stay_days"].median()),
        "train_severity_counts": {str(key): int(value) for key, value in severity_counts.items()},
        "current_severity": str(claim["severity"]),
        "current_stay_days": float(claim["length_of_stay_days"]),
    }


def build_claim_comparison(claims, features, claim_id, split_column):
    if split_column not in {"split_time", "split_facility"}:
        raise ValueError("split_column harus split_time atau split_facility.")
    _require_columns(
        claims,
        [
            "id_klaim",
            split_column,
            "patient_token",
            "submitted_at",
            "service_date",
            "service_type",
            "facility_id",
            "ward_class",
            "severity",
            "length_of_stay_days",
            "claim_total_idr",
            "diagnosis_primary",
            "diagnosis_codes",
            "procedure_codes",
            "item_codes",
            *COST_COLUMNS,
        ],
        "Data klaim",
    )
    _require_columns(
        features,
        [
            "id_klaim",
            split_column,
            "cost_reference_level",
            "expected_cost_train_median_idr",
            "cost_ratio_train_median",
        ],
        "Data fitur",
    )
    if claims["id_klaim"].duplicated().any() or features["id_klaim"].duplicated().any():
        raise ValueError("id_klaim harus unik pada data klaim dan fitur.")
    claim_matches = claims.loc[claims["id_klaim"].astype(str).eq(str(claim_id))]
    feature_matches = features.loc[features["id_klaim"].astype(str).eq(str(claim_id))]
    if len(claim_matches) != 1 or len(feature_matches) != 1:
        raise ValueError(f"Klaim {claim_id} harus ditemukan tepat satu kali di data dan fitur.")
    claim = claim_matches.iloc[0]
    feature = feature_matches.iloc[0]
    if claim[split_column] != feature[split_column]:
        raise ValueError("Partisi klaim dan fitur tidak cocok.")
    train = claims.loc[claims[split_column].eq("train")]
    if train.empty:
        raise ValueError(f"Tidak ada data latih pada {split_column}.")
    prior_claims = _prior_claim_context(claims, claim, split_column)
    return {
        "cost": _cost_comparison(train, claim, feature),
        "similar_claim": _similar_claim_comparison(prior_claims, claim),
        "repeat": _repeat_claim_comparison(prior_claims, claim),
        "severity": _severity_comparison(train, claim),
    }
