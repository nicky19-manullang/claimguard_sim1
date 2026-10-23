#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and normalize the separate synthetic claim-table dataset (schema v2)."""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass

import pandas as pd


FILES = {
    "claims": "klaim.csv",
    "costs": "klaim_biaya.csv",
    "diagnoses": "klaim_diagnosis.csv",
    "procedures": "klaim_tindakan.csv",
    "items": "klaim_item.csv",
    "reviews": "hasil_verifikasi.csv",
    "dictionary": "kamus_data.csv",
}
REQUIRED_COLUMNS = {
    "claims": {
        "id_klaim", "id_peserta_token", "id_faskes", "tgl_pengajuan", "tgl_pelayanan",
        "jenis_layanan", "kelas_rawat", "severity", "lama_rawat_hari", "versi_grouper",
        "versi_tarif", "total_klaim_idr",
    },
    "costs": {"id_klaim", "komponen", "nilai", "satuan"},
    "diagnoses": {"id_klaim", "urutan", "kode_diagnosis", "jenis"},
    "procedures": {"id_klaim", "urutan", "kode_tindakan"},
    "items": {"id_klaim", "urutan", "kode_item", "tipe_item", "jumlah", "harga_satuan_idr"},
    "reviews": {
        "id_klaim", "hasil", "kode_alasan", "deskripsi_alasan", "status",
        "tgl_pemeriksaan", "perlu_klarifikasi",
    },
    "dictionary": {"tabel", "kolom", "tipe_format", "keterangan"},
}
FEATURE_FORBIDDEN_COLUMNS = {
    "hasil", "kode_alasan", "deskripsi_alasan", "status",
    "tgl_pemeriksaan", "perlu_klarifikasi", "is_anomaly",
}
ICD10_PATTERN = re.compile(r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$")
ICD9_PATTERN = re.compile(r"^[0-9]{2}(?:\.[0-9]{1,2})?$")
MONEY_TOLERANCE_IDR = 0.01


class DataValidationError(ValueError):
    """Raised when the v2 tables fail schema or consistency checks."""


@dataclass
class V2Dataset:
    claims: pd.DataFrame
    reviews: pd.DataFrame
    report: dict


def _require_columns(name, frame):
    missing = sorted(REQUIRED_COLUMNS[name] - set(frame.columns))
    if missing:
        raise DataValidationError(f"{FILES[name]}: kolom wajib hilang: {', '.join(missing)}")


def _require_nonblank(name, frame, columns):
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise DataValidationError(f"{FILES[name]}: kolom {column} berisi nilai kosong.")


def _numeric(name, frame, column, *, integer=False, nonnegative=True):
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise DataValidationError(f"{FILES[name]}: kolom {column} harus berisi angka valid.")
    if nonnegative and (values < 0).any():
        raise DataValidationError(f"{FILES[name]}: kolom {column} tidak boleh negatif.")
    if integer and (values % 1 != 0).any():
        raise DataValidationError(f"{FILES[name]}: kolom {column} harus bilangan bulat.")
    frame[column] = values


def _date_values(name, frame, column, *, allow_blank=False):
    raw = frame[column].astype(str).str.strip()
    blank = raw.eq("")
    if blank.any() and not allow_blank:
        raise DataValidationError(f"{FILES[name]}: kolom {column} tidak boleh kosong.")
    parsed = pd.to_datetime(raw.mask(blank), format="%Y-%m-%d", errors="coerce")
    if parsed[~blank].isna().any():
        raise DataValidationError(f"{FILES[name]}: kolom {column} harus berformat YYYY-MM-DD.")
    return parsed


def _validate_unique(name, frame, columns):
    if frame.duplicated(columns).any():
        key = " + ".join(columns)
        raise DataValidationError(f"{FILES[name]}: kunci {key} duplikat.")


def _validate_child_keys(name, frame, claim_ids, key_columns):
    _require_nonblank(name, frame, key_columns)
    if not set(frame["id_klaim"]).issubset(claim_ids):
        raise DataValidationError(f"{FILES[name]}: ada id_klaim yang tidak terdapat di klaim.csv.")
    _validate_unique(name, frame, key_columns)


def _aggregate_codes(frame, code_column, *, kinds=None):
    if kinds is None:
        return frame.groupby("id_klaim", sort=False)[code_column].agg("|".join)
    return frame.groupby(["id_klaim", kinds], sort=False)[code_column].agg("|".join)


def validate_and_normalize(tables):
    """Validate v2 table relations and return one feature row per claim plus isolated reviews."""
    missing_tables = sorted(set(FILES) - set(tables))
    if missing_tables:
        raise DataValidationError(f"Tabel wajib belum dimuat: {', '.join(missing_tables)}")

    frames = {name: tables[name].copy() for name in FILES}
    for name, frame in frames.items():
        _require_columns(name, frame)
        if frame.empty:
            raise DataValidationError(f"{FILES[name]} tidak boleh kosong.")

    claims = frames["claims"]
    _require_nonblank("claims", claims, REQUIRED_COLUMNS["claims"])
    _validate_unique("claims", claims, ["id_klaim"])
    claim_ids = set(claims["id_klaim"])

    for column in ("tgl_pengajuan", "tgl_pelayanan"):
        claims[f"_{column}"] = _date_values("claims", claims, column)
    if (claims["_tgl_pelayanan"] > claims["_tgl_pengajuan"]).any():
        raise DataValidationError("klaim.csv: tanggal pelayanan tidak boleh setelah tanggal pengajuan.")
    _numeric("claims", claims, "lama_rawat_hari", integer=True)
    _numeric("claims", claims, "total_klaim_idr")
    if not claims["jenis_layanan"].isin({"RJTL", "RITL"}).all():
        raise DataValidationError("klaim.csv: jenis_layanan harus RJTL atau RITL.")

    outpatient = claims["jenis_layanan"].eq("RJTL")
    outpatient_ok = (
        claims["kelas_rawat"].eq("-")
        & claims["severity"].eq("-")
        & claims["lama_rawat_hari"].eq(0)
    )
    inpatient_ok = (
        claims["kelas_rawat"].isin({"1", "2", "3"})
        & claims["severity"].isin({"I", "II", "III"})
        & claims["lama_rawat_hari"].ge(1)
    )
    if not outpatient_ok.loc[outpatient].all() or not inpatient_ok.loc[~outpatient].all():
        raise DataValidationError("klaim.csv: kelas, severity, atau lama rawat tidak konsisten dengan jenis layanan.")

    costs = frames["costs"]
    _require_nonblank("costs", costs, ["id_klaim", "komponen", "nilai", "satuan"])
    _validate_child_keys("costs", costs, claim_ids, ["id_klaim", "komponen"])
    if not costs["satuan"].eq("IDR").all():
        raise DataValidationError("klaim_biaya.csv: satuan biaya harus IDR.")
    _numeric("costs", costs, "nilai")
    cost_totals = costs.groupby("id_klaim")["nilai"].sum()
    if set(cost_totals.index) != claim_ids:
        raise DataValidationError("klaim_biaya.csv: setiap klaim harus memiliki rincian biaya.")
    declared_totals = claims.set_index("id_klaim")["total_klaim_idr"]
    if ((cost_totals - declared_totals).abs() > MONEY_TOLERANCE_IDR).any():
        raise DataValidationError("klaim_biaya.csv: jumlah komponen tidak cocok dengan total_klaim_idr.")
    component_names = set(costs["komponen"])
    if not {"obat", "alkes"}.issubset(component_names):
        raise DataValidationError("klaim_biaya.csv: komponen obat dan alkes wajib tersedia.")
    cost_matrix = costs.pivot(index="id_klaim", columns="komponen", values="nilai").fillna(0)
    for component in ("obat", "alkes"):
        if cost_matrix[component].isna().any():
            raise DataValidationError(f"klaim_biaya.csv: komponen {component} tidak valid.")

    diagnoses = frames["diagnoses"]
    _require_nonblank("diagnoses", diagnoses, ["id_klaim", "urutan", "kode_diagnosis", "jenis"])
    _validate_child_keys("diagnoses", diagnoses, claim_ids, ["id_klaim", "urutan"])
    _numeric("diagnoses", diagnoses, "urutan", integer=True)
    if diagnoses["urutan"].lt(1).any():
        raise DataValidationError("klaim_diagnosis.csv: urutan harus dimulai dari 1.")
    if not diagnoses["jenis"].isin({"utama", "sekunder"}).all():
        raise DataValidationError("klaim_diagnosis.csv: jenis harus utama atau sekunder.")
    diagnosis_codes = diagnoses["kode_diagnosis"].astype(str).str.upper()
    if not diagnosis_codes.map(lambda value: bool(ICD10_PATTERN.fullmatch(value))).all():
        raise DataValidationError("klaim_diagnosis.csv: ditemukan format kode diagnosis yang tidak valid.")
    primary_counts = diagnoses.loc[diagnoses["jenis"].eq("utama")].groupby("id_klaim").size()
    if not primary_counts.reindex(claim_ids, fill_value=0).eq(1).all():
        raise DataValidationError("klaim_diagnosis.csv: setiap klaim harus memiliki tepat satu diagnosis utama.")

    procedures = frames["procedures"]
    _require_nonblank("procedures", procedures, ["id_klaim", "urutan", "kode_tindakan"])
    _validate_child_keys("procedures", procedures, claim_ids, ["id_klaim", "urutan"])
    _numeric("procedures", procedures, "urutan", integer=True)
    if procedures["urutan"].lt(1).any():
        raise DataValidationError("klaim_tindakan.csv: urutan harus dimulai dari 1.")
    procedure_codes = procedures["kode_tindakan"].astype(str).str.upper()
    if not procedure_codes.map(lambda value: bool(ICD9_PATTERN.fullmatch(value))).all():
        raise DataValidationError("klaim_tindakan.csv: ditemukan format kode tindakan yang tidak valid.")

    items = frames["items"]
    _require_nonblank(
        "items", items,
        ["id_klaim", "urutan", "kode_item", "tipe_item", "jumlah", "harga_satuan_idr"],
    )
    _validate_child_keys("items", items, claim_ids, ["id_klaim", "urutan"])
    _numeric("items", items, "urutan", integer=True)
    _numeric("items", items, "jumlah", integer=True)
    _numeric("items", items, "harga_satuan_idr")
    if items["urutan"].lt(1).any() or items["jumlah"].lt(1).any():
        raise DataValidationError("klaim_item.csv: urutan dan jumlah harus bernilai minimal 1.")
    if not items["tipe_item"].isin({"obat", "alkes"}).all():
        raise DataValidationError("klaim_item.csv: tipe_item harus obat atau alkes.")
    items["nilai_baris_idr"] = items["jumlah"] * items["harga_satuan_idr"]
    item_totals = items.assign(
        komponen=items["tipe_item"].map({"obat": "obat", "alkes": "alkes"})
    ).groupby(["id_klaim", "komponen"])["nilai_baris_idr"].sum().unstack(fill_value=0)
    for component in ("obat", "alkes"):
        if component not in item_totals:
            item_totals[component] = 0.0
        item_totals[component] = item_totals[component].reindex(cost_matrix.index, fill_value=0)
        delta = (item_totals[component] - cost_matrix[component]).abs()
        if (delta > MONEY_TOLERANCE_IDR).any():
            raise DataValidationError(
                f"klaim_item.csv: nilai item {component} tidak cocok dengan komponen biaya."
            )

    reviews = frames["reviews"]
    _require_nonblank("reviews", reviews, ["id_klaim"])
    _validate_child_keys("reviews", reviews, claim_ids, ["id_klaim"])
    if set(reviews["id_klaim"]) != claim_ids:
        raise DataValidationError("hasil_verifikasi.csv: harus memiliki tepat satu baris per klaim.")
    valid_status = reviews["status"].isin({"final", "proses"})
    if not valid_status.all():
        raise DataValidationError("hasil_verifikasi.csv: status harus final atau proses.")
    final = reviews["status"].eq("final")
    if not reviews.loc[final, "hasil"].isin({"Disetujui", "Dikembalikan"}).all():
        raise DataValidationError("hasil_verifikasi.csv: status final harus memiliki hasil yang dikenal.")
    if reviews.loc[~final, "hasil"].astype(str).str.strip().ne("").any():
        raise DataValidationError("hasil_verifikasi.csv: hasil untuk status proses harus kosong.")
    if not reviews["perlu_klarifikasi"].isin({"Y", "N", ""}).all():
        raise DataValidationError("hasil_verifikasi.csv: perlu_klarifikasi harus Y, N, atau kosong.")
    review_dates = _date_values("reviews", reviews, "tgl_pemeriksaan", allow_blank=True)
    review_by_claim = reviews.set_index("id_klaim")["tgl_pemeriksaan"].astype(str)
    submission_by_claim = claims.set_index("id_klaim")["tgl_pengajuan"].astype(str)
    present_review_date = review_by_claim.str.strip().ne("")
    review_date_parsed = pd.to_datetime(
        review_by_claim[present_review_date], format="%Y-%m-%d", errors="coerce"
    )
    if review_date_parsed.isna().any():
        raise DataValidationError("hasil_verifikasi.csv: tgl_pemeriksaan harus kosong atau YYYY-MM-DD.")
    submitted_for_review = pd.to_datetime(
        submission_by_claim.loc[review_date_parsed.index], format="%Y-%m-%d"
    )
    if (review_date_parsed.to_numpy() < submitted_for_review.to_numpy()).any():
        raise DataValidationError("hasil_verifikasi.csv: tanggal pemeriksaan sebelum pengajuan.")
    if (final & reviews["tgl_pemeriksaan"].astype(str).str.strip().eq("")).any():
        raise DataValidationError("hasil_verifikasi.csv: hasil final harus memiliki tgl_pemeriksaan.")

    dictionary = frames["dictionary"]
    _require_nonblank("dictionary", dictionary, REQUIRED_COLUMNS["dictionary"])
    dictionary_notes = " ".join(dictionary["keterangan"].astype(str)).upper()
    if "DATA SINTETIS" not in dictionary_notes or "BUKAN LABEL FRAUD/ANOMALI" not in dictionary_notes:
        raise DataValidationError(
            "kamus_data.csv: catatan sintetis dan batas label hasil verifikasi harus tetap tercantum."
        )

    diagnosis_sorted = diagnoses.sort_values(["id_klaim", "urutan"])
    primary = diagnosis_sorted.loc[diagnosis_sorted["jenis"].eq("utama")].set_index("id_klaim")[
        "kode_diagnosis"
    ].rename("diagnosis_primary")
    secondary = _aggregate_codes(
        diagnosis_sorted.loc[diagnosis_sorted["jenis"].eq("sekunder")], "kode_diagnosis"
    ).rename("diagnosis_secondary")
    procedure_list = _aggregate_codes(
        procedures.sort_values(["id_klaim", "urutan"]), "kode_tindakan"
    ).rename("procedure_codes")
    item_list = _aggregate_codes(
        items.sort_values(["id_klaim", "urutan"]), "kode_item"
    ).rename("item_codes")

    normalized = claims.set_index("id_klaim").rename(
        columns={
            "id_peserta_token": "patient_token",
            "id_faskes": "facility_id",
            "tgl_pengajuan": "submitted_at",
            "tgl_pelayanan": "service_date",
            "jenis_layanan": "service_type",
            "kelas_rawat": "ward_class",
            "severity": "severity",
            "lama_rawat_hari": "length_of_stay_days",
            "versi_grouper": "grouper_version",
            "versi_tarif": "tariff_version",
            "total_klaim_idr": "claim_total_idr",
        }
    )
    normalized = normalized.drop(columns=["_tgl_pengajuan", "_tgl_pelayanan"], errors="ignore")
    normalized = normalized.join(primary).join(secondary).join(procedure_list).join(item_list)
    normalized[["diagnosis_secondary", "procedure_codes", "item_codes"]] = normalized[
        ["diagnosis_secondary", "procedure_codes", "item_codes"]
    ].fillna("")
    normalized["diagnosis_codes"] = normalized.apply(
        lambda row: "|".join(
            code for code in [row["diagnosis_primary"], row["diagnosis_secondary"]] if code
        ),
        axis=1,
    )
    normalized["procedure_count"] = procedures.groupby("id_klaim").size().reindex(
        normalized.index, fill_value=0
    )
    normalized["item_count"] = items.groupby("id_klaim").size().reindex(
        normalized.index, fill_value=0
    )
    for component in sorted(component_names):
        column = "cost_" + re.sub(r"[^a-zA-Z0-9]+", "_", component).strip("_").lower() + "_idr"
        if column in normalized.columns:
            raise DataValidationError(f"Nama komponen biaya menghasilkan kolom duplikat: {column}.")
        normalized[column] = cost_matrix[component].reindex(normalized.index, fill_value=0)
    normalized = normalized.reset_index()
    forbidden = FEATURE_FORBIDDEN_COLUMNS & set(normalized.columns)
    if forbidden:
        raise DataValidationError(f"Kolom hasil verifikasi/label tidak boleh menjadi fitur: {sorted(forbidden)}")
    if len(normalized) != len(claims) or normalized["id_klaim"].duplicated().any():
        raise DataValidationError("Normalisasi gagal mempertahankan satu baris unik per klaim.")

    report = {
        "dataset": "synthetic_claim_tables_v2",
        "claims": int(len(claims)),
        "child_rows": {
            "costs": int(len(costs)),
            "diagnoses": int(len(diagnoses)),
            "procedures": int(len(procedures)),
            "items": int(len(items)),
        },
        "review_rows_kept_separate": int(len(reviews)),
        "dictionary_entries": int(len(dictionary)),
        "review_status_counts": {
            str(key): int(value) for key, value in reviews["status"].value_counts().items()
        },
        "service_type_counts": {
            str(key): int(value) for key, value in claims["jenis_layanan"].value_counts().items()
        },
        "checks": {
            "claim_id_unique": True,
            "child_foreign_keys_valid": True,
            "claim_totals_reconcile": True,
            "item_totals_reconcile": True,
            "one_primary_diagnosis_per_claim": True,
            "review_data_excluded_from_features": True,
        },
        "note": "Dataset sintetis; hasil verifikasi bukan label fraud/anomali.",
    }
    return V2Dataset(claims=normalized, reviews=reviews.copy(), report=report)


def load_v2(root):
    """Load source CSVs from <root>/data and normalize them without modifying inputs."""
    data_dir = os.path.join(root, "data")
    tables = {}
    for name, filename in FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File wajib tidak ditemukan: {path}")
        tables[name] = pd.read_csv(path, keep_default_na=False, dtype=str)
    return validate_and_normalize(tables)


def run(root, outdir=None):
    dataset = load_v2(root)
    output_dir = outdir or os.path.join(root, "prep_v2")
    os.makedirs(output_dir, exist_ok=True)
    claims_path = os.path.join(output_dir, "claims_v2_flat.csv.gz")
    report_path = os.path.join(output_dir, "data_quality_v2.json")
    dataset.claims.to_csv(claims_path, index=False, compression="gzip")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(dataset.report, stream, ensure_ascii=False, indent=2)
    return claims_path, report_path, dataset.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="out", help="Folder output utama yang berisi subfolder data.")
    parser.add_argument("--outdir", default=None, help="Folder output; default: <root>/prep_v2.")
    args = parser.parse_args()
    claims_path, report_path, report = run(args.root, args.outdir)
    print(
        f"OK: {report['claims']} klaim tervalidasi; "
        f"hasil fitur={claims_path}; laporan={report_path}"
    )


if __name__ == "__main__":
    main()
