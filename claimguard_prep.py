#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 2: preprocessing aman dari kebocoran

Dua lapis:
  1. clean_stateless(df): langkah yang tidak belajar dari data (normalisasi & validasi kode ICD,
     penanganan missing, pseudonimisasi ID, penguraian daftar kode, episode). Aman dijalankan
     pada seluruh data.
  2. PrepPipeline.fit(train).transform(df): langkah yang belajar dari data (kosakata kode yang
     dikenal, referensi biaya dan lama rawat per (diagnosis, severity)). fit HANYA pada split latih.

Pemakaian:
    python claimguard_prep.py --root out            # membuat out/prep/claims_prep_{faskes,waktu}.csv.gz
"""
import argparse
import hashlib
import hmac
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

from claimguard_eval import assert_no_label_columns

ICD10 = re.compile(r"^[A-Z]\d{2}(\.\d{1,2})?$")
ICD9 = re.compile(r"^\d{2}(\.\d{1,2})?$")
START = pd.Timestamp("2025-01-01")


# ----------------------------------------------------------------------------- util
def pseudonymize(series, salt, prefix):
    """ID -> token HMAC-SHA256 terpotong. Salt disimpan terpisah dari data."""
    cache = {}

    def h(x):
        if x not in cache:
            cache[x] = prefix + hmac.new(salt.encode(), str(x).encode(), hashlib.sha256).hexdigest()[:10]
        return cache[x]

    return series.map(h)


def norm_icd10(code):
    c = str(code).strip().upper().replace(",", ".")
    if c and "." not in c and len(c) > 3:
        c = c[:3] + "." + c[3:]
    return c


def norm_icd9(code):
    c = str(code).strip().replace(",", ".")
    if c and "." not in c and len(c) > 2:
        c = c[:2] + "." + c[2:]
    return c


def split_codes(s):
    return [x for x in str(s).split("|") if x.strip() != ""]


# ------------------------------------------------------------------ langkah tanpa fit
def clean_stateless(df, salt="claimguard-demo-salt", gap_days=1):
    assert_no_label_columns(df)
    d = df.copy()
    rep = {}

    # --- kode diagnosis utama
    raw = d["diagnosis_utama"].astype(str)
    d["dx"] = raw.map(norm_icd10)
    d["dx_format_fixed"] = (raw != d["dx"]).astype(int)
    d["dx_invalid"] = (~d["dx"].str.match(ICD10)).astype(int)
    rep["dx_diperbaiki"] = int(d["dx_format_fixed"].sum())
    rep["dx_tetap_invalid"] = int(d["dx_invalid"].sum())

    # --- diagnosis sekunder & tindakan (daftar)
    d["sec_list"] = d["diagnosis_sekunder"].map(lambda s: [norm_icd10(x) for x in split_codes(s)])
    raw_p = d["tindakan"].map(split_codes)
    d["proc_list"] = raw_p.map(lambda l: [norm_icd9(x) for x in l])
    d["proc_format_fixed"] = [int(a != b) for a, b in zip(raw_p, d["proc_list"])]
    d["proc_invalid"] = d["proc_list"].map(lambda l: int(any(not ICD9.match(x) for x in l)))
    rep["tindakan_diperbaiki"] = int(d["proc_format_fixed"].sum())
    rep["tindakan_tetap_invalid"] = int(d["proc_invalid"].sum())
    d["item_list"] = d["obat_alkes"].map(split_codes)

    # --- missing
    d["doctor_missing"] = (d["doctor_id"].astype(str).str.strip() == "").astype(int)
    d["items_missing"] = (d["item_list"].map(len) == 0).astype(int)
    rep["doctor_kosong"] = int(d["doctor_missing"].sum())
    rep["obat_alkes_kosong"] = int(d["items_missing"].sum())
    d["doctor_id"] = d["doctor_id"].where(d["doctor_missing"] == 0, "UNKNOWN")

    # --- pseudonimisasi (data riil nanti: hapus kolom mentah)
    d["patient_key"] = pseudonymize(d["patient_id"], salt, "P")
    d["doctor_key"] = np.where(d["doctor_missing"] == 1, "UNKNOWN", pseudonymize(d["doctor_id"], salt, "D"))
    d = d.drop(columns=["patient_id", "doctor_id", "diagnosis_utama", "diagnosis_sekunder", "tindakan", "obat_alkes"])

    # --- numerik & tanggal
    for c in ("tanggal_masuk", "tanggal_pulang", "tanggal_klaim"):
        d[c] = pd.to_datetime(d[c])
    d["masuk_d"] = (d["tanggal_masuk"] - START).dt.days
    d["pulang_d"] = (d["tanggal_pulang"] - START).dt.days
    d["klaim_d"] = (d["tanggal_klaim"] - START).dt.days
    d["klaim_lag"] = d["klaim_d"] - d["pulang_d"]
    d["log_nilai"] = np.log(d["nilai_klaim"].clip(lower=1))
    d["n_items"] = d["item_list"].map(len)
    d["n_sec"] = d["sec_list"].map(len)
    d["n_proc"] = d["proc_list"].map(len)
    d["dx3"] = d["dx"].str[:3]
    d["days_since_prev_claim"] = pd.to_numeric(d["days_since_prev_claim"], errors="coerce")
    d["is_weekend_masuk"] = (d["tanggal_masuk"].dt.weekday >= 5).astype(int)
    d["bulan_masuk"] = d["tanggal_masuk"].dt.month
    d["severity_num"] = d["severity_inacbg"].map({"0": 0, "I": 1, "II": 2, "III": 3})

    # --- episode: pasien + faskes + diagnosis (3 karakter) + jendela waktu
    keys = ["patient_key", "facility_id", "dx3"]
    d = d.sort_values(keys + ["masuk_d", "klaim_d", "claim_id"]).reset_index(drop=True)
    cm = d.groupby(keys, sort=False)["pulang_d"].cummax()
    prev_end = cm.groupby([d[k] for k in keys], sort=False).shift(1)
    new_ep = prev_end.isna() | ((d["masuk_d"] - prev_end) > gap_days)
    d["episode_id"] = "EP-" + (new_ep.cumsum()).astype(str).str.zfill(6)
    ge = d.groupby("episode_id", sort=False)
    d["episode_size"] = ge["claim_id"].transform("size")          # fitur batch retrospektif
    d["episode_rank"] = ge.cumcount()                              # urutan dalam episode (tanggal masuk)
    d["episode_total_nilai"] = ge["nilai_klaim"].transform("sum")
    d["episode_span_days"] = ge["pulang_d"].transform("max") - ge["masuk_d"].transform("min")
    rep["n_episode"] = int(d["episode_id"].nunique())
    rep["episode_multi_klaim"] = int((d.groupby("episode_id")["claim_id"].size() > 1).sum())
    d = d.sort_values("claim_id").reset_index(drop=True)
    assert_no_label_columns(d)
    return d, rep


# --------------------------------------------------------------------- langkah dengan fit
class PrepPipeline:
    """Belajar hanya dari data latih: kosakata kode dan referensi biaya/lama rawat."""

    def __init__(self, min_count=5, min_group=20):
        self.min_count, self.min_group = min_count, min_group
        self.fitted = False

    def fit(self, tr):
        mc = self.min_count
        self.dx_vocab = set(tr["dx"].value_counts()[lambda s: s >= mc].index)
        self.proc_vocab = set(pd.Series([p for l in tr["proc_list"] for p in l]).value_counts()[lambda s: s >= mc].index)
        self.item_vocab = set(pd.Series([i for l in tr["item_list"] for i in l]).value_counts()[lambda s: s >= mc].index)
        # referensi biaya/lama rawat: (dx, severity) -> dx -> global
        g = lambda keys, col: tr.groupby(keys)[col].agg(["median", "size"])
        self.ref = {}
        for col in ("log_nilai", "lama_rawat"):
            a = g(["dx", "severity_num"], col)
            b = g(["dx"], col)
            self.ref[col] = dict(
                by_dx_sev=a[a["size"] >= self.min_group]["median"].to_dict(),
                by_dx=b[b["size"] >= self.min_group]["median"].to_dict(),
                glob=float(tr[col].median()))
        mad = (tr["log_nilai"] - tr.set_index(["dx", "severity_num"]).index.map(self.ref["log_nilai"]["by_dx_sev"]).to_numpy(dtype=float))
        self.log_scale = float(np.nanmedian(np.abs(mad[~np.isnan(mad)])) * 1.4826) or 0.3
        self.fitted = True
        return self

    def _ref(self, d, col):
        r = self.ref[col]
        a = pd.Series(list(zip(d["dx"], d["severity_num"]))).map(r["by_dx_sev"])
        b = d["dx"].map(r["by_dx"])
        return a.fillna(b).fillna(r["glob"]).to_numpy(dtype=float)

    def transform(self, d):
        assert self.fitted, "panggil fit() pada data latih dulu"
        o = d.copy()
        o["dx_known"] = o["dx"].isin(self.dx_vocab).astype(int)
        o["n_proc_unknown"] = o["proc_list"].map(lambda l: sum(p not in self.proc_vocab for p in l))
        o["n_item_unknown"] = o["item_list"].map(lambda l: sum(i not in self.item_vocab for i in l))
        o["ref_log_nilai"] = self._ref(o, "log_nilai")
        o["ref_lama_rawat"] = self._ref(o, "lama_rawat")
        o["resid_log_nilai_naif"] = o["log_nilai"] - o["ref_log_nilai"]       # model expected cost penuh: tahap 4
        o["resid_lama_rawat_naif"] = o["lama_rawat"] - o["ref_lama_rawat"]
        assert_no_label_columns(o)
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--gap", type=int, default=1, help="jeda hari maksimum antar klaim dalam satu episode")
    ap.add_argument("--salt", default="claimguard-demo-salt")
    a = ap.parse_args()

    claims = pd.read_csv(os.path.join(a.root, "data", "claims.csv"), keep_default_na=False)
    splits = pd.read_csv(os.path.join(a.root, "data", "splits.csv"))
    clean, rep = clean_stateless(claims, salt=a.salt, gap_days=a.gap)
    out = os.path.join(a.root, "prep")
    os.makedirs(out, exist_ok=True)
    report = {"stateless": rep, "per_skema": {}}
    for scheme, col in (("faskes", "split_faskes"), ("waktu", "split_waktu")):
        d = clean.merge(splits[["claim_id", col]].rename(columns={col: "split"}), on="claim_id")
        pipe = PrepPipeline().fit(d[d["split"] == "train"])
        full = pipe.transform(d)
        for c in ("sec_list", "proc_list", "item_list"):
            full[c] = full[c].map("|".join)
        full.to_csv(os.path.join(out, f"claims_prep_{scheme}.csv.gz"), index=False)
        with open(os.path.join(out, f"pipeline_{scheme}.pkl"), "wb") as f:
            pickle.dump(pipe, f)
        tr, te = full[full["split"] == "train"], full[full["split"] == "test"]
        report["per_skema"][scheme] = {
            "n_train": int(len(tr)), "n_test": int((full["split"] == "test").sum()),
            "kosakata_dx": len(pipe.dx_vocab), "kosakata_tindakan": len(pipe.proc_vocab),
            "kosakata_item": len(pipe.item_vocab),
            "test_dx_tidak_dikenal": int((te["dx_known"] == 0).sum()),
            "test_item_tidak_dikenal": int(te["n_item_unknown"].sum()),
        }
    with open(os.path.join(out, "prep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
