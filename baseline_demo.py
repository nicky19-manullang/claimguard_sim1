#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uji harness evaluasi dengan 3 baseline sederhana (BUKAN model final):
  1. acak
  2. z-score robust log(nilai_klaim) per (diagnosis, severity), di-fit hanya pada split latih
  3. aturan repeat naif: diagnosis sama dengan klaim sebelumnya pasien dan jarak <= 14 hari
Dipakai untuk memastikan metrik masuk akal dan skema split tidak bocor.
"""
import sys

import numpy as np
import pandas as pd

from claimguard_eval import Evaluator, assert_no_label_columns, format_report

ROOT = sys.argv[1] if len(sys.argv) > 1 else "out"


def norm_dx(s):
    s = s.fillna("").str.strip().str.upper().str.replace(",", ".", regex=False)
    return s.where(s.str.contains(".", regex=False) | (s.str.len() <= 3), s.str[:3] + "." + s.str[3:])


claims = pd.read_csv(f"{ROOT}/data/claims.csv", keep_default_na=False)
splits = pd.read_csv(f"{ROOT}/data/splits.csv")
assert_no_label_columns(claims)  # fitur tidak boleh mengandung label
claims["dx"] = norm_dx(claims["diagnosis_utama"])
claims["logv"] = np.log(claims["nilai_klaim"].clip(lower=1))
df = claims.merge(splits, on="claim_id")
ev = Evaluator(ROOT)
rng = np.random.default_rng(0)

# 1) acak
random_scores = pd.Series(rng.random(len(df)), index=df["claim_id"])

# 3) aturan repeat naif (memakai riwayat pasien, tidak butuh fit)
prev = df.set_index("claim_id")["dx"]
prev_dx = df["prev_claim_id"].map(prev).fillna("")
gap = pd.to_numeric(df["days_since_prev_claim"], errors="coerce").fillna(999)
rule = ((df["dx"] == prev_dx) & (gap <= 14)).astype(float)
rule_scores = pd.Series(rule.to_numpy(), index=df["claim_id"])


def costz_scores(scheme):
    col = {"faskes": "split_faskes", "waktu": "split_waktu"}[scheme]
    tr = df[df[col] == "train"]
    g = tr.groupby(["dx", "severity_inacbg"])["logv"].agg(med="median", mad=lambda x: np.median(np.abs(x - np.median(x))))
    g["mad"] = g["mad"].clip(lower=0.05) * 1.4826
    m = df.merge(g, left_on=["dx", "severity_inacbg"], right_index=True, how="left")
    z = ((m["logv"] - m["med"]) / m["mad"]).fillna(0.0)
    return pd.Series(z.to_numpy(), index=m["claim_id"])


for scheme in ("faskes", "waktu"):
    for name, sc in (("ACAK", random_scores), ("COST-Z (fit di train)", costz_scores(scheme)),
                     ("ATURAN REPEAT NAIF", rule_scores)):
        print(f"\n######## {name}")
        res = ev.evaluate(sc, scheme=scheme, split="test", budget=0.05, n_boot=100)
        print(format_report(res))
