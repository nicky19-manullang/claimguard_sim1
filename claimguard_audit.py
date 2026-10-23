#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 9: audit operasional dan kesiapan deployment

Tujuan:
  menilai apakah model siap dipakai untuk review operasional, bukan hanya untuk eksperimen.
  Fokus pada beban review, trade-off precision/recall, FPR, konsentrasi per faskes,
  dan ambang yang paling sesuai untuk operasional.

Masukan:
  out/calibrated/calibrated_{skema}_{mode}.csv.gz
  out/detect/detect_{skema}_{mode}.csv.gz
  out/data/claims.csv
  out/eval_only/labels.csv

Keluaran:
  out/audit/audit_{skema}_{mode}.csv.gz
  out/audit/audit_report.md
"""
import argparse
import os

import numpy as np
import pandas as pd

LEVELS = [0.01, 0.03, 0.05, 0.10]
PRIORITY_LABELS = ["Critical", "High", "Medium", "Low"]


def md(df, fmt="{:.3f}"):
    L = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
         "|" + "---|" * (len(df.columns) + 1)]
    for i, r in df.iterrows():
        L.append("| " + " | ".join([
            str(i),
            *(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("-" if pd.isna(v) else str(v)) for v in r)
        ]) + " |")
    return "\n".join(L)


def build_threshold_rows(T):
    rows = []
    y1 = T["is_anomaly"] == 1
    y0 = T["is_anomaly"] == 0
    n_all = len(T)
    n_anom = y1.sum()
    n_norm = y0.sum()
    for a in LEVELS:
        flagged = T["p_value"] <= a
        n_flagged = int(flagged.sum())
        review_share = n_flagged / n_all
        precision = float(T.loc[flagged, "is_anomaly"].mean()) if n_flagged else np.nan
        recall = float(T.loc[y1, "p_value"].le(a).mean()) if n_anom else np.nan
        fpr = float(T.loc[y0, "p_value"].le(a).mean()) if n_norm else np.nan
        rows.append({
            "threshold": a,
            "n_flagged": n_flagged,
            "review_share": review_share,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "expected_review_cost": n_flagged,
            "flagged_anomaly": int(T.loc[flagged, "is_anomaly"].sum()) if n_flagged else 0,
            "flagged_normal": int((~y1[flagged]).sum()) if n_flagged else 0,
        })
    return pd.DataFrame(rows).set_index("threshold")


def summarize_levels(T):
    rows = []
    for level in PRIORITY_LABELS:
        s = T[T["tingkat"] == level]
        rows.append({
            "tingkat": level,
            "n_klaim": int(len(s)),
            "share": len(s) / len(T),
            "precision": float(s["is_anomaly"].mean()) if len(s) else np.nan,
            "recall_vs_all": float(s["is_anomaly"].sum() / max((T["is_anomaly"] == 1).sum(), 1)),
            "fpr": float(((s["is_anomaly"] == 0).sum()) / max((T["is_anomaly"] == 0).sum(), 1)) if len(s) else np.nan,
        })
    return pd.DataFrame(rows).set_index("tingkat")


def facility_summary(T):
    n_all = len(T)
    by_fac = T[T["is_anomaly"] == 0].groupby("facility_id").agg(
        normal_claims=("claim_id", "size"),
        fpr_05=("p_value", lambda s: float((s <= 0.05).mean())),
        fpr_01=("p_value", lambda s: float((s <= 0.01).mean())),
    ).reset_index()
    by_fac["review_share"] = by_fac["normal_claims"] / max(by_fac["normal_claims"].sum(), 1)
    return by_fac.sort_values(["fpr_05", "normal_claims"], ascending=[False, False]).head(10)


def read_operational_table(root, scheme, mode):
    suf = f"{scheme}_{mode}"
    detect = pd.read_csv(os.path.join(root, "detect", f"detect_{suf}.csv.gz"))
    cal = pd.read_csv(os.path.join(root, "calibrated", f"calibrated_{suf}.csv.gz"))
    labels = pd.read_csv(os.path.join(root, "eval_only", "labels.csv"), keep_default_na=False)
    claims = pd.read_csv(os.path.join(root, "data", "claims.csv"), usecols=["claim_id", "facility_id", "jenis_layanan"])

    d = detect.merge(cal[["claim_id", "split", "p_value", "skor_0_100", "tingkat"]], on=["claim_id", "split"], how="inner")
    d = d.merge(labels[["claim_id", "is_anomaly", "anomaly_type", "difficulty", "hard_negative_type"]], on="claim_id", how="left")
    d = d.merge(claims, on="claim_id", how="left")
    test = d[d["split"] == "test"].copy().reset_index(drop=True)
    return test


def recommended_threshold(summary_df):
    cand = summary_df[summary_df["fpr"] <= 0.05].sort_values(["review_share", "recall"], ascending=[True, False])
    if cand.empty:
        return summary_df.sort_values("review_share", ascending=True).iloc[0]
    return cand.iloc[0]


def build_report_for_scheme_mode(root, scheme, mode):
    T = read_operational_table(root, scheme, mode)
    if T.empty:
        return None

    body = []
    thr = build_threshold_rows(T)
    lvl = summarize_levels(T)
    fac = facility_summary(T)

    # recommended threshold
    rec = recommended_threshold(thr)
    rec_level = float(rec.name)
    rec_row = thr.loc[rec.name]

    # summary headline
    body.append(f"## Skema `{scheme}`, mode `{mode}`\n")
    body.append(f"- Total test claims: {len(T):,}\n")
    body.append(f"- Anomali: {int((T['is_anomaly']==1).sum()):,}\n")
    body.append(f"- Normal: {int((T['is_anomaly']==0).sum()):,}\n")
    body.append(f"- Rekomendasi ambang operasional: p <= {rec_level:.2f} ({rec_row['review_share']:.1%} review burden, FPR={rec_row['fpr']:.3f}, Recall={rec_row['recall']:.3f})\n")

    # threshold table
    body.append("### 1) Trade-off review burden vs recall/FPR\n")
    body.append(md(thr[["n_flagged", "review_share", "precision", "recall", "fpr"]].rename(columns={
        "n_flagged": "flagged",
        "review_share": "review_share",
        "precision": "precision",
        "recall": "recall",
        "fpr": "fpr",
    })) + "\n")

    body.append("### 2) Tingkat prioritas dari skor terkalibrasi\n")
    body.append(md(lvl.rename(columns={
        "n_klaim": "n_claims",
        "share": "share",
        "precision": "precision",
        "recall_vs_all": "recall_vs_all",
        "fpr": "fpr",
    })) + "\n")

    body.append("### 3) Konsentrasi per fasilitas pada klaim normal\n")
    fac_cols = fac[["facility_id", "normal_claims", "fpr_05", "fpr_01"]].copy()
    fac_cols = fac_cols.rename(columns={"normal_claims": "normal_claims", "fpr_05": "FPR@5%", "fpr_01": "FPR@1%"})
    body.append(md(fac_cols.set_index("facility_id")) + "\n")

    # hard negatives
    hard = T[(T["hard_negative_type"] != "none") & (T["hard_negative_type"].notna())]
    if not hard.empty:
        rows = []
        for a in LEVELS:
            d = hard[hard["p_value"] <= a]
            rows.append({
                "threshold": a,
                "hard_negative_flagged": int(len(d)),
                "share_flagged": len(d) / len(hard),
                "median_score": float(d["skor_0_100"].median()) if len(d) else np.nan,
            })
        hard_df = pd.DataFrame(rows).set_index("threshold")
        body.append("### 4) Hard negatives yang masih ditandai\n")
        body.append(md(hard_df) + "\n")
    else:
        body.append("### 4) Hard negatives yang masih ditandai\n")
        body.append("Tidak ada hard negative pada dataset test.\n")

    body.append("### 5) Penilaian operasional\n")
    body.append(
        "- Model ini masuk kategori layak untuk review tertarget pada ambang yang disarankan bila review burden di bawah ~5-10% dan FPR pada klaim normal tetap terkendali.\n"
        "- Jika prioritas adalah deteksi agresif untuk kasus berisiko tinggi, gunakan ambang 1%-3% untuk review kritis.\n"
        "- Jika prioritas adalah throughput dan biaya review yang lebih rendah, gunakan ambang 5% dengan pengawasan FPR per fasilitas.\n"
    )

    return "\n".join(body), thr, lvl, fac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--schemes", default="faskes,waktu")
    ap.add_argument("--modes", default="retro,asof")
    a = ap.parse_args()

    outdir = os.path.join(a.root, "audit")
    os.makedirs(outdir, exist_ok=True)

    sections = [
        "# Audit operasional dan kesiapan deployment (tahap 9)\n",
        "Dokumen ini mengevaluasi apakah skor terkalibrasi siap dipakai untuk review manusia dalam operasional. Fokus bukan pada akurasi murni tetapi pada trade-off: review burden, FPR, recall, konsentrasi per fasilitas, dan ambang yang dapat dipertanggungjawabkan.\n",
    ]

    for scheme in a.schemes.split(","):
        for mode in a.modes.split(","):
            res = build_report_for_scheme_mode(a.root, scheme, mode)
            if res is None:
                sections.append(f"\n## Skema `{scheme}`, mode `{mode}`\nTidak ada data test untuk audit.\n")
                continue
            txt, thr, lvl, fac = res
            sections.append(txt)
            out_path = os.path.join(outdir, f"audit_{scheme}_{mode}.csv.gz")
            thr.reset_index().to_csv(out_path, index=False)
            print(f"[{scheme}/{mode}] audit siap -> {out_path}", flush=True)

    report_path = os.path.join(outdir, "audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    print(f"[ok] audit report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
