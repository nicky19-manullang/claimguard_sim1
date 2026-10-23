#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 6: skor terkalibrasi 0-100 dengan conformal anomaly detection

Masukan : out/detect/detect_{skema}_{mode}.csv.gz  (kolom risk_raw = skor hibrida tahap 5)
Keluaran: out/calibrated/calibrated_{skema}_{mode}.csv.gz   claim_id, split, kelompok, p_value, skor_0_100, tingkat
          out/calibrated/calibrator_{skema}_{mode}.pkl      objek Calibrator siap pakai
          out/calibrated/kalibrasi_report.md

Gagasan
-------
p-value konformal klaim x = (1 + #{klaim normal kalibrasi dengan skor >= skor x}) / (n + 1).
Kecil = klaim itu lebih ekstrem daripada hampir semua klaim normal. Jika klaim normal baru bisa
dipertukarkan (exchangeable) dengan klaim kalibrasi, maka P(p <= a) <= a untuk klaim normal,
sehingga ambang p <= a menandai paling banyak a dari klaim normal (laju false positive terkontrol).

* Kalibrasi: klaim NORMAL (is_anomaly = 0, termasuk hard negatives) pada split VAL saja.
  Detektor (tahap 5) di-fit di train, jadi val belum pernah dipakai untuk memfit skor.
  Satu-satunya pemakaian val sebelumnya: memilih penggabung (maks vs bobot) lewat CV, bukan membentuk skor.
* Seri skor dipecah acak (smoothed conformal p-value) supaya p-value tidak terlalu konservatif.
* Mondrian: kalibrasi terpisah per kelompok (default: jenis_layanan RJTL/RITL). Kelompok dengan
  klaim normal kalibrasi < min_cal memakai kalibrasi gabungan.
* skor_0_100 = 100 * (1 - p). Tingkat: Critical p<=0,01 ; High p<=0,03 ; Medium p<=0,05 ; Low selainnya.
* Label hanya dipakai untuk (a) memilih klaim normal di val, (b) evaluasi di test.

    python claimguard_calibrate.py --root out
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

TIERS = [("Critical", 0.01), ("High", 0.03), ("Medium", 0.05)]
LEVELS = [0.01, 0.03, 0.05, 0.10]


class Calibrator:
    def __init__(self, min_cal=500, seed=0):
        self.min_cal, self.seed = min_cal, seed

    def fit(self, score, group, is_normal_val):
        score, group = np.asarray(score, float), np.asarray(group)
        m = np.asarray(is_normal_val, bool)
        self.pool = np.sort(score[m])
        self.groups = {}
        for g in np.unique(group):
            s = score[m & (group == g)]
            if len(s) >= self.min_cal:
                self.groups[g] = np.sort(s)
        return self

    def p_values(self, score, group, seed=None):
        score, group = np.asarray(score, float), np.asarray(group)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        u = rng.random(len(score))
        p = np.empty(len(score))
        for g in np.unique(group):
            i = group == g
            cal = self.groups.get(g, self.pool)
            n = len(cal)
            ge = n - np.searchsorted(cal, score[i], side="left")      # cal >= s
            gt = n - np.searchsorted(cal, score[i], side="right")     # cal >  s
            p[i] = (1 + gt + u[i] * (ge - gt)) / (n + 1)               # ties: acak
        return p

    @staticmethod
    def to_score(p):
        return 100.0 * (1.0 - np.asarray(p))

    @staticmethod
    def to_tier(p):
        p = np.asarray(p)
        out = np.full(len(p), "Low", dtype=object)
        for name, a in reversed(TIERS):
            out[p <= a] = name
        return out


def md(df, fmt="{:.3f}"):
    L = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |", "|" + "---|" * (len(df.columns) + 1)]
    for i, r in df.iterrows():
        L.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("-" if pd.isna(v) else str(v)) for v in r]) + " |")
    return "\n".join(L)


def boot_fpr(fac, p, y0, a, n_boot=300, seed=0):
    """CI95 bootstrap klaster per faskes untuk FPR (klaim normal dengan p <= a)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(fac)
    idx = {f: np.where((fac == f) & y0)[0] for f in ids}
    flag = p <= a
    cnt = np.array([flag[idx[f]].sum() for f in ids], float)
    tot = np.array([len(idx[f]) for f in ids], float)
    v = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ids), len(ids))
        v.append(cnt[k].sum() / max(tot[k].sum(), 1))
    return np.percentile(v, [2.5, 97.5])


def run(root, scheme, mode, groupby, min_cal, seed, outdir):
    suf = f"{scheme}_{mode}"
    D = pd.read_csv(os.path.join(root, "detect", f"detect_{suf}.csv.gz"))[["claim_id", "split", "risk_raw"]]
    cl = pd.read_csv(os.path.join(root, "data", "claims.csv"), usecols=["claim_id", "facility_id", "jenis_layanan", "jenis_faskes"])
    lab = pd.read_csv(os.path.join(root, "eval_only", "labels.csv"), keep_default_na=False)
    df = D.merge(cl, on="claim_id").merge(lab, on="claim_id")
    df["grp"] = df[groupby] if groupby != "none" else "ALL"
    va, te = df["split"] == "val", df["split"] == "test"
    normal_val = va & (df["is_anomaly"] == 0)

    cal = Calibrator(min_cal=min_cal, seed=seed).fit(df["risk_raw"], df["grp"], normal_val)
    df["p_value"] = cal.p_values(df["risk_raw"], df["grp"])
    df["skor_0_100"] = Calibrator.to_score(df["p_value"])
    df["tingkat"] = Calibrator.to_tier(df["p_value"])
    # pembanding: kalibrasi marginal (tanpa Mondrian)
    cal_m = Calibrator(min_cal=10 ** 9, seed=seed).fit(df["risk_raw"], df["grp"], normal_val)
    df["p_marg"] = cal_m.p_values(df["risk_raw"], df["grp"])

    T = df[te].reset_index(drop=True)
    y1, y0 = (T["is_anomaly"] == 1).to_numpy(), (T["is_anomaly"] == 0).to_numpy()
    fac = T["facility_id"].to_numpy()
    R = []

    # 1. validitas: FPR empiris pada klaim normal vs a
    rows = []
    for name, col in (("Mondrian (" + groupby + ")", "p_value"), ("Marginal", "p_marg")):
        for a in LEVELS:
            f = (T[col] <= a).to_numpy()
            lo, hi = boot_fpr(fac, T[col].to_numpy(), y0, a, seed=seed)
            rows.append({"kalibrasi": name, "a (dijanjikan)": a, "FPR empiris": f[y0].mean(),
                         "CI95": f"[{lo:.3f}, {hi:.3f}]", "rasio FPR/a": f[y0].mean() / a,
                         "Precision": T.loc[f & y1 | f & y0, "is_anomaly"].mean() if f.any() else np.nan,
                         "Recall": f[y1].mean(), "ditandai": f.mean()})
    V = pd.DataFrame(rows).set_index("kalibrasi")
    R.append("### 1. Validitas: FPR empiris pada klaim normal (test) vs level yang dijanjikan\n")
    R.append(md(V) + "\n")

    # 2. FPR per kelompok (Mondrian)
    rows = []
    for g, s in T.groupby("grp"):
        r = {"kelompok": g, "n normal": int((s["is_anomaly"] == 0).sum()), "n anomali": int((s["is_anomaly"] == 1).sum())}
        for a in (0.01, 0.05):
            r[f"FPR@{a}"] = float((s.loc[s["is_anomaly"] == 0, "p_value"] <= a).mean())
            r[f"Recall@{a}"] = float((s.loc[s["is_anomaly"] == 1, "p_value"] <= a).mean())
        rows.append(r)
    R.append("### 2. Per kelompok (Mondrian)\n")
    R.append(md(pd.DataFrame(rows).set_index("kelompok")) + "\n")

    # 3. per tingkat
    rows = []
    for t in ["Critical", "High", "Medium", "Low"]:
        s = T[T["tingkat"] == t]
        rows.append({"tingkat": t, "klaim": len(s), "% klaim": len(s) / len(T), "anomali": int(s["is_anomaly"].sum()),
                     "precision": s["is_anomaly"].mean() if len(s) else np.nan,
                     "% anomali tertangkap": s["is_anomaly"].sum() / max(y1.sum(), 1),
                     "FPR": ((s["is_anomaly"] == 0).sum()) / max(y0.sum(), 1)})
    R.append("### 3. Per tingkat prioritas (tiap tingkat berdiri sendiri, tidak kumulatif)\n")
    R.append(md(pd.DataFrame(rows).set_index("tingkat")) + "\n")

    # 4. recall per modus x kesulitan pada p <= 0.05 dan p <= 0.01
    for a in (0.05, 0.01):
        f = (T["p_value"] <= a).to_numpy()
        tab = {}
        for m in sorted(set(T["anomaly_type"]) - {"none"}):
            tab[m] = {d: float(f[(T["anomaly_type"] == m) & (T["difficulty"] == d)].mean()) for d in ("easy", "medium", "hard")}
        R.append(f"### 4. Recall per modus x kesulitan pada p <= {a}\n")
        R.append(md(pd.DataFrame(tab).T.rename_axis("modus")) + "\n")

    # 5. hard negatives
    hn = {}
    for a in (0.01, 0.03, 0.05):
        hn[f"p<={a}"] = {t: float((T.loc[T["hard_negative_type"] == t, "p_value"] <= a).mean())
                         for t in sorted(set(T["hard_negative_type"]) - {"none"})}
    R.append("### 5. Tingkat ditandai pada hard negatives\n")
    R.append(md(pd.DataFrame(hn).rename_axis("tipe")) + "\n")

    # 6. dispersi FPR antar faskes (uji asumsi exchangeability per faskes)
    rows = []
    fr = T[y0].groupby("facility_id").agg(n=("p_value", "size"), f05=("p_value", lambda x: (x <= 0.05).mean()),
                                          f01=("p_value", lambda x: (x <= 0.01).mean()))
    for a, c in ((0.05, "f05"), (0.01, "f01")):
        rows.append({"a": a, "faskes": len(fr), "FPR median": fr[c].median(), "FPR p90": fr[c].quantile(0.9),
                     "FPR maks": fr[c].max(), "faskes dengan FPR > 2a": int((fr[c] > 2 * a).sum())})
    R.append("### 6. Sebaran FPR antar faskes (klaim normal)\n")
    R.append(md(pd.DataFrame(rows).set_index("a")) + "\n")

    os.makedirs(outdir, exist_ok=True)
    df[["claim_id", "split", "grp", "p_value", "skor_0_100", "tingkat"]].rename(columns={"grp": "kelompok"}).to_csv(
        os.path.join(outdir, f"calibrated_{suf}.csv.gz"), index=False)
    with open(os.path.join(outdir, f"calibrator_{suf}.pkl"), "wb") as f:
        pickle.dump(cal, f)
    n_cal = {g: len(v) for g, v in cal.groups.items()} or {"pool": len(cal.pool)}
    head = f"\n## Skema `{scheme}`, mode `{mode}` (kalibrasi: klaim normal val per `{groupby}`, n = {n_cal}, pool = {len(cal.pool)})\n"
    return head + "\n".join(R), V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--schemes", default="faskes,waktu")
    ap.add_argument("--modes", default="retro,asof")
    ap.add_argument("--groupby", default="jenis_layanan", choices=["jenis_layanan", "jenis_faskes", "none"])
    ap.add_argument("--min-cal", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    outdir = os.path.join(a.root, "calibrated")
    R = ["# Laporan kalibrasi konformal (tahap 6)\n",
         "Semua angka dari split **test**. Kalibrasi memakai klaim normal (termasuk hard negatives) di split **val**; "
         "detektor tahap 5 tidak diubah. p-value konformal dengan pemecahan seri acak; skor 0-100 = 100 x (1 - p). "
         "Tingkat: Critical p<=0,01; High p<=0,03; Medium p<=0,05; Low selainnya. "
         "Jaminan FPR <= a hanya berlaku jika klaim normal baru dapat dipertukarkan dengan klaim kalibrasi; "
         "di skema faskes (faskes baru) dan skema waktu asumsi itu tidak penuh, jadi dicek empiris dan dilaporkan apa adanya.\n"]
    for scheme in a.schemes.split(","):
        for mode in a.modes.split(","):
            txt, V = run(a.root, scheme, mode, a.groupby, a.min_cal, a.seed, outdir)
            R.append(txt)
            m = V[V.index.str.startswith("Mondrian")].set_index("a (dijanjikan)")
            print(f"[{scheme}/{mode}] FPR empiris (dijanjikan 1%/3%/5%/10%): "
                  + " / ".join(f"{m.loc[x, 'FPR empiris']:.3f}" for x in (0.01, 0.03, 0.05, 0.10) if x in m.index)
                  + f" | recall@5% = {m.loc[0.05, 'Recall']:.3f}", flush=True)
    with open(os.path.join(outdir, "kalibrasi_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))


if __name__ == "__main__":
    main()