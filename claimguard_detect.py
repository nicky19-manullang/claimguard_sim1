#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 5: mesin deteksi hibrida

Empat detektor berbasis domain (satu per modus) + ensemble anomaly detection tak terawasi:

  d_repeat  aturan    : jarak ke klaim sebelumnya relatif terhadap pola wajar diagnosis x kemiripan biaya x item
  d_clone   similarity: kemiripan klaim terdekat (MinHash/LSH + embedding), dinormalisasi per diagnosis
  d_infl    residual  : residu terhadap expected cost (case-mix) x proporsi obat/alkes
  d_upc     severity  : kejutan severity (severity tercatat vs prediksi dari diagnosis dan lama rawat)
  u_ens     tak terawasi: rata-rata persentil Isolation Forest, ECOD, COPOD (PyOD) atas fitur gabungan

Setiap detektor dipetakan ke persentil (0-1) terhadap distribusi DATA LATIH. Penggabungan:
  - "maks"       : nilai tertinggi antar detektor (tanpa pelatihan)
  - "bobot val"  : regresi logistik dengan bobot >= 0 (monoton) yang dipelajari pada split VAL (label biner anomali),
                   atas transformasi ekor -log(1 - persentil)
Fit komponen apa pun: hanya split latih. Bobot: hanya split val. Metrik akhir: split test (via claimguard_eval).
Fitur faskes (uc_fac_*) sengaja TIDAK dipakai (tidak stabil antar split, lihat laporan fitur).

    python claimguard_detect.py --root out
"""
import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

from claimguard_eval import Evaluator, assert_no_label_columns, average_precision, order_desc

MODES = ["repeat_billing", "claim_cloning", "inflated_bills", "upcoding"]
RULE_COLS = ["cl_best_score_soft_z", "cl_emb_sim_z", "cl_best_score_z", "ib_resid_x_obat", "ib_resid_z",
             "uc_sev_surprise", "uc_los_short", "uc_n_severe_dx"]
UNSUP_COLS = ["rp_gap_ratio_c", "rp_val_ratio", "rp_item_jac", "rp_cnt_14d", "rp_same_doctor",
              "cl_best_score_z", "cl_best_score_soft_z", "cl_emb_sim_z", "cl_best_same_doctor", "cl_n_near_z",
              "ib_resid_z", "ib_obat_share_z", "uc_sev_surprise", "uc_sev_nll", "uc_n_severe_dx",
              "uc_n_addon", "uc_proc_rarity", "uc_los_ratio"]
RULE_P = ["p_repeat", "p_clone", "p_infl", "p_upc"]
HYB_P = RULE_P + ["p_ens"]
NAMES = {"p_repeat": "Aturan repeat", "p_clone": "Similarity cloning", "p_infl": "Residual inflated",
         "p_upc": "Kejutan severity", "p_if": "Isolation Forest", "p_ecod": "ECOD", "p_copod": "COPOD",
         "p_ens": "Ensemble tak terawasi"}


class Ecdf:
    """Persentil (0-1) relatif terhadap distribusi data latih."""

    def fit(self, x):
        self.s = np.sort(np.asarray(x, float))
        return self

    def __call__(self, x):
        return np.searchsorted(self.s, np.asarray(x, float), side="right") / len(self.s)


def tail(p):
    return -np.log(1.0 - np.clip(p, 0, 1) + 1e-3)


def fit_weights(X, y, l2=0.01):
    """Regresi logistik dengan bobot >= 0 (monoton) dan intersep bebas."""
    k = X.shape[1]

    def f(th):
        w, b = th[:k], th[k]
        z = X @ w + b
        loss = np.mean(np.logaddexp(0, z) - y * z) + l2 * np.sum(w ** 2)
        gz = (1 / (1 + np.exp(-z)) - y) / len(y)
        return loss, np.r_[X.T @ gz + 2 * l2 * w, gz.sum()]

    r = minimize(f, np.r_[np.full(k, 0.3), -4.0], jac=True, method="L-BFGS-B",
                 bounds=[(0, None)] * k + [(None, None)])
    return r.x[:k], float(r.x[k])


class HybridDetector:
    def __init__(self, seed=0, contamination=0.03):
        self.seed, self.contamination = seed, contamination

    @staticmethod
    def _derive(F):
        F = F.copy()
        F["rp_gap_ratio_c"] = F["rp_gap_ratio_typ"].clip(-1, 5)
        F["uc_los_short"] = -F["uc_los_ratio"]
        return F

    def _rules_raw(self, F):
        P = lambda c: self.cp[c](F[c])
        ri = (F["jenis_layanan"] == "RITL").to_numpy()
        out = pd.DataFrame(index=F.index)
        out["p_repeat"] = (F["rp_has_prev"].to_numpy() * np.clip(1 - F["rp_gap_ratio_typ"].to_numpy(), 0, 1)
                           * (0.5 + 0.5 * F["rp_val_ratio"].to_numpy()) * (0.5 + 0.5 * F["rp_item_jac"].to_numpy()))
        out["p_clone"] = 0.4 * P("cl_best_score_soft_z") + 0.4 * P("cl_emb_sim_z") + 0.2 * P("cl_best_score_z")
        out["p_infl"] = 0.7 * P("ib_resid_x_obat") + 0.3 * P("ib_resid_z")
        out["p_upc"] = np.where(ri, 0.7 * P("uc_sev_surprise") + 0.15 * P("uc_los_short") + 0.15 * P("uc_n_severe_dx"), 0.0)
        return out

    def _unsup_raw(self, F):
        X = np.clip(F[UNSUP_COLS].to_numpy(float), self.lo, self.hi)
        return pd.DataFrame({"p_if": -self.iforest.score_samples(X),
                             "p_ecod": np.nan_to_num(self.ecod.decision_function(X)),
                             "p_copod": np.nan_to_num(self.copod.decision_function(X))}, index=F.index)

    def fit(self, F, tr):
        F = self._derive(F)
        self.cp = {c: Ecdf().fit(F.loc[tr, c]) for c in RULE_COLS}
        raw = self._rules_raw(F)
        self.dp = {k: Ecdf().fit(raw.loc[tr, k]) for k in raw}
        Xtr = F.loc[tr, UNSUP_COLS].to_numpy(float)
        self.lo, self.hi = np.percentile(Xtr, [0.5, 99.5], axis=0)
        Xc = np.clip(Xtr, self.lo, self.hi)
        self.iforest = IsolationForest(n_estimators=300, max_samples=1024, random_state=self.seed, n_jobs=-1).fit(Xc)
        self.ecod = ECOD(contamination=self.contamination).fit(Xc)
        self.copod = COPOD(contamination=self.contamination).fit(Xc)
        ru = self._unsup_raw(F)
        self.up = {k: Ecdf().fit(ru.loc[tr, k]) for k in ru}
        ens = np.mean([self.up[k](ru[k]) for k in ru], axis=0)
        self.up_ens = Ecdf().fit(ens[np.asarray(tr)])
        return self

    def scores(self, F):
        F = self._derive(F)
        raw, ru = self._rules_raw(F), self._unsup_raw(F)
        out = pd.DataFrame(index=F.index)
        for k in raw:
            out[k] = self.dp[k](raw[k])
        for k in ru:
            out[k] = self.up[k](ru[k])
        out["p_ens"] = self.up_ens(np.mean([out[k] for k in ("p_if", "p_ecod", "p_copod")], axis=0))
        return out


def combiner(P, y_val, val, cols):
    X = tail(P.loc[val, cols].to_numpy(float))
    w, b = fit_weights(X, y_val.astype(float))
    return dict(zip(cols, w)), b


def apply_combiner(P, cols, w, b):
    return tail(P[cols].to_numpy(float)) @ np.array([w[c] for c in cols]) + b


def md(df, fmt="{:.3f}"):
    L = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |", "|" + "---|" * (len(df.columns) + 1)]
    for i, r in df.iterrows():
        L.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("-" if pd.isna(v) else str(v)) for v in r]) + " |")
    return "\n".join(L)


def run_config(root, scheme, mode, ev, labels, seed, n_boot, outdir):
    t0 = time.time()
    suf = "_asof" if mode == "asof" else ""
    F = pd.read_csv(os.path.join(root, "features", f"features_{scheme}{suf}.csv.gz"))
    assert_no_label_columns(F)
    tr, va = (F["split"] == "train").to_numpy(), (F["split"] == "val").to_numpy()
    det = HybridDetector(seed=seed).fit(F, tr)
    P = det.scores(F)
    P.insert(0, "claim_id", F["claim_id"])
    y = F[["claim_id"]].merge(labels[["claim_id", "is_anomaly"]], on="claim_id")["is_anomaly"].to_numpy()

    wr, br = combiner(P, y[va], va, RULE_P)
    wh, bh = combiner(P, y[va], va, HYB_P)
    # pilih penggabung (bobot val vs maks) dengan cross-validation per faskes PADA VAL, bukan dari test
    fac = F[["claim_id"]].merge(ev.df[["claim_id", "facility_id"]], on="claim_id")["facility_id"].to_numpy()
    vi = np.where(va)[0]
    Xv, yv = tail(P.loc[va, HYB_P].to_numpy(float)), y[va].astype(float)
    oof = np.zeros(len(vi))
    for a_, b_ in GroupKFold(n_splits=5).split(vi, groups=fac[vi]):
        w_, bb_ = fit_weights(Xv[a_], yv[a_])
        oof[b_] = Xv[b_] @ w_ + bb_
    ap_learn = average_precision(yv, oof)
    ap_max = average_precision(yv, P.loc[va, HYB_P].max(axis=1).to_numpy())
    use_learned = bool(ap_learn > ap_max)
    S = {NAMES[c]: P[c] for c in ("p_repeat", "p_clone", "p_infl", "p_upc", "p_if", "p_ecod", "p_copod", "p_ens")}
    S["Aturan, maks"] = P[RULE_P].max(axis=1)
    S["Aturan, bobot val"] = pd.Series(apply_combiner(P, RULE_P, wr, br), index=P.index)
    S["Hibrida, maks"] = P[HYB_P].max(axis=1)
    S["Hibrida, bobot val"] = pd.Series(apply_combiner(P, HYB_P, wh, bh), index=P.index)
    S["Hibrida, dipilih (CV val)"] = S["Hibrida, bobot val"] if use_learned else S["Hibrida, maks"]

    rows, res_all = [], {}
    for name, s in S.items():
        sc = pd.Series(np.asarray(s, float), index=F["claim_id"].to_numpy())
        r = ev.evaluate(sc, scheme=scheme, split="test", budget=0.05, n_boot=n_boot, seed=seed)
        res_all[name] = r
        row = {"PR-AUC": r["pr_auc"], "CI95": f"[{r['ci95']['pr_auc'][0]:.3f}, {r['ci95']['pr_auc'][1]:.3f}]",
               "Precision@5%": r["precision"], "Recall@5%": r["recall"], "FPR": r["fpr"],
               "Review 80%": r["review_ratio_r80"]}
        row.update({f"rec_{m.split('_')[0]}": r["recall_by_mode"].get(m, np.nan) for m in MODES})
        rows.append((name, row))
    T = pd.DataFrame({n: r for n, r in rows}).T
    T.index.name = "varian"

    final = S["Hibrida, dipilih (CV val)"]
    P["risk_raw"] = np.asarray(final, float)
    P.insert(1, "split", F["split"])
    os.makedirs(outdir, exist_ok=True)
    P.to_csv(os.path.join(outdir, f"detect_{scheme}_{mode}.csv.gz"), index=False)
    with open(os.path.join(outdir, f"detector_{scheme}_{mode}.pkl"), "wb") as f:
        pickle.dump(dict(detector=det, w_rule=wr, b_rule=br, w_hybrid=wh, b_hybrid=bh), f)
    # recall tingkat pasangan (repeat/cloning): terdeteksi jika salinan ATAU klaim sumbernya ikut ditandai (anggaran 5%)
    te = (F["split"] == "test").to_numpy()
    n_te = int(te.sum())
    o = order_desc(np.asarray(final, float)[te], seed)
    flagged = set(F.loc[te, "claim_id"].to_numpy()[o[:int(math.ceil(0.05 * n_te))]])
    Lt = labels[labels["claim_id"].isin(set(F.loc[te, "claim_id"])) & labels["anomaly_type"].isin(["repeat_billing", "claim_cloning"])]
    pair = {m: float(np.mean([(c in flagged) or (r in flagged) for c, r in zip(g["claim_id"], g["related_claim_id"])]))
            for m, g in Lt.groupby("anomaly_type")}
    info = dict(weights_rule={NAMES[k]: round(v, 3) for k, v in wr.items()}, bias_rule=round(br, 3),
                weights_hybrid={NAMES[k]: round(v, 3) for k, v in wh.items()}, bias_hybrid=round(bh, 3),
                cv_ap_learned=round(ap_learn, 3), cv_ap_max=round(ap_max, 3), dipilih="bobot val" if use_learned else "maks",
                pair_recall=pair,
                detik=round(time.time() - t0, 1))
    return T, res_all, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--schemes", default="faskes,waktu")
    ap.add_argument("--modes", default="retro,asof")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    labels = pd.read_csv(os.path.join(a.root, "eval_only", "labels.csv"), keep_default_na=False)
    ev = Evaluator(a.root)
    outdir = os.path.join(a.root, "detect")
    R = ["# Laporan mesin deteksi hibrida (tahap 5)\n",
         "Semua angka dari split **test**, anggaran review 5% klaim teratas, CI 95% bootstrap per faskes. Komponen di-fit pada latih, "
         "bobot hibrida dipelajari pada val (label biner anomali), test hanya untuk pelaporan. `rec_*` = recall per modus pada anggaran 5%. "
         "Varian bukan dipilih berdasarkan test: seluruh daftar ablation dilaporkan.\n",
         "Mode `retro` = batch retrospektif; `asof` = hanya membandingkan dengan klaim yang sudah diajukan lebih dulu.\n"]
    summary = []
    for scheme in a.schemes.split(","):
        for mode in a.modes.split(","):
            T, res, info = run_config(a.root, scheme, mode, ev, labels, a.seed, a.boot, outdir)
            R.append(f"\n## Skema `{scheme}`, mode `{mode}`\n")
            R.append(md(T) + "\n")
            R.append(f"Bobot hibrida (bobot val): {json.dumps(info['weights_hybrid'], ensure_ascii=False)}, intersep {info['bias_hybrid']}\n")
            R.append(f"Bobot aturan saja (bobot val): {json.dumps(info['weights_rule'], ensure_ascii=False)}, intersep {info['bias_rule']}\n")
            R.append(f"Pemilihan penggabung (CV per faskes pada val, average precision): bobot val={info['cv_ap_learned']}, maks={info['cv_ap_max']} -> dipilih **{info['dipilih']}**.\n")
            R.append("Recall tingkat pasangan pada anggaran 5% (repeat/cloning terdeteksi jika salinan atau klaim sumbernya ditandai): "
                     + ", ".join(f"{k}={v:.3f}" for k, v in info["pair_recall"].items()) + "\n")
            h = res["Hibrida, dipilih (CV val)"]
            R.append(f"Tingkat ditandai pada hard negatives (hibrida terpilih; makin rendah makin baik): "
                     + ", ".join(f"{k}={v:.3f}" for k, v in h["flag_rate_hard_negative"].items()) + "\n")
            R.append("Recall hibrida per modus x kesulitan (anggaran 5%):\n")
            R.append(md(pd.DataFrame(h["recall_mode_x_difficulty"]).T.rename_axis("modus")) + "\n")
            summary.append((scheme, mode, h, res["Aturan, bobot val"], res["Ensemble tak terawasi"]))
            print(f"[{scheme}/{mode}] dipilih={info['dipilih']} hibrida PR-AUC={h['pr_auc']:.3f} P@5%={h['precision']:.3f} R@5%={h['recall']:.3f} "
                  f"| aturan={res['Aturan, bobot val']['pr_auc']:.3f} | tak terawasi={res['Ensemble tak terawasi']['pr_auc']:.3f} ({info['detik']}s)", flush=True)
    with open(os.path.join(outdir, "deteksi_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))


if __name__ == "__main__":
    main()