#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 0: modul evaluasi

Detektor apa pun cukup menghasilkan skor per klaim (makin tinggi = makin berisiko):

    from claimguard_eval import Evaluator, format_report
    ev = Evaluator("out")
    res = ev.evaluate(scores, scheme="faskes", split="test", budget=0.05)
    print(format_report(res))

`scores`: pandas Series berindeks claim_id, atau DataFrame dengan kolom claim_id dan score.

Aturan anti-kebocoran
---------------------
* Fitur HANYA boleh dari data/claims.csv. Panggil assert_no_label_columns(df_fitur).
* Label dipakai hanya di modul ini (eval_only/labels.csv).
* Split per faskes atau per waktu, tidak pernah acak (lihat data/splits.csv).
* Ambang/anggaran review dipilih di split val, dilaporkan di split test.
* Seri skor (ties) dipecah acak dengan seed tetap, bukan menurut urutan baris.
"""
import json
import math
import os

import numpy as np
import pandas as pd

KS = (50, 100, 200, 500)
RECALL_TARGETS = (0.5, 0.8)
LABEL_COLS = {"is_anomaly", "anomaly_type", "difficulty", "hard_negative_type",
              "related_claim_id", "injected_signal", "case_group", "risky", "dominant_modes"}
SCHEME_COL = {"faskes": "split_faskes", "waktu": "split_waktu"}


def assert_no_label_columns(df):
    bad = LABEL_COLS & set(df.columns)
    if bad:
        raise ValueError(f"Kolom label/eval_only terdeteksi di fitur: {sorted(bad)}")


def average_precision(y, s):
    """PR-AUC (average precision) dengan penanganan ties pada ambang yang sama."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    P = y.sum()
    if P == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys)
    n = np.arange(1, len(ys) + 1)
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    prec = tp[last] / n[last]
    rec = tp[last] / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def order_desc(s, seed=0):
    rng = np.random.default_rng(seed)
    return np.lexsort((rng.random(len(s)), -np.asarray(s, float)))


def _core(y, s, budget, thr, seed):
    """Metrik inti (dipakai juga oleh bootstrap)."""
    n = len(y)
    P = float(y.sum())
    N = n - P
    o = order_desc(s, seed)
    tp = np.cumsum(y[o])
    if thr is None:
        kb = int(math.ceil(budget * n))
        flag = np.zeros(n, bool)
        flag[o[:kb]] = True
    else:
        flag = s >= thr
        kb = int(flag.sum())
    hit = float(y[flag].sum())
    k80 = int(np.searchsorted(tp / max(P, 1), 0.8) + 1) if P > 0 else n
    return dict(
        pr_auc=average_precision(y, s),
        flag_rate=kb / n,
        precision=hit / max(kb, 1),
        recall=hit / P if P else float("nan"),
        fpr=float((flag & (y == 0)).sum()) / max(N, 1),
        review_ratio_r80=k80 / n,
    ), flag, o, tp


class Evaluator:
    def __init__(self, root="out"):
        d, e = os.path.join(root, "data"), os.path.join(root, "eval_only")
        claims = pd.read_csv(os.path.join(d, "claims.csv"), usecols=["claim_id", "facility_id"])
        splits = pd.read_csv(os.path.join(d, "splits.csv"))
        labels = pd.read_csv(os.path.join(e, "labels.csv"), keep_default_na=False)
        self.df = claims.merge(splits, on="claim_id").merge(labels, on="claim_id")

    @staticmethod
    def _norm_scores(scores):
        if isinstance(scores, pd.Series):
            scores = scores.rename("score").rename_axis("claim_id").reset_index()
        if not {"claim_id", "score"} <= set(scores.columns):
            raise ValueError("scores harus punya kolom claim_id dan score")
        return scores[["claim_id", "score"]]

    def evaluate(self, scores, scheme="faskes", split="test", budget=0.05,
                 calibrate_on_val=False, n_boot=200, seed=0):
        sc = self._norm_scores(scores)
        df = self.df.merge(sc, on="claim_id", how="left")
        if df["score"].isna().any():
            raise ValueError(f"{int(df['score'].isna().sum())} klaim tidak punya skor")
        col = SCHEME_COL[scheme]
        te = df[df[col] == split].reset_index(drop=True)
        thr = None
        if calibrate_on_val:
            va = df[df[col] == "val"]
            thr = float(np.quantile(va["score"], 1 - budget))
        y = te["is_anomaly"].to_numpy(float)
        s = te["score"].to_numpy(float)
        core, flag, o, tp = _core(y, s, budget, thr, seed)
        P = y.sum()
        n = len(y)

        res = dict(scheme=scheme, split=split, budget=budget, calibrate_on_val=calibrate_on_val,
                   n=int(n), n_pos=int(P), prevalence=float(P / n))
        res.update(core)
        res["pr_auc_lift"] = res["pr_auc"] / res["prevalence"]
        for K in KS:
            if K <= n:
                res[f"precision@{K}"] = float(y[o[:K]].mean())
                res[f"recall@{K}"] = float(tp[K - 1] / P)
        for t in RECALL_TARGETS:
            k = int(np.searchsorted(tp / P, t) + 1)
            res[f"review_ratio_r{int(t * 100)}"] = k / n
            res[f"lift_vs_random_r{int(t * 100)}"] = t / (k / n)

        mode, diff, hn = te["anomaly_type"].to_numpy(), te["difficulty"].to_numpy(), te["hard_negative_type"].to_numpy()
        modes = sorted(m for m in set(mode) if m != "none")
        res["recall_by_mode"] = {m: float(flag[mode == m].mean()) for m in modes}
        res["recall_by_difficulty"] = {d: float(flag[diff == d].mean()) for d in ("easy", "medium", "hard") if (diff == d).any()}
        res["recall_mode_x_difficulty"] = {
            m: {d: float(flag[(mode == m) & (diff == d)].mean()) for d in ("easy", "medium", "hard") if ((mode == m) & (diff == d)).any()}
            for m in modes}
        res["pr_auc_by_mode"] = {}
        for m in modes:
            keep = (mode == m) | (y == 0)
            res["pr_auc_by_mode"][m] = average_precision((mode[keep] == m).astype(float), s[keep])
        res["flag_rate_hard_negative"] = {t: float(flag[hn == t].mean()) for t in sorted(set(hn)) if t != "none"}

        # bootstrap klaster per faskes
        if n_boot:
            rng = np.random.default_rng(seed + 1)
            facs = te["facility_id"].to_numpy()
            groups = {f: np.where(facs == f)[0] for f in np.unique(facs)}
            keys = list(groups)
            vals = {k: [] for k in ("pr_auc", "precision", "recall", "fpr", "review_ratio_r80")}
            for b in range(n_boot):
                pick = rng.choice(len(keys), size=len(keys), replace=True)
                idx = np.concatenate([groups[keys[i]] for i in pick])
                if y[idx].sum() == 0:
                    continue
                c, *_ = _core(y[idx], s[idx], budget, thr, seed + b)
                for k in vals:
                    vals[k].append(c[k])
            res["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                           for k, v in vals.items() if v}
        return res


def aggregate(results):
    """Rata-rata dan simpangan baku metrik utama dari beberapa seed/run."""
    keys = ["pr_auc", "precision", "recall", "fpr", "review_ratio_r80"]
    return {k: {"mean": float(np.mean([r[k] for r in results])),
                "sd": float(np.std([r[k] for r in results], ddof=1)) if len(results) > 1 else 0.0}
            for k in keys}


def format_report(r):
    f = lambda x: f"{x:.3f}"
    ci = r.get("ci95", {})
    cs = lambda k: f" [{f(ci[k][0])}, {f(ci[k][1])}]" if k in ci else ""
    mode = f"ambang dari val (flag {r['flag_rate']:.1%})" if r["calibrate_on_val"] else f"anggaran review top {r['budget']:.0%}"
    L = [f"== Evaluasi | split {r['scheme']}/{r['split']} | {mode} ==",
         f"klaim {r['n']:,} | anomali {r['n_pos']:,} (prevalensi {r['prevalence']:.2%})",
         f"PR-AUC {f(r['pr_auc'])}{cs('pr_auc')} (lift {r['pr_auc_lift']:.1f}x vs acak)",
         f"Precision {f(r['precision'])}{cs('precision')} | Recall {f(r['recall'])}{cs('recall')} | FPR {f(r['fpr'])}{cs('fpr')}"]
    ks = [f"P@{K}={f(r[f'precision@{K}'])}" for K in KS if f"precision@{K}" in r]
    L.append("  ".join(ks))
    L.append(f"Beban review: tangkap 50% anomali = periksa {r['review_ratio_r50']:.1%} klaim (lift {r['lift_vs_random_r50']:.1f}x); "
             f"80% = {r['review_ratio_r80']:.1%} (lift {r['lift_vs_random_r80']:.1f}x){cs('review_ratio_r80')}")
    L.append("Recall per modus: " + ", ".join(f"{k}={f(v)}" for k, v in r["recall_by_mode"].items()))
    L.append("Recall per kesulitan: " + ", ".join(f"{k}={f(v)}" for k, v in r["recall_by_difficulty"].items()))
    L.append("PR-AUC per modus: " + ", ".join(f"{k}={f(v)}" for k, v in r["pr_auc_by_mode"].items()))
    L.append("Tingkat ditandai pada hard negatives (makin rendah makin baik): " +
             ", ".join(f"{k}={f(v)}" for k, v in r["flag_rate_hard_negative"].items()))
    return "\n".join(L)


if __name__ == "__main__":
    print(__doc__)
