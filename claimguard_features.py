#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 4: feature engineering per modus

Prinsip anti-kebocoran
----------------------
* Input hanya out/prep/claims_prep_<skema>.csv.gz (tidak berisi label). assert_no_label_columns dipanggil.
* Semua komponen yang BELAJAR (model expected cost, model severity, kosakata/embedding, referensi per diagnosis,
  statistik rekan faskes) di-fit hanya pada split latih.
* Untuk baris latih, prediksi model memakai cross-fitting (5 fold, dikelompokkan per faskes) agar fitur latih tidak
  terlalu optimistis dibanding val/test.
* Fitur relasional (klaim pembanding, riwayat pasien, profil faskes) dihitung secara retrospektif pada batch klaim
  (tanpa label). Pada skema faskes tidak ada pasangan lintas split; pada skema waktu pasangan boleh melintas
  karena di praktik klaim lama sudah ada saat klaim baru ditinjau.

Fitur per modus (awalan kolom)
------------------------------
rp_  repeat billing : jarak ke klaim sebelumnya (pasien+faskes+dx), tumpang tindih, rasio biaya, kemiripan item,
                      jumlah klaim dalam 14 hari, jarak dibanding pola wajar diagnosis (dari data latih)
cl_  claim cloning  : MinHash + LSH atas himpunan kode, kemiripan embedding (SVD), kedekatan biaya, jumlah
                      duplikat dekat, dinormalisasi per diagnosis (agar protokol baku tidak ikut ditandai)
ib_  inflated bills : residu terhadap expected cost (gradient boosting, case-mix), residu terkoreksi faskes,
                      proporsi obat/alkes dibanding wajar diagnosis
uc_  upcoding       : kejutan severity (severity tercatat vs prediksi dari diagnosis, lama rawat, usia),
                      kelangkaan pasangan diagnosis-tindakan, dx berat, profil severity faskes vs rekan,
                      pergeseran bauran severity per bulan

Dua mode (flag --modes)
-----------------------
retro : batch retrospektif; klaim pembanding dan profil faskes memakai semua klaim (audit periodik).
asof  : simulasi penyaringan online; setiap klaim hanya dibandingkan dengan klaim yang SUDAH DIAJUKAN lebih dulu
        (tanggal_klaim lebih kecil atau sama). Angkanya biasanya lebih rendah dan lebih realistis.

    python claimguard_features.py --root out --modes retro,asof
"""
import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GroupKFold

from claimguard_eval import assert_no_label_columns

SEVERE_DX = {"A41.9", "J96.0", "N17.9", "E87.1", "D65"}
COMPLICATION_DX = {"T81.4", "T81.0", "I26.9"}
ADDON = ["96.71", "87.03", "88.91", "99.04", "88.72"]
AGE = {"0-17": 0, "18-44": 1, "45-59": 2, "60+": 3}
FTYPE = {"RS_A": 0, "RS_B": 1, "RS_C": 2, "RS_D": 3, "Klinik": 4}
WINDOW_DAYS = 200

GROUPS = {
    "repeat_billing": "rp_", "claim_cloning": "cl_", "inflated_bills": "ib_", "upcoding": "uc_",
}


# =============================================================================== util
def load_prep(root, scheme):
    d = pd.read_csv(os.path.join(root, "prep", f"claims_prep_{scheme}.csv.gz"), keep_default_na=False)
    for c in ("days_since_prev_claim", "resid_log_nilai_naif", "resid_lama_rawat_naif", "ref_log_nilai",
              "ref_lama_rawat", "log_nilai"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("sec_list", "proc_list", "item_list"):
        d[c] = d[c].map(lambda s: [x for x in str(s).split("|") if x])
    return d.sort_values("claim_id").reset_index(drop=True)


def mad_scale(x, floor):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return floor
    return max(float(np.median(np.abs(x - np.median(x))) * 1.4826), floor)


def oof_fit_predict(make_model, X, y, tr_idx, other_idx, groups, proba=False, n_splits=5):
    """Prediksi cross-fit untuk baris latih (OOF per kelompok faskes), model penuh-latih untuk sisanya."""
    X = np.asarray(X)
    y = np.asarray(y)
    first = make_model().fit(X[tr_idx[:50]], y[tr_idx[:50]])  # hanya untuk menentukan bentuk keluaran
    k = first.predict_proba(X[:1]).shape[1] if proba else 1
    out = np.full((len(X), k), np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups[tr_idx]))))
    for a, b in gkf.split(tr_idx, groups=groups[tr_idx]):
        m = make_model().fit(X[tr_idx[a]], y[tr_idx[a]])
        p = m.predict_proba(X[tr_idx[b]]) if proba else m.predict(X[tr_idx[b]])[:, None]
        out[tr_idx[b]] = _pad(p, m, k, proba)
    if len(other_idx):
        m = make_model().fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[other_idx]) if proba else m.predict(X[other_idx])[:, None]
        out[other_idx] = _pad(p, m, k, proba)
    return out if proba else out[:, 0]


def _pad(p, model, k, proba):
    if not proba or p.shape[1] == k:
        return p
    full = np.zeros((p.shape[0], k))
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = p[:, j]
    return full


# ========================================================================= 1. repeat
def feat_repeat(d, tr, asof=False):
    keys = ["patient_key", "facility_id", "dx3"]
    x = d[["claim_id"] + keys + ["dx", "masuk_d", "pulang_d", "klaim_d", "nilai_klaim", "doctor_key", "item_list"]].copy()
    order = ["klaim_d", "masuk_d", "claim_id"] if asof else ["masuk_d", "klaim_d", "claim_id"]   # as-of: urutan pengajuan
    x = x.sort_values(keys + order).reset_index(drop=True)
    g = x.groupby(keys, sort=False)
    cm = g["pulang_d"].cummax()
    prev_end = cm.groupby([x[k] for k in keys], sort=False).shift(1)
    gap = (x["masuk_d"] - prev_end).to_numpy(float)
    prev_val = g["nilai_klaim"].shift(1).to_numpy(float)
    val = x["nilai_klaim"].to_numpy(float)
    prev_doc = g["doctor_key"].shift(1).fillna("").to_numpy()
    prev_klaim = g["klaim_d"].shift(1).to_numpy(float)
    prev_items = g["item_list"].shift(1).to_numpy(dtype=object)

    def jac(a, b):
        if not isinstance(a, list) or not isinstance(b, list) or (not a and not b):
            return np.nan
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb)

    item_jac = np.array([jac(a, b) for a, b in zip(x["item_list"], prev_items)])
    cnt14 = np.zeros(len(x))
    md = x["masuk_d"].to_numpy()
    for _, idx in g.indices.items():
        if len(idx) > 1:
            m = md[idx]
            if asof:   # hanya klaim yang sudah diajukan lebih dulu (urutan pengajuan), jarak tanggal layanan <= 14 hari
                cnt14[idx] = [int((np.abs(m[:a] - m[a]) <= 14).sum()) for a in range(len(m))]
            else:
                cnt14[idx] = np.arange(len(m)) - np.searchsorted(m, m - 14, side="left")

    # pola wajar diagnosis (fit: data latih saja) -> jeda tipikal antar klaim pasien yang sama
    trm = x["claim_id"].isin(set(d.loc[tr, "claim_id"])).to_numpy()
    pos = (~np.isnan(gap)) & (gap >= 0) & trm
    typ = pd.Series(gap[pos]).groupby(x.loc[pos, "dx"].to_numpy()).agg(["median", "size"])
    typ = typ[typ["size"] >= 30]["median"].to_dict()
    typ_gap = x["dx"].map(typ).fillna(30.0).to_numpy(float)

    has_prev = ~np.isnan(gap)
    out = pd.DataFrame({
        "claim_id": x["claim_id"],
        "rp_has_prev": has_prev.astype(int),
        "rp_gap_prev": np.where(has_prev, gap, 999.0),
        "rp_overlap": (has_prev & (gap < 0)).astype(int),
        "rp_gap_ratio_typ": np.where(has_prev, np.clip(gap, -30, None) / np.maximum(typ_gap, 1.0), 99.0),
        "rp_val_ratio": np.where(has_prev, np.minimum(val, prev_val) / np.maximum(np.maximum(val, prev_val), 1), 0.0),
        "rp_item_jac": np.nan_to_num(item_jac, nan=0.0),
        "rp_same_doctor": ((x["doctor_key"].to_numpy() == prev_doc) & (prev_doc != "UNKNOWN") & (prev_doc != "")).astype(int),
        "rp_cnt_14d": cnt14,
        "rp_klaim_gap": np.where(has_prev, x["klaim_d"].to_numpy(float) - prev_klaim, 999.0),
    })
    # gabungan: dekat + biaya mirip + item mirip (dalam pola kasar)
    out["rp_combo"] = np.where(out["rp_gap_prev"] <= 14, out["rp_val_ratio"] * (0.5 + 0.5 * out["rp_item_jac"]), 0.0)
    # klaim kasus yang sama di faskes lain dalam 14 hari
    cross = np.zeros(len(d))
    dd = d[["patient_key", "dx3", "facility_id", "masuk_d", "klaim_d"]].reset_index(drop=True)
    for _, idx in dd.groupby(["patient_key", "dx3"], sort=False).indices.items():
        if len(idx) < 2 or dd["facility_id"].to_numpy()[idx].tolist().count(dd["facility_id"].to_numpy()[idx][0]) == len(idx):
            continue
        f, m, kk = dd["facility_id"].to_numpy()[idx], dd["masuk_d"].to_numpy()[idx], dd["klaim_d"].to_numpy()[idx]
        for a in range(len(idx)):
            cond = (f != f[a]) & (np.abs(m - m[a]) <= 14)
            if asof:
                cond &= kk <= kk[a]
            cross[idx[a]] = int(cond.sum())
    out = out.merge(pd.DataFrame({"claim_id": d["claim_id"], "rp_cross_fac_14d": cross}), on="claim_id")
    return out


# ========================================================================== 2. clone
def minhash(tok_lists, K=64, seed=7):
    rng = np.random.default_rng(seed)
    p = 2147483647
    a = rng.integers(1, p, size=K).astype(np.uint64)
    b = rng.integers(0, p, size=K).astype(np.uint64)
    cache = {}
    sig = np.zeros((len(tok_lists), K), dtype=np.uint64)
    for i, toks in enumerate(tok_lists):
        hs = []
        for t in toks:
            if t not in cache:
                cache[t] = zlib.crc32(t.encode()) % p
            hs.append(cache[t])
        h = np.array(hs, dtype=np.uint64)
        sig[i] = ((a[:, None] * h[None, :] + b[:, None]) % np.uint64(p)).min(axis=1)
    return sig


def lsh_pairs(sig, fac_codes, masuk, bands=16, rows=4, window=25):
    n = len(sig)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(masuk, kind="stable")] = np.arange(n)
    chunks = []
    with np.errstate(over="ignore"):
        for b in range(bands):
            cols = sig[:, b * rows:(b + 1) * rows]
            key = cols[:, 0].copy()
            for r in range(1, rows):
                key = key * np.uint64(1000003) + cols[:, r]
            df = pd.DataFrame({"k": key, "f": fac_codes, "r": rank}).sort_values(["f", "k", "r"])
            kk, ff, ii = df["k"].to_numpy(), df["f"].to_numpy(), df.index.to_numpy()
            for off in range(1, min(window, n - 1) + 1):
                m = (kk[off:] == kk[:-off]) & (ff[off:] == ff[:-off])
                if m.any():
                    chunks.append(np.minimum(ii[:-off][m], ii[off:][m]).astype(np.int64) * n + np.maximum(ii[:-off][m], ii[off:][m]))
    codes = np.unique(np.concatenate(chunks)) if chunks else np.array([], dtype=np.int64)
    return codes // n, codes % n


def feat_clone(d, tr, asof=False, root_seed=7):
    n = len(d)
    toks = [list(it) + ["P" + p for p in pr] + ["S" + s for s in se] + ["D" + dx]
            for it, pr, se, dx in zip(d["item_list"], d["proc_list"], d["sec_list"], d["dx"])]
    vocab = {t: i for i, t in enumerate(sorted({t for l in toks for t in l}))}
    r_, c_ = zip(*[(i, vocab[t]) for i, l in enumerate(toks) for t in set(l)])
    B = sparse.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(n, len(vocab)))
    size = np.asarray(B.sum(1)).ravel()

    fac_codes = d["facility_id"].astype("category").cat.codes.to_numpy()
    masuk = d["masuk_d"].to_numpy()
    patient = d["patient_key"].to_numpy()
    logv = d["log_nilai"].to_numpy()
    doctor = d["doctor_key"].to_numpy()

    sig = minhash(toks, K=72, seed=root_seed)
    I, J = lsh_pairs(sig, fac_codes, masuk, bands=24, rows=3)   # 24 band x 3 baris: peka sampai Jaccard ~0,4
    keep = (patient[I] != patient[J]) & (np.abs(masuk[I] - masuk[J]) <= WINDOW_DAYS)
    I, J = I[keep], J[keep]
    inter = np.zeros(len(I), np.float32)
    for s in range(0, len(I), 200_000):
        sl = slice(s, s + 200_000)
        inter[sl] = np.asarray(B[I[sl]].multiply(B[J[sl]]).sum(1)).ravel()
    jac = inter / np.maximum(size[I] + size[J] - inter, 1)
    close = np.exp(-np.abs(logv[I] - logv[J]) / 0.05)
    sc = jac * close
    dl = np.abs(logv[I] - logv[J])
    sc2 = jac * np.exp(-dl / 0.15)                      # kernel biaya lebih longgar (klaim salinan "hard" bergeser ~7%)
    jn = np.where(dl <= 0.15, jac, 0.0)
    kl = d["klaim_d"].to_numpy()
    fw = (kl[J] <= kl[I]) if asof else np.ones(len(I), bool)   # as-of: i hanya boleh dibandingkan dengan j yang diajukan lebih dulu
    bw = (kl[I] <= kl[J]) if asof else np.ones(len(I), bool)
    dr = lambda a: np.r_[a[fw], a[bw]]
    P = pd.DataFrame({"i": np.r_[I[fw], J[bw]], "j": np.r_[J[fw], I[bw]], "jac": dr(jac), "close": dr(close), "sc": dr(sc)})
    soft = pd.DataFrame({"i": np.r_[I[fw], J[bw]], "sc2": dr(sc2), "jn": dr(jn)}).groupby("i").max()
    best = P.sort_values("sc", ascending=False).drop_duplicates("i").set_index("i")
    near = P[(P["jac"] >= 0.7) & (P["close"] >= 0.5)].groupby("i").size()
    exact = P[(P["jac"] >= 0.999) & (P["close"] >= 0.5)].groupby("i").size()
    out = pd.DataFrame({"claim_id": d["claim_id"].to_numpy()})
    out["cl_best_score"] = best["sc"].reindex(range(n)).fillna(0).to_numpy()
    out["cl_best_jac"] = best["jac"].reindex(range(n)).fillna(0).to_numpy()
    out["cl_best_score_soft"] = soft["sc2"].reindex(range(n)).fillna(0).to_numpy()
    out["cl_best_jac_costnear"] = soft["jn"].reindex(range(n)).fillna(0).to_numpy()
    out["cl_best_close"] = best["close"].reindex(range(n)).fillna(0).to_numpy()
    out["cl_n_near"] = near.reindex(range(n)).fillna(0).to_numpy()
    out["cl_n_exact"] = exact.reindex(range(n)).fillna(0).to_numpy()
    bj = best["j"].reindex(range(n))
    has = bj.notna().to_numpy()
    bjj = bj.fillna(0).astype(int).to_numpy()
    out["cl_best_same_doctor"] = (has & (doctor[bjj] == doctor) & (doctor != "UNKNOWN")).astype(int)
    out["cl_best_date_gap"] = np.where(has, np.abs(masuk[bjj] - masuk), 999)

    # embedding semantik: TF-IDF + SVD(16) dipelajari dari data latih
    trm = tr
    tf = TfidfTransformer().fit(B[trm])
    svd = TruncatedSVD(n_components=16, random_state=0).fit(tf.transform(B[trm]))
    E = svd.transform(tf.transform(B))
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    emb = np.zeros(n)
    for _, idx in d.reset_index(drop=True).groupby(["facility_id", "dx"], sort=False).indices.items():
        if len(idx) < 2:
            continue
        S = E[idx] @ E[idx].T
        ok = (patient[idx][:, None] != patient[idx][None, :]) & (np.abs(masuk[idx][:, None] - masuk[idx][None, :]) <= WINDOW_DAYS)
        if asof:
            ok &= kl[idx][None, :] <= kl[idx][:, None]
        cl = np.exp(-np.abs(logv[idx][:, None] - logv[idx][None, :]) / 0.05)
        emb[idx] = np.where(ok, S * cl, -1).max(1)
    out["cl_emb_sim"] = np.clip(emb, 0, 1)

    # normalisasi per diagnosis (fit: latih) -> protokol baku punya kemiripan tinggi yang sah
    dxs = d["dx"].to_numpy()
    for col in ("cl_best_score", "cl_best_score_soft", "cl_best_jac_costnear", "cl_emb_sim", "cl_n_near"):
        s = out[col].to_numpy()
        stats = {}
        for dx in np.unique(dxs[trm]):
            v = s[trm & (dxs == dx)]
            if len(v) >= 30:
                stats[dx] = (float(np.median(v)), mad_scale(v, 0.05))
        gl = (float(np.median(s[trm])), mad_scale(s[trm], 0.05))
        out[col + "_z"] = [(v - stats.get(dx, gl)[0]) / stats.get(dx, gl)[1] for v, dx in zip(s, dxs)]
    return out


# ======================================================================== 3. inflated
def _basic_X(d, dxmap):
    n_addon = d["proc_list"].map(lambda l: sum(p in ADDON for p in l)).to_numpy()
    n_comp = d["sec_list"].map(lambda l: sum(s in COMPLICATION_DX for s in l)).to_numpy()
    return pd.DataFrame({
        "dx": d["dx"].map(dxmap).fillna(-1).astype(int), "ftype": d["jenis_faskes"].map(FTYPE).astype(int),
        "setting": (d["jenis_layanan"] == "RITL").astype(int), "sev": d["severity_num"].astype(int),
        "los": d["lama_rawat"].astype(int), "n_proc": d["n_proc"].astype(int), "n_addon": n_addon,
        "n_comp": n_comp, "age": d["kelompok_umur"].map(AGE).fillna(1).astype(int),
        "sex": (d["jenis_kelamin"] == "P").astype(int),
    })


def feat_inflated(d, tr, asof=False):
    tr_idx, ot_idx = np.where(tr)[0], np.where(~tr)[0]
    dxmap = {dx: i for i, dx in enumerate(sorted(d.loc[tr, "dx"].unique()))}
    X = _basic_X(d, dxmap)
    cat = [c in ("dx", "ftype") for c in X.columns]
    mk = lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.1,
                                               categorical_features=cat, random_state=0)
    y = d["log_nilai"].to_numpy()
    pred = oof_fit_predict(mk, X.to_numpy(), y, tr_idx, ot_idx, d["facility_id"].to_numpy())
    resid = y - pred
    out = pd.DataFrame({"claim_id": d["claim_id"], "ib_pred_log": pred, "ib_resid": resid})
    # koreksi level faskes (median residu klaim faskes itu sendiri; tanpa label)
    if asof:   # median residu hanya dari klaim faskes itu yang sudah diajukan lebih dulu (min. 20 klaim)
        s_ = pd.DataFrame({"fac": d["facility_id"].to_numpy(), "k": d["klaim_d"].to_numpy(), "r": resid, "i": np.arange(len(d))}).sort_values(["fac", "k", "i"])
        em = s_.groupby("fac")["r"].transform(lambda x: x.expanding(min_periods=20).median().shift(1))
        fac_med = pd.Series(em.to_numpy(), index=s_["i"].to_numpy()).sort_index().fillna(0.0).to_numpy()
    else:
        fac_med = pd.Series(resid).groupby(d["facility_id"].to_numpy()).transform("median").to_numpy()
    out["ib_fac_resid_med"] = fac_med
    out["ib_resid_fac_adj"] = resid - fac_med
    scale = {s: mad_scale(resid[tr & (d["jenis_layanan"] == s).to_numpy()], 0.05) for s in ("RJTL", "RITL")}
    out["ib_resid_z"] = out["ib_resid_fac_adj"] / d["jenis_layanan"].map(scale).to_numpy()
    # proporsi obat/alkes dibanding wajar diagnosis (inflasi menaikkan obat/alkes, komplikasi sah menaikkan semuanya)
    share = (d["biaya_obat_alkes"] / d["nilai_klaim"].clip(lower=1)).to_numpy()
    out["ib_obat_share"] = share
    dxs = d["dx"].to_numpy()
    z = np.zeros(len(d))
    gl = (float(np.median(share[tr])), mad_scale(share[tr], 0.02))
    st = {dx: (float(np.median(share[tr & (dxs == dx)])), mad_scale(share[tr & (dxs == dx)], 0.02))
          for dx in np.unique(dxs[tr]) if (tr & (dxs == dx)).sum() >= 30}
    out["ib_obat_share_z"] = [(v - st.get(dx, gl)[0]) / st.get(dx, gl)[1] for v, dx in zip(share, dxs)]
    out["ib_resid_x_obat"] = out["ib_resid_z"].clip(lower=0) * out["ib_obat_share_z"].clip(lower=0)
    return out


# ======================================================================== 4. upcoding
def feat_upcoding(d, tr, asof=False):
    n = len(d)
    ri = (d["jenis_layanan"] == "RITL").to_numpy()
    tr_idx = np.where(tr & ri)[0]
    ot_idx = np.where(~tr & ri)[0]
    dxmap = {dx: i for i, dx in enumerate(sorted(d.loc[tr, "dx"].unique()))}
    Xb = _basic_X(d, dxmap)
    Xs = Xb[["dx", "ftype", "los", "age", "sex"]]           # tanpa n_sec/n_proc/dx berat: itu yang bisa dimanipulasi
    cat = [c in ("dx", "ftype") for c in Xs.columns]
    mk = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, categorical_features=cat, random_state=0)
    y = (d["severity_num"].to_numpy() - 1).clip(0, 2)
    P = oof_fit_predict(mk, Xs.to_numpy(), y, tr_idx, ot_idx, d["facility_id"].to_numpy(), proba=True)
    out = pd.DataFrame({"claim_id": d["claim_id"]})
    sev = d["severity_num"].to_numpy()
    exp = np.where(ri, (P * np.arange(1, 4)[None, :]).sum(1), 0.0)
    pobs = np.where(ri, P[np.arange(n), np.clip(sev - 1, 0, 2)], 1.0)
    out["uc_sev_surprise"] = np.where(ri, sev - exp, 0.0)
    out["uc_sev_nll"] = np.where(ri, -np.log(np.clip(pobs, 1e-4, 1)), 0.0)
    out["uc_p_sev_ge_obs"] = np.where(ri, np.take_along_axis(np.cumsum(P[:, ::-1], 1)[:, ::-1], np.clip(sev - 1, 0, 2)[:, None], 1).ravel(), 1.0)
    out["uc_los_ratio"] = np.where(ri, d["lama_rawat"].to_numpy() / np.maximum(d["ref_lama_rawat"].to_numpy(), 1), 1.0)
    out["uc_n_severe_dx"] = d["sec_list"].map(lambda l: sum(s in SEVERE_DX for s in l)).to_numpy()
    out["uc_n_sec"] = d["n_sec"].to_numpy()
    out["uc_n_addon"] = d["proc_list"].map(lambda l: sum(p in ADDON for p in l)).to_numpy()

    # kelangkaan pasangan diagnosis-tindakan (fit: latih)
    rows = [(dx, p) for dx, pl in zip(d.loc[tr, "dx"], d.loc[tr, "proc_list"]) for p in pl]
    pc = pd.DataFrame(rows, columns=["dx", "p"]).groupby(["dx", "p"]).size().to_dict()
    ndx = d.loc[tr].groupby("dx").size().to_dict()
    rar, unseen = [], []
    for dx, pl in zip(d["dx"], d["proc_list"]):
        vals = [-np.log((pc.get((dx, p), 0) + 0.5) / (ndx.get(dx, 0) + 1.0)) for p in pl]
        rar.append(max(vals) if vals else 0.0)
        unseen.append(int(any(pc.get((dx, p), 0) < 3 for p in pl)))
    out["uc_proc_rarity"] = rar
    out["uc_pair_unseen"] = unseen

    # profil severity faskes vs rekan sejenis (tanpa label), dengan SHRINKAGE: faskes dengan sedikit klaim ditarik ke
    # rata-rata rekan sejenis. Mode as-of hanya memakai klaim yang sudah diajukan lebih dulu.
    hi = ((sev >= 2) & ri).astype(float)
    fac = d["facility_id"].to_numpy()
    ftype = d["jenis_faskes"].to_numpy()
    SHRINK = 40.0
    df = pd.DataFrame({"fac": fac, "ft": ftype, "hi": hi, "ri": ri, "m": d["bulan_masuk"].to_numpy(),
                       "k": d["klaim_d"].to_numpy(), "i": np.arange(n)})
    fs = df[df.ri].groupby("fac").agg(n=("hi", "size"), share=("hi", "mean"), ft=("ft", "first"))
    trfac = set(d.loc[tr, "facility_id"])
    peer = fs[fs.index.isin(trfac) & (fs.n >= 20)].groupby("ft")["share"].agg(["mean", "std"])
    gl_mean, gl_sd = float(fs["share"].mean()), float(fs["share"].std() or 0.1)
    mu_of, sd_of = {}, {}
    for ft in df["ft"].unique():
        mu_of[ft] = float(peer["mean"].get(ft, gl_mean))
        s_ = peer["std"].get(ft, np.nan)
        sd_of[ft] = max(float(s_) if not np.isnan(s_) else gl_sd, 0.05)
    mu_row = df["ft"].map(mu_of).to_numpy()
    sd_row = df["ft"].map(sd_of).to_numpy()
    shift = np.zeros(n)
    if not asof:
        shr = pd.Series({f: (r.n * r.share + SHRINK * mu_of[r.ft]) / (r.n + SHRINK) for f, r in fs.iterrows()})
        shr_row = pd.Series(fac).map(shr).fillna(pd.Series(mu_row)).to_numpy()
        fm = df[df.ri].groupby(["fac", "m"]).agg(c=("hi", "size"), k=("hi", "sum"))
        key = pd.MultiIndex.from_arrays([fac, d["bulan_masuk"].to_numpy()])
        c_m = fm["c"].reindex(key).to_numpy()
        k_m = fm["k"].reindex(key).to_numpy()
    else:
        R = df[df.ri].sort_values(["fac", "k", "i"]).copy()
        gR = R.groupby("fac")
        R["n_prior"] = gR.cumcount()
        R["k_prior"] = gR["hi"].cumsum() - R["hi"]
        R["shr"] = (R["k_prior"] + SHRINK * R["ft"].map(mu_of).to_numpy()) / (R["n_prior"] + SHRINK)
        gm = R.groupby(["fac", "m"])
        R["c_m"] = gm.cumcount()
        R["k_m"] = gm["hi"].cumsum() - R["hi"]
        ix = R["i"].to_numpy()
        shr_row, c_m, k_m = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        shr_row[ix], c_m[ix], k_m[ix] = R["shr"].to_numpy(), R["c_m"].to_numpy(), R["k_m"].to_numpy()
        shr_row = np.where(np.isnan(shr_row), mu_row, shr_row)
    z = (shr_row - mu_row) / sd_row
    p = np.clip(shr_row, 0.02, 0.98)
    ok = ri & (np.nan_to_num(c_m, nan=0) >= 10)
    shift[ok] = ((k_m[ok] / c_m[ok]) - p[ok]) / np.sqrt(p[ok] * (1 - p[ok]) / c_m[ok])
    out["uc_fac_hi_sev_z"] = np.where(ri, z, 0.0)
    out["uc_fac_month_shift"] = shift
    return out


# ======================================================================== orkestrasi
def build_features(d, train_mask, asof=False):
    assert_no_label_columns(d)
    cols = ["claim_id", "split", "jenis_layanan"] + ([] if asof else ["episode_size"])   # episode_size = fitur batch (retro saja)
    F = d[cols].rename(columns={"episode_size": "ctx_episode_size"}).copy()
    for name, fn in (("repeat", feat_repeat), ("clone", feat_clone), ("inflated", feat_inflated), ("upcoding", feat_upcoding)):
        F = F.merge(fn(d, train_mask, asof=asof), on="claim_id", how="left")
    assert_no_label_columns(F)
    return F


def auc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.nan_to_num(np.asarray(pos, float), nan=-1e9), np.nan_to_num(np.asarray(neg, float), nan=-1e9)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def md(df, fmt="{:.3f}"):
    L = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |", "|" + "---|" * (len(df.columns) + 1)]
    for i, r in df.iterrows():
        L.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("-" if pd.isna(v) else str(v)) for v in r]) + " |")
    return "\n".join(L)


HN = {"repeat_billing": "legit_recurring", "claim_cloning": "legit_similar",
      "inflated_bills": "legit_high_cost", "upcoding": "legit_atypical_coding"}


def feature_report(F, labels, scheme):
    """AUC tiap fitur per modus pada val (dan latih sebagai pembanding), per kesulitan, dan terhadap hard negative.
    Label dipakai HANYA untuk diagnostik fitur; split val dipakai untuk pemilihan, bukan test."""
    D = F.merge(labels, on="claim_id")
    feats = [c for c in F.columns if c[:3] in ("rp_", "cl_", "ib_", "uc_")]
    rows, top = [], []
    for mode, pre in GROUPS.items():
        for split in ("train", "val"):
            S = D[D["split"] == split]
            pop = S[S["jenis_layanan"] == "RITL"] if mode == "upcoding" else S
            pos, neg = pop[pop["anomaly_type"] == mode], pop[pop["is_anomaly"] == 0]
            for f in [c for c in feats if c.startswith(pre)]:
                a = auc(pos[f], neg[f])
                rows.append(dict(modus=mode, fitur=f, split=split, auc=a))
    R = pd.DataFrame(rows).pivot_table(index=["modus", "fitur"], columns="split", values="auc").reset_index()
    R["arah"] = np.where(R["val"] >= 0.5, "+", "-")
    R["auc_val_arah"] = np.where(R["val"] >= 0.5, R["val"], 1 - R["val"])
    R["selisih_train_val"] = (np.where(R["train"] >= 0.5, R["train"], 1 - R["train"]) - R["auc_val_arah"])
    S = D[D["split"] == "val"]
    detail = []
    for mode, pre in GROUPS.items():
        pop = S[S["jenis_layanan"] == "RITL"] if mode == "upcoding" else S
        norm = pop[pop["is_anomaly"] == 0]
        sub = R[R["modus"] == mode].sort_values("auc_val_arah", ascending=False).head(4)
        for _, r in sub.iterrows():
            f, sgn = r["fitur"], (1 if r["arah"] == "+" else -1)
            row = {"modus": mode, "fitur": f, "arah": r["arah"], "auc_val": r["auc_val_arah"]}
            for df_ in ("easy", "medium", "hard"):
                p_ = pop[(pop["anomaly_type"] == mode) & (pop["difficulty"] == df_)]
                a = auc(sgn * p_[f], sgn * norm[f])
                row[df_] = a
            hn = pop[pop["hard_negative_type"] == HN[mode]]
            row["vs_" + HN[mode]] = auc(sgn * pop[pop["anomaly_type"] == mode][f], sgn * hn[f])
            detail.append(row)
    return R, pd.DataFrame(detail).set_index("modus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--schemes", default="faskes,waktu")
    ap.add_argument("--modes", default="retro,asof", help="retro (batch) dan/atau asof (online, hanya klaim yang sudah diajukan)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    labels = pd.read_csv(os.path.join(a.root, "eval_only", "labels.csv"), keep_default_na=False)
    outdir = os.path.join(a.root, "features")
    os.makedirs(outdir, exist_ok=True)
    report = ["# Laporan fitur per modus (tahap 4)\n",
              "AUC dihitung pada split **val** (latih hanya pembanding untuk mendeteksi overfit). Negatif = semua klaim normal "
              "(upcoding: klaim rawat inap normal). Angka 0,5 = acak. Label hanya dipakai untuk diagnostik fitur, tidak masuk fitur.\n",
              "Mode `retro` = batch retrospektif; mode `asof` = klaim hanya dibandingkan dengan klaim yang sudah diajukan lebih dulu.\n"]
    for scheme in a.schemes.split(","):
        d = load_prep(a.root, scheme)
        tr = (d["split"] == "train").to_numpy()
        for mode in a.modes.split(","):
            asof = mode == "asof"
            suf = "_asof" if asof else ""
            F = build_features(d, tr, asof=asof)
            F.to_csv(os.path.join(outdir, f"features_{scheme}{suf}.csv.gz"), index=False)
            R, detail = feature_report(F, labels, scheme)
            R.to_csv(os.path.join(outdir, f"auc_fitur_{scheme}{suf}.csv"), index=False)
            report.append(f"\n## Skema `{scheme}`, mode `{mode}`\n")
            report.append(f"{F.shape[1] - 3} kolom fitur, {len(F):,} klaim.\n")
            report.append("### Fitur terbaik per modus (val): AUC menurut kesulitan dan terhadap hard negative\n")
            report.append(md(detail) + "\n")
            report.append("### Semua fitur (AUC val sesuai arah; selisih latih-val besar = waspada overfit)\n")
            report.append(md(R.set_index("modus")[["fitur", "arah", "auc_val_arah", "selisih_train_val"]].sort_values(["modus", "auc_val_arah"], ascending=[True, False]).reset_index().set_index("modus")) + "\n")
            print(f"[{scheme}/{mode}] selesai: {F.shape}", flush=True)
    with open(os.path.join(outdir, "fitur_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    if not a.quiet:
        print("\n".join(report))


if __name__ == "__main__":
    main()