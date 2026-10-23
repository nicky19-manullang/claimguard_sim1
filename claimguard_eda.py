#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 3: EDA yang mendesain fitur (HANYA data latih)

Dua tujuan:
  (a) memvalidasi realisme simulator (biaya ~ lognormal, lama rawat naik menurut severity, pola waktu, profil faskes),
  (b) menentukan sinyal tiap modus: seberapa baik satu sinyal sederhana memisahkan anomali dari klaim normal
      dan dari hard negatives. Label dipakai di sini hanya sebagai diagnostik desain fitur pada split latih.

    python claimguard_eda.py --root out --scheme faskes
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew

# palet kategorikal tervalidasi (slot 1-4, mode terang) + abu untuk "normal"
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9a9a96"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 11, "axes.titlecolor": INK,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
MODES = ["repeat_billing", "claim_cloning", "inflated_bills", "upcoding"]
SEVERE_DX = {"A41.9", "J96.0", "N17.9", "E87.1", "D65"}


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    L = ["| " + " | ".join(str(c) for c in [df.index.name or ""] + cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("-" if pd.isna(v) else str(v)) for v in r]
        L.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(L)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos = np.nan_to_num(pos, nan=-1e9)
    neg = np.nan_to_num(neg, nan=-1e9)
    r = rankdata(np.r_[pos, neg])
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ sinyal per modus
def signal_repeat(d):
    keys = ["patient_key", "facility_id", "dx3"]
    x = d.sort_values(keys + ["masuk_d", "claim_id"]).copy()
    cm = x.groupby(keys, sort=False)["pulang_d"].cummax()
    prev_end = cm.groupby([x[k] for k in keys], sort=False).shift(1)
    prev_val = x.groupby(keys, sort=False)["nilai_klaim"].shift(1)
    gap = (x["masuk_d"] - prev_end).fillna(999)
    vr = (np.minimum(x["nilai_klaim"], prev_val) / np.maximum(x["nilai_klaim"], prev_val)).fillna(0)
    out = pd.DataFrame({"claim_id": x["claim_id"], "repeat_gap": -gap.clip(-30, 400),
                        "repeat_combo": np.where(gap <= 14, vr, 0.0)})
    return out


def signal_clone(d, window=200):
    tok = []
    for it, pr, se in zip(d["item_list"], d["proc_list"], d["sec_list"]):
        t = [x for x in str(it).split("|") if x] + ["P" + x for x in str(pr).split("|") if x] + ["S" + x for x in str(se).split("|") if x]
        tok.append(t)
    vocab = {t: i for i, t in enumerate(sorted({t for l in tok for t in l}))}
    score = np.zeros(len(d))
    grp = d.groupby(["facility_id", "dx"]).indices
    patient, masuk, logv = d["patient_key"].to_numpy(), d["masuk_d"].to_numpy(), d["log_nilai"].to_numpy()
    for _, idx in grp.items():
        if len(idx) < 2:
            continue
        X = np.zeros((len(idx), len(vocab)), np.float32)
        for r, i in enumerate(idx):
            for t in tok[i]:
                X[r, vocab[t]] = 1
        inter = X @ X.T
        s = X.sum(1)
        jac = inter / np.maximum(s[:, None] + s[None, :] - inter, 1)
        close = np.exp(-np.abs(logv[idx][:, None] - logv[idx][None, :]) / 0.05)
        ok = (patient[idx][:, None] != patient[idx][None, :]) & (np.abs(masuk[idx][:, None] - masuk[idx][None, :]) <= window)
        m = np.where(ok, jac * close, 0.0)
        score[idx] = m.max(1)
    return pd.DataFrame({"claim_id": d["claim_id"], "clone_sim": score})


def signal_upcode(d):
    dd = d.copy()
    sec_cnt = dd["sec_list"].map(lambda s: sum(x in SEVERE_DX for x in str(s).split("|")))
    # kelangkaan tindakan terhadap diagnosisnya (dari data latih)
    rows = [(dx, p) for dx, pl in zip(dd["dx"], dd["proc_list"]) for p in str(pl).split("|") if p]
    pr = pd.DataFrame(rows, columns=["dx", "p"])
    share = (pr.groupby(["dx", "p"]).size() / dd.groupby("dx").size()).dropna().to_dict()
    rar = [max([-np.log(max(share.get((dx, p), 1e-3), 1e-3)) for p in str(pl).split("|") if p] or [0.0])
           for dx, pl in zip(dd["dx"], dd["proc_list"])]
    return pd.DataFrame({"claim_id": dd["claim_id"], "los_pendek_utk_severity": -dd["resid_lama_rawat_naif"],
                         "jumlah_dx_berat": sec_cnt, "tindakan_langka": rar})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--scheme", default="faskes", choices=["faskes", "waktu"])
    a = ap.parse_args()
    out = os.path.join(a.root, "eda")
    os.makedirs(out, exist_ok=True)

    prep = pd.read_csv(os.path.join(a.root, "prep", f"claims_prep_{a.scheme}.csv.gz"), keep_default_na=False)
    labels = pd.read_csv(os.path.join(a.root, "eval_only", "labels.csv"), keep_default_na=False)
    truth = pd.read_csv(os.path.join(a.root, "eval_only", "facility_truth.csv"))
    for c in ("days_since_prev_claim", "resid_log_nilai_naif", "resid_lama_rawat_naif", "log_nilai"):
        prep[c] = pd.to_numeric(prep[c], errors="coerce")
    tr = prep[prep["split"] == "train"].reset_index(drop=True)          # EDA hanya data latih
    n = len(tr)
    R = [f"# EDA data latih (skema `{a.scheme}`, {n:,} klaim)\n",
         "Semua statistik dihitung hanya dari split latih. Label dipakai hanya untuk diagnostik sinyal (bagian 6).\n"]

    # ---------------------------------------------------------------- 1 ringkasan
    R.append("## 1. Ringkasan\n")
    R.append(f"- Klaim {n:,} | faskes {tr['facility_id'].nunique()} | pasien {tr['patient_key'].nunique():,} | rawat inap {(tr['jenis_layanan']=='RITL').mean():.1%}")
    R.append(f"- Nilai klaim median Rp {tr['nilai_klaim'].median():,.0f} (rawat jalan Rp {tr.loc[tr.jenis_layanan=='RJTL','nilai_klaim'].median():,.0f}; rawat inap Rp {tr.loc[tr.jenis_layanan=='RITL','nilai_klaim'].median():,.0f})")
    R.append(f"- Lama rawat inap median {tr.loc[tr.jenis_layanan=='RITL','lama_rawat'].median():.0f} hari | jeda pengajuan klaim median {tr['klaim_lag'].median():.0f} hari")
    R.append(f"- Episode multi-klaim (pasien+faskes+dx, jeda ≤1 hari): {(tr.groupby('episode_id').size()>1).sum():,}\n")

    # ---------------------------------------------------------------- 2 biaya per kelompok
    R.append("## 2. Profil biaya dan lama rawat per diagnosis utama\n")
    g = tr.groupby(["dx", "jenis_layanan"]).agg(n=("claim_id", "size"), median_rp=("nilai_klaim", "median"),
                                                 p25=("nilai_klaim", lambda s: s.quantile(.25)),
                                                 p75=("nilai_klaim", lambda s: s.quantile(.75)),
                                                 cv=("nilai_klaim", lambda s: s.std() / s.mean()),
                                                 los_med=("lama_rawat", "median")).reset_index().sort_values("median_rp")
    g.to_csv(os.path.join(out, "profil_biaya_per_dx.csv"), index=False)
    gi = g.set_index("dx")
    R.append(md(gi[["jenis_layanan", "n", "median_rp", "p25", "p75", "cv", "los_med"]].assign(
        median_rp=lambda x: x.median_rp.round(0), p25=lambda x: x.p25.round(0), p75=lambda x: x.p75.round(0)), "{:,.2f}"))
    low = g[(g["cv"] < 0.2) & (g["n"] >= 100)]["dx"].tolist()
    R.append(f"\nKelompok dengan CV biaya rendah (< 0,2, n ≥ 100): {', '.join(low)}. Ini klaim yang wajar sangat seragam (protokol baku), "
             "sehingga klaim sah di kelompok ini memang mirip satu sama lain (dasar hard negative `legit_similar` dan `legit_recurring`). "
             "Persalinan normal (O80) tetap bervariasi karena severity dan lama rawat.\n")

    # ---------------------------------------------------------------- 3 realisme
    R.append("## 3. Validasi realisme simulator\n")
    base = tr[tr["n_sec"] >= 0]
    sk = tr.groupby(["dx", "severity_num"])["log_nilai"].agg(lambda s: skew(s) if len(s) >= 30 else np.nan).dropna()
    ri = tr[tr.jenis_layanan == "RITL"]
    sev = ri.groupby(["dx", "severity_num"]).agg(c=("nilai_klaim", "median"), l=("lama_rawat", "median")).unstack()
    r2 = (sev["c"][2] / sev["c"][1]).median()
    r3 = (sev["c"][3] / sev["c"][1]).median()
    l3 = (sev["l"][3] / sev["l"][1]).median()
    rj = tr[tr.jenis_layanan == "RJTL"]
    wk = rj["is_weekend_masuk"].mean()
    ftype = tr.groupby("jenis_faskes")["resid_log_nilai_naif"].median()
    sevmix = ri.groupby("jenis_faskes")["severity_num"].apply(lambda s: (s == 3).mean())
    checks = pd.DataFrame([
        ("Skewness log(biaya) per (dx, severity), median", f"{sk.median():.2f}", "dekat 0 (≈ lognormal)", "ya" if abs(sk.median()) < 0.6 else "periksa"),
        ("Rasio biaya severity II / I (median antar dx)", f"{r2:.2f}", "≈ 1,35 + efek lama rawat", "ya" if 1.2 < r2 < 1.9 else "periksa"),
        ("Rasio biaya severity III / I", f"{r3:.2f}", "≈ 1,9 + efek lama rawat", "ya" if 1.6 < r3 < 3.0 else "periksa"),
        ("Rasio lama rawat severity III / I", f"{l3:.2f}", "> 1 (naik menurut severity)", "ya" if l3 > 1.2 else "periksa"),
        ("Porsi rawat jalan pada akhir pekan", f"{wk:.1%}", "< 28,6% (pola hari kerja)", "ya" if wk < 2 / 7 else "periksa"),
        ("Selisih biaya naif RS_A vs Klinik (log)", f"{ftype.get('RS_A', np.nan) - ftype.get('Klinik', np.nan):+.2f}", "RS_A lebih mahal", "ya" if ftype.get('RS_A', 0) > ftype.get('Klinik', 0) else "periksa"),
        ("Porsi severity III: RS_A vs RS_D (rawat inap)", f"{sevmix.get('RS_A', np.nan):.1%} vs {sevmix.get('RS_D', np.nan):.1%}", "RS_A lebih tinggi", "ya" if sevmix.get('RS_A', 0) > sevmix.get('RS_D', 0) else "periksa"),
    ], columns=["Pemeriksaan", "Hasil", "Harapan", "Lolos?"]).set_index("Pemeriksaan")
    R.append(md(checks) + "\n")

    # ---------------------------------------------------------------- 4 waktu
    R.append("## 4. Pola waktu\n")
    mon = tr.groupby("bulan_masuk").size()
    wd = tr.groupby(tr["tanggal_masuk"].str.slice(0, 10).pipe(pd.to_datetime).dt.weekday).size()
    lag = tr["klaim_lag"]
    R.append(f"- Klaim per bulan (masuk): {', '.join(f'{int(k)}={int(v):,}' for k, v in mon.items())}")
    R.append(f"- Jeda pengajuan klaim: median {lag.median():.0f}, p90 {lag.quantile(.9):.0f}, maks {lag.max():.0f} hari")
    R.append(f"- Hari-hari dalam seminggu (Sen..Min): {', '.join(str(int(v)) for v in wd.values)}")
    R.append("- Catatan artefak: penurunan Desember (dan sedikit Januari) berasal dari penyensoran simulator "
             "(klaim yang diajukan setelah 31 Des 2025 tidak dibangkitkan), bukan pola musiman. "
             "Jangan diinterpretasikan sebagai tren, dan ingat bahwa split waktu `test` (Nov-Des) sedikit terpengaruh.\n")

    # ---------------------------------------------------------------- 5 faskes
    R.append("## 5. Profil faskes\n")
    fac = tr.groupby(["facility_id", "jenis_faskes"]).agg(
        n=("claim_id", "size"), porsi_ri=("jenis_layanan", lambda s: (s == "RITL").mean()),
        residu_biaya_median=("resid_log_nilai_naif", "median"),
        porsi_dokter_teratas=("doctor_key", lambda s: s.value_counts(normalize=True).iloc[0]),
        jeda_klaim_median=("klaim_lag", "median")).reset_index()
    lab_tr = tr[["claim_id", "facility_id"]].merge(labels[["claim_id", "is_anomaly"]], on="claim_id")
    fac = fac.merge(lab_tr.groupby("facility_id")["is_anomaly"].mean().rename("tingkat_anomali"), on="facility_id")
    fac = fac.merge(truth[["facility_id", "risky", "dominant_modes"]], on="facility_id")
    fac.to_csv(os.path.join(out, "profil_faskes.csv"), index=False)
    spread = fac["residu_biaya_median"].std()
    R.append(f"- {len(fac)} faskes latih; simpangan baku residu biaya median antar faskes = {spread:.3f} (variasi sah; bukan anomali).")
    R.append(f"- Tingkat anomali rata-rata faskes berisiko {fac.loc[fac.risky==1,'tingkat_anomali'].mean():.1%} vs lainnya {fac.loc[fac.risky==0,'tingkat_anomali'].mean():.1%}.\n")

    # ---------------------------------------------------------------- 6 sinyal per modus
    R.append("## 6. Sinyal tiap modus (AUC satu sinyal, split latih)\n")
    sig = tr[["claim_id"]].copy()
    sig["biaya_residu_z"] = (tr["resid_log_nilai_naif"] / tr["resid_log_nilai_naif"].abs().median() / 1.4826).to_numpy()
    sig = sig.merge(signal_repeat(tr), on="claim_id").merge(signal_clone(tr), on="claim_id").merge(
        signal_upcode(tr), on="claim_id").merge(labels, on="claim_id")
    sig["jenis_layanan"] = tr.set_index("claim_id").loc[sig["claim_id"], "jenis_layanan"].to_numpy()
    signals = ["repeat_gap", "repeat_combo", "clone_sim", "biaya_residu_z", "los_pendek_utk_severity", "jumlah_dx_berat", "tindakan_langka"]
    normal = sig[sig.is_anomaly == 0]
    mat = {}
    for m in MODES:
        pos = sig[sig.anomaly_type == m]
        nn = normal[normal.jenis_layanan == "RITL"] if m == "upcoding" else normal
        mat[m] = {s: auc(pos[s], nn[s]) for s in signals}
    A = pd.DataFrame(mat)
    A.index.name = "sinyal"
    A.to_csv(os.path.join(out, "auc_sinyal_per_modus.csv"))
    R.append("AUC tiap sinyal terhadap seluruh klaim normal (0,5 = acak; upcoding dibandingkan dengan rawat inap normal):\n")
    R.append(md(A) + "\n")

    primary = {"repeat_billing": "repeat_combo", "claim_cloning": "clone_sim", "inflated_bills": "biaya_residu_z",
               "upcoding": "jumlah_dx_berat"}
    rows = []
    for m in MODES:
        for s in {primary[m]} | ({"los_pendek_utk_severity", "tindakan_langka", "biaya_residu_z"} if m == "upcoding" else set()):
            nn = normal[normal.jenis_layanan == "RITL"] if m == "upcoding" else normal
            row = {"modus": m, "sinyal": s}
            for dfc in ("easy", "medium", "hard"):
                row[dfc] = auc(sig[(sig.anomaly_type == m) & (sig.difficulty == dfc)][s], nn[s])
            for hn in ("legit_recurring", "legit_revisit", "legit_similar", "legit_high_cost"):
                row[f"vs_{hn}"] = auc(sig[sig.anomaly_type == m][s], sig[sig.hard_negative_type == hn][s])
            rows.append(row)
    P = pd.DataFrame(rows).set_index(["modus"])
    P.to_csv(os.path.join(out, "auc_sinyal_utama.csv"))
    R.append("AUC sinyal utama per tingkat kesulitan, dan terhadap tiap jenis hard negative (anomali modus tsb vs klaim sah yang mirip):\n")
    R.append(md(P) + "\n")

    # ---------------------------------------------------------------- 7 grafik
    # fig 1: biaya median per diagnosis (skala log), batang = median, garis = p25-p75
    gg = g.sort_values("median_rp")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(gg["dx"], gg["median_rp"], color=BLUE, height=0.55)
    ax.errorbar(gg["median_rp"], gg["dx"], xerr=[gg["median_rp"] - gg["p25"], gg["p75"] - gg["median_rp"]], fmt="none", ecolor=INK2, lw=1, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Nilai klaim (Rp, skala log): median dan rentang p25-p75")
    ax.set_title("Biaya klaim per diagnosis utama (data latih)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_biaya_per_dx.png"), dpi=140)
    plt.close(fig)

    # fig 2: pola waktu
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].plot(mon.index, mon.values, color=BLUE, lw=2, marker="o", ms=4)
    ax[0].set_title("Klaim per bulan masuk")
    ax[0].set_xlabel("Bulan")
    ax[0].set_ylim(0)
    ax[1].bar(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"], wd.values, color=BLUE, width=0.6)
    ax[1].set_title("Klaim per hari masuk")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_pola_waktu.png"), dpi=140)
    plt.close(fig)

    # fig 3: profil faskes
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for r, c, lab in ((0, GREY, "Faskes biasa"), (1, ORANGE, "Faskes berisiko (disimulasikan)")):
        s = fac[fac.risky == r]
        ax.scatter(s["residu_biaya_median"], s["tingkat_anomali"] * 100, s=40 + s["n"] / 40, color=c, edgecolor="#fcfcfb", linewidth=1.5, label=lab, alpha=0.95)
    ax.set_xlabel("Residu biaya median (log) terhadap referensi (diagnosis, severity)")
    ax.set_ylabel("Tingkat anomali (%)")
    ax.set_title("Variasi biaya sah antar faskes vs tingkat anomali")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_profil_faskes.png"), dpi=140)
    plt.close(fig)

    # fig 4: pemisahan sinyal utama per modus
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    xlim = {"repeat_billing": (0, 1.0), "claim_cloning": (0, 1.0), "inflated_bills": (-4, 12), "upcoding": (0, 3)}
    xlabel = {"repeat_billing": "repeat_combo (kemiripan biaya jika jarak ≤14 hari)", "claim_cloning": "clone_sim (Jaccard x kedekatan biaya)",
              "inflated_bills": "biaya_residu_z", "upcoding": "jumlah dx berat"}
    hn_map = {"repeat_billing": "legit_recurring", "claim_cloning": "legit_similar", "inflated_bills": "legit_high_cost", "upcoding": "legit_high_cost"}
    for ax, m in zip(axes.ravel(), MODES):
        s = primary[m]
        lo, hi = xlim[m]
        bins = np.linspace(lo, hi, 26 if m != "upcoding" else 7)
        pop = sig[sig.jenis_layanan == "RITL"] if m == "upcoding" else sig
        for sub, c, lab in ((pop[pop.is_anomaly == 0], GREY, "normal"),
                            (pop[pop.hard_negative_type == hn_map[m]], ORANGE, f"hard negative ({hn_map[m]})"),
                            (pop[pop.anomaly_type == m], BLUE, "anomali")):
            ax.hist(np.clip(sub[s], lo, hi), bins=bins, density=True, histtype="step", color=c, lw=2, label=lab)
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel(xlabel[m])
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Pemisahan sinyal utama (kepadatan, sumbu y log)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_pemisahan_sinyal.png"), dpi=140)
    plt.close(fig)

    R.append("## 7. Grafik\n\n`fig1_biaya_per_dx.png`, `fig2_pola_waktu.png`, `fig3_profil_faskes.png`, `fig4_pemisahan_sinyal.png`\n")
    with open(os.path.join(out, "eda_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))


if __name__ == "__main__":
    main()
