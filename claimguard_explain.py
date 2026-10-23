#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Tahap 7: explainability / interpretability

Masukan:
  out/detect/detect_{skema}_{mode}.csv.gz
  out/features/features_{skema}{_asof}.csv.gz
  out/calibrated/calibrated_{skema}_{mode}.csv.gz
  out/data/claims.csv
  out/eval_only/labels.csv

Keluaran:
  out/explain/explain_{skema}_{mode}.csv.gz
  out/explain/explain_report.md

Tujuan:
  memberi penjelasan per klaim yang ditandai: komponen detektor mana yang paling dominan,
  fitur apa yang paling mendorong skor tinggi, dan bagaimana pemicu itu berhubungan dengan
  modus anomali yang dipelajari (repeat / clone / inflated / upcoding).
"""
import argparse
import os

import numpy as np
import pandas as pd

DET_COLS = ["p_repeat", "p_clone", "p_infl", "p_upc", "p_ens"]

DRIVER_MAP = {
    "repeat_billing": {
        "label": "Repeat billing",
        "detectors": ["p_repeat"],
        "features": ["rp_gap_ratio_typ", "rp_val_ratio", "rp_item_jac", "rp_same_doctor", "rp_cnt_14d", "rp_combo"],
    },
    "claim_cloning": {
        "label": "Claim cloning",
        "detectors": ["p_clone"],
        "features": ["cl_best_score_soft_z", "cl_emb_sim_z", "cl_best_score_z", "cl_best_same_doctor", "cl_n_near_z", "cl_best_date_gap"],
    },
    "inflated_bills": {
        "label": "Inflated bills",
        "detectors": ["p_infl"],
        "features": ["ib_resid_x_obat", "ib_resid_z", "ib_obat_share_z", "ib_resid", "ib_fac_resid_med"],
    },
    "upcoding": {
        "label": "Upcoding",
        "detectors": ["p_upc"],
        "features": ["uc_sev_surprise", "uc_los_short", "uc_n_severe_dx", "uc_proc_rarity", "uc_p_sev_ge_obs"],
    },
    "ensemble": {
        "label": "Ensemble / cross-pattern",
        "detectors": ["p_ens"],
        "features": ["p_if", "p_ecod", "p_copod", "rp_gap_ratio_typ", "cl_emb_sim_z", "ib_resid_z", "uc_sev_surprise"],
    },
}


def md(df, fmt="{:.3f}"):
    L = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
         "|" + "---|" * (len(df.columns) + 1)]
    for i, r in df.iterrows():
        L.append("| " + " | ".join([
            str(i),
            *(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("-" if pd.isna(v) else str(v)) for v in r)
        ]) + " |")
    return "\n".join(L)


def fmt_num(v):
    if pd.isna(v):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)


def mode_from_driver(driver):
    if driver in DRIVER_MAP:
        return DRIVER_MAP[driver]["label"]
    return driver


def explain_row(row):
    mode = row["top_driver_mode"]
    cfg = DRIVER_MAP.get(mode, DRIVER_MAP["ensemble"])
    feature_names = cfg["features"]
    feature_values = []
    for c in feature_names:
        if c in row and pd.notna(row[c]):
            feature_values.append((c, float(row[c])))
    if not feature_values:
        return "Sinyal utama tidak dapat diuraikan dari fitur yang tersedia."
    feature_values = sorted(feature_values, key=lambda kv: abs(kv[1]), reverse=True)[:3]
    top = feature_values[0]
    top_name, top_val = top
    if mode == "repeat_billing":
        return (
            f"{cfg['label']} dominan ({row['top_driver_score']:.3f}) dengan {top_name}={fmt_num(top_val)}; "
            f"nilai lain: {', '.join(f'{n}={fmt_num(v)}' for n, v in feature_values[1:]) or 'tidak ada'}; "
            f"karakteristik ini sesuai pola pengulangan/overlap pasien dan item yang berulang."
        )
    if mode == "claim_cloning":
        return (
            f"{cfg['label']} dominan ({row['top_driver_score']:.3f}) dengan {top_name}={fmt_num(top_val)}; "
            f"nilai lain: {', '.join(f'{n}={fmt_num(v)}' for n, v in feature_values[1:]) or 'tidak ada'}; "
            f"indikasi duplikasi/salinan dengan kemiripan tinggi antar klaim dan tanggal/doctor yang serupa."
        )
    if mode == "inflated_bills":
        return (
            f"{cfg['label']} dominan ({row['top_driver_score']:.3f}) dengan {top_name}={fmt_num(top_val)}; "
            f"nilai lain: {', '.join(f'{n}={fmt_num(v)}' for n, v in feature_values[1:]) or 'tidak ada'}; "
            f"ini konsisten dengan biaya yang lebih tinggi dari ekspektasi, terutama pada obat/alkes dan komponen biaya utama."
        )
    if mode == "upcoding":
        return (
            f"{cfg['label']} dominan ({row['top_driver_score']:.3f}) dengan {top_name}={fmt_num(top_val)}; "
            f"nilai lain: {', '.join(f'{n}={fmt_num(v)}' for n, v in feature_values[1:]) or 'tidak ada'}; "
            f"pola ini menunjukkan lonjakan severity dan prosedur/diagnosis yang tidak konsisten dengan lama rawat."
        )
    return (
        f"Sinyal ensemble dominan ({row['top_driver_score']:.3f}) dengan {top_name}={fmt_num(top_val)}; "
        f"nilai lain: {', '.join(f'{n}={fmt_num(v)}' for n, v in feature_values[1:]) or 'tidak ada'}; "
        f"kombinasi outlier lintas fitur menunjukkan pola tidak wajar yang tidak semata-mata berasal dari satu modus tunggal."
    )


def pick_top_driver(row):
    score_map = {k: float(row.get(k, 0.0) or 0.0) for k in DET_COLS}
    top_name, top_score = max(score_map.items(), key=lambda kv: kv[1])
    if top_name == "p_ens" and top_score < 0.5 and len(score_map) > 0:
        # jika p_ens bukan dominan secara kuat, balik ke komponen aturan yang lebih jelas
        ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_score = ordered[0]
    if top_name == "p_repeat":
        mode = "repeat_billing"
    elif top_name == "p_clone":
        mode = "claim_cloning"
    elif top_name == "p_infl":
        mode = "inflated_bills"
    elif top_name == "p_upc":
        mode = "upcoding"
    else:
        mode = "ensemble"
    return mode, top_name, top_score


def run(root, scheme, mode, budget=0.05, top_n=25):
    suf = f"{scheme}_{mode}"
    detect = pd.read_csv(os.path.join(root, "detect", f"detect_{suf}.csv.gz"))
    features = pd.read_csv(os.path.join(root, "features", f"features_{scheme}{'_asof' if mode == 'asof' else ''}.csv.gz"))
    calibrated = pd.read_csv(os.path.join(root, "calibrated", f"calibrated_{suf}.csv.gz"))
    labels = pd.read_csv(os.path.join(root, "eval_only", "labels.csv"), keep_default_na=False)
    claims = pd.read_csv(
        os.path.join(root, "data", "claims.csv"),
        usecols=["claim_id", "facility_id", "jenis_layanan"],
    ).rename(columns={"facility_id": "facility_id_claim", "jenis_layanan": "jenis_layanan_claim"})

    merged = detect.merge(features, on=["claim_id", "split"], how="inner")
    merged = merged.merge(calibrated[["claim_id", "split", "p_value", "skor_0_100", "tingkat"]], on=["claim_id", "split"], how="inner")
    merged = merged.merge(labels[["claim_id", "anomaly_type", "difficulty", "hard_negative_type", "is_anomaly"]], on="claim_id", how="left")
    merged = merged.merge(claims, on="claim_id", how="left")

    test = merged[merged["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame(), pd.DataFrame(), "Tidak ada data test untuk menjelaskan."

    flagged = test[(test["p_value"] <= budget)].copy()
    if flagged.empty:
        flagged = test.sort_values("risk_raw", ascending=False).head(top_n).copy()
    else:
        flagged = flagged.sort_values("p_value").head(top_n).copy()

    out_rows = []
    for _, row in flagged.iterrows():
        mode_name, top_name, top_score = pick_top_driver(row)
        facility_id = row.get("facility_id", row.get("facility_id_claim", "-"))
        jenis_layanan = row.get("jenis_layanan", row.get("jenis_layanan_claim", "-"))
        r = {
            "claim_id": row["claim_id"],
            "split": row["split"],
            "facility_id": facility_id,
            "jenis_layanan": jenis_layanan,
            "anomaly_type": row.get("anomaly_type", "none"),
            "difficulty": row.get("difficulty", "-"),
            "hard_negative_type": row.get("hard_negative_type", "none"),
            "is_anomaly": row.get("is_anomaly", 0),
            "p_value": float(row["p_value"]),
            "skor_0_100": float(row["skor_0_100"]),
            "tingkat": row["tingkat"],
            "top_driver": top_name,
            "top_driver_mode": mode_name,
            "top_driver_score": float(top_score),
            "top_feature": "",
            "top_feature_value": np.nan,
            "explanation": "",
            "risk_raw": float(row["risk_raw"]),
        }
        cfg = DRIVER_MAP.get(mode_name, DRIVER_MAP["ensemble"])
        feature_candidates = []
        for c in cfg["features"]:
            if c in row and pd.notna(row[c]):
                val = float(row[c])
                feature_candidates.append((c, val))
                r[c] = val
        if feature_candidates:
            feature_candidates = sorted(feature_candidates, key=lambda kv: abs(kv[1]), reverse=True)
            r["top_feature"] = feature_candidates[0][0]
            r["top_feature_value"] = feature_candidates[0][1]
        r["explanation"] = explain_row(r)
        out_rows.append(r)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out, pd.DataFrame(), "Tidak ada klaim yang dipilih untuk penjelasan."

    out = out.sort_values(["p_value", "risk_raw"], ascending=[True, False]).reset_index(drop=True)
    summary_rows = []
    for key, g in out.groupby("top_driver_mode"):
        summary_rows.append({
            "modus": key,
            "n_klaim": int(len(g)),
            "p_value_rata2": float(g["p_value"].mean()),
            "skor_rata2": float(g["skor_0_100"].mean()),
            "precision_likely": float(g["is_anomaly"].mean()) if "is_anomaly" in g else np.nan,
        })
    summary = pd.DataFrame(summary_rows).set_index("modus") if summary_rows else pd.DataFrame()
    return out, summary, f"{len(out)} klaim terpilih pada skema `{scheme}` mode `{mode}`."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    ap.add_argument("--schemes", default="faskes,waktu")
    ap.add_argument("--modes", default="retro,asof")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    outdir = os.path.join(a.root, "explain")
    os.makedirs(outdir, exist_ok=True)
    report = [
        "# Laporan penjelasan klaim (tahap 7)\n",
        "Daftar ini menyorot klaim yang diprediksi paling mencurigakan pada split **test** berdasarkan p-value konformal. "
        "Setiap baris menjelaskan komponen detektor paling dominan dan fitur-fitur yang mendorong skor tinggi.\n",
    ]

    all_rows = []
    for scheme in a.schemes.split(","):
        for mode in a.modes.split(","):
            rows, summary, info = run(a.root, scheme, mode, budget=a.budget, top_n=a.top)
            if rows.empty:
                report.append(f"\n## Skema `{scheme}`, mode `{mode}`\n")
                report.append(f"{info}\n")
                continue
            rows = rows.copy()
            rows["scheme"] = scheme
            rows["mode"] = mode
            all_rows.append(rows)
            report.append(f"\n## Skema `{scheme}`, mode `{mode}`\n")
            report.append(f"{info}\n")
            report.append(f"**Distribusi komponen dominan**\n")
            if not summary.empty:
                report.append(md(summary.reset_index()) + "\n")
            else:
                report.append("Tidak ada ringkasan pembagi modus.\n")
            report.append("**Contoh klaim terpilih**\n")
            topk = rows[["claim_id", "facility_id", "jenis_layanan", "p_value", "skor_0_100", "tingkat", "top_driver_mode", "top_driver", "top_feature", "explanation"]].head(10)
            report.append(md(topk.set_index("claim_id")) + "\n")
            outfile = os.path.join(outdir, f"explain_{scheme}_{mode}.csv.gz")
            rows.to_csv(outfile, index=False)
            print(f"[{scheme}/{mode}] {len(rows)} klaim dijelaskan -> {outfile}", flush=True)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(os.path.join(outdir, "explain_all.csv.gz"), index=False)

    report_path = os.path.join(outdir, "explain_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"[ok] Laporan lengkap: {report_path}", flush=True)


if __name__ == "__main__":
    main()
