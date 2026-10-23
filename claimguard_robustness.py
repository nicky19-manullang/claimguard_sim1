#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tahap 8: uji multi-seed + variasi prevalensi 1-5%.

Menjalankan pipeline lengkap untuk beberapa kombinasi seed dan prevalence,
melaporkan FPR empiris terkalibrasi dan variasi antar run.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PY = r"C:/Users/admin/AppData/Local/Programs/Python/Python313/python.exe"
SEEDS = [0, 7, 13, 21, 42]
PREVALENCES = [0.01, 0.02, 0.03, 0.04, 0.05]


def run_cmd(args):
    print(f"[RUN] {' '.join(args)}")
    cp = subprocess.run(args, capture_output=True, text=True)
    if cp.stdout:
        print(cp.stdout.strip())
    if cp.stderr:
        print(cp.stderr.strip(), file=sys.stderr)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return cp.stdout


def parse_fpr_block(text):
    pat = re.compile(r"\[(.*?)\/(.*?)\] FPR empiris .*?:\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)\s*\| recall@5% =\s*([0-9.]+)")
    out = []
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            scheme, mode, a1, a2, a3, a4, recall = m.groups()
            out.append({
                "scheme": scheme.strip(),
                "mode": mode.strip(),
                "fpr_1": float(a1),
                "fpr_3": float(a2),
                "fpr_5": float(a3),
                "fpr_10": float(a4),
                "recall_5": float(recall),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    ap.add_argument("--prevalence", nargs="*", type=float, default=None)
    ap.add_argument("--root", default="robustness")
    a = ap.parse_args()

    seeds = a.seed if a.seed else SEEDS
    prevalences = a.prevalence if a.prevalence else PREVALENCES
    root = Path(a.root)
    root.mkdir(exist_ok=True)

    rows = []
    for prev in prevalences:
        for seed in seeds:
            out_dir = root / f"pre_{prev:.2f}_seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== prevalence={prev} seed={seed} out={out_dir} ===")
            run_cmd([PY, "claimguard_sim.py", "--seed", str(seed), "--prevalence", str(prev), "--out", str(out_dir)])
            run_cmd([PY, "claimguard_prep.py", "--root", str(out_dir)])
            run_cmd([PY, "claimguard_features.py", "--root", str(out_dir), "--modes", "retro,asof"])
            run_cmd([PY, "claimguard_detect.py", "--root", str(out_dir), "--schemes", "faskes,waktu", "--modes", "retro,asof", "--seed", str(seed)])
            out = run_cmd([PY, "claimguard_calibrate.py", "--root", str(out_dir), "--schemes", "faskes,waktu", "--modes", "retro,asof", "--seed", str(seed)])
            vals = parse_fpr_block(out)
            if not vals:
                raise RuntimeError(f"Tidak ada output FPR untuk prev={prev} seed={seed}")
            for item in vals:
                rows.append({
                    "prevalence": prev,
                    "seed": seed,
                    "scheme": item["scheme"],
                    "mode": item["mode"],
                    "fpr_1": item["fpr_1"],
                    "fpr_3": item["fpr_3"],
                    "fpr_5": item["fpr_5"],
                    "fpr_10": item["fpr_10"],
                    "recall_5": item["recall_5"],
                })

    df = __import__("pandas").DataFrame(rows)
    summary = df.groupby(["scheme", "mode"]).agg(
        mean_fpr_1=("fpr_1", "mean"),
        std_fpr_1=("fpr_1", "std"),
        mean_fpr_3=("fpr_3", "mean"),
        std_fpr_3=("fpr_3", "std"),
        mean_fpr_5=("fpr_5", "mean"),
        std_fpr_5=("fpr_5", "std"),
        mean_fpr_10=("fpr_10", "mean"),
        std_fpr_10=("fpr_10", "std"),
        mean_recall_5=("recall_5", "mean"),
        std_recall_5=("recall_5", "std"),
    ).reset_index()

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    summary.to_csv(root / "robustness_summary.csv", index=False)

    print("\n=== PER RUN ===")
    print(df.to_string(index=False))
    df.to_csv(root / "robustness_detail.csv", index=False)


if __name__ == "__main__":
    main()
