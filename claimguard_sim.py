#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClaimGuard JKN - Simulator Data Klaim v1.1 (Tahap 1: data simulasi berbasis domain)

Prinsip desain
--------------
* TIDAK memakai data peserta JKN riil. Semua ID, tanggal, biaya, dan kode obat/alkes sintetis.
* Parameter biaya per kelompok kasus bersifat ILUSTRATIF (bukan tarif resmi). Kalibrasi
  dengan Permenkes tarif JKN dan statistik agregat publik sebelum dipakai di proposal.
* Klaim normal dibangkitkan dulu, baru anomali disuntik (4 modus x 3 tingkat kesulitan).
* Hard negatives: klaim wajar yang menyerupai anomali (kunjungan ulang sah, dialisis/kemo
  berulang, protokol baku yang memang mirip, kasus komplikasi yang sah mahal).
* Label HANYA untuk evaluasi, disimpan terpisah di folder eval_only/.
* Split disediakan per faskes dan per waktu (bukan acak).

Output (folder --out)
---------------------
data/claims.csv        data klaim mentah (input sistem)
data/facilities.csv    atribut publik faskes
data/splits.csv        split per faskes dan per waktu
eval_only/labels.csv   label anomali + hard negative (JANGAN dipakai sebagai fitur)
eval_only/facility_truth.csv  profil perilaku faskes yang disimulasikan
metadata.json          parameter, seed, ringkasan
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-12-31")
MODES = ["repeat_billing", "claim_cloning", "inflated_bills", "upcoding"]
MODE_SHARE = {"repeat_billing": 0.25, "claim_cloning": 0.25, "inflated_bills": 0.30, "upcoding": 0.20}
DIFFS = ["easy", "medium", "hard"]
DIFF_P = [0.35, 0.40, 0.25]


def D(n):
    return pd.Timedelta(days=int(n))


# ----------------------------------------------------------------------------
# Parameter domain (ILUSTRATIF)
# ----------------------------------------------------------------------------
SEV_COST = {1: 1.0, 2: 1.35, 3: 1.9}
SEV_LOS = {1: 1.0, 2: 1.3, 3: 1.8}
SEV_LABEL = {1: "I", 2: "II", 3: "III"}
FAC_COST = {"RS_A": 1.15, "RS_B": 1.05, "RS_C": 1.0, "RS_D": 0.92, "Klinik": 0.85}
FAC_SEV = {
    "RS_A": [0.45, 0.35, 0.20],
    "RS_B": [0.55, 0.31, 0.14],
    "RS_C": [0.65, 0.27, 0.08],
    "RS_D": [0.72, 0.22, 0.06],
    "Klinik": [1.0, 0.0, 0.0],
}
TYPE_P = {"RS_A": 0.07, "RS_B": 0.20, "RS_C": 0.30, "RS_D": 0.16, "Klinik": 0.27}
TYPE_VOL = {"RS_A": 2.0, "RS_B": 1.5, "RS_C": 1.0, "RS_D": 0.7, "Klinik": 0.8}
ALL_TYPES = ("RS_A", "RS_B", "RS_C", "RS_D", "Klinik")
RS_ALL = ("RS_A", "RS_B", "RS_C", "RS_D")
RS_ABC = ("RS_A", "RS_B", "RS_C")

COMMON_DX = ["I10", "E11.9", "E78.5", "K21.9", "D64.9", "N18.3", "J45.9"]
SEVERE_DX = ["A41.9", "J96.0", "N17.9", "E87.1", "D65"]
COMPLICATION_DX = ["T81.4", "T81.0", "I26.9"]
ADDON_PROCS = {  # kode ICD-9-CM: (nama, biaya IDR)
    "96.71": ("Ventilasi mekanik invasif", 6_000_000),
    "87.03": ("CT scan kepala", 1_800_000),
    "88.91": ("MRI otak", 3_500_000),
    "99.04": ("Transfusi PRC", 1_200_000),
    "88.72": ("Ekokardiografi", 900_000),
}
# (setting, bedah) -> porsi biaya [tindakan, obat_alkes, kamar, lain]
SHARES = {
    ("RJTL", False): (0.25, 0.45, 0.00, 0.30),
    ("RJTL", True): (0.55, 0.20, 0.00, 0.25),
    ("RITL", False): (0.25, 0.25, 0.35, 0.15),
    ("RITL", True): (0.40, 0.22, 0.25, 0.13),
}


def _g(gid, nama, setting, dx, proc, med, los, w, kind="acute", surgical=False,
       protocol=False, types=None, obst=False, n_items=None):
    if types is None:
        types = ALL_TYPES if setting == "RJTL" else RS_ALL
    if n_items is None:
        n_items = 3 if setting == "RJTL" else 5
    sigma = 0.05 if protocol else (0.15 if kind != "acute" else (0.22 if surgical else 0.28))
    gi = int(gid[1:])
    pool = [f"OBT-{gi:02d}{k:02d}" for k in range(8)]
    if surgical or setting == "RITL":
        pool = [f"ALK-{gi:02d}00", f"ALK-{gi:02d}01"] + pool
    return dict(gid=gid, nama=nama, setting=setting, dx=dx, proc=proc, med=med, los=los, w=w,
                kind=kind, surgical=surgical, protocol=protocol, types=types, obst=obst,
                n_items=n_items, sigma=sigma, pool=pool)


GROUPS = [
    _g("G01", "ISPA akut", "RJTL", "J06.9", [], 250_000, 0, 14),
    _g("G02", "Gastritis", "RJTL", "K29.7", [], 400_000, 0, 6),
    _g("G03", "Dermatitis", "RJTL", "L30.9", [], 300_000, 0, 5),
    _g("G04", "Nyeri punggung bawah", "RJTL", "M54.5", ["93.39"], 450_000, 0, 4),
    _g("G05", "Asma", "RJTL", "J45.9", [], 500_000, 0, 3),
    _g("G06", "Katarak (fakoemulsifikasi)", "RJTL", "H25.9", ["13.41"], 4_500_000, 0, 2.5,
       surgical=True, protocol=True, types=RS_ABC),
    _g("G07", "Hipertensi kontrol rutin", "RJTL", "I10", [], 350_000, 0, 0, kind="monthly"),
    _g("G08", "DM tipe 2 kontrol rutin", "RJTL", "E11.9", [], 600_000, 0, 0, kind="monthly"),
    _g("G09", "Hemodialisis", "RJTL", "N18.5", ["39.95"], 1_050_000, 0, 0, kind="dialysis",
       protocol=True, types=RS_ABC, n_items=3),
    _g("G10", "Kemoterapi payudara", "RJTL", "C50.9", ["99.25"], 6_000_000, 0, 0, kind="chemo",
       types=RS_ABC),
    _g("G11", "Diare / gastroenteritis", "RITL", "A09", [], 2_500_000, 3, 5),
    _g("G12", "Demam berdarah dengue", "RITL", "A91", [], 3_500_000, 4, 4),
    _g("G13", "Pneumonia", "RITL", "J18.9", [], 5_500_000, 5, 4),
    _g("G14", "Infeksi saluran kemih", "RITL", "N39.0", [], 3_000_000, 4, 2.5),
    _g("G15", "Gagal jantung", "RITL", "I50.9", ["88.72"], 8_000_000, 6, 2.5),
    _g("G16", "Stroke iskemik", "RITL", "I63.9", ["87.03"], 9_000_000, 7, 2),
    _g("G17", "Apendisitis akut", "RITL", "K35.8", ["47.09"], 7_000_000, 3, 2, surgical=True),
    _g("G18", "Persalinan normal", "RITL", "O80", ["73.59"], 3_200_000, 2, 6, protocol=True,
       obst=True, n_items=3),
    _g("G19", "Sectio caesarea", "RITL", "O82", ["74.1"], 8_500_000, 4, 3.5, surgical=True,
       obst=True),
    _g("G20", "Fraktur femur", "RITL", "S72.0", ["79.35"], 22_000_000, 6, 1, surgical=True,
       types=RS_ABC),
    _g("G21", "Kontrol pasca rawat", "RJTL", "Z09", [], 200_000, 0, 0, kind="followup"),
]
GBY = {g["gid"]: g for g in GROUPS}
POOL_W = {n: np.array([0.75 ** i for i in range(n)]) for n in (8, 10)}


class Simulator:
    def __init__(self, seed, n_fac, n_pat, prevalence, acute_rate=2.0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_fac = n_fac
        self.n_pat = n_pat
        self.prevalence = prevalence
        self.acute_rate = acute_rate
        self.claims = []
        self._uid = 0
        self._acute_cache = {}
        self.make_facilities()
        self.make_patients()

    # ------------------------------------------------------------------ util
    def uid(self):
        self._uid += 1
        return self._uid

    def pick_doctor(self, fac, avoid=None):
        docs, w = fac["doctors"], fac["doc_w"]
        for _ in range(10):
            d = docs[int(self.rng.choice(len(docs), p=w))]
            if d != avoid:
                return d
        return docs[0]

    # ------------------------------------------------------------ facilities
    def make_facilities(self):
        rng = self.rng
        n = self.n_fac
        types = list(rng.choice(list(TYPE_P), size=n, p=list(TYPE_P.values())))
        for i, t in enumerate(["RS_A", "RS_B", "RS_B", "RS_C", "RS_C", "RS_D", "Klinik", "Klinik"][:n]):
            types[i] = t
        self.fac = []
        for i, t in enumerate(types):
            n_doc = 3 + int(rng.poisson(6)) if t != "Klinik" else 2 + int(rng.poisson(1))
            dw = np.array([1.0 / (k + 1) ** 0.8 for k in range(n_doc)])
            self.fac.append(dict(
                idx=i, id=f"FSK-{i + 1:03d}", type=t,
                w=float(rng.lognormal(0, 0.6) * TYPE_VOL[t]),
                markup=float(rng.lognormal(0, 0.07)),
                lag_scale=float(rng.uniform(1.5, 4.0)),
                sev_p=list(rng.dirichlet(np.array(FAC_SEV[t]) * 60 + 1e-6)) if t != "Klinik" else [1, 0, 0],
                doctors=[f"DR-{i + 1:03d}-{k + 1:02d}" for k in range(n_doc)],
                doc_w=dw / dw.sum(),
                mode_mult={m: 1.0 for m in MODES}, risky=False, dominant=[],
            ))
        risky_n = max(3, int(round(0.18 * n)))
        for i in rng.choice(n, size=risky_n, replace=False):
            f = self.fac[int(i)]
            modes = MODES if f["type"] != "Klinik" else MODES[:3]
            k = int(rng.choice([1, 2], p=[0.6, 0.4]))
            f["dominant"] = [str(m) for m in rng.choice(modes, size=min(k, len(modes)), replace=False)]
            f["risky"] = True
            for m in f["dominant"]:
                f["mode_mult"][m] = 8.0
        w = np.array([f["w"] for f in self.fac])
        self.fac_w_all = w / w.sum()
        big = np.array([f["w"] if f["type"] in RS_ABC else 0.0 for f in self.fac])
        self.fac_w_big = big / big.sum()

    # -------------------------------------------------------------- patients
    def make_patients(self):
        rng = self.rng
        P = self.n_pat
        self.sex = np.where(rng.random(P) < 0.52, "P", "L")
        self.age = rng.choice(4, size=P, p=[0.22, 0.40, 0.23, 0.15])
        self.repro = (self.sex == "P") & (self.age == 1) & (rng.random(P) < 0.30)
        htn_p = np.array([0.0, 0.04, 0.22, 0.40])[self.age]
        dm_p = np.array([0.0, 0.02, 0.12, 0.20])[self.age]
        hd_p = np.array([0.0, 0.0, 0.008, 0.012])[self.age]
        ch_p = np.where((self.sex == "P") & (self.age >= 2), 0.015, 0.0)
        self.htn = rng.random(P) < htn_p
        self.dm = rng.random(P) < dm_p
        self.hd = rng.random(P) < hd_p
        self.chemo = (~self.hd) & (rng.random(P) < ch_p)
        self.home = np.zeros(P, dtype=int)
        for p in range(P):
            wts = self.fac_w_big if (self.hd[p] or self.chemo[p]) else self.fac_w_all
            self.home[p] = int(rng.choice(self.n_fac, p=wts))
        self.fac_patients = {i: [] for i in range(self.n_fac)}
        for p in range(P):
            self.fac_patients[int(self.home[p])].append(p)
        perm = rng.permutation(P)
        self.pid = [f"PSN-{perm[p] + 1:06d}" for p in range(P)]

    # --------------------------------------------------------- claim builder
    def draw_items(self, g):
        rng = self.rng
        pool = g["pool"]
        if g["protocol"]:
            core = list(pool[:3])
            if rng.random() < 0.25:
                core.append(pool[3])
            return core
        n = len(pool)
        k = int(np.clip(rng.poisson(g["n_items"]), 2, min(8, n)))
        w = POOL_W[n] / POOL_W[n].sum()
        idx = rng.choice(n, size=k, replace=False, p=w)
        return [pool[i] for i in sorted(idx)]

    def build_claim(self, fac, pi, g, masuk, doctor, hn="", sev=None):
        rng = self.rng
        inpatient = g["setting"] == "RITL"
        if sev is None:
            sev = int(rng.choice([1, 2, 3], p=fac["sev_p"])) if inpatient else 1
        sec = []
        if inpatient:
            n_sec = {1: min(int(rng.poisson(0.4)), 2),
                     2: 1 + min(int(rng.poisson(0.6)), 2),
                     3: 1 + min(int(rng.poisson(1.0)), 3)}[sev]
            sec = [str(x) for x in rng.choice(COMMON_DX, size=n_sec, replace=False)]
            if sev == 3 and rng.random() < 0.5:
                sec[0] = str(rng.choice(SEVERE_DX))
            sec = [s for s in dict.fromkeys(sec) if s != g["dx"]]
        los = 0
        if inpatient and g["los"] > 0:
            mean = g["los"] * SEV_LOS[sev]
            los = 1 + int(rng.poisson(max(mean - 1, 0.2)))
        base = g["med"] * SEV_COST[sev] * FAC_COST[fac["type"]] * fac["markup"]
        los_term = 1.0
        if inpatient and g["los"] > 0:
            los_term = 0.75 + 0.25 * (los / max(g["los"] * SEV_LOS[sev], 1))
        total = base * los_term * float(np.exp(rng.normal(0, g["sigma"])))
        if inpatient and not g["protocol"] and not hn and rng.random() < 0.03:
            los = int(max(los * rng.uniform(2.2, 3.0), los + 3))
            total *= rng.uniform(1.5, 2.4)
            sec.append(str(rng.choice(COMPLICATION_DX)))
            hn = "legit_high_cost"
        if not hn and g["protocol"]:
            hn = "legit_recurring" if g["kind"] == "dialysis" else "legit_similar"
        # v1.1: kode berat / tindakan atipikal yang SAH pada klaim normal (mis. kondisi sudah teratasi,
        # pemeriksaan penunjang atas indikasi). Tanpa ini, "dx berat" membocorkan upcoding.
        proc = list(g["proc"])
        addon_cost = 0.0
        if inpatient and not hn and not g["protocol"] and rng.random() < 0.09:
            hn = "legit_atypical_coding"
            if rng.random() < 0.55:
                sec.append(str(rng.choice(SEVERE_DX)))
            else:
                cand = [k for k in ("87.03", "88.91", "99.04", "88.72") if k not in proc]
                if cand:
                    a = str(rng.choice(cand))
                    proc.append(a)
                    addon_cost = float(np.round(ADDON_PROCS[a][1] * rng.uniform(0.9, 1.1) / 100.0) * 100.0)
            sec = list(dict.fromkeys(sec))
        sh = np.array(SHARES[(g["setting"], g["surgical"])], dtype=float)
        sh = sh * np.exp(rng.normal(0, 0.12, 4))
        sh = sh / sh.sum()
        comp = [float(np.round(total * s / 100.0) * 100.0) for s in sh]
        comp[0] += addon_cost
        pulang = masuk + D(los)
        lag = int(np.clip(rng.gamma(2.2, fac["lag_scale"]) + 1, 1, 45))
        return dict(uid=self.uid(), pi=pi, fac=fac["idx"], doctor=doctor, gid=g["gid"],
                    setting=g["setting"], sev=sev, dx=g["dx"], sec=sec, proc=proc,
                    items=self.draw_items(g), masuk=masuk, los=los, pulang=pulang,
                    klaim=pulang + D(lag), comp=comp, hn=hn, anom="", diff="", signal={},
                    related=None)

    def acute_groups(self, ftype, repro):
        key = (ftype, bool(repro))
        if key not in self._acute_cache:
            gs = [g for g in GROUPS if g["kind"] == "acute" and ftype in g["types"]
                  and (repro or not g["obst"])]
            w = np.array([g["w"] for g in gs], float)
            self._acute_cache[key] = (gs, w / w.sum())
        return self._acute_cache[key]

    def add(self, c):
        if c["klaim"] <= END and c["masuk"] >= START:
            self.claims.append(c)
            return True
        return False

    # ------------------------------------------------------- base generation
    def generate_base(self):
        rng = self.rng
        for p in range(self.n_pat):
            home = self.fac[int(self.home[p])]
            had_obst = False
            for _ in range(int(rng.poisson(self.acute_rate))):
                fac = home if rng.random() < 0.85 else self.fac[int(rng.integers(self.n_fac))]
                gs, pw = self.acute_groups(fac["type"], self.repro[p])
                g = gs[int(rng.choice(len(gs), p=pw))]
                if g["obst"]:
                    if had_obst:
                        continue
                    had_obst = True
                masuk = START + D(rng.integers(0, 365))
                if g["setting"] == "RJTL" and masuk.weekday() >= 5 and rng.random() < 0.6:
                    masuk += D(7 - masuk.weekday())
                c = self.build_claim(fac, p, g, masuk, self.pick_doctor(fac))
                if not self.add(c):
                    continue
                if c["setting"] == "RITL" and rng.random() < 0.12:
                    ctl = self.build_claim(fac, p, GBY["G21"], c["pulang"] + D(rng.integers(7, 22)),
                                           c["doctor"] if rng.random() < 0.7 else self.pick_doctor(fac),
                                           hn="legit_revisit")
                    self.add(ctl)
                elif c["setting"] == "RJTL" and not g["protocol"] and rng.random() < 0.04:
                    rv = self.build_claim(fac, p, g, c["masuk"] + D(rng.integers(7, 22)),
                                          c["doctor"], hn="legit_revisit")
                    f = float(rng.uniform(0.55, 0.85))
                    rv["comp"] = [float(np.round(x * f / 100.0) * 100.0) for x in rv["comp"]]
                    self.add(rv)
            # kronis bulanan
            for flag, gid in ((self.htn[p], "G07"), (self.dm[p], "G08")):
                if not flag:
                    continue
                g = GBY[gid]
                fac = home if g["types"] == ALL_TYPES or home["type"] in g["types"] else home
                doc = self.pick_doctor(fac)
                d = START + D(rng.integers(0, 36))
                while d <= END:
                    if rng.random() < 0.9:
                        self.add(self.build_claim(fac, p, g, d, doc if rng.random() < 0.7 else self.pick_doctor(fac)))
                    d += D(rng.integers(27, 36))
            # hemodialisis
            if self.hd[p]:
                g = GBY["G09"]
                pattern = [(0, 3), (1, 4), (2, 5)][int(rng.integers(3))]
                if rng.random() < 0.4:
                    pattern = (pattern[0], pattern[0] + 2, pattern[0] + 4)
                d = START + D(rng.integers(0, 150))
                stop = END if rng.random() > 0.1 else START + D(rng.integers(150, 365))
                doc = self.pick_doctor(home)
                while d <= stop:
                    if d.weekday() in pattern and rng.random() > 0.04:
                        self.add(self.build_claim(home, p, g, d, doc))
                    d += D(1)
            # kemoterapi
            if self.chemo[p]:
                g = GBY["G10"]
                d = START + D(rng.integers(0, 200))
                for _ in range(int(rng.integers(4, 7))):
                    if d > END:
                        break
                    self.add(self.build_claim(home, p, g, d, self.pick_doctor(home), hn="legit_recurring"))
                    d += D(21)

    # --------------------------------------------------------------- injeksi
    def _eligible(self, pred):
        return np.array([i for i, c in enumerate(self.claims) if pred(c)], dtype=int)

    def _pick(self, idx, mode, n):
        if len(idx) == 0 or n <= 0:
            return np.array([], dtype=int)
        w = np.array([self.fac[self.claims[i]["fac"]]["mode_mult"][mode] for i in idx], float)
        w /= w.sum()
        n = min(n, len(idx))
        return self.rng.choice(idx, size=n, replace=False, p=w)

    def _total(self, c):
        return float(sum(c["comp"]))

    def inflate(self, c, diff):
        rng = self.rng
        r = {"easy": rng.uniform(2.5, 4.0), "medium": rng.uniform(1.6, 2.2),
             "hard": rng.uniform(1.25, 1.5)}[diff]
        delta = (r - 1.0) * self._total(c)
        c["comp"][1] += float(np.round(0.7 * delta / 100.0) * 100.0)
        c["comp"][3] += float(np.round(0.3 * delta / 100.0) * 100.0)
        c["anom"], c["diff"], c["signal"] = "inflated_bills", diff, {"cost_ratio": round(float(r), 3)}

    def upcode(self, c, diff):
        rng = self.rng
        old = c["sev"]
        if diff == "easy":
            new = 3
            c["sec"] += [str(x) for x in rng.choice(SEVERE_DX, size=2, replace=False)]
            addons = ["96.71"]
        elif diff == "medium":
            new = min(old + 1, 3)
            cand = [k for k in ("87.03", "88.91", "99.04") if k not in c["proc"]]
            use_dx, use_proc = rng.random() < 0.6, rng.random() < 0.6
            if not (use_dx or use_proc):
                use_dx = True
            if use_dx:
                c["sec"].append(str(rng.choice(SEVERE_DX)))
            addons = [str(rng.choice(cand))] if (use_proc and cand) else []
        else:
            # hard (v1.1): hanya severity naik + biaya naik; tanpa dx berat atau tindakan atipikal tambahan
            new = min(old + 1, 3)
            addons = []
        c["sec"] = list(dict.fromkeys(c["sec"]))
        f = SEV_COST[new] / SEV_COST[old]
        c["comp"] = [float(np.round(x * f / 100.0) * 100.0) for x in c["comp"]]
        for a in addons:
            c["proc"].append(a)
            c["comp"][0] += float(np.round(ADDON_PROCS[a][1] * rng.uniform(0.9, 1.1) / 100.0) * 100.0)
        c["sev"] = new
        c["anom"], c["diff"] = "upcoding", diff
        c["signal"] = {"severity_from": SEV_LABEL[old], "severity_to": SEV_LABEL[new], "addon_proc": addons}

    def make_repeat(self, src, diff):
        rng = self.rng
        n = copy.deepcopy(src)
        fac = self.fac[n["fac"]]
        n["uid"] = self.uid()
        n["hn"], n["anom"], n["diff"], n["related"] = "", "repeat_billing", diff, src["uid"]
        if diff == "easy":
            n["klaim"] = src["klaim"] + D(rng.integers(1, 6))
            n["signal"] = {"date_gap_days": 0, "cost_ratio": 1.0}
        elif diff == "medium":
            shift = int(rng.integers(1, max(src["los"], 3) + 1)) if src["setting"] == "RITL" else int(rng.integers(1, 4))
            n["masuk"] = src["masuk"] + D(shift)
            n["pulang"] = n["masuk"] + D(n["los"])
            n["klaim"] = n["pulang"] + D(rng.integers(1, 10))
            f = rng.uniform(0.97, 1.03, 4)
            n["comp"] = [float(np.round(x * k / 100.0) * 100.0) for x, k in zip(src["comp"], f)]
            if n["sec"]:
                n["sec"] = n["sec"][:-1]
            n["signal"] = {"date_gap_days": shift, "cost_ratio": round(float(f.mean()), 3)}
        else:
            gap = int(rng.integers(4, 11))
            n["masuk"] = src["pulang"] + D(gap)
            n["los"] = max(1, int(round(src["los"] * 0.5))) if src["setting"] == "RITL" else 0
            n["pulang"] = n["masuk"] + D(n["los"])
            n["klaim"] = n["pulang"] + D(rng.integers(1, 10))
            f = float(rng.uniform(0.70, 0.95))
            n["comp"] = [float(np.round(x * f / 100.0) * 100.0) for x in src["comp"]]
            n["doctor"] = self.pick_doctor(fac, avoid=src["doctor"])
            if len(n["items"]) > 2:
                n["items"] = n["items"][:-1]
            n["signal"] = {"date_gap_days": gap, "cost_ratio": round(f, 3)}
        return n if (n["klaim"] <= END and n["masuk"] <= END) else None

    def _shift_date(self, d, off):
        nd = d + D(off)
        if nd < START or nd > END - D(60):
            nd = d - D(off)
        if nd < START or nd > END - D(60):
            nd = d
        return nd

    def make_clone(self, src, diff):
        rng = self.rng
        n = copy.deepcopy(src)
        fac = self.fac[n["fac"]]
        g = GBY[src["gid"]]
        cands = self.fac_patients[n["fac"]]
        pi = None
        for _ in range(30):
            q = int(cands[int(rng.integers(len(cands)))])
            if q != src["pi"] and (not g["obst"] or self.repro[q]):
                pi = q
                break
        if pi is None:
            return None
        n["uid"] = self.uid()
        n["pi"] = pi
        n["hn"], n["anom"], n["diff"], n["related"] = "", "claim_cloning", diff, src["uid"]
        pool = [x for x in g["pool"] if x not in src["items"]]

        def swap(k):
            its = list(n["items"])
            for _ in range(min(k, len(its) - 1, len(pool))):
                j = int(rng.integers(len(its)))
                its[j] = pool[int(rng.integers(len(pool)))]
            return list(dict.fromkeys(its))

        if diff == "easy":
            off, jit = int(rng.integers(0, 15)), 0.003
        elif diff == "medium":
            off, jit = int(rng.integers(-45, 46)), 0.03
            n["items"] = swap(1)
        else:
            off, jit = int(rng.integers(-150, 151)), 0.07
            n["items"] = swap(2)
            n["doctor"] = self.pick_doctor(fac, avoid=src["doctor"])
            if n["sec"]:
                n["sec"] = n["sec"][:-1]
            if n["los"] > 1:
                n["los"] += int(rng.choice([-1, 0, 1]))
        n["masuk"] = self._shift_date(src["masuk"], off)
        n["pulang"] = n["masuk"] + D(n["los"])
        n["klaim"] = n["pulang"] + D(rng.integers(1, 15))
        f = 1 + rng.uniform(-jit, jit, 4)
        n["comp"] = [float(np.round(x * k / 100.0) * 100.0) for x, k in zip(src["comp"], f)]
        n["signal"] = {"cost_ratio": round(float(f.mean()), 3), "item_overlap": "see related claim"}
        return n

    def inject(self):
        rng = self.rng
        N = len(self.claims)
        total = int(round(self.prevalence * N / (1 - self.prevalence)))
        n_mode = {m: int(round(total * MODE_SHARE[m])) for m in MODES}
        used = set()
        clean = lambda c: c["hn"] in ("", "legit_similar")

        # 1) modifikasi di tempat
        idx = self._eligible(lambda c: clean(c) and c["setting"] == "RITL" and not GBY[c["gid"]]["protocol"] and c["sev"] < 3)
        for i in self._pick(idx, "upcoding", n_mode["upcoding"]):
            self.upcode(self.claims[int(i)], str(rng.choice(DIFFS, p=DIFF_P)))
            used.add(int(i))
        idx = self._eligible(lambda c: clean(c) and sum(c["comp"]) >= 200_000 and c["anom"] == "")
        idx = np.array([i for i in idx if int(i) not in used], dtype=int)
        for i in self._pick(idx, "inflated_bills", n_mode["inflated_bills"]):
            self.inflate(self.claims[int(i)], str(rng.choice(DIFFS, p=DIFF_P)))
            used.add(int(i))

        # 2) klaim baru (salinan) dari sumber yang bersih
        new = []
        idx = self._eligible(lambda c: clean(c) and c["anom"] == "")
        idx = np.array([i for i in idx if int(i) not in used], dtype=int)
        picks = self._pick(idx, "repeat_billing", n_mode["repeat_billing"])
        for i in picks:
            used.add(int(i))
            r = self.make_repeat(self.claims[int(i)], str(rng.choice(DIFFS, p=DIFF_P)))
            if r:
                new.append(r)
        idx = self._eligible(lambda c: clean(c) and c["anom"] == "" and sum(c["comp"]) >= 1_000_000)
        idx = np.array([i for i in idx if int(i) not in used], dtype=int)
        for i in self._pick(idx, "claim_cloning", n_mode["claim_cloning"]):
            used.add(int(i))
            r = self.make_clone(self.claims[int(i)], str(rng.choice(DIFFS, p=DIFF_P)))
            if r:
                new.append(r)
        self.claims.extend(new)

    # -------------------------------------------------------------- dataframe
    def to_frames(self):
        rng = self.rng
        rows = []
        for c in self.claims:
            fac = self.fac[c["fac"]]
            comp = c["comp"]
            rows.append(dict(
                _uid=c["uid"], _related=c["related"],
                patient_id=self.pid[c["pi"]],
                jenis_kelamin=self.sex[c["pi"]],
                kelompok_umur=["0-17", "18-44", "45-59", "60+"][int(self.age[c["pi"]])],
                facility_id=fac["id"], jenis_faskes=fac["type"], doctor_id=c["doctor"],
                jenis_layanan=c["setting"], diagnosis_utama=c["dx"],
                diagnosis_sekunder="|".join(c["sec"]), tindakan="|".join(c["proc"]),
                obat_alkes="|".join(c["items"]),
                severity_inacbg=SEV_LABEL[c["sev"]] if c["setting"] == "RITL" else "0",
                tanggal_masuk=c["masuk"], tanggal_pulang=c["pulang"], tanggal_klaim=c["klaim"],
                lama_rawat=c["los"], biaya_tindakan=comp[0], biaya_obat_alkes=comp[1],
                biaya_kamar=comp[2], biaya_lain=comp[3], nilai_klaim=float(sum(comp)),
                _anom=c["anom"], _diff=c["diff"], _hn=c["hn"],
                _signal=json.dumps(c["signal"], ensure_ascii=False), _gid=c["gid"],
            ))
        df = pd.DataFrame(rows)
        n = len(df)
        # claim_id dibuat SETELAH diurutkan menurut tanggal agar urutan ID tidak membocorkan label
        df["_r"] = rng.random(n)
        df = df.sort_values(["tanggal_klaim", "_r"]).reset_index(drop=True)
        df["claim_id"] = [f"CLM-{i + 1:07d}" for i in range(n)]
        uid2id = dict(zip(df["_uid"], df["claim_id"]))
        df["_related_id"] = df["_related"].map(lambda u: uid2id.get(u, "") if pd.notna(u) else "")

        # noise (merata di semua baris, termasuk anomali)
        m = rng.random(n) < 0.012
        df.loc[m, "doctor_id"] = ""
        m = rng.random(n) < 0.008
        df.loc[m, "obat_alkes"] = ""

        def mangle(s):
            if not s:
                return s
            parts = s.split("|")
            k = int(rng.integers(len(parts)))
            p = parts[k]
            op = int(rng.integers(4))
            if op == 0:
                p = p.lower()
            elif op == 1:
                p = p.replace(".", "")
            elif op == 2:
                p = p + " "
            else:
                p = p.replace(".", ",")
            parts[k] = p
            return "|".join(parts)

        for col in ("diagnosis_utama", "tindakan"):
            mm = np.where(rng.random(n) < 0.006)[0]
            for i in mm:
                df.at[i, col] = mangle(df.at[i, col])

        # atribut riwayat (hanya dari klaim sebelumnya milik pasien yang sama)
        df["_d"] = (df["tanggal_masuk"] - START).dt.days
        df = df.sort_values(["patient_id", "tanggal_masuk", "tanggal_klaim", "claim_id"]).reset_index(drop=True)
        df["prev_claim_id"] = df.groupby("patient_id")["claim_id"].shift(1).fillna("")
        prev_pulang = df.groupby("patient_id")["tanggal_pulang"].shift(1)
        df["days_since_prev_claim"] = (df["tanggal_masuk"] - prev_pulang).dt.days

        def freq30(s):
            d = s.to_numpy()
            return pd.Series(np.searchsorted(d, d, side="right") - np.searchsorted(d, d - 30, side="left") - 1,
                             index=s.index)

        df["service_freq_30d"] = df.groupby("patient_id")["_d"].transform(freq30)
        df = df.sort_values("claim_id").reset_index(drop=True)

        fac_df = pd.DataFrame([dict(facility_id=f["id"], jenis_faskes=f["type"]) for f in self.fac])
        truth = pd.DataFrame([dict(facility_id=f["id"], jenis_faskes=f["type"], risky=int(f["risky"]),
                                   dominant_modes="|".join(f["dominant"]),
                                   cost_markup=round(f["markup"], 4)) for f in self.fac])
        # split per faskes (stratifikasi risky) dan per waktu
        pattern = ["train", "train", "train", "val", "test"]
        fsplit = {}
        order = self.rng.permutation(self.n_fac)
        for grp in (True, False):
            ids = [int(i) for i in order if self.fac[int(i)]["risky"] == grp]
            for k, i in enumerate(ids):
                fsplit[self.fac[i]["id"]] = pattern[k % 5]
        splits = pd.DataFrame(dict(
            claim_id=df["claim_id"],
            split_faskes=df["facility_id"].map(fsplit),
            split_waktu=np.where(df["tanggal_klaim"] <= pd.Timestamp("2025-08-31"), "train",
                         np.where(df["tanggal_klaim"] <= pd.Timestamp("2025-10-31"), "val", "test")),
        ))
        labels = pd.DataFrame(dict(
            claim_id=df["claim_id"],
            is_anomaly=(df["_anom"] != "").astype(int),
            anomaly_type=df["_anom"].replace("", "none"),
            difficulty=df["_diff"].replace("", "none"),
            hard_negative_type=df["_hn"].replace("", "none"),
            related_claim_id=df["_related_id"],
            injected_signal=df["_signal"].where(df["_anom"] != "", ""),
            case_group=df["_gid"],
        ))
        claims = df[["claim_id", "patient_id", "jenis_kelamin", "kelompok_umur", "facility_id",
                     "jenis_faskes", "doctor_id", "jenis_layanan", "diagnosis_utama",
                     "diagnosis_sekunder", "tindakan", "obat_alkes", "severity_inacbg",
                     "tanggal_masuk", "tanggal_pulang", "tanggal_klaim", "lama_rawat",
                     "biaya_tindakan", "biaya_obat_alkes", "biaya_kamar", "biaya_lain",
                     "nilai_klaim", "prev_claim_id", "days_since_prev_claim",
                     "service_freq_30d"]].copy()
        for c in ("tanggal_masuk", "tanggal_pulang", "tanggal_klaim"):
            claims[c] = claims[c].dt.strftime("%Y-%m-%d")
        for c in ("biaya_tindakan", "biaya_obat_alkes", "biaya_kamar", "biaya_lain", "nilai_klaim"):
            claims[c] = claims[c].astype(np.int64)
        return claims, fac_df, splits, labels, truth


def run_checks(claims, labels, splits, prevalence):
    res = {}
    res["claim_id_unik"] = bool(claims["claim_id"].is_unique)
    comp = claims[["biaya_tindakan", "biaya_obat_alkes", "biaya_kamar", "biaya_lain"]].sum(axis=1)
    res["nilai_klaim_eq_komponen"] = bool((comp == claims["nilai_klaim"]).all())
    a, b, c = (pd.to_datetime(claims[k]) for k in ("tanggal_masuk", "tanggal_pulang", "tanggal_klaim"))
    res["urutan_tanggal_valid"] = bool(((a <= b) & (b <= c)).all())
    res["tidak_ada_biaya_negatif"] = bool((claims[["biaya_tindakan", "biaya_obat_alkes", "biaya_kamar", "biaya_lain"]] >= 0).all().all())
    prev = float(labels["is_anomaly"].mean())
    res["prevalensi_aktual"] = round(prev, 4)
    res["prevalensi_dekat_target"] = bool(abs(prev - prevalence) < 0.01)
    rank = np.arange(len(labels))
    r = float(np.corrcoef(rank, labels["is_anomaly"].to_numpy())[0, 1])
    res["korelasi_urutan_id_vs_label"] = round(r, 4)
    res["id_tidak_bocor"] = bool(abs(r) < 0.03)
    res["ada_anomali_di_tiap_split_faskes"] = bool(
        labels.merge(splits, on="claim_id").groupby("split_faskes")["is_anomaly"].sum().gt(0).all())
    return res


def main():
    ap = argparse.ArgumentParser(description="Simulator data klaim ClaimGuard JKN")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--facilities", type=int, default=60)
    ap.add_argument("--patients", type=int, default=12000)
    ap.add_argument("--prevalence", type=float, default=0.03, help="porsi klaim anomali (0.01-0.05)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    sim = Simulator(a.seed, a.facilities, a.patients, a.prevalence)
    sim.generate_base()
    n_base = len(sim.claims)
    sim.inject()
    claims, fac_df, splits, labels, truth = sim.to_frames()

    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "eval_only"), exist_ok=True)
    claims.to_csv(os.path.join(a.out, "data", "claims.csv"), index=False)
    fac_df.to_csv(os.path.join(a.out, "data", "facilities.csv"), index=False)
    splits.to_csv(os.path.join(a.out, "data", "splits.csv"), index=False)
    labels.to_csv(os.path.join(a.out, "eval_only", "labels.csv"), index=False)
    truth.to_csv(os.path.join(a.out, "eval_only", "facility_truth.csv"), index=False)

    checks = run_checks(claims, labels, splits, a.prevalence)
    summ = {
        "n_klaim": int(len(claims)), "n_klaim_dasar": int(n_base),
        "n_anomali": int(labels["is_anomaly"].sum()),
        "anomali_per_modus_kesulitan": labels[labels.is_anomaly == 1].groupby(["anomaly_type", "difficulty"]).size().unstack(fill_value=0).to_dict("index"),
        "hard_negative": labels["hard_negative_type"].value_counts().to_dict(),
        "checks": checks,
    }
    meta = {"seed": a.seed, "facilities": a.facilities, "patients": a.patients,
            "prevalence_target": a.prevalence, "window": [str(START.date()), str(END.date())],
            "catatan": "Parameter biaya ilustratif, bukan tarif resmi. Data sintetis penuh.",
            "ringkasan": summ}
    with open(os.path.join(a.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=int)
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=int))


if __name__ == "__main__":
    main()
