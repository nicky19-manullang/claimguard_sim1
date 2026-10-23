#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-route ClaimGuard demo dashboard with an evidence-based claims queue."""

import base64
import io
from pathlib import Path
from numbers import Real

import pandas as pd
import streamlit as st
from PIL import Image, ImageChops
from streamlit_option_menu import option_menu

from claimguard_review_evidence_v2 import build_claim_comparison
from claimguard_review_assistant import build_review_assistant_content
from claimguard_workflow_demo import (
    EVIDENCE_ITEMS,
    RECOMMENDATION_FEEDBACK,
    REVIEWER_ROLES,
    WORKFLOW_ACTIONS,
    get_workflow_status,
    read_workflow_log,
    record_workflow_action,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OLD_CLAIM_COLUMNS = [
    "claim_id",
    "facility_id",
    "jenis_layanan",
    "patient_id",
    "tanggal_klaim",
    "diagnosis_utama",
    "diagnosis_sekunder",
    "tindakan",
    "obat_alkes",
    "nilai_klaim",
    "lama_rawat",
]
OLD_DETECTOR_COLUMNS = ["p_repeat", "p_clone", "p_infl", "p_upc", "p_ens"]
OLD_DETECTOR_LABELS = {
    "p_repeat": "Klaim berulang",
    "p_clone": "Kemiripan antar-klaim",
    "p_infl": "Biaya di atas pola",
    "p_upc": "Tingkat keparahan dan lama rawat",
    "p_ens": "Pola gabungan",
}
OLD_DRIVER_TEXT = {
    "repeat_billing": "Pola klaim berulang",
    "claim_cloning": "Kemiripan antar-klaim",
    "inflated_bills": "Biaya di atas pola",
    "upcoding": "Tingkat keparahan dan lama rawat",
    "ensemble": "Pola gabungan",
}
TIER_LABELS = {"Critical": "Sangat tinggi", "High": "Tinggi", "Medium": "Menengah", "Low": "Rendah"}
V2_QUEUE_FILES = {
    "time": ("split_time", "review_queue_v2_time.csv.gz"),
    "facility": ("split_facility", "review_queue_v2_facility.csv.gz"),
}
V2_FEATURE_FILES = {"time": "features_v2_time.csv.gz", "facility": "features_v2_facility.csv.gz"}
V2_SPLIT_OPTIONS = {
    "Urut berdasarkan waktu pengajuan": ("time", "split_time"),
    "Kelompokkan berdasarkan fasilitas": ("facility", "split_facility"),
}
V2_PARTITIONS = {
    "Data uji": "test",
    "Data acuan": "train",
    "Data validasi": "val",
}
V2_COST_LABELS = {
    "cost_akomodasi_idr": "Akomodasi",
    "cost_alkes_idr": "Alat kesehatan",
    "cost_konsultasi_idr": "Konsultasi",
    "cost_obat_idr": "Obat",
    "cost_penunjang_idr": "Penunjang",
    "cost_tindakan_idr": "Tindakan",
}
V2_SIGNAL_LABELS = {
    "repeat_rank_percentile": "Pola klaim berulang",
    "cloning_rank_percentile": "Kemiripan dengan klaim lain",
    "inflated_cost_rank_percentile": "Perbandingan biaya",
    "severity_los_unusual_rank_percentile": "Tingkat keparahan dan lama rawat",
}
V2_SERVICE_LABELS = {"RJTL": "Rawat jalan (RJTL)", "RITL": "Rawat inap (RITL)"}
WORKFLOW_LOG = OUT / "workflow_records_demo_v2.csv"
LEGACY_WORKFLOW_LOG = OUT / "workflow_records_demo.csv"
DEFAULT_SCHEME_LABEL = "Urut berdasarkan waktu pengajuan"
DEFAULT_PARTITION_LABEL = "Data uji"


@st.cache_data
def _load_csv(path: str, **kwargs) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File yang diperlukan tidak ditemukan: {file_path}")
    return pd.read_csv(file_path, **kwargs)


def _required_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} tidak memiliki kolom wajib: {', '.join(missing)}")


def _load_v2_review_data(scheme: str):
    prep = OUT / "prep_v2"
    queue_column, queue_file = V2_QUEUE_FILES[scheme]
    queue = _load_csv(str(prep / queue_file))
    claims = _load_csv(str(prep / "claims_v2_flat_with_splits.csv.gz"), keep_default_na=False)
    features = _load_csv(str(prep / V2_FEATURE_FILES[scheme]), keep_default_na=False)
    _required_columns(
        queue,
        [
            "id_klaim", queue_column, "priority_rank", "submitted_at", "service_type",
            "facility_id", "priority_score", "top_driver", "review_recommended", "evidence",
            "interpretation", *V2_SIGNAL_LABELS,
        ],
        "Antrean V2",
    )
    if queue["id_klaim"].duplicated().any() or claims["id_klaim"].duplicated().any():
        raise ValueError("ID klaim duplikat pada tabel V2.")
    return queue, claims, features, queue_column, V2_SPLIT_OPTIONS[
        "Urut berdasarkan waktu pengajuan" if scheme == "time"
        else "Kelompokkan berdasarkan fasilitas"
    ][1]


def _workflow_log() -> pd.DataFrame:
    return read_workflow_log(WORKFLOW_LOG, LEGACY_WORKFLOW_LOG)


def _format_number(value) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    if isinstance(value, Real):
        return format(value, ",.0f").replace(",", ".")
    return str(value)


def _format_money(value) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    return f"Rp {_format_number(value)}"


def _service_name(value) -> str:
    return V2_SERVICE_LABELS.get(str(value), str(value))


def _load_cropped_logo(path: Path) -> Image.Image:
    source = Image.open(path).convert("RGBA")
    alpha = source.getchannel("A")
    if alpha.getextrema()[0] < 250:
        # PNG transparan: pangkas berdasarkan alpha dan pertahankan transparansinya.
        image = source
        content = alpha.point(lambda value: 255 if value > 8 else 0).getbbox()
    else:
        # Gambar berlatar putih: pangkas bagian putih di sekelilingnya.
        image = source.convert("RGB")
        background = Image.new("RGB", image.size, (255, 255, 255))
        difference = ImageChops.difference(image, background).convert("L")
        content = difference.point(lambda value: 255 if value > 32 else 0).getbbox()
    if content is None:
        raise ValueError(f"Logo tidak memiliki konten yang dapat ditampilkan: {path}")
    left, top, right, bottom = content
    padding_x = max(4, (right - left) // 30)
    padding_y = max(4, (bottom - top) // 12)
    crop_box = (
        max(0, left - padding_x),
        max(0, top - padding_y),
        min(image.width, right + padding_x),
        min(image.height, bottom + padding_y),
    )
    return image.crop(crop_box)


@st.cache_data
def _logo_data_uri(path: str) -> str:
    """Logo yang sudah dipangkas, dikodekan base64 agar bisa disisipkan ke HTML."""
    buffer = io.BytesIO()
    _load_cropped_logo(Path(path)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _logo_html(path: Path, alt: str, fallback: str) -> str:
    if path.is_file():
        try:
            return f'<img src="{_logo_data_uri(str(path))}" alt="{alt}">'
        except Exception:
            pass
    return f'<span class="cg-logo-fallback">{fallback}</span>'


def _brand_html(path: Path) -> str:
    """Logo ClaimGuard di kiri atas; kalau file tidak ada, pakai perisai + teks."""
    if path.is_file():
        try:
            return (
                '<div class="cg-brand"><img class="cg-brand-logo" '
                f'src="{_logo_data_uri(str(path))}" alt="ClaimGuard"></div>'
            )
        except Exception:
            pass
    return f'<div class="cg-brand">{SHIELD_SVG}<div class="cg-wordmark">CLAIM<span>GUARD</span></div></div>'


def _hero_art_html(path: Path) -> str:
    """Logo besar di tengah lingkaran konsentris; fallback ke ilustrasi perisai lama."""
    if path.is_file():
        try:
            return (
                f'<div class="cg-hero-art">{HERO_RINGS_SVG}<div class="cg-hero-glow"></div>'
                f'<img class="cg-hero-logo" src="{_logo_data_uri(str(path))}" alt="ClaimGuard"></div>'
            )
        except Exception:
            pass
    return f'<div class="cg-hero-art">{HERO_ART_SVG}</div>'


def _render_header() -> None:
    st.title("ClaimGuard | Pemeriksaan Klaim")
    st.caption(
        "Satu antrean untuk melihat klaim yang perlu diperiksa, memahami alasannya, "
        "dan mencatat tindak lanjut."
    )
    st.warning(
        "Ini adalah demonstrasi dengan data buatan. Hasilnya hanya membantu menentukan urutan "
        "pemeriksaan, bukan bukti kecurangan atau keputusan klaim. Kemampuan pada data nyata "
        "belum diuji."
    )


def _open_analysis() -> None:
    st.session_state["nav_menu"] = "Pemeriksaan klaim"


def _go_home() -> None:
    st.session_state["nav_menu"] = "Beranda"


def _render_unavailable_page(page: str) -> None:
    st.title(page)
    st.info(
        f"Menu {page.casefold()} belum tersedia pada prototipe ini. "
        "Yang dapat digunakan saat ini adalah halaman beranda dan pemeriksaan klaim "
        "dengan data simulasi."
    )


LANDING_CSS = """
<style>
.stApp {
    background:
        radial-gradient(1000px 520px at 95% -8%, rgba(22,118,106,.10), transparent 62%),
        radial-gradient(800px 420px at -5% 35%, rgba(47,109,181,.06), transparent 60%),
        #f7f9f9;
}
@keyframes cgfade { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
.cg-serif { font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; }
.cg-top { display:flex; align-items:center; justify-content:space-between; padding:.2rem 0 1.6rem; }
.cg-brand { display:flex; align-items:center; gap:.65rem; }
.cg-brand svg { width:30px; height:30px; flex:none; }
.cg-brand-logo { height:66px; width:auto; max-width:240px; object-fit:contain; display:block; }
.cg-wordmark { font-size:1.2rem; font-weight:800; letter-spacing:.06em; color:#12303f; }
.cg-wordmark span { color:#16766a; }
.cg-partners { display:flex; align-items:center; gap:1.8rem; }
.cg-partners img { height:38px; max-width:170px; object-fit:contain; mix-blend-mode:multiply; opacity:.92; }
.cg-logo-fallback { color:#4a606b; font-size:.95rem; font-weight:700; }

.cg-hero2 { position:relative; overflow:hidden; border-radius:28px; padding:4.6rem 3.4rem 4.2rem;
    background:linear-gradient(115deg,#e3f1ee 0%,#ecf3f6 52%,#f2f5fa 100%);
    animation:cgfade .7s ease both; }
.cg-hero-art { position:absolute; right:-10px; top:50%; transform:translateY(-50%);
    width:470px; height:470px; pointer-events:none; }
.cg-hero-glow { position:absolute; left:53.2%; top:50%; width:340px; height:340px;
    transform:translate(-50%,-50%); border-radius:50%;
    background:radial-gradient(circle, rgba(255,255,255,.92) 0%, rgba(255,255,255,.55) 45%, rgba(255,255,255,0) 72%); }
.cg-hero-logo { position:absolute; left:53.2%; top:50%; transform:translate(-50%,-50%);
    height:270px; width:auto; max-width:250px; object-fit:contain;
    filter:drop-shadow(0 16px 26px rgba(18,48,63,.18)); animation:cgfloat 6s ease-in-out infinite; }
@keyframes cgfloat { 0%,100% { transform:translate(-50%,-50%); } 50% { transform:translate(-50%,calc(-50% - 8px)); } }
.cg-hero-copy { position:relative; max-width:610px; }
.cg-tag { color:#16766a; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.cg-hero-title { margin:1rem 0 1.2rem; font-size:3.4rem; line-height:1.08; font-weight:600;
    color:#10303e; letter-spacing:-.018em; padding:0; }
.cg-hero-title em { font-style:italic; color:#116b60;
    background:linear-gradient(transparent 64%, rgba(22,118,106,.22) 64%); padding:0 .06em; }
.cg-lead { max-width:540px; color:#46606c; font-size:1.12rem; line-height:1.8; margin:0; }
.cg-meta { display:flex; flex-wrap:wrap; align-items:center; gap:.9rem; margin-top:1.8rem;
    color:#5b727d; font-size:.88rem; font-weight:600; }
.cg-meta i { width:4px; height:4px; border-radius:50%; background:#9bb0b8; display:inline-block; }

.cg-kicker { color:#16766a; font-size:.74rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; margin:4rem 0 .45rem; }
.cg-h2 { font-size:2rem; font-weight:600; color:#10303e; line-height:1.2; margin:0 0 .55rem; letter-spacing:-.01em; }
.cg-sub { color:#5a6f7a; max-width:640px; line-height:1.75; margin:0 0 1.6rem; font-size:1rem; }

.cg-feat { animation:cgfade .7s ease both; }
.cg-feat-num { color:#16766a; font-weight:700; font-size:.85rem; letter-spacing:.06em; }
.cg-feat h3 { margin:.4rem 0 .45rem; color:#10303e; font-size:1.14rem; font-weight:700; padding:0; }
.cg-feat p { margin:0; color:#5a6f7a; line-height:1.75; font-size:.96rem; }

.cg-path { border-top:3px solid #16766a; padding-top:1.2rem; margin-bottom:1.1rem; animation:cgfade .7s ease both; }
.cg-path.v2 { border-top-color:#2f6db5; }
.cg-path-label { color:#16766a; font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.cg-path.v2 .cg-path-label { color:#2f6db5; }
.cg-path h3 { margin:.45rem 0 .55rem; font-size:1.55rem; color:#10303e; font-weight:600; padding:0;
    font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; }
.cg-path p { margin:0 0 .75rem; min-height:3.5rem; color:#46606c; line-height:1.75; font-size:.98rem; }
.cg-path ul { margin:0 0 .5rem; padding-left:1.1rem; color:#5a6f7a; line-height:1.85; font-size:.93rem; }

.cg-step { border-top:1px solid #d3dee1; padding-top:.95rem; animation:cgfade .7s ease both; }
.cg-step b { color:#16766a; font-size:.85rem; letter-spacing:.06em; }
.cg-step h4 { margin:.3rem 0 .3rem; color:#10303e; font-size:1.03rem; font-weight:700; padding:0; }
.cg-step p { margin:0; color:#5a6f7a; font-size:.91rem; line-height:1.7; }

.cg-notice { margin:3.4rem 0 .5rem; padding-top:1.1rem; border-top:1px solid #d3dee1;
    color:#6b7f89; font-size:.85rem; line-height:1.7; max-width:900px; }

.stButton > button[kind="primary"] { background:linear-gradient(135deg,#14665c,#12594f); border:none;
    border-radius:10px; font-weight:600; padding:.7rem 1.1rem; box-shadow:0 6px 16px rgba(18,89,79,.22);
    transition:transform .15s ease, box-shadow .15s ease; }
.stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#12594f,#0e4a42);
    transform:translateY(-1px); box-shadow:0 10px 22px rgba(18,89,79,.28); }

@media (max-width: 900px) {
    .cg-top { flex-direction:column; align-items:flex-start; gap:1rem; }
    .cg-hero2 { padding:2.6rem 1.6rem; border-radius:22px; }
    .cg-hero-art { display:none; }
    .cg-hero-title { font-size:2.3rem; }
}
</style>
"""

SHIELD_SVG = (
    '<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="cgg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#16766a"/><stop offset="1" stop-color="#2f6db5"/>'
    '</linearGradient></defs>'
    '<path d="M20 2.5 34 7.8v10.6c0 9.1-5.8 15.6-14 19.1C11.8 34 6 27.5 6 18.4V7.8z" '
    'fill="url(#cgg)"/>'
    '<path d="m13.5 20.2 4.7 4.7 8.6-9.3" fill="none" stroke="#fff" stroke-width="3" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

HERO_ART_SVG = (
    '<svg viewBox="0 0 470 470" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="cga" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#16766a"/><stop offset="1" stop-color="#2f6db5"/>'
    '</linearGradient></defs>'
    '<g fill="none" stroke="#16766a">'
    '<circle cx="250" cy="235" r="90" stroke-opacity=".22"/>'
    '<circle cx="250" cy="235" r="145" stroke-opacity=".16"/>'
    '<circle cx="250" cy="235" r="200" stroke-opacity=".11"/>'
    '<circle cx="250" cy="235" r="255" stroke-opacity=".07"/>'
    '</g>'
    '<g transform="translate(150 125) scale(5.5)">'
    '<path d="M20 2.5 34 7.8v10.6c0 9.1-5.8 15.6-14 19.1C11.8 34 6 27.5 6 18.4V7.8z" '
    'fill="url(#cga)" fill-opacity=".10" stroke="url(#cga)" stroke-width=".35" stroke-opacity=".7"/>'
    '<path d="m13.5 20.2 4.7 4.7 8.6-9.3" fill="none" stroke="url(#cga)" stroke-width=".7" '
    'stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".8"/></g>'
    '<g fill="#16766a">'
    '<circle cx="340" cy="95" r="4" fill-opacity=".45"/>'
    '<circle cx="95" cy="300" r="3" fill-opacity=".35"/>'
    '<circle cx="395" cy="330" r="5" fill="#2f6db5" fill-opacity=".35"/>'
    '<circle cx="175" cy="70" r="3.5" fill="#2f6db5" fill-opacity=".4"/>'
    '</g></svg>'
)


HERO_RINGS_SVG = (
    '<svg viewBox="0 0 470 470" xmlns="http://www.w3.org/2000/svg">'
    '<g fill="none" stroke="#16766a">'
    '<circle cx="250" cy="235" r="130" stroke-opacity=".20"/>'
    '<circle cx="250" cy="235" r="175" stroke-opacity=".15"/>'
    '<circle cx="250" cy="235" r="220" stroke-opacity=".10"/>'
    '<circle cx="250" cy="235" r="265" stroke-opacity=".06"/>'
    '</g>'
    '<g fill="#16766a">'
    '<circle cx="355" cy="90" r="4" fill-opacity=".45"/>'
    '<circle cx="90" cy="320" r="3" fill-opacity=".35"/>'
    '<circle cx="410" cy="335" r="5" fill="#2f6db5" fill-opacity=".35"/>'
    '<circle cx="165" cy="62" r="3.5" fill="#2f6db5" fill-opacity=".4"/>'
    '</g></svg>'
)


def _render_landing_page() -> None:
    st.markdown(LANDING_CSS, unsafe_allow_html=True)

    bpjs = _logo_html(ROOT / "asset" / "logo_bpjs.png", "BPJS Kesehatan", "BPJS Kesehatan")
    jkn = _logo_html(ROOT / "asset" / "logo_jkn.jpeg", "JKN", "JKN")
    claim_logo = next(
        (
            candidate
            for candidate in (ROOT / "asset" / "logo_claimrmf.png", ROOT / "asset" / "logo_claim.png")
            if candidate.is_file()
        ),
        ROOT / "asset" / "logo_claimrmf.png",
    )
    st.markdown(
        '<div class="cg-top">'
        f'{_brand_html(claim_logo)}'
        f'<div class="cg-partners">{bpjs}{jkn}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<section class="cg-hero2">'
        f'{_hero_art_html(claim_logo)}'
        '<div class="cg-hero-copy">'
        '<div class="cg-tag">Prototipe demonstrasi</div>'
        '<h1 class="cg-hero-title cg-serif">Pemeriksaan klaim yang lebih terarah, '
        'dengan <em>bukti yang jelas</em></h1>'
        '<p class="cg-lead">ClaimGuard membantu petugas melihat klaim yang perlu diperhatikan, '
        'memahami alasan serta pembandingnya, lalu mencatat tindak lanjut. '
        'Pemeriksaan dan keputusan tetap berada di tangan petugas berwenang.</p>'
        '<div class="cg-meta"><span>Data sintetis</span><i></i><span>Satu alur pemeriksaan</span><i></i>'
        '<span>Keputusan tetap di petugas</span></div>'
        '</div></section>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="cg-kicker">Kemampuan</div>'
        '<div class="cg-h2 cg-serif">Satu alur kerja, tiga kegiatan utama</div>',
        unsafe_allow_html=True,
    )
    features = st.columns(3, gap="large")
    feature_items = [
        ("01", "Melihat antrean pemeriksaan",
         "Klaim berprioritas ditampilkan beserta petunjuk yang perlu diperiksa."),
        ("02", "Memahami alasan dan pembanding",
         "Pola, rincian klaim, dan perbandingan membantu petugas menilai konteksnya."),
        ("03", "Mencatat hasil dan tindak lanjut",
         "Hasil pemeriksaan petugas dan simulasi klarifikasi dengan faskes tercatat rapi."),
    ]
    for column, (number, title, description) in zip(features, feature_items):
        with column:
            st.markdown(
                f'<div class="cg-feat"><div class="cg-feat-num">{number}</div>'
                f'<h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="cg-kicker">Mulai bekerja</div>'
        '<div class="cg-h2 cg-serif">Satu tempat untuk memeriksa klaim</div>'
        '<p class="cg-sub">ClaimGuard menampilkan klaim yang perlu diperhatikan beserta alasan '
        'dan bukti pembanding. Petugas meninjau hasilnya, lalu meminta klarifikasi kepada faskes '
        'melalui kanal resmi bila diperlukan.</p>',
        unsafe_allow_html=True,
    )
    st.button(
        "Mulai pemeriksaan klaim",
        type="primary",
        width="stretch",
        key="open_claim_review",
        on_click=_open_analysis,
    )

    st.markdown(
        '<div class="cg-kicker">Alur kerja</div>'
        '<div class="cg-h2 cg-serif">Cara kerja pemeriksaan</div>'
        '<p class="cg-sub">Semua klaim mengikuti satu proses yang sederhana dan konsisten.</p>',
        unsafe_allow_html=True,
    )
    steps = st.columns(4, gap="large")
    step_items = [
        ("1", "Tinjau antrean", "Klaim diurutkan menurut prioritas pemeriksaan."),
        ("2", "Pelajari bukti", "Lihat alasan utama, pembanding biaya, dan klaim yang mirip."),
        ("3", "Catat hasil", "Petugas mencatat bukti yang ditinjau dan hasil pemeriksaan."),
        ("4", "Tindak lanjut", "Simulasi klarifikasi faskes sebelum diteruskan ke proses resmi."),
    ]
    for column, (number, title, description) in zip(steps, step_items):
        with column:
            st.markdown(
                f'<div class="cg-step"><b>{number}</b><h4>{title}</h4><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="cg-notice"><strong>Bukan portal resmi BPJS Kesehatan.</strong> '
        'Logo ditampilkan sebagai identitas konteks pada mockup, bukan pernyataan afiliasi, '
        'persetujuan, atau integrasi. Seluruh data yang digunakan adalah data sintetis, '
        'dan hasilnya bukan bukti kecurangan.</div>',
        unsafe_allow_html=True,
    )


def _render_claim_workflow(
    claim_id: str,
    analysis_source: str,
    analysis_path: str,
    split_name: str,
    assistant_content: dict,
) -> None:
    st.markdown("#### Pemeriksaan petugas dan tindak lanjut")
    st.caption(
        "Pilih peran dan hasil pemeriksaan. ClaimGuard menyiapkan ringkasan sinyal serta saran "
        "bukti untuk ditinjau; petugas tetap memeriksa dokumen, memilih bukti yang benar-benar "
        "ditinjau, dan menentukan apakah klarifikasi diperlukan."
    )
    st.warning(
        "Ini simulasi dengan data buatan. Permintaan dan tanggapan faskes tidak dikirim melalui "
        "kanal resmi dan versi klaim yang dikoreksi tidak disimpan. Meneruskan klaim ke proses "
        "resmi bukan berarti klaim disetujui atau pembayaran dilakukan."
    )
    with st.expander("Ringkasan temuan dan saran ClaimGuard", expanded=True):
        st.caption(
            "Draf otomatis berdasarkan sinyal dan pembanding data yang tersedia; bukan model "
            "AI generatif, bukan pemeriksaan dokumen, dan bukan kesimpulan ketidaksesuaian."
        )
        for finding in assistant_content["findings"]:
            st.markdown(f"- {finding}")
        st.markdown("**Bukti yang disarankan untuk diperiksa petugas**")
        st.write(", ".join(assistant_content["suggested_evidence"]))
        st.caption(
            "Saran ini tidak otomatis dicatat sebagai bukti yang telah ditinjau. "
            "Petugas harus memilih bukti setelah memeriksanya."
        )
    workflow_log = read_workflow_log(WORKFLOW_LOG, LEGACY_WORKFLOW_LOG)
    status = get_workflow_status(
        workflow_log,
        analysis_source,
        analysis_path,
        claim_id,
        split_name,
    )
    st.metric("Status tindak lanjut", status)

    available_actions = WORKFLOW_ACTIONS[status]
    if not available_actions:
        st.info(
            "Tahap simulasi selesai setelah klaim diteruskan ke proses resmi. Tahapan berikutnya "
            "dapat mencakup pembayaran sesuai hasil dan kewenangan resmi; dashboard tidak "
            "menentukan persetujuan atau memproses pembayaran."
        )
    else:
        actor = available_actions[0]["actor"]
        is_staff_review = actor == "Petugas"
        action_options = [option["action"] for option in available_actions]
        action_key = f"workflow_decision_{analysis_source}_{analysis_path}_{claim_id}_{split_name}"
        action_label = st.selectbox(
            "Hasil pemeriksaan petugas" if is_staff_review else "Tanggapan faskes (simulasi)",
            action_options,
            key=action_key,
        )
        selected_action_index = action_options.index(action_label)
        selected_action = available_actions[selected_action_index]
        is_clarification = "klarifikasi" in action_label.casefold()
        notes_key = (
            f"workflow_notes_{analysis_source}_{analysis_path}_{claim_id}_{split_name}_"
            f"{selected_action_index}"
        )
        default_notes = ""
        if is_staff_review and is_clarification:
            default_notes = assistant_content["clarification_request"]
            if "tambahan" in action_label.casefold():
                default_notes = (
                    "Sebagai klarifikasi tambahan, " + default_notes[0].lower() + default_notes[1:]
                )
        elif is_staff_review:
            if action_label == "Tanggapan cukup; lanjut ke proses resmi":
                default_notes = (
                    "Petugas telah meninjau tanggapan faskes dan menilai informasi yang diberikan "
                    "cukup untuk diteruskan ke proses resmi. Keputusan pembayaran tetap mengikuti "
                    "kewenangan dan proses resmi."
                )
            else:
                default_notes = (
                    "Petugas meninjau rincian klaim dan ringkasan pembanding. "
                    "Hasil pemeriksaan dicatat untuk diteruskan ke proses resmi; "
                    "keputusan pembayaran tetap mengikuti kewenangan dan proses resmi."
                )
        with st.form(f"workflow_form_{analysis_source}_{analysis_path}_{claim_id}_{split_name}"):
            if is_staff_review:
                reviewer_role = st.selectbox(
                    "Peran pemeriksa",
                    REVIEWER_ROLES,
                    key=f"workflow_role_{analysis_source}_{analysis_path}_{claim_id}_{split_name}",
                )
                evidence_reviewed = st.multiselect(
                    "Bukti yang benar-benar telah ditinjau petugas",
                    EVIDENCE_ITEMS,
                    key=f"workflow_evidence_{analysis_source}_{analysis_path}_{claim_id}_{split_name}",
                )
                feedback = st.selectbox(
                    "Apakah petunjuk ClaimGuard membantu pemeriksaan?",
                    RECOMMENDATION_FEEDBACK,
                    key=f"workflow_feedback_{analysis_source}_{analysis_path}_{claim_id}_{split_name}",
                )
            else:
                reviewer_role = ""
                evidence_reviewed = []
                feedback = ""
            notes_label = "Catatan pemeriksa" if is_staff_review else "Rincian dokumen atau penjelasan"
            notes = st.text_area(
                notes_label,
                value=default_notes,
                max_chars=2000,
                key=notes_key,
            )
            submitted = st.form_submit_button(
                "Simpan hasil pemeriksaan" if is_staff_review else "Catat tanggapan faskes"
            )
        if submitted:
            if is_staff_review and not evidence_reviewed:
                st.error("Pilih setidaknya satu bukti yang telah ditinjau.")
            elif len(notes.strip()) < 10:
                st.error("Catatan harus berisi sedikitnya 10 karakter.")
            else:
                record_workflow_action(
                    WORKFLOW_LOG,
                    analysis_source,
                    analysis_path,
                    claim_id,
                    split_name,
                    actor,
                    selected_action["action"],
                    notes,
                    reviewer_role=reviewer_role,
                    evidence_reviewed=evidence_reviewed,
                    review_outcome=selected_action["review_outcome"],
                    recommendation_feedback=feedback,
                    legacy_path=LEGACY_WORKFLOW_LOG,
                )
                st.success("Catatan simulasi tersimpan. Tidak ada pesan yang dikirim ke faskes.")
                st.rerun()

    claim_workflow = workflow_log.loc[
        workflow_log["analysis_source"].eq(analysis_source)
        & workflow_log["analysis_path"].eq(analysis_path)
        & workflow_log["claim_id"].eq(str(claim_id))
        & workflow_log["split_name"].eq(split_name)
    ]
    if not claim_workflow.empty:
        st.markdown("**Riwayat langkah simulasi**")
        history = claim_workflow[
            [
                "recorded_at_utc",
                "actor",
                "evidence_reviewed",
                "review_outcome",
                "action",
                "status_after_action",
                "notes",
            ]
        ].rename(
            columns={
                "recorded_at_utc": "Waktu pencatatan",
                "actor": "Pihak simulasi",
                "evidence_reviewed": "Bukti yang ditinjau",
                "review_outcome": "Hasil pemeriksaan",
                "action": "Langkah",
                "status_after_action": "Status setelah langkah",
                "notes": "Catatan",
            }
        )
        st.dataframe(history, width="stretch", hide_index=True)


def _render_legacy_ml() -> None:
    st.subheader("Jalur 1: Model ML pada simulator lama")
    st.caption(
        "Jalur ini mencari pola yang perlu diperiksa menggunakan aturan dan model machine learning. "
        "Hasilnya dilengkapi skor prioritas dan alasan yang tersedia dari simulator."
    )
    with st.sidebar:
        scheme_label = st.selectbox("Cara membagi data", ["Periode waktu", "Fasilitas kesehatan"])
        scheme = "waktu" if scheme_label == "Periode waktu" else "faskes"
        mode_label = st.selectbox(
            "Informasi yang digunakan",
            ["Tersedia saat klaim diajukan", "Termasuk informasi setelah pengajuan (audit)"],
        )
        mode = "asof" if mode_label == "Tersedia saat klaim diajukan" else "retro"
        capacity = st.slider("Batas jumlah klaim dalam antrean (%)", 1, 10, 5)
        show_metrics = st.checkbox(
            "Tampilkan hasil uji pada data simulasi",
            value=False,
            help="Data buatan ini hanya dipakai untuk mengukur hasil uji, bukan untuk membuat prediksi.",
        )

    suffix = f"{scheme}_{mode}"
    calibrated = _load_csv(str(OUT / "calibrated" / f"calibrated_{suffix}.csv.gz"))
    claims = _load_csv(str(OUT / "data" / "claims.csv"), usecols=OLD_CLAIM_COLUMNS)
    detector_path = OUT / "detect" / f"detect_{suffix}.csv.gz"
    explanations_path = OUT / "explain" / f"explain_{suffix}.csv.gz"
    detectors = _load_csv(str(detector_path)) if detector_path.is_file() else pd.DataFrame()
    explanations = _load_csv(str(explanations_path)) if explanations_path.is_file() else pd.DataFrame()
    _required_columns(
        calibrated,
        ["claim_id", "split", "p_value", "skor_0_100", "tingkat"],
        "Hasil kalibrasi model lama",
    )
    if claims["claim_id"].duplicated().any() or calibrated.duplicated(["claim_id", "split"]).any():
        raise ValueError("ID klaim atau pasangan klaim/split duplikat pada hasil model lama.")

    calibrated = calibrated.copy()
    calibrated["claim_id"] = calibrated["claim_id"].astype(str)
    claims["claim_id"] = claims["claim_id"].astype(str)
    data = calibrated.merge(claims, on="claim_id", how="left", validate="many_to_one")
    if data["nilai_klaim"].isna().any():
        raise ValueError("Sebagian hasil model tidak memiliki rincian klaim yang sesuai.")
    if not detectors.empty:
        _required_columns(detectors, ["claim_id", "split", *OLD_DETECTOR_COLUMNS], "Hasil detektor")
        detector_data = detectors[["claim_id", "split", *OLD_DETECTOR_COLUMNS]].copy()
        detector_data["claim_id"] = detector_data["claim_id"].astype(str)
        data = data.merge(
            detector_data,
            on=["claim_id", "split"],
            how="left",
            validate="one_to_one",
        )
        detector_columns = [column for column in OLD_DETECTOR_COLUMNS if column in data]
        data["sinyal_utama"] = data[detector_columns].fillna(0).idxmax(axis=1).map(OLD_DETECTOR_LABELS)
    else:
        data["sinyal_utama"] = "Belum tersedia"

    if not explanations.empty and "claim_id" in explanations:
        explanations = explanations.copy()
        explanations["claim_id"] = explanations["claim_id"].astype(str)
        if explanations["claim_id"].duplicated().any():
            raise ValueError("Penjelasan model lama memiliki ID klaim duplikat.")
        explanation_columns = [
            column for column in [
                "claim_id", "top_driver_mode", "top_feature", "top_feature_value", "explanation",
            ] if column in explanations
        ]
        data = data.merge(
            explanations[explanation_columns],
            on="claim_id",
            how="left",
            validate="many_to_one",
        )

    test = data.loc[data["split"].eq("test")].copy()
    if test.empty:
        st.error("Data uji model lama tidak tersedia untuk pilihan ini.")
        return
    test["p_value"] = pd.to_numeric(test["p_value"], errors="coerce")
    test["skor_0_100"] = pd.to_numeric(test["skor_0_100"], errors="coerce")
    if test[["p_value", "skor_0_100"]].isna().any().any():
        raise ValueError("Skor atau p-value tidak valid pada hasil model lama.")
    review_count = max(1, (capacity * len(test) + 99) // 100)
    queue = test.sort_values(["p_value", "claim_id"], ascending=[True, True]).head(review_count).copy()
    queue["prioritas"] = queue["tingkat"].map(TIER_LABELS).fillna(queue["tingkat"])

    if show_metrics:
        labels = _load_csv(str(OUT / "eval_only" / "labels.csv"), keep_default_na=False)
        _required_columns(labels, ["claim_id", "is_anomaly"], "Label evaluasi sintetis")
        labels = labels[["claim_id", "is_anomaly"]].copy()
        labels["claim_id"] = labels["claim_id"].astype(str)
        test = test.merge(labels, on="claim_id", how="left", validate="one_to_one")
        if test["is_anomaly"].isna().any():
            raise ValueError("Ada klaim uji tanpa label simulasi untuk evaluasi.")
        flagged_ids = set(queue["claim_id"])
        normal = test.loc[test["is_anomaly"].astype(str).eq("0")]
        anomaly = test.loc[test["is_anomaly"].astype(str).eq("1")]
        false_positive_rate = normal["claim_id"].isin(flagged_ids).mean() if len(normal) else float("nan")
        recall = anomaly["claim_id"].isin(flagged_ids).mean() if len(anomaly) else float("nan")
    else:
        false_positive_rate = None
        recall = None

    summary_tab, claims_tab, quality_tab = st.tabs(
        ["Gambaran hasil", "Daftar klaim untuk ditinjau", "Batasan dan hasil uji"]
    )
    with summary_tab:
        columns = st.columns(3)
        columns[0].metric("Klaim simulasi yang diuji", _format_number(len(test)))
        columns[1].metric("Klaim dalam antrean", _format_number(len(queue)))
        columns[2].metric("Batas tinjauan", f"{capacity}%")
        st.caption(
            f"Pengaturan: {scheme_label}; {mode_label}. Klaim diurutkan berdasarkan tingkat "
            "prioritas dari hasil pengujian. Skor 0–100 bukan kemungkinan kecurangan."
        )
        if show_metrics:
            metric_columns = st.columns(2)
            metric_columns[0].metric("Klaim wajar yang ikut ditandai", f"{false_positive_rate:.1%}")
            metric_columns[1].metric("Kasus buatan yang masuk antrean", f"{recall:.1%}")
            st.caption(
                "Hasil ini hanya mengukur data uji buatan pada simulator, bukan kemampuan "
                "mengenali masalah pada klaim nyata."
            )
        tier_counts = (
            queue["prioritas"]
            .value_counts()
            .reindex(["Sangat tinggi", "Tinggi", "Menengah", "Rendah"], fill_value=0)
            .rename_axis("Prioritas")
            .to_frame("Jumlah klaim")
        )
        st.markdown("**Jumlah klaim menurut tingkat prioritas**")
        st.bar_chart(tier_counts)
        facility_counts = (
            queue.groupby("facility_id", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
            .rename("Jumlah klaim dalam antrean")
        )
        st.markdown("**Fasilitas dengan klaim terbanyak dalam antrean**")
        st.dataframe(facility_counts.reset_index(), width="stretch", hide_index=True)
        st.caption("Jumlah klaim dalam antrean bukan ukuran tingkat kecurangan fasilitas.")

    with claims_tab:
        facilities = ["Semua"] + sorted(queue["facility_id"].astype(str).unique().tolist())
        selected_facility = st.selectbox("Fasilitas", facilities, key="legacy_facility")
        search = st.text_input("Cari nomor klaim", key="legacy_search").strip().casefold()
        filtered = queue.copy()
        if selected_facility != "Semua":
            filtered = filtered.loc[filtered["facility_id"].astype(str).eq(selected_facility)]
        if search:
            filtered = filtered.loc[filtered["claim_id"].astype(str).str.casefold().str.contains(search, regex=False)]

        display = filtered[
            ["claim_id", "facility_id", "jenis_layanan", "tanggal_klaim", "nilai_klaim", "sinyal_utama", "prioritas"]
        ].rename(
            columns={
                "claim_id": "Nomor klaim",
                "facility_id": "Fasilitas",
                "jenis_layanan": "Jenis layanan",
                "tanggal_klaim": "Tanggal pengajuan",
                "nilai_klaim": "Nilai klaim (simulasi)",
                "sinyal_utama": "Alasan utama untuk ditinjau",
                "prioritas": "Prioritas",
            }
        )
        if "Nilai klaim (simulasi)" in display:
            display["Nilai klaim (simulasi)"] = display["Nilai klaim (simulasi)"].map(_format_money)
        st.dataframe(display, width="stretch", hide_index=True)
        if filtered.empty:
            st.info("Tidak ada klaim yang sesuai dengan filter.")
        else:
            selected_id = st.selectbox(
                "Klaim untuk ditinjau",
                filtered["claim_id"].astype(str).tolist(),
                key="legacy_claim",
            )
            selected = filtered.loc[filtered["claim_id"].astype(str).eq(selected_id)].iloc[0]
            st.markdown(f"#### Ringkasan klaim {selected_id}")
            st.info(
                "Alasan ini berasal dari model pada simulator. Periksa dokumen dan konteks layanan; "
                "masuk antrean bukan berarti klaim bermasalah."
            )
            metrics = st.columns(4)
            metrics[0].metric("Skor prioritas (0–100)", f"{selected['skor_0_100']:.1f}")
            metrics[1].metric("Prioritas", selected["prioritas"])
            metrics[2].metric("Alasan utama", selected["sinyal_utama"])
            metrics[3].metric("Nilai klaim", _format_money(selected["nilai_klaim"]))
            details = st.columns(2)
            for index, (column, label) in enumerate([
                ("facility_id", "Kode fasilitas"),
                ("jenis_layanan", "Jenis layanan"),
                ("tanggal_klaim", "Tanggal pengajuan"),
                ("patient_id", "ID peserta (simulasi)"),
                ("diagnosis_utama", "Diagnosis utama"),
                ("diagnosis_sekunder", "Diagnosis tambahan"),
                ("tindakan", "Tindakan atau prosedur"),
                ("lama_rawat", "Lama rawat"),
            ]):
                details[index % 2].markdown(
                    f"**{label}**  \n{selected.get(column, 'Tidak tersedia')}"
                )
            detector_values = [
                {"Sinyal": OLD_DETECTOR_LABELS[column], "Nilai": float(selected[column])}
                for column in OLD_DETECTOR_COLUMNS
                if column in selected and pd.notna(selected[column])
            ]
            if detector_values:
                with st.expander("Lihat rincian pola yang ditemukan"):
                    st.dataframe(pd.DataFrame(detector_values), width="stretch", hide_index=True)
            if pd.notna(selected.get("explanation")):
                st.markdown("**Alasan dari model**")
                st.write(str(selected["explanation"]))
            if pd.notna(selected.get("top_feature")):
                st.caption(
                    f"Fitur utama: {selected['top_feature']} "
                    f"({selected.get('top_feature_value', 'tidak tersedia')})."
                )
            st.divider()
            _render_claim_workflow(
                selected_id,
                "legacy_ml",
                f"{scheme}_{mode}",
                str(selected["split"]),
            )

    with quality_tab:
        st.markdown(
            "Jalur ini menggabungkan beberapa cara untuk mencari pola yang tidak biasa. "
            "Setiap pola adalah petunjuk untuk ditinjau, bukan bukti kesalahan."
        )
        st.markdown(
            "Hasil uji berasal dari data buatan. Angka tersebut belum menunjukkan seberapa baik "
            "sistem bekerja pada klaim JKN nyata."
        )
        robustness_path = ROOT / "robustness" / "robustness_summary.csv"
        if robustness_path.is_file():
            robustness = _load_csv(str(robustness_path))
            st.markdown("**Ringkasan kestabilan pada data simulasi**")
            st.dataframe(robustness, width="stretch", hide_index=True)
        st.caption(
            "Data baru belum bisa dimasukkan untuk dianalisis otomatis. Data tersebut perlu "
            "diperiksa dan diuji terlebih dahulu."
        )


def _render_v2() -> None:
    st.subheader("Antrean pemeriksaan klaim")
    st.caption(
        "Klaim diurutkan berdasarkan sinyal pada data simulasi. Untuk setiap klaim, ClaimGuard "
        "menampilkan alasan dan pembanding yang dapat ditinjau petugas. Petunjuk ini bukan "
        "bukti kesalahan atau kecurangan."
    )
    with st.sidebar:
        scheme_label = st.selectbox("Cara mengelompokkan klaim", list(V2_SPLIT_OPTIONS), key="v2_scheme")
        scheme, split_column = V2_SPLIT_OPTIONS[scheme_label]
        partition_label = st.selectbox("Pilih kumpulan data", list(V2_PARTITIONS), key="v2_partition")
        partition = V2_PARTITIONS[partition_label]
        recommended_only = st.checkbox(
            "Hanya tampilkan klaim yang disarankan untuk ditinjau",
            value=True,
            key="v2_recommended",
        )
        search = st.text_input("Cari nomor klaim atau kode fasilitas", key="v2_search").strip().casefold()

    prep = OUT / "prep_v2"
    queue_column, queue_file = V2_QUEUE_FILES[scheme]
    queue = _load_csv(str(prep / queue_file))
    claims = _load_csv(str(prep / "claims_v2_flat_with_splits.csv.gz"), keep_default_na=False)
    features = _load_csv(str(prep / V2_FEATURE_FILES[scheme]), keep_default_na=False)
    _required_columns(
        queue,
        [
            "id_klaim", queue_column, "priority_rank", "submitted_at", "service_type",
            "facility_id", "priority_score", "top_driver", "review_recommended", "evidence",
            "interpretation", *V2_SIGNAL_LABELS,
        ],
        "Antrean V2",
    )
    if queue["id_klaim"].duplicated().any() or claims["id_klaim"].duplicated().any():
        raise ValueError("ID klaim duplikat pada tabel V2.")
    visible = queue.loc[queue[queue_column].eq(partition)].copy()
    if recommended_only:
        visible = visible.loc[visible["review_recommended"].astype(bool)]
    if search:
        visible = visible.loc[
            visible["id_klaim"].astype(str).str.casefold().str.contains(search, regex=False)
            | visible["facility_id"].astype(str).str.casefold().str.contains(search, regex=False)
        ]
    visible = visible.sort_values(["priority_rank", "id_klaim"], kind="stable")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Klaim dalam daftar", _format_number(len(visible)))
    metric_columns[1].metric(
        "Disarankan untuk ditinjau",
        _format_number(int(queue.loc[queue[queue_column].eq(partition), "review_recommended"].astype(bool).sum())),
    )
    metric_columns[2].metric("Kumpulan data", partition_label)
    st.caption(
        "Urutan ini membandingkan klaim dalam data buatan. Skor bukan kemungkinan kecurangan."
    )
    display = visible[
        ["priority_rank", "id_klaim", "submitted_at", "service_type", "facility_id", "top_driver"]
    ].rename(
        columns={
            "priority_rank": "Urutan",
            "id_klaim": "Nomor klaim",
            "submitted_at": "Tanggal pengajuan",
            "service_type": "Jenis layanan",
            "facility_id": "Fasilitas",
            "top_driver": "Alasan utama untuk ditinjau",
        }
    )
    display["Jenis layanan"] = display["Jenis layanan"].map(_service_name)
    st.dataframe(display, width="stretch", hide_index=True)
    if visible.empty:
        st.info("Tidak ditemukan klaim yang sesuai dengan filter.")
        return

    selected_id = st.selectbox(
        "Klaim untuk ditinjau",
        visible["id_klaim"].astype(str).tolist(),
        key="v2_claim",
    )
    row = visible.loc[visible["id_klaim"].astype(str).eq(selected_id)].iloc[0]
    claim_matches = claims.loc[claims["id_klaim"].astype(str).eq(selected_id)]
    if len(claim_matches) != 1:
        raise ValueError(f"Klaim {selected_id} harus ditemukan tepat satu kali.")
    claim = claim_matches.iloc[0]
    evidence = build_claim_comparison(claims, features, selected_id, split_column)
    assistant_content = build_review_assistant_content(
        claim.to_dict(),
        row.to_dict(),
        evidence,
    )
    st.markdown(f"#### Klaim yang dipilih: {selected_id}")
    st.info(
        f"Alasan utama untuk meninjau: {row['top_driver']}. {row['evidence']}. "
        "Ini hanya petunjuk, bukan bukti kesalahan. Pemeriksa tetap perlu melihat dokumen dan konteksnya."
    )
    details_tab, comparison_tab, workflow_tab = st.tabs(
        [
            "Data klaim",
            "Bandingkan dengan klaim lain",
            "Pemeriksaan dan tindak lanjut",
        ]
    )
    with details_tab:
        columns = st.columns(3)
        for index, (key, label) in enumerate(
            [
                ("submitted_at", "Tanggal pengajuan"),
                ("service_date", "Tanggal pelayanan"),
                ("service_type", "Jenis layanan"),
                ("facility_id", "Kode fasilitas (simulasi)"),
                ("claim_total_idr", "Total klaim"),
                ("length_of_stay_days", "Lama rawat (hari)"),
                ("diagnosis_primary", "Diagnosis utama"),
                ("diagnosis_secondary", "Diagnosis tambahan"),
                ("procedure_codes", "Kode tindakan atau prosedur"),
                ("item_codes", "Kode obat atau alat kesehatan"),
            ]
        ):
            value = claim[key]
            if key == "service_type":
                value = _service_name(value)
            if key == "claim_total_idr":
                value = _format_money(value)
            columns[index % 3].markdown(f"**{label}**  \n{value if value != '' else 'Tidak tersedia'}")
        component_table = pd.DataFrame(
            [
                {"Komponen biaya": label, "Nilai": _format_money(claim[column])}
                for column, label in V2_COST_LABELS.items()
            ]
        )
        st.dataframe(component_table, width="stretch", hide_index=True)
    with comparison_tab:
        cost = evidence["cost"]
        metrics = st.columns(3)
        metrics[0].metric("Total klaim", _format_money(cost["current_total_idr"]))
        metrics[1].metric("Nilai tengah klaim pembanding", _format_money(cost["train_median_total_idr"]))
        metrics[2].metric(
            "Rasio terhadap pembanding",
            f"{cost['ratio_to_train_median']:.2f} kali".replace(".", ","),
        )
        st.caption(
            f"Pembanding: {cost['group_label']} ({_format_number(cost['reference_count'])} klaim acuan). "
            "Nilai ini berasal dari data simulasi, bukan tarif resmi."
        )
        cost_table = pd.DataFrame(
            [
                {
                    "Komponen": V2_COST_LABELS[item["component"]],
                    "Klaim terpilih": _format_money(item["current_idr"]),
                    "Nilai tengah klaim acuan": _format_money(item["train_median_idr"]),
                    "Selisih": _format_money(item["difference_idr"]),
                }
                for item in cost["components"]
            ]
        )
        st.dataframe(cost_table, width="stretch", hide_index=True)
        prior = evidence["repeat"]["previous_claim"]
        st.markdown("**Klaim sebelumnya dari peserta yang sama**")
        if prior is None:
            st.caption("Tidak ditemukan klaim terdahulu dalam konteks data yang digunakan.")
        else:
            st.write(
                f"{prior['claim_id']} — tanggal {prior['submitted_at']}, "
                f"total {_format_money(prior['claim_total_idr'])}; "
                f"{evidence['repeat']['prior_count_30d']} klaim terdahulu dalam 30 hari."
            )
        similar = evidence["similar_claim"]
        st.markdown("**Klaim sebelumnya dengan rincian paling mirip**")
        if similar is None:
            st.caption("Tidak ditemukan klaim terdahulu dengan kode yang sama untuk dibandingkan.")
        else:
            sim_metrics = st.columns(3)
            sim_metrics[0].metric("Klaim pembanding", similar["claim_id"])
            sim_metrics[1].metric("Tanggal pengajuan", similar["submitted_at"])
            sim_metrics[2].metric("Kemiripan kode", f"{similar['similarity']:.0%}")
            st.caption("Klaim yang mirip belum tentu salah atau merupakan duplikasi.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Kelompok kode": "Kode yang sama", "Kode": ", ".join(similar["shared_codes"]) or "-"},
                        {
                            "Kelompok kode": "Hanya pada klaim terpilih",
                            "Kode": ", ".join(similar["current_only_codes"]) or "-",
                        },
                        {
                            "Kelompok kode": "Hanya pada klaim pembanding",
                            "Kode": ", ".join(similar["comparison_only_codes"]) or "-",
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        severity = evidence["severity"]
        if severity is not None:
            st.caption(
                f"Dari {_format_number(severity['reference_count'])} klaim acuan untuk "
                f"{severity['group_label']}, "
                f"median lama rawat {severity['train_median_stay_days']:.1f} hari; "
                f"klaim terpilih mencatat severity {severity['current_severity']} dan "
                f"{severity['current_stay_days']:g} hari rawat. Ini konteks statistik, bukan penilaian klinis."
            )
    with workflow_tab:
        _render_claim_workflow(
            selected_id,
            "v2",
            scheme,
            partition,
            assistant_content,
        )


st.set_page_config(
    page_title="ClaimGuard | Dashboard Pemeriksaan Klaim",
    page_icon="CG",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #f4f7f8; color: #18364b; }
    [data-testid="stMainBlockContainer"] { max-width: 1380px; padding-top: 1.5rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dce5e8; }
    [data-testid="stSidebar"] * { color: #18364b; }
    h1, h2, h3, h4 { color: #18364b; }
    h1 { font-size: 1.9rem; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce5e8;
        border-radius: 7px; padding: .8rem 1rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #dce5e8; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    nav_menu = option_menu(
        "Menu utama",
        ["Beranda", "Pemeriksaan klaim", "Analitik", "Laporan", "Manajemen", "Pengaturan"],
        icons=["house", "check-circle", "bar-chart", "file-text", "gear", "tools"],
        menu_icon="cast",
        default_index=0,
        key="nav_menu",
    )

if nav_menu == "Beranda":
    _render_landing_page()
elif nav_menu == "Pemeriksaan klaim":
    st.sidebar.button("Kembali ke beranda", width="stretch", on_click=_go_home)
    _render_header()
    _render_v2()
else:
    _render_unavailable_page(nav_menu)

st.caption(
    "Prototipe demonstrasi ClaimGuard. Tidak terhubung dengan BPJS Kesehatan atau sistem klaim "
    "resmi. Analisis memakai data sintetis; hasilnya hanya petunjuk untuk pemeriksaan petugas."
)