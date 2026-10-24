#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-route ClaimGuard demo dashboard with an evidence-based claims queue."""

import base64
import html
import io
import json
import platform
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from numbers import Real

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageChops
from streamlit_option_menu import option_menu

from claimguard_pages.common import *  # noqa: F401,F403
from claimguard_pages.analitik import _render_analytics
from claimguard_pages.laporan import _render_reports
from claimguard_pages.manajemen import _render_management
from claimguard_pages.pengaturan import _render_settings
from claimguard_review_evidence_v2 import build_claim_comparison
from claimguard_review_assistant import build_review_assistant_content
from claimguard_workflow_demo import (
    EVIDENCE_ITEMS,
    INITIAL_STATUS,
    OFFICIAL_PROCESS_STATUS,
    RECOMMENDATION_FEEDBACK,
    REVIEWER_ROLES,
    WAITING_FOR_FACILITY_STATUS,
    WAITING_FOR_STAFF_STATUS,
    WORKFLOW_ACTIONS,
    get_workflow_status,
    read_workflow_log,
    record_workflow_action,
)


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


def _get_page_icon():
    """Favicon ClaimGuard dari berkas asset, dipotong rapi dan simetris agar jernih di tab browser."""
    for candidate in (ROOT / "asset" / "logo_claimrmf.png", ROOT / "asset" / "logo_claim.png"):
        if candidate.is_file():
            try:
                source = Image.open(candidate).convert("RGBA")
                alpha = source.getchannel("A")
                bbox = alpha.getbbox()
                if bbox:
                    left, top, right, bottom = bbox
                    cx = (left + right) / 2
                    cy = (top + bottom) / 2
                    side = max(right - left, bottom - top) * 1.08
                    half = side / 2
                    box = (
                        max(0, int(cx - half)),
                        max(0, int(cy - half)),
                        min(source.width, int(cx + half)),
                        min(source.height, int(cy + half)),
                    )
                    return source.crop(box)
                return source
            except Exception:
                pass
    return "🛡️"


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
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800;14..32,900&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

/* ── Landing page overrides ── */
.stApp {
    background:
        radial-gradient(ellipse 1100px 550px at 88% -6%, rgba(22,118,106,.07), transparent 62%),
        radial-gradient(ellipse 850px 480px at -6% 38%, rgba(47,109,181,.045), transparent 55%),
        radial-gradient(ellipse 500px 500px at 50% 105%, rgba(22,118,106,.025), transparent 55%),
        linear-gradient(172deg, #eef3f4 0%, #f4f7f8 35%, #f9fafb 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}

/* ── Animations ── */
@keyframes cgfade  { from { opacity:0; transform:translateY(16px) } to { opacity:1; transform:none } }
@keyframes cgfadeR { from { opacity:0; transform:translateX(16px) } to { opacity:1; transform:none } }
@keyframes cgfloat { 0%,100% { transform:translate(-50%,-50%) scale(1) } 50% { transform:translate(-50%,calc(-50% - 9px)) scale(1.008) } }
@keyframes cgGlow  { 0%,100% { opacity:.55 } 50% { opacity:.8 } }
@keyframes cgLine  { from { width:0 } to { width:100% } }

/* ── Top bar ── */
.cg-top { display:flex; align-items:center; justify-content:space-between; padding:.4rem 0 2.2rem; }
.cg-brand { display:flex; align-items:center; gap:.65rem; }
.cg-brand svg { width:30px; height:30px; flex:none; }
.cg-brand-logo { height:66px; width:auto; max-width:240px; object-fit:contain; display:block;
    filter: drop-shadow(0 2px 6px rgba(18,48,63,.07)); }
.cg-wordmark { font-size:1.15rem; font-weight:800; letter-spacing:.06em; color:#0b2430; }
.cg-wordmark span { color:#16766a; }
.cg-partners { display:flex; align-items:center; gap:1.6rem; }
.cg-partners img { height:36px; max-width:160px; object-fit:contain; mix-blend-mode:multiply;
    opacity:.82; transition: all .3s cubic-bezier(.4,0,.2,1); }
.cg-partners img:hover { opacity:1; transform: scale(1.05); }
.cg-logo-fallback { color:#4a606b; font-size:.92rem; font-weight:700; }

/* ── Hero banner ── */
.cg-hero2 { position:relative; overflow:hidden; border-radius:22px;
    padding:4.4rem 3.4rem 4rem;
    background: linear-gradient(128deg, #e5f0ed 0%, #ebf1f5 48%, #f0f3f8 100%);
    box-shadow: 0 1px 2px rgba(18,48,63,.03), 0 4px 20px rgba(18,48,63,.05), 0 12px 48px rgba(18,48,63,.04);
    animation:cgfade .65s ease both; }
.cg-hero2::before { content:''; position:absolute; inset:0;
    background: linear-gradient(135deg, transparent 55%, rgba(22,118,106,.025) 100%); pointer-events:none; }
.cg-hero-art { position:absolute; right:-10px; top:50%; transform:translateY(-50%);
    width:470px; height:470px; pointer-events:none; }
.cg-hero-glow { position:absolute; left:53.2%; top:50%; width:340px; height:340px;
    transform:translate(-50%,-50%); border-radius:50%;
    background:radial-gradient(circle, rgba(255,255,255,.96) 0%, rgba(255,255,255,.55) 42%, rgba(255,255,255,0) 68%);
    animation: cgGlow 5s ease-in-out infinite; }
.cg-hero-logo { position:absolute; left:53.2%; top:50%; transform:translate(-50%,-50%);
    height:260px; width:auto; max-width:245px; object-fit:contain;
    filter:drop-shadow(0 18px 36px rgba(18,48,63,.14)); animation:cgfloat 6s ease-in-out infinite; }
.cg-hero-copy { position:relative; max-width:580px; z-index:1; }
.cg-tag { color:#16766a; font-size:.68rem; font-weight:700; letter-spacing:.2em; text-transform:uppercase;
    background: rgba(22,118,106,.07); padding:.32rem .8rem; border-radius:20px; display:inline-block; }
.cg-hero-title { margin:1.1rem 0 1.3rem; font-size:3rem; line-height:1.1; font-weight:700;
    color:#0a2530; letter-spacing:-.025em; padding:0; }
.cg-hero-title em { font-style:italic; color:#126b60;
    background:linear-gradient(transparent 60%, rgba(22,118,106,.14) 60%); padding:0 .06em;
    text-decoration-skip-ink:none; }
.cg-lead { max-width:520px; color:#3d5a68; font-size:1.04rem; line-height:1.8; margin:0; font-weight:400; }
.cg-meta { display:flex; flex-wrap:wrap; align-items:center; gap:.8rem; margin-top:2rem;
    color:#5b727d; font-size:.78rem; font-weight:600; }
.cg-meta span { background:rgba(22,118,106,.06); padding:.28rem .65rem; border-radius:14px;
    transition: background .2s ease; }
.cg-meta span:hover { background:rgba(22,118,106,.1); }
.cg-meta i { width:3px; height:3px; border-radius:50%; background:#a3bcc4; display:inline-block; }

/* ── Section kickers & headings ── */
.cg-kicker { color:#16766a; font-size:.68rem; font-weight:700; letter-spacing:.2em;
    text-transform:uppercase; margin:4.5rem 0 .55rem;
    display:inline-block; background:rgba(22,118,106,.06); padding:.28rem .7rem; border-radius:14px; }
.cg-h2 { font-size:1.9rem; font-weight:700; color:#0a2530; line-height:1.18;
    margin:0 0 .55rem; letter-spacing:-.02em; }
.cg-sub { color:#4d6a78; max-width:620px; line-height:1.78; margin:0 0 1.6rem; font-size:.98rem; font-weight:400; }

/* ── Feature cards ── */
.cg-feat { animation:cgfade .65s ease both; padding:1.5rem 1.5rem 1.5rem 1.5rem;
    border-radius:16px; position:relative; overflow:hidden;
    background:rgba(255,255,255,.62); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    box-shadow: 0 1px 2px rgba(18,48,63,.03), 0 4px 16px rgba(18,48,63,.035);
    transition: transform .3s cubic-bezier(.4,0,.2,1), box-shadow .3s cubic-bezier(.4,0,.2,1); }
.cg-feat::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg, #16766a, #2f6db5); border-radius:3px 0 0 3px; opacity:.7;
    transition: opacity .3s ease, width .3s ease; }
.cg-feat:hover { transform:translateY(-5px); box-shadow: 0 8px 32px rgba(18,48,63,.08); }
.cg-feat:hover::before { opacity:1; width:4px; }
.cg-feat-icon { width:40px; height:40px; border-radius:10px;
    background:linear-gradient(135deg, rgba(22,118,106,.1), rgba(47,109,181,.08));
    display:flex; align-items:center; justify-content:center; margin-bottom:.6rem; flex-shrink:0; }
.cg-feat-icon svg { width:20px; height:20px; }
.cg-feat-num { color:#16766a; font-weight:700; font-size:.72rem; letter-spacing:.1em;
    background:rgba(22,118,106,.07); padding:.18rem .5rem; border-radius:6px; display:inline-block; }
.cg-feat h3 { margin:.5rem 0 .4rem; color:#0a2530; font-size:1.06rem; font-weight:700; padding:0;
    letter-spacing:-.005em; }
.cg-feat p  { margin:0; color:#4d6a78; line-height:1.72; font-size:.92rem; font-weight:400; }

/* ── Path cards ── */
.cg-path { padding:1.5rem; border-radius:16px; margin-bottom:1rem; animation:cgfade .65s ease both;
    border-left:3px solid #16766a; border-top:none;
    background:rgba(255,255,255,.55); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
    box-shadow: 0 1px 3px rgba(18,48,63,.025), 0 4px 14px rgba(18,48,63,.035); }
.cg-path.v2 { border-left-color:#2f6db5; }
.cg-path-label { color:#16766a; font-size:.68rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.cg-path.v2 .cg-path-label { color:#2f6db5; }
.cg-path h3 { margin:.4rem 0 .5rem; font-size:1.4rem; color:#0a2530; font-weight:600; padding:0;
    font-family:'Playfair Display','Iowan Old Style','Palatino Linotype',Georgia,serif; }
.cg-path p  { margin:0 0 .7rem; min-height:3rem; color:#3d5a68; line-height:1.72; font-size:.94rem; }
.cg-path ul { margin:0 0 .5rem; padding-left:1.05rem; color:#4d6a78; line-height:1.85; font-size:.91rem; }

/* ── Step timeline cards ── */
.cg-step { padding:1.2rem 1.3rem; border-radius:14px; position:relative;
    background:rgba(255,255,255,.58); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    box-shadow: 0 1px 2px rgba(18,48,63,.025), 0 3px 12px rgba(18,48,63,.035);
    animation:cgfade .6s ease both;
    transition: transform .25s cubic-bezier(.4,0,.2,1), box-shadow .25s cubic-bezier(.4,0,.2,1); }
.cg-step::after { content:''; position:absolute; bottom:0; left:12%; right:12%; height:2px;
    background:linear-gradient(90deg, transparent, rgba(22,118,106,.12), transparent); }
.cg-step:hover { transform:translateY(-3px); box-shadow: 0 6px 24px rgba(18,48,63,.07); }
.cg-step-num { color:#fff; font-size:.72rem; font-weight:700; letter-spacing:.06em;
    background:linear-gradient(135deg, #16766a, #1a8a7c); width:30px; height:30px; border-radius:10px;
    display:inline-flex; align-items:center; justify-content:center;
    box-shadow: 0 2px 8px rgba(22,118,106,.2); }
.cg-step h4  { margin:.5rem 0 .3rem; color:#0a2530; font-size:.98rem; font-weight:700; padding:0; letter-spacing:-.005em; }
.cg-step p   { margin:0; color:#4d6a78; font-size:.88rem; line-height:1.7; font-weight:400; }

/* ── Notice / disclaimer ── */
.cg-notice { margin:3.5rem 0 .5rem; padding:1.2rem 1.4rem; border-radius:14px;
    background:rgba(107,127,137,.04); color:#6b7f89; font-size:.82rem; line-height:1.75;
    max-width:900px; border-left:3px solid rgba(107,127,137,.15); }

/* ── Footer ── */
.cg-footer { margin-top:4rem; padding:2.2rem 0 1rem; border-top:1px solid rgba(18,48,63,.06);
    text-align:center; animation:cgfade .7s ease both; }
.cg-footer-brand { font-size:.78rem; font-weight:700; letter-spacing:.1em; color:#16766a;
    text-transform:uppercase; margin-bottom:.4rem; }
.cg-footer-sub { font-size:.76rem; color:#8a9faa; line-height:1.6; max-width:500px; margin:0 auto; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #16766a 0%, #12594f 60%, #0e4a42 100%);
    border:none; border-radius:12px; font-weight:600; padding:.8rem 1.6rem;
    box-shadow: 0 2px 6px rgba(18,89,79,.12), 0 6px 20px rgba(18,89,79,.16);
    transition: all .3s cubic-bezier(.4,0,.2,1);
    font-family:'Inter',sans-serif; letter-spacing:.01em; font-size:.92rem; }
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #12594f, #0e4a42, #0a3a34);
    transform:translateY(-2px); box-shadow: 0 4px 12px rgba(18,89,79,.15), 0 10px 30px rgba(18,89,79,.22); }
.stButton > button[kind="primary"]:active { transform:translateY(0); }

@media (max-width:900px) {
    .cg-top   { flex-direction:column; align-items:flex-start; gap:1rem; }
    .cg-hero2 { padding:2.4rem 1.4rem; border-radius:18px; }
    .cg-hero-art  { display:none; }
    .cg-hero-title { font-size:2.1rem; }
    .cg-feat, .cg-step { padding:1.1rem; }
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
                f'<div class="cg-feat">'
                f'<div class="cg-feat-num">{number}</div>'
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
    steps = st.columns(5, gap="medium")
    step_items = [
        ("1", "Pelayanan & Pengajuan", "Peserta menerima layanan; faskes menyiapkan dan mengajukan klaim."),
        ("2", "Analisis ClaimGuard", "Klaim dianalisis sinyal statistik dan diurutkan dalam antrean prioritas."),
        ("3", "Pemeriksaan Petugas", "Verifikator memeriksa dokumen asli dan alasan pembanding dari sistem."),
        ("4", "Siklus Klarifikasi", "Jika ada keraguan, klarifikasi diminta ke faskes dan ditinjau ulang."),
        ("5", "Proses Resmi", "Klaim yang wajar dan lengkap dicatat diteruskan ke sistem resmi."),
    ]
    for column, (number, title, description) in zip(steps, step_items):
        with column:
            st.markdown(
                f'<div class="cg-step"><div class="cg-step-num">{number}</div>'
                f'<h4>{title}</h4><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="cg-notice"><strong>Bukan portal resmi BPJS Kesehatan.</strong> '
        'Logo ditampilkan sebagai identitas konteks pada mockup, bukan pernyataan afiliasi, '
        'persetujuan, atau integrasi. Seluruh data yang digunakan adalah data sintetis, '
        'dan hasilnya bukan bukti kecurangan.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="cg-footer">'
        '<div class="cg-footer-brand">ClaimGuard</div>'
        '<div class="cg-footer-sub">'
        'Prototipe demonstrasi &middot; Data sintetis &middot; '
        'Tidak terhubung dengan sistem klaim resmi'
        '</div></div>',
        unsafe_allow_html=True,
    )


def _render_workflow_stepper(claim: dict, priority_row: dict, status: str) -> None:
    is_clarifying = status == WAITING_FOR_FACILITY_STATUS
    is_recheck = status == WAITING_FOR_STAFF_STATUS
    is_official = status == OFFICIAL_PROCESS_STATUS
    is_initial = status == INITIAL_STATUS

    s1_sub = claim.get("service_date") or "Tercatat"
    s2_sub = claim.get("submitted_at") or "Diajukan"
    rank = priority_row.get("priority_rank", "-")
    s3_sub = f"Prioritas #{rank}"

    if is_initial:
        s4_cls = "active"
        s4_icon = "4"
        s4_sub = "Sedang ditinjau"
    elif is_recheck:
        s4_cls = "active"
        s4_icon = "4"
        s4_sub = "Pemeriksaan ulang"
    else:
        s4_cls = "done"
        s4_icon = "✓"
        s4_sub = "Telah diperiksa"

    if is_official:
        s5_cls = "done"
        s5_icon = "✓"
        s5_sub = "Diteruskan resmi"
    elif is_clarifying:
        s5_cls = "clarify"
        s5_icon = "5"
        s5_sub = "Menunggu faskes"
    elif is_recheck:
        s5_cls = "pending"
        s5_icon = "5"
        s5_sub = "Verifikasi ulang"
    else:
        s5_cls = "pending"
        s5_icon = "5"
        s5_sub = "Menunggu review"

    line34_cls = "done" if not is_initial else ""
    line45_cls = "done" if is_official else ("clarify" if is_clarifying else "")

    html_stepper = f"""
    <div class="cg-stepper">
        <div class="cg-stepper-item done">
            <div class="cg-stepper-icon">✓</div>
            <div class="cg-stepper-text">
                <span class="cg-stepper-label">1. Pelayanan Peserta</span>
                <span class="cg-stepper-sub">{html.escape(str(s1_sub))}</span>
            </div>
        </div>
        <div class="cg-stepper-line done"></div>
        <div class="cg-stepper-item done">
            <div class="cg-stepper-icon">✓</div>
            <div class="cg-stepper-text">
                <span class="cg-stepper-label">2. Pengajuan Faskes</span>
                <span class="cg-stepper-sub">{html.escape(str(s2_sub))}</span>
            </div>
        </div>
        <div class="cg-stepper-line done"></div>
        <div class="cg-stepper-item done">
            <div class="cg-stepper-icon">✓</div>
            <div class="cg-stepper-text">
                <span class="cg-stepper-label">3. Analisis ClaimGuard</span>
                <span class="cg-stepper-sub">{html.escape(str(s3_sub))}</span>
            </div>
        </div>
        <div class="cg-stepper-line {line34_cls}"></div>
        <div class="cg-stepper-item {s4_cls}">
            <div class="cg-stepper-icon">{s4_icon}</div>
            <div class="cg-stepper-text">
                <span class="cg-stepper-label">4. Review Verifikator</span>
                <span class="cg-stepper-sub">{html.escape(str(s4_sub))}</span>
            </div>
        </div>
        <div class="cg-stepper-line {line45_cls}"></div>
        <div class="cg-stepper-item {s5_cls}">
            <div class="cg-stepper-icon">{s5_icon}</div>
            <div class="cg-stepper-text">
                <span class="cg-stepper-label">5. Tindak Lanjut Resmi</span>
                <span class="cg-stepper-sub">{html.escape(str(s5_sub))}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html_stepper, unsafe_allow_html=True)


def _render_claim_workflow(
    claim_id: str,
    analysis_source: str,
    analysis_path: str,
    split_name: str,
    assistant_content: dict | None = None,
) -> None:
    if assistant_content is None:
        assistant_content = {
            "findings": ["Hasil analisis klaim."],
            "suggested_evidence": ["Dokumen pendukung"],
            "clarification_request": "Mohon konfirmasi kelengkapan klaim.",
        }

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
    settings = _load_settings()
    with st.expander("Ringkasan temuan dan saran ClaimGuard", expanded=settings["assistant_expanded"]):
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

    # Role banner reflecting the current workflow actor
    if status == INITIAL_STATUS:
        st.markdown(
            '<div class="cg-role-banner verifikator">'
            '<span class="cg-role-badge verifikator">VERIFIKATOR</span>'
            '<div><div class="cg-role-title">Tahap: Pemeriksaan Petugas BPJS Kesehatan</div>'
            '<div class="cg-role-desc">Klaim siap ditinjau. Teliti dokumen fisik/asli berdasarkan petunjuk AI ClaimGuard, lalu tentukan apakah klaim wajar atau membutuhkan klarifikasi ke faskes.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    elif status == WAITING_FOR_FACILITY_STATUS:
        st.markdown(
            '<div class="cg-role-banner faskes">'
            '<span class="cg-role-badge faskes">FASKES</span>'
            '<div><div class="cg-role-title">Tahap: Menunggu Tanggapan Fasilitas Kesehatan (Simulasi)</div>'
            '<div class="cg-role-desc">Petugas telah mencatat permintaan klarifikasi. Pada simulasi ini, Anda bertindak sebagai pihak Faskes (RS/Klinik) untuk memberikan respons dan kelengkapan dokumen balik ke verifikator.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    elif status == WAITING_FOR_STAFF_STATUS:
        st.markdown(
            '<div class="cg-role-banner verifikator">'
            '<span class="cg-role-badge verifikator">VERIFIKATOR</span>'
            '<div><div class="cg-role-title">Tahap: Pemeriksaan Ulang Tanggapan Faskes oleh Petugas</div>'
            '<div class="cg-role-desc">Faskes telah memberikan dokumen/penjelasan. Verifikator meninjau apakah informasi tersebut sudah cukup untuk diteruskan ke proses resmi atau butuh klarifikasi tambahan.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    elif status == OFFICIAL_PROCESS_STATUS:
        st.markdown(
            '<div class="cg-role-banner official">'
            '<span class="cg-role-badge official">PROSES RESMI</span>'
            '<div><div class="cg-role-title">Tahap Selesai: Klaim Telah Diteruskan ke Proses Resmi</div>'
            '<div class="cg-role-desc">Pemeriksaan di ClaimGuard telah tuntas dicatat. Berkas diteruskan mengikuti kanal resmi; persetujuan dan pembayaran tetap mengikuti kewenangan resmi.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.metric("Status tindak lanjut saat ini", status)

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
            "Hasil pemeriksaan petugas" if is_staff_review else "Pilihan tanggapan pihak faskes (simulasi)",
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
        else:
            if "dokumen" in action_label.casefold():
                default_notes = (
                    "Fasilitas kesehatan telah melengkapi resume medis dan berkas penunjang yang diminta "
                    "untuk diverifikasi ulang oleh petugas."
                )
            elif "penjelasan" in action_label.casefold():
                default_notes = (
                    "Fasilitas kesehatan memberikan penjelasan klinis mengenai kesesuaian diagnosis dan tindakan "
                    "yang telah dilakukan sesuai standar prosedur operasional."
                )
            else:
                default_notes = (
                    "Fasilitas kesehatan telah mencatat koreksi administratif terhadap rincian klaim sesuai masukan verifikator."
                )

        with st.form(f"workflow_form_{analysis_source}_{analysis_path}_{claim_id}_{split_name}"):
            if is_staff_review:
                role_options = list(REVIEWER_ROLES)
                reviewer_role = st.selectbox(
                    "Peran pemeriksa",
                    role_options,
                    index=(
                        role_options.index(settings["default_reviewer_role"])
                        if settings["default_reviewer_role"] in role_options
                        else 0
                    ),
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
            notes_label = "Catatan pemeriksa" if is_staff_review else "Rincian tanggapan faskes (penjelasan atau dokumen)"
            notes = st.text_area(
                notes_label,
                value=default_notes,
                max_chars=2000,
                key=notes_key,
            )
            submitted = st.form_submit_button(
                "Simpan hasil pemeriksaan verifikator" if is_staff_review else "Kirim tanggapan faskes (simulasi)"
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
        st.markdown(
            '<div class="cg-thread">'
            '<div class="cg-thread-title">📋 Riwayat Disposisi & Korespondensi Klaim</div>',
            unsafe_allow_html=True,
        )
        thread_html = []
        for _, step in claim_workflow.sort_values("recorded_at_utc", kind="stable").iterrows():
            actor_name = str(step["actor"])
            is_staff = actor_name == "Petugas"
            dot_cls = "official" if "resmi" in str(step.get("action", "")).casefold() else ("faskes" if not is_staff else "")
            badge_cls = "official" if "resmi" in str(step.get("action", "")).casefold() else ("faskes" if not is_staff else "verifikator")
            actor_role_tag = f'<span class="cg-role-badge {badge_cls}">{html.escape(actor_name)}</span>'
            time_str = str(step.get("recorded_at_utc", ""))[:19]
            role_text = f" &middot; {step['reviewer_role']}" if is_staff and step.get("reviewer_role") else ""
            notes_esc = html.escape(str(step.get("notes", "")))
            action_esc = html.escape(str(step.get("action", "")))
            evidence_val = step.get("evidence_reviewed")
            evidence_esc = (
                f"<div style='font-size:.76rem; color:#7a919c; margin-top:3px;'>"
                f"Bukti ditinjau: {html.escape(str(evidence_val))}</div>"
                if evidence_val
                else ""
            )

            thread_html.append(
                f'<div class="cg-thread-item">'
                f'<div class="cg-thread-dot {dot_cls}"></div>'
                f'<div class="cg-thread-header">'
                f'<span class="cg-thread-actor">{actor_role_tag}{html.escape(role_text)}</span>'
                f'<span class="cg-thread-time">{time_str} UTC</span>'
                f'</div>'
                f'<div class="cg-thread-action">{action_esc}</div>'
                f'{evidence_esc}'
                f'<div class="cg-thread-notes">{notes_esc}</div>'
                f'</div>'
            )
        st.markdown("".join(thread_html) + "</div>", unsafe_allow_html=True)
        st.caption(
            f"{len(claim_workflow)} langkah simulasi tercatat untuk klaim ini. "
            "Riwayat lengkap beserta alur waktunya ada di menu Laporan."
        )
        st.button(
            "Lihat riwayat di Laporan",
            key=f"open_report_{analysis_source}_{analysis_path}_{claim_id}_{split_name}",
            on_click=_open_report,
            args=(str(claim_id),),
        )


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
    settings = _load_settings()
    st.session_state.setdefault("v2_scheme", settings["default_scheme_label"])
    st.session_state.setdefault("v2_partition", settings["default_partition_label"])
    st.session_state.setdefault("v2_recommended", settings["default_recommended_only"])
    with st.sidebar:
        scheme_label = st.selectbox("Cara mengelompokkan klaim", list(V2_SPLIT_OPTIONS), key="v2_scheme")
        scheme, split_column = V2_SPLIT_OPTIONS[scheme_label]
        partition_label = st.selectbox("Pilih kumpulan data", list(V2_PARTITIONS), key="v2_partition")
        partition = V2_PARTITIONS[partition_label]
        recommended_only = st.checkbox(
            "Hanya tampilkan klaim yang disarankan untuk ditinjau",
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
    workflow_log = read_workflow_log(WORKFLOW_LOG, LEGACY_WORKFLOW_LOG)
    current_status = get_workflow_status(
        workflow_log,
        "v2",
        scheme,
        selected_id,
        partition,
    )
    _render_workflow_stepper(claim.to_dict(), row.to_dict(), current_status)
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
    page_icon=_get_page_icon(),
    layout="wide",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800;14..32,900&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

    /* ════════════════════════════════════════════════
       BASE LAYER
       ════════════════════════════════════════════════ */
    .stApp {
        background:
            radial-gradient(ellipse 1000px 500px at 85% -5%, rgba(22,118,106,.055), transparent 58%),
            radial-gradient(ellipse 800px 450px at -5% 35%, rgba(47,109,181,.04), transparent 52%),
            linear-gradient(172deg, #eef3f4 0%, #f4f7f8 35%, #f9fafb 100%);
        color: #18364b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1380px;
        padding-top: 1.6rem;
    }

    /* ════════════════════════════════════════════════
       SIDEBAR — frosted glass
       ════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,.78) 0%, rgba(248,250,251,.82) 100%) !important;
        backdrop-filter: blur(24px) saturate(1.5);
        -webkit-backdrop-filter: blur(24px) saturate(1.5);
        border-right: none !important;
        box-shadow: 1px 0 24px rgba(18,48,63,.035);
    }
    [data-testid="stSidebar"]::after {
        content: ''; position: absolute; left:0; top:0; bottom:0; width:3px;
        background: linear-gradient(180deg, #16766a 0%, #2f6db5 50%, #16766a 100%);
        opacity: .35; border-radius: 0 2px 2px 0;
    }
    [data-testid="stSidebar"] * { color: #18364b; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* ── option_menu ── */
    .nav-link {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: .86rem !important;
        border-radius: 10px !important;
        margin: 1px 4px !important;
        padding: .52rem .85rem !important;
        transition: all .25s cubic-bezier(.4,0,.2,1) !important;
        position: relative;
    }
    .nav-link:hover {
        background: rgba(22,118,106,.05) !important;
        transform: translateX(3px);
    }
    .nav-link-selected {
        background: rgba(22,118,106,.07) !important;
        color: #16766a !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: inset 3px 0 0 #16766a !important;
    }

    /* ════════════════════════════════════════════════
       TYPOGRAPHY
       ════════════════════════════════════════════════ */
    h1, h2, h3, h4 {
        color: #0a2530;
        font-family: 'Inter', sans-serif;
        letter-spacing: -.015em;
    }
    h1 { font-size: 1.8rem; font-weight: 700; }
    h2 { font-weight: 600; }
    h3 { font-weight: 600; }
    p, li, span, label, .stCaption, .stMarkdown {
        font-family: 'Inter', sans-serif;
    }

    /* ════════════════════════════════════════════════
       METRIC CARDS — seamless, blends with background
       ════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.48) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.7) !important;
        border-radius: 14px !important;
        padding: .95rem 1.15rem !important;
        box-shadow: 0 2px 10px rgba(18, 48, 63, 0.025) !important;
        transition: all .25s ease !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        display: none !important;
        content: none !important;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.7) !important;
        box-shadow: 0 4px 18px rgba(18, 48, 63, 0.045) !important;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        margin-bottom: 3px !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        color: #5a6f7a !important;
        font-weight: 600 !important;
        font-size: .73rem !important;
        line-height: 1.25 !important;
        text-transform: uppercase !important;
        letter-spacing: .04em !important;
        white-space: normal !important;
        text-overflow: unset !important;
        overflow: visible !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.45rem !important;
        color: #0a2530 !important;
        letter-spacing: -.01em !important;
    }
    [data-testid="stMetricValue"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap;
    }

    /* ════════════════════════════════════════════════
       DATAFRAME
       ════════════════════════════════════════════════ */
    div[data-testid="stDataFrame"] {
        border: none !important;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(18,48,63,.025), 0 3px 14px rgba(18,48,63,.035);
    }

    /* ════════════════════════════════════════════════
       TABS — pill style
       ════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        background: rgba(255,255,255,.45);
        border-radius: 12px;
        padding: 3px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: inset 0 1px 3px rgba(18,48,63,.03);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: .86rem;
        padding: .48rem .95rem;
        transition: all .25s cubic-bezier(.4,0,.2,1);
        color: #5a6f7a;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #16766a !important; }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,.9) !important;
        color: #16766a !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 4px rgba(18,48,63,.06);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ════════════════════════════════════════════════
       EXPANDER
       ════════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: .9rem;
        border-radius: 12px;
    }
    details {
        border: none !important;
        background: rgba(255,255,255,.45) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 1px 2px rgba(18,48,63,.02), 0 2px 10px rgba(18,48,63,.03);
    }

    /* ════════════════════════════════════════════════
       FORM INPUTS
       ════════════════════════════════════════════════ */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        border-radius: 10px !important;
        border-color: rgba(18,48,63,.09) !important;
        background: rgba(255,255,255,.5) !important;
        transition: all .25s cubic-bezier(.4,0,.2,1);
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {
        border-color: #16766a !important;
        box-shadow: 0 0 0 3px rgba(22,118,106,.07) !important;
        background: rgba(255,255,255,.7) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #16766a !important;
    }
    .stCheckbox label span[data-testid="stCheckbox"] {
        border-radius: 5px;
    }

    /* ════════════════════════════════════════════════
       BUTTONS
       ════════════════════════════════════════════════ */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        border-radius: 10px;
        font-weight: 500;
        font-size: .88rem;
        transition: all .25s cubic-bezier(.4,0,.2,1);
        border: 1px solid rgba(18,48,63,.07);
        background: rgba(255,255,255,.5);
    }
    .stButton > button:hover {
        border-color: rgba(22,118,106,.18);
        box-shadow: 0 2px 10px rgba(22,118,106,.07);
        background: rgba(255,255,255,.75);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #16766a 0%, #12594f 60%, #0e4a42 100%);
        border: none; border-radius: 12px; font-weight: 600;
        box-shadow: 0 2px 6px rgba(18,89,79,.1), 0 6px 20px rgba(18,89,79,.15);
        transition: all .3s cubic-bezier(.4,0,.2,1);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #12594f, #0e4a42, #0a3a34);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(18,89,79,.12), 0 10px 30px rgba(18,89,79,.2);
    }
    .stButton > button[kind="primary"]:active { transform: translateY(0); }

    /* forms */
    .stForm { border: none !important; padding: 0 !important; }
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #16766a, #12594f) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(22,118,106,.15);
    }

    /* ════════════════════════════════════════════════
       ALERTS
       ════════════════════════════════════════════════ */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    /* ════════════════════════════════════════════════
       MISC
       ════════════════════════════════════════════════ */
    hr { border-color: rgba(18,48,63,.05) !important; }

    /* scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(18,48,63,.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(18,48,63,.18); }

    /* page-enter animation */
    @keyframes cgPageFade { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }
    .main .block-container { animation: cgPageFade .35s ease both; }

    /* captions */
    .stCaption p {
        font-size: .8rem !important;
        color: #7a919c !important;
    }

    /* multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(22,118,106,.08) !important;
        border-radius: 6px !important;
    }

    /* ════════════════════════════════════════════════
       WORKFLOW STEPPER
       ════════════════════════════════════════════════ */
    .cg-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255,255,255,.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin: .8rem 0 1.4rem;
        box-shadow: 0 1px 3px rgba(18,48,63,.03), 0 4px 16px rgba(18,48,63,.04);
        position: relative;
        overflow-x: auto;
    }
    .cg-stepper-item {
        display: flex;
        align-items: center;
        gap: .7rem;
        flex: 1;
        min-width: 130px;
    }
    .cg-stepper-icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: .82rem;
        font-weight: 700;
        flex-shrink: 0;
        transition: all .25s ease;
    }
    .cg-stepper-item.done .cg-stepper-icon {
        background: linear-gradient(135deg, #16766a, #115e54);
        color: white;
        box-shadow: 0 2px 8px rgba(22,118,106,.25);
    }
    .cg-stepper-item.active .cg-stepper-icon {
        background: linear-gradient(135deg, #2f6db5, #1e4e87);
        color: white;
        box-shadow: 0 0 0 4px rgba(47,109,181,.18), 0 2px 8px rgba(47,109,181,.3);
        animation: cgPulse 2s infinite;
    }
    .cg-stepper-item.clarify .cg-stepper-icon {
        background: linear-gradient(135deg, #d97706, #b45309);
        color: white;
        box-shadow: 0 0 0 4px rgba(217,119,6,.18), 0 2px 8px rgba(217,119,6,.3);
        animation: cgPulse 2s infinite;
    }
    .cg-stepper-item.pending .cg-stepper-icon {
        background: rgba(18,48,63,.08);
        color: #7a919c;
    }
    .cg-stepper-text {
        display: flex;
        flex-direction: column;
    }
    .cg-stepper-label {
        font-size: .78rem;
        font-weight: 600;
        color: #12303f;
        line-height: 1.2;
    }
    .cg-stepper-sub {
        font-size: .7rem;
        color: #7a919c;
        margin-top: 2px;
    }
    .cg-stepper-line {
        height: 2px;
        flex: 0 0 24px;
        margin: 0 8px;
        border-radius: 1px;
        background: rgba(18,48,63,.1);
    }
    .cg-stepper-line.done {
        background: #16766a;
    }
    .cg-stepper-line.clarify {
        background: #d97706;
    }
    @keyframes cgPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.06); }
    }

    /* ════════════════════════════════════════════════
       ROLE CARDS & AUDIT THREAD
       ════════════════════════════════════════════════ */
    .cg-role-banner {
        border-radius: 12px;
        padding: .85rem 1.15rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: .8rem;
    }
    .cg-role-banner.verifikator {
        background: linear-gradient(135deg, rgba(22,118,106,.08), rgba(47,109,181,.06));
        border-left: 4px solid #16766a;
    }
    .cg-role-banner.faskes {
        background: linear-gradient(135deg, rgba(217,119,6,.1), rgba(245,158,11,.06));
        border-left: 4px solid #d97706;
    }
    .cg-role-banner.official {
        background: linear-gradient(135deg, rgba(16,185,129,.1), rgba(5,150,105,.06));
        border-left: 4px solid #10b981;
    }
    .cg-role-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 6px;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .cg-role-badge.verifikator {
        background: rgba(22, 118, 106, 0.12);
        color: #12594f;
    }
    .cg-role-badge.faskes {
        background: rgba(217, 119, 6, 0.14);
        color: #b45309;
    }
    .cg-role-badge.official {
        background: rgba(16, 185, 129, 0.14);
        color: #047857;
    }
    .cg-role-title {
        font-weight: 700;
        font-size: .88rem;
        color: #0a2530;
    }
    .cg-role-desc {
        font-size: .78rem;
        color: #5a6f7a;
        margin-top: 2px;
        line-height: 1.4;
    }

    /* Thread Timeline */
    .cg-thread {
        margin-top: 1.5rem;
        padding: 1.2rem;
        background: rgba(255,255,255,.55);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(18,48,63,.03);
    }
    .cg-thread-title {
        font-size: .84rem;
        font-weight: 700;
        color: #12303f;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: 1rem;
    }
    .cg-thread-item {
        position: relative;
        padding-left: 1.8rem;
        padding-bottom: 1.2rem;
        border-left: 2px solid rgba(18,48,63,.1);
    }
    .cg-thread-item:last-child {
        border-left: 2px solid transparent;
        padding-bottom: 0;
    }
    .cg-thread-dot {
        position: absolute;
        left: -7px;
        top: 2px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #16766a;
        box-shadow: 0 0 0 3px rgba(22,118,106,.2);
    }
    .cg-thread-dot.faskes {
        background: #d97706;
        box-shadow: 0 0 0 3px rgba(217,119,6,.2);
    }
    .cg-thread-dot.official {
        background: #10b981;
        box-shadow: 0 0 0 3px rgba(16,185,129,.2);
    }
    .cg-thread-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: .25rem;
    }
    .cg-thread-actor {
        font-weight: 600;
        font-size: .84rem;
        color: #0a2530;
    }
    .cg-thread-time {
        font-size: .74rem;
        color: #7a919c;
    }
    .cg-thread-action {
        font-size: .82rem;
        font-weight: 600;
        color: #16766a;
        margin-bottom: .35rem;
    }
    .cg-thread-notes {
        font-size: .82rem;
        color: #435b66;
        background: rgba(255,255,255,.6);
        padding: .5rem .75rem;
        border-radius: 8px;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    nav_menu = option_menu(
        "Menu utama",
        ["Beranda", "Pemeriksaan klaim", "Analitik", "Laporan", "Manajemen", "Pengaturan"],
        icons=["house", "check2-circle", "bar-chart-line", "file-earmark-text", "kanban", "sliders"],
        menu_icon="grid-fill",
        default_index=0,
        key="nav_menu",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#5a6f7a", "font-size": "14px"},
            "nav-link": {
                "font-size": "13px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "8px 12px",
                "border-radius": "9px",
                "color": "#334e5c",
                "--hover-color": "rgba(22, 118, 106, 0.07)",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #16766a 0%, #12594f 100%)",
                "color": "#ffffff",
                "font-weight": "600",
                "box-shadow": "0 2px 8px rgba(22, 118, 106, 0.2)",
            },
        },
    )

if nav_menu == "Beranda":
    _render_landing_page()
elif nav_menu == "Pemeriksaan klaim":
    st.sidebar.button("Kembali ke beranda", width="stretch", on_click=_go_home)
    _render_header()
    _render_v2()
elif nav_menu == "Analitik":
    _render_analytics()
elif nav_menu == "Laporan":
    _render_reports()
elif nav_menu == "Manajemen":
    _render_management()
elif nav_menu == "Pengaturan":
    _render_settings()
else:
    _render_unavailable_page(nav_menu)

st.caption(
    "Prototipe demonstrasi ClaimGuard. Tidak terhubung dengan BPJS Kesehatan atau sistem klaim "
    "resmi. Analisis memakai data sintetis; hasilnya hanya petunjuk untuk pemeriksaan petugas."
)