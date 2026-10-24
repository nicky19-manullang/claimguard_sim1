"""Bagian bersama untuk halaman-halaman ClaimGuard (konstanta, data, grafik, pengaturan)."""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from claimguard_workflow_demo import (
    EVIDENCE_ITEMS,
    RECOMMENDATION_FEEDBACK,
    REVIEWER_ROLES,
    WORKFLOW_ACTIONS,
    get_workflow_status,
    read_workflow_log,
    record_workflow_action,
)

ROOT = Path(__file__).resolve().parent.parent
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




# ---------------------------------------------------------------------------
# Halaman Analitik, Laporan, Manajemen, dan Pengaturan
# ---------------------------------------------------------------------------
BRAND = {
    "teal": "#16766a",
    "blue": "#2f6db5",
    "amber": "#c98a1a",
    "rose": "#b4506b",
    "grey": "#9bb0b8",
    "slate": "#6b7f89",
}
KIND_ORDER = ["Belum ditinjau", "Menunggu petugas", "Menunggu faskes", "Selesai (diteruskan)", "Lainnya"]
KIND_COLORS = {
    "Belum ditinjau": BRAND["grey"],
    "Menunggu petugas": BRAND["amber"],
    "Menunggu faskes": BRAND["blue"],
    "Selesai (diteruskan)": BRAND["teal"],
    "Lainnya": BRAND["slate"],
}
WORKFLOW_COLUMNS = [
    "recorded_at_utc",
    "analysis_source",
    "analysis_path",
    "claim_id",
    "split_name",
    "actor",
    "reviewer_role",
    "evidence_reviewed",
    "review_outcome",
    "action",
    "status_after_action",
    "recommendation_feedback",
    "notes",
]
DEFAULT_SETTINGS = {
    "default_scheme_label": DEFAULT_SCHEME_LABEL,
    "default_partition_label": DEFAULT_PARTITION_LABEL,
    "default_recommended_only": True,
    "overdue_days": 3,
    "default_reviewer_role": "",
    "assistant_expanded": True,
}
SETTINGS_FILE = OUT / "dashboard_settings.json"
BACKUP_DIR = OUT / "backup_riwayat"
JAKARTA_OFFSET = pd.Timedelta(hours=7)

PAGE_CSS = """
<style>
.stApp {
    background:
        radial-gradient(ellipse 1000px 500px at 85% -5%, rgba(22,118,106,.055), transparent 58%),
        radial-gradient(ellipse 800px 450px at -5% 35%, rgba(47,109,181,.04), transparent 52%),
        linear-gradient(172deg, #eef3f4 0%, #f4f7f8 35%, #f9fafb 100%) !important;
}
@keyframes cgfade { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
.cg-ph { margin:.2rem 0 1.2rem; animation:cgfade .4s ease both; }
.cg-ph-kicker { color:#16766a; font-size:.74rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.cg-ph-title { margin:.3rem 0 .4rem; color:#0a2530; font-size:2rem; line-height:1.2; font-weight:700;
    letter-spacing:-.015em; font-family:'Inter', sans-serif; }
.cg-ph-desc { max-width:800px; margin:0; color:#5a6f7a; font-size:.92rem; line-height:1.65; }
.cg-sec { margin:1.4rem 0 .3rem; color:#0a2530; font-size:1.05rem; font-weight:700; }
.cg-hint { color:#6b7f89; font-size:.82rem; line-height:1.55; margin:.1rem 0 .6rem; }

/* ════════════════════════════════════════════════
   SEAMLESS METRIC CARDS (NO HARSH BORDERS)
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
    color: #0a2530 !important;
    font-weight: 700 !important;
    font-size: 1.45rem !important;
    letter-spacing: -.01em !important;
}
[data-testid="stMetricValue"] > div {
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: nowrap;
}

/* ════════════════════════════════════════════════
   TABS — TEAL BRANDING (NO RED UNDERLINE)
   ════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: rgba(255,255,255,.45) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}
.stTabs [data-baseweb="tab"] {
    color: #5a6f7a !important;
    font-weight: 500 !important;
    font-size: .86rem !important;
    padding: .45rem .95rem !important;
    border-radius: 9px !important;
    border: none !important;
    transition: all .2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #16766a !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,.85) !important;
    color: #16766a !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 4px rgba(18,48,63,.06) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Timeline */
.cg-tl { position:relative; margin:.6rem 0 1rem; padding-left:1.4rem; border-left:2px solid rgba(18,48,63,.1); }
.cg-tl-item { position:relative; padding:0 0 1.15rem .4rem; }
.cg-tl-item::before { content:""; position:absolute; left:calc(-1.4rem - 6px); top:.3rem; width:10px; height:10px;
    border-radius:50%; background:#16766a; box-shadow:0 0 0 4px rgba(22,118,106,.15); }
.cg-tl-time { color:#7a8f99; font-size:.78rem; }
.cg-tl-title { margin:.1rem 0; color:#10303e; font-weight:700; font-size:.95rem; }
.cg-tl-meta { color:#5a6f7a; font-size:.85rem; line-height:1.6; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background:linear-gradient(135deg,#16766a,#12594f); border:none; border-radius:10px; font-weight:600;
    box-shadow: 0 2px 8px rgba(22,118,106,.2);
}
</style>
"""


def _page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="cg-ph">'
        f'<div class="cg-ph-kicker">{html.escape(kicker)}</div>'
        f'<div class="cg-ph-title">{html.escape(title)}</div>'
        f'<p class="cg-ph-desc">{html.escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def _section(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="cg-sec">{html.escape(title)}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<div class="cg-hint">{html.escape(hint)}</div>', unsafe_allow_html=True)


def _format_percent(value) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    return f"{value:.1f}%".replace(".", ",")


# ----------------------------- pengaturan --------------------------------
def _load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(stored, dict):
        return settings
    if stored.get("default_scheme_label") in V2_SPLIT_OPTIONS:
        settings["default_scheme_label"] = stored["default_scheme_label"]
    if stored.get("default_partition_label") in V2_PARTITIONS:
        settings["default_partition_label"] = stored["default_partition_label"]
    if isinstance(stored.get("default_recommended_only"), bool):
        settings["default_recommended_only"] = stored["default_recommended_only"]
    if isinstance(stored.get("overdue_days"), int) and 1 <= stored["overdue_days"] <= 60:
        settings["overdue_days"] = stored["overdue_days"]
    if stored.get("default_reviewer_role") in REVIEWER_ROLES:
        settings["default_reviewer_role"] = stored["default_reviewer_role"]
    if isinstance(stored.get("assistant_expanded"), bool):
        settings["assistant_expanded"] = stored["assistant_expanded"]
    return settings


def _save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)


# ------------------------------- grafik ----------------------------------
def _show_chart(chart) -> None:
    st.altair_chart(chart, use_container_width=True)


def _style_chart(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#46606c",
            domainColor="#d3dee1",
            gridColor="#e8eef0",
            tickColor="#d3dee1",
            labelFontSize=12,
        )
        .configure_legend(labelColor="#46606c", labelFontSize=12)
    )


def _bar_chart(frame: pd.DataFrame, category: str, value: str, *, color: str, horizontal: bool = False):
    data = frame[[category, value]].copy()
    tooltip = [
        alt.Tooltip(field=category, type="nominal"),
        alt.Tooltip(field=value, type="quantitative"),
    ]
    if horizontal:
        chart = (
            alt.Chart(data)
            .mark_bar(color=color, cornerRadiusEnd=4, size=18)
            .encode(
                y=alt.Y(
                    field=category, type="nominal", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=280),
                ),
                x=alt.X(field=value, type="quantitative", title=None),
                tooltip=tooltip,
            )
            .properties(height=max(160, 36 * len(data) + 24), padding={"top": 10, "bottom": 10, "left": 5, "right": 15})
        )
    else:
        chart = (
            alt.Chart(data)
            .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(
                    field=category, type="nominal", sort=None, title=None,
                    axis=alt.Axis(labelAngle=0, labelLimit=110),
                ),
                y=alt.Y(field=value, type="quantitative", title=None),
                tooltip=tooltip,
            )
            .properties(height=280, padding={"top": 10, "bottom": 10, "left": 5, "right": 5})
        )
    return _style_chart(chart)


def _line_chart(frame: pd.DataFrame, x: str, y: str, series: str, colors: dict):
    domain = list(colors)
    chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(field=x, type="temporal", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y(field=y, type="quantitative", title=None),
            color=alt.Color(
                field=series, type="nominal",
                scale=alt.Scale(domain=domain, range=[colors[name] for name in domain]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip(field=x, type="temporal", format="%b %Y"),
                alt.Tooltip(field=series, type="nominal"),
                alt.Tooltip(field=y, type="quantitative"),
            ],
        )
        .properties(height=300)
    )
    return _style_chart(chart)


def _donut_chart(frame: pd.DataFrame, category: str, value: str, palette: list[str]):
    chart = (
        alt.Chart(frame)
        .mark_arc(innerRadius=46, outerRadius=76)
        .encode(
            theta=alt.Theta(field=value, type="quantitative"),
            color=alt.Color(
                field=category, type="nominal",
                scale=alt.Scale(range=palette),
                legend=alt.Legend(title=None, orient="bottom", labelLimit=250),
            ),
            tooltip=[
                alt.Tooltip(field=category, type="nominal"),
                alt.Tooltip(field=value, type="quantitative"),
            ],
        )
        .properties(height=280, padding={"top": 18, "bottom": 8, "left": 8, "right": 8})
    )
    return _style_chart(chart)


# ------------------------- data riwayat langkah --------------------------
def _workflow_frame() -> pd.DataFrame:
    """Riwayat langkah simulasi dengan kolom yang dijamin ada dan waktu dalam WIB."""
    raw = read_workflow_log(WORKFLOW_LOG, LEGACY_WORKFLOW_LOG)
    frame = pd.DataFrame(raw).copy() if raw is not None else pd.DataFrame()
    for column in WORKFLOW_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    for column in WORKFLOW_COLUMNS:
        frame[column] = frame[column].fillna("")
    frame["claim_id"] = frame["claim_id"].astype(str)
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at_utc"], errors="coerce", utc=True)
    frame["waktu_wib"] = frame["recorded_at"].dt.tz_convert(None) + JAKARTA_OFFSET
    return frame.sort_values("recorded_at", kind="stable").reset_index(drop=True)


def _status_kinds() -> dict:
    """Kelompokkan setiap status alur kerja menurut siapa yang berikutnya bertindak."""
    kinds = {}
    for status, options in WORKFLOW_ACTIONS.items():
        if not options:
            kinds[status] = "Selesai (diteruskan)"
        elif options[0]["actor"] == "Petugas":
            kinds[status] = "Menunggu petugas"
        else:
            kinds[status] = "Menunggu faskes"
    return kinds


def _kind_of(status, kinds: dict) -> str:
    if status is None or (not isinstance(status, str) and pd.isna(status)) or status == "":
        return "Belum ditinjau"
    return kinds.get(status, "Lainnya")


def _latest_per_claim(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis_source", "analysis_path", "split_name", "claim_id"]
    return frame.drop_duplicates(keys, keep="last")


def _count_evidence(series: pd.Series) -> pd.DataFrame:
    rows = []
    texts = series.astype(str)
    for item in EVIDENCE_ITEMS:
        rows.append({"Bukti": item, "Jumlah pemeriksaan": int(texts.str.contains(item, regex=False).sum())})
    result = pd.DataFrame(rows, columns=["Bukti", "Jumlah pemeriksaan"])
    return result.loc[result["Jumlah pemeriksaan"] > 0]


# ---------------------- cakupan data antrean V2 --------------------------
def _scope_controls(prefix: str) -> dict:
    settings = _load_settings()
    scheme_options = list(V2_SPLIT_OPTIONS)
    partition_options = list(V2_PARTITIONS)
    columns = st.columns([1.3, 1, 1.2], vertical_alignment="bottom")
    with columns[0]:
        scheme_label = st.selectbox(
            "Cara mengelompokkan klaim",
            scheme_options,
            index=scheme_options.index(settings["default_scheme_label"]),
            key=f"{prefix}_scheme",
        )
    with columns[1]:
        partition_label = st.selectbox(
            "Kumpulan data",
            partition_options,
            index=partition_options.index(settings["default_partition_label"]),
            key=f"{prefix}_partition",
        )
    with columns[2]:
        recommended_only = st.checkbox(
            "Hanya klaim yang disarankan ditinjau",
            value=settings["default_recommended_only"],
            key=f"{prefix}_recommended",
        )
    return {
        "scheme_label": scheme_label,
        "scheme": V2_SPLIT_OPTIONS[scheme_label][0],
        "partition_label": partition_label,
        "partition": V2_PARTITIONS[partition_label],
        "recommended_only": recommended_only,
    }


@st.cache_data(show_spinner=False)
def _load_scope(scheme: str, partition: str) -> pd.DataFrame:
    queue, claims, _features, queue_column, _split_column = _load_v2_review_data(scheme)
    scoped = queue.loc[queue[queue_column].eq(partition)].copy()
    scoped["id_klaim"] = scoped["id_klaim"].astype(str)
    scoped["review_recommended"] = scoped["review_recommended"].astype(bool)
    scoped["priority_score"] = pd.to_numeric(scoped["priority_score"], errors="coerce")
    scoped["submitted_at"] = pd.to_datetime(scoped["submitted_at"], errors="coerce")
    extra = claims[["id_klaim", "claim_total_idr", "length_of_stay_days"]].copy()
    extra["id_klaim"] = extra["id_klaim"].astype(str)
    extra["claim_total_idr"] = pd.to_numeric(extra["claim_total_idr"], errors="coerce")
    extra["length_of_stay_days"] = pd.to_numeric(extra["length_of_stay_days"], errors="coerce")
    scoped = scoped.merge(extra, on="id_klaim", how="left")
    scoped["jenis_layanan"] = scoped["service_type"].map(_service_name)
    return scoped.reset_index(drop=True)


# ----------------------------- navigasi ----------------------------------
def _go_page(page: str) -> None:
    st.session_state["nav_menu"] = page


def _open_report(claim_id: str) -> None:
    st.session_state["report_focus_claim"] = str(claim_id)
    st.session_state["nav_menu"] = "Laporan"


def _open_claim(claim_id: str, scheme_label: str, partition_label: str) -> None:
    st.session_state["v2_scheme"] = scheme_label
    st.session_state["v2_partition"] = partition_label
    st.session_state["v2_recommended"] = False
    st.session_state["v2_search"] = ""
    st.session_state["v2_claim"] = str(claim_id)
    st.session_state["nav_menu"] = "Pemeriksaan klaim"


# --------------------------- sumber data ---------------------------------
def _data_source_table(count_rows: bool = False) -> pd.DataFrame:
    prep = OUT / "prep_v2"
    asset = ROOT / "asset"
    sources = [
        ("Antrean V2 (waktu)", prep / V2_QUEUE_FILES["time"][1], True),
        ("Antrean V2 (fasilitas)", prep / V2_QUEUE_FILES["facility"][1], True),
        ("Data klaim V2", prep / "claims_v2_flat_with_splits.csv.gz", True),
        ("Fitur V2 (waktu)", prep / V2_FEATURE_FILES["time"], True),
        ("Fitur V2 (fasilitas)", prep / V2_FEATURE_FILES["facility"], True),
        ("Riwayat langkah simulasi", WORKFLOW_LOG, False),
        ("Riwayat lama (legacy)", LEGACY_WORKFLOW_LOG, False),
        ("Pengaturan dashboard", SETTINGS_FILE, False),
        ("Logo ClaimGuard", asset / "logo_claimrmf.png", False),
        ("Logo BPJS Kesehatan", asset / "logo_bpjs.png", False),
        ("Logo JKN", asset / "logo_jkn.jpeg", False),
    ]
    rows = []
    for name, path, required in sources:
        exists = path.is_file()
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            relative = str(path)
        row = {
            "Sumber": name,
            "Berkas": relative,
            "Status": "Tersedia" if exists else ("Tidak ditemukan" if required else "Belum ada (opsional)"),
            "Ukuran (KB)": round(path.stat().st_size / 1024, 1) if exists else None,
            "Terakhir diubah": (
                datetime.fromtimestamp(path.stat().st_mtime).strftime("%d-%m-%Y %H:%M") if exists else ""
            ),
            "Jumlah baris": None,
        }
        if count_rows and exists and path.suffix in {".gz", ".csv"}:
            try:
                row["Jumlah baris"] = len(_load_csv(str(path)))
            except Exception:
                row["Jumlah baris"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# Ekspor semua nama (termasuk yang berawalan _) agar bisa diimpor dengan "import *".
__all__ = [_n for _n in dir() if not _n.startswith("__")]
