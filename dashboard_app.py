from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageChops

ROOT = Path("out")
ROBUSTNESS_ROOT = Path("robustness")
ASSET_ROOT = Path("asset")
DETECTOR_COLUMNS = ["p_repeat", "p_clone", "p_infl", "p_upc"]
DETECTOR_LABELS = {
    "p_repeat": "Klaim berulang",
    "p_clone": "Kemiripan antar-klaim",
    "p_infl": "Biaya di atas pola",
    "p_upc": "Tingkat keparahan tidak selaras",
}
DETECTOR_GUIDANCE = {
    "Klaim berulang": "Ada pola klaim pasien yang berdekatan dan cukup mirip. Pastikan layanan tersebut bukan kunjungan atau episode yang sah.",
    "Kemiripan antar-klaim": "Ada klaim lain dengan kombinasi informasi yang mirip. Kemiripan dapat terjadi secara wajar pada layanan dengan protokol serupa.",
    "Biaya di atas pola": "Nilai klaim tampak lebih tinggi dari pola pembanding pada data simulasi. Periksa komponen biaya dan bukti layanan.",
    "Tingkat keparahan tidak selaras": "Tingkat keparahan tampak berbeda dari pola diagnosis dan lama rawat pada data simulasi. Pastikan penilaian klinis dan kodingnya sesuai.",
    "Sinyal gabungan": "Beberapa karakteristik klaim tampak tidak biasa secara bersamaan. Periksa detail klaim dan bukti pendukung.",
    "Belum tersedia": "Sinyal utama belum tersedia. Periksa detail klaim dan bukti pendukung secara manual.",
}
TIER_LABELS = {"Critical": "Sangat tinggi", "High": "Tinggi", "Medium": "Menengah", "Low": "Rendah"}
CLAIM_COLUMNS = [
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
EXPLANATION_COLUMNS = [
    "claim_id",
    "top_driver_mode",
    "top_driver",
    "top_feature",
    "top_feature_value",
    "explanation",
]

st.set_page_config(page_title="ClaimGuard | Audit Klaim", page_icon="CG", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --cg-ink: #18364b;
        --cg-blue: #07549a;
        --cg-green: #078b58;
        --cg-amber: #c77716;
        --cg-paper: #f3f6f5;
        --cg-line: #d8e2e0;
        --cg-muted: #5e706f;
    }
    .stApp { background: var(--cg-paper); color: var(--cg-ink); }
    [data-testid="stHeader"] { background: rgba(243, 246, 245, 0.94); }
    [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid var(--cg-line); color: var(--cg-ink); }
    [data-testid="stSidebar"] * { color: var(--cg-ink); }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div:has(> input[role="combobox"]) {
        background: #ffffff !important; border: 1px solid var(--cg-line) !important; border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"] { color: var(--cg-ink) !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] div:has(> div > input[type="range"]) {
        background-color: var(--cg-green) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] input[type="range"] { accent-color: var(--cg-green) !important; }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] input { accent-color: var(--cg-green); }
    [data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 1.4rem; }
    h1, h2, h3, h4 { color: var(--cg-ink); letter-spacing: 0; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.35rem; }
    h3 { font-size: 1.05rem; }
    p, label, [data-testid="stCaption"] { color: var(--cg-muted); }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid var(--cg-line); border-radius: 6px;
        padding: 12px 15px; min-height: 104px;
    }
    [data-testid="stMetricLabel"] { color: var(--cg-muted); font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: var(--cg-ink); font-size: 1.55rem; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: var(--cg-green) !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] .react-aria-SelectionIndicator {
        animation: none !important; transition: none !important;
        background-color: var(--cg-green) !important; background-image: none !important;
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--cg-line); border-radius: 6px; }
    .cg-brand { border-bottom: 1px solid var(--cg-line); padding: 0 0 1rem 0; margin-bottom: 1rem; }
    .cg-kicker { color: var(--cg-green) !important; font-size: .8rem; font-weight: 700; text-transform: uppercase; }
    .cg-title { color: var(--cg-ink); font-size: 1.8rem; font-weight: 750; line-height: 1.1; }
    .cg-subtitle { color: var(--cg-muted); font-size: .95rem; margin-top: .3rem; }
    .cg-notice { background: #eaf3f8; border-left: 4px solid var(--cg-blue); border-radius: 3px; padding: .8rem 1rem; color: var(--cg-ink); margin: .5rem 0 1.2rem; }
    .cg-section-note { color: var(--cg-muted); font-size: .86rem; margin-top: -.55rem; }
    .cg-risk-note { background: #fff5e7; border-left: 4px solid var(--cg-amber); border-radius: 3px; padding: .75rem .9rem; color: #573a19; }
    .cg-risk-note * { color: #573a19; }
    .cg-stat-note { color: var(--cg-muted); font-size: .82rem; margin: -.6rem 0 .8rem; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
        .cg-title { font-size: 1.45rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(path: str, **kwargs) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path, **kwargs)


@st.cache_data
def load_logo(path: str) -> Image.Image | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    image = Image.open(file_path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image, background).convert("L")
    bounds = difference.point(lambda value: 255 if value > 28 else 0).getbbox()
    return image.crop(bounds) if bounds else image


@st.cache_data
def load_brand_logos(bpjs_path: str, jkn_path: str) -> Image.Image | None:
    bpjs = load_logo(bpjs_path)
    jkn = load_logo(jkn_path)
    if bpjs is None or jkn is None:
        return bpjs or jkn
    bpjs.thumbnail((155, 60), Image.Resampling.LANCZOS)
    jkn.thumbnail((70, 70), Image.Resampling.LANCZOS)
    gap, padding = 14, 4
    canvas = Image.new("RGB", (bpjs.width + gap + jkn.width, max(bpjs.height, jkn.height) + padding * 2), "white")
    canvas.paste(bpjs, (0, (canvas.height - bpjs.height) // 2))
    canvas.paste(jkn, (bpjs.width + gap, (canvas.height - jkn.height) // 2))
    return canvas


def format_currency(value) -> str:
    if pd.isna(value):
        return "Tidak tersedia"
    return f"Rp {value:,.0f}".replace(",", ".")


def show_claim_field(container, label: str, value) -> None:
    text = "Tidak tersedia" if pd.isna(value) or value == "" else str(value)
    container.markdown(f"**{label}**  \n{text}")


logo_group, brand = st.columns([1.6, 4.4], vertical_alignment="center")
with logo_group:
    logos = load_brand_logos(str(ASSET_ROOT / "logo_bpjs.png"), str(ASSET_ROOT / "logo_jkn.jpeg"))
    if logos is not None:
        st.image(logos, width=235)
with brand:
    st.markdown(
        '<div class="cg-brand"><div class="cg-kicker">Analitik risiko pelayanan JKN</div>'
        '<div class="cg-title">ClaimGuard</div>'
        '<div class="cg-subtitle">Ringkasan klaim, prioritas pemeriksaan, dan alasan penandaan</div></div>',
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="cg-notice"><strong>Prototipe berbasis data sintetis.</strong> '
    'Prioritas risiko membantu menentukan klaim yang ditinjau; bukan bukti kecurangan, '
    'dan tidak mengubah atau menahan proses klaim.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Pengaturan tampilan")
    scheme_label = st.selectbox("Cara uji", ["Periode waktu", "Fasilitas kesehatan"])
    scheme = "waktu" if scheme_label == "Periode waktu" else "faskes"
    mode_label = st.selectbox("Perbandingan klaim", ["Saat diajukan", "Audit retrospektif"])
    mode = "asof" if mode_label == "Saat diajukan" else "retro"
    review_capacity_pct = st.slider("Kapasitas review (%)", 1, 10, 5, step=1)
    st.caption("Jumlah klaim dalam antrean mengikuti kapasitas ini.")
    show_eval_metrics = st.checkbox(
        "Tampilkan metrik evaluasi simulasi",
        value=False,
        help="Mengaktifkan label sintetis hanya untuk menghitung FPR dan recall pada data uji. Bukan metrik klaim nyata.",
    )

slider_progress = (review_capacity_pct - 1) / 9 * 100
st.markdown(
    f"""<style>
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-orientation="horizontal"]:not([role="group"]) > div:first-child {{
        background-image: linear-gradient(to right, var(--cg-green) 0%, var(--cg-green) {slider_progress:.4f}%, var(--cg-line) {slider_progress:.4f}%, var(--cg-line) 100%) !important;
    }}
    </style>""",
    unsafe_allow_html=True,
)

cal = load_csv(str(ROOT / "calibrated" / f"calibrated_{scheme}_{mode}.csv.gz"))
if cal.empty:
    st.warning("Data hasil analisis belum tersedia untuk pilihan ini. Jalankan pipeline terlebih dahulu.")
    st.stop()

claims = load_csv(str(ROOT / "data" / "claims.csv"), usecols=CLAIM_COLUMNS)
explain = load_csv(str(ROOT / "explain" / f"explain_{scheme}_{mode}.csv.gz"))
detectors = load_csv(str(ROOT / "detect" / f"detect_{scheme}_{mode}.csv.gz"))
cal["claim_id"] = cal["claim_id"].astype(str)
cal["p_value"] = pd.to_numeric(cal["p_value"], errors="coerce")
cal["skor_0_100"] = pd.to_numeric(cal["skor_0_100"], errors="coerce")
cal["split"] = cal["split"].fillna("unknown")
cal = cal.merge(claims, on="claim_id", how="left")
if not detectors.empty and set(["claim_id", *DETECTOR_COLUMNS]).issubset(detectors.columns):
    score_data = detectors[["claim_id", *DETECTOR_COLUMNS]].copy()
    score_data["claim_id"] = score_data["claim_id"].astype(str)
    cal = cal.merge(score_data, on="claim_id", how="left")
    cal["sinyal_utama"] = cal[DETECTOR_COLUMNS].fillna(0).idxmax(axis=1).map(DETECTOR_LABELS)
else:
    cal["sinyal_utama"] = "Belum tersedia"

test = cal.loc[cal["split"] == "test"].copy()
if test.empty:
    st.warning("Data test tidak tersedia untuk kombinasi ini.")
    st.stop()

test_size = len(test)
review_limit = max(1, (review_capacity_pct * test_size + 99) // 100)
flagged = test.sort_values(["p_value", "claim_id"], ascending=[True, True]).head(review_limit).copy()
test["is_flagged"] = test["claim_id"].isin(flagged["claim_id"])
review_burden = len(flagged) / test_size

if show_eval_metrics:
    labels = load_csv(str(ROOT / "eval_only" / "labels.csv"), keep_default_na=False)
else:
    labels = pd.DataFrame()

if not labels.empty and {"claim_id", "is_anomaly"}.issubset(labels.columns):
    eval_labels = labels[["claim_id", "is_anomaly"]].copy()
    eval_labels["claim_id"] = eval_labels["claim_id"].astype(str)
    test = test.merge(eval_labels, on="claim_id", how="left")
    known = test["is_anomaly"].notna()
    normal = test.loc[known & (test["is_anomaly"].astype(str) == "0")]
    anomalies = test.loc[known & (test["is_anomaly"].astype(str) == "1")]
    false_positive_rate = normal["is_flagged"].mean() if len(normal) else float("nan")
    recall = anomalies["is_flagged"].mean() if len(anomalies) else float("nan")
else:
    false_positive_rate = float("nan")
    recall = float("nan")
eval_metrics_value = (
    f"{false_positive_rate:.1%} / {recall:.1%}"
    if show_eval_metrics and not labels.empty
    else "Tidak aktif" if not show_eval_metrics else "Label tidak tersedia"
)

tab_monitor, tab_claims, tab_robustness = st.tabs(["Ringkasan", "Review klaim", "Uji keandalan"])

with tab_monitor:
    st.caption(
        f"Uji {scheme_label.lower()} · {mode_label.lower()} · periode uji · "
        f"antrean dibatasi top {review_capacity_pct}%"
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total klaim uji", f"{test_size:,}", help="Jumlah klaim simulasi pada bagian data yang dipakai untuk uji akhir.")
    metric_cols[1].metric("Klaim untuk ditinjau", f"{len(flagged):,}", help="Klaim dengan skor risiko tertinggi sesuai kapasitas review.")
    metric_cols[2].metric("Porsi antrean", f"{review_burden:.1%}", help="Porsi klaim yang masuk daftar prioritas; bukan persentase klaim curang.")
    if show_eval_metrics:
        st.metric("FPR / recall simulasi", eval_metrics_value, help="FPR adalah klaim normal yang ikut ditandai; recall adalah anomali simulasi yang masuk antrean. Bukan hasil data JKN nyata.")

    st.subheader("Apa yang perlu diperhatikan")
    st.markdown(
        '<div class="cg-section-note">Peringkat menunjukkan urutan pemeriksaan yang disarankan. '
        'Petugas tetap memeriksa bukti dan aturan yang berlaku.</div>',
        unsafe_allow_html=True,
    )
    chart_col, facility_col = st.columns([0.9, 1.1])
    with chart_col:
        tier_counts = (
            flagged["tingkat"].value_counts()
            .reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
            .rename(index={"Critical": "Sangat tinggi", "High": "Tinggi", "Medium": "Menengah", "Low": "Rendah"})
            .rename_axis("Prioritas")
            .to_frame("Klaim dalam antrean")
        )
        st.markdown("#### Tingkat risiko dalam antrean")
        st.bar_chart(tier_counts, color="#078b58")
    with facility_col:
        st.markdown("#### Fasilitas dengan antrean terbanyak")
        facility_summary = (
            flagged.groupby("facility_id", dropna=False)
            .agg(
                prioritas=("claim_id", "size"),
                layanan=("jenis_layanan", lambda values: values.value_counts().index[0] if len(values) else "-"),
            )
            .reset_index()
            .sort_values("prioritas", ascending=False)
            .head(12)
        )
        st.dataframe(
            facility_summary.rename(
                columns={"facility_id": "Fasilitas", "prioritas": "Klaim prioritas", "layanan": "Layanan terbanyak"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            '<div class="cg-stat-note">Jumlah ini menunjukkan banyaknya klaim prioritas, bukan tingkat kecurangan fasilitas. '
            'Fasilitas dengan lebih banyak klaim dapat memiliki antrean lebih besar.</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Contoh klaim prioritas")
    queue = flagged.sort_values(["p_value", "skor_0_100"], ascending=[True, False]).head(12)
    queue_columns = ["claim_id", "facility_id", "jenis_layanan", "sinyal_utama", "skor_0_100", "tingkat"]
    queue = queue[[column for column in queue_columns if column in queue.columns]].copy()
    if "skor_0_100" in queue:
        queue["skor_0_100"] = queue["skor_0_100"].map(lambda value: f"{value:.1f}")
    if "tingkat" in queue:
        queue["tingkat"] = queue["tingkat"].map(TIER_LABELS).fillna(queue["tingkat"])
    queue = queue.rename(columns={
        "claim_id": "ID klaim", "facility_id": "Fasilitas", "jenis_layanan": "Jenis layanan",
        "sinyal_utama": "Sinyal utama", "skor_0_100": "Skor risiko / 100", "tingkat": "Prioritas",
    })
    st.dataframe(queue, use_container_width=True, hide_index=True)
    st.caption("Skor risiko adalah peringkat relatif dari model simulasi, bukan probabilitas atau keputusan kecurangan.")

with tab_claims:
    st.subheader("Cari klaim dalam antrean")
    st.markdown('<div class="cg-section-note">Pilih satu klaim untuk melihat ringkasan dan alasan prioritasnya.</div>', unsafe_allow_html=True)
    facilities = sorted(flagged["facility_id"].dropna().astype(str).unique()) if "facility_id" in flagged else []
    tier_options = [tier for tier in TIER_LABELS.values() if tier in flagged["tingkat"].map(TIER_LABELS).unique()]
    services = sorted(flagged["jenis_layanan"].dropna().astype(str).unique()) if "jenis_layanan" in flagged else []
    signals = sorted(flagged["sinyal_utama"].dropna().astype(str).unique()) if "sinyal_utama" in flagged else []
    with st.expander("Filter antrean", expanded=True):
        search_col, facility_col = st.columns([1.3, 1])
        with search_col:
            claim_search = st.text_input("Cari ID klaim", placeholder="Contoh: CLM-000...").strip().lower()
        with facility_col:
            facility_filter = st.selectbox("Fasilitas", ["Semua"] + facilities)
        tier_col, service_col, signal_col = st.columns(3)
        with tier_col:
            tier_filter = st.selectbox("Prioritas", ["Semua"] + tier_options)
        with service_col:
            service_filter = st.selectbox("Jenis layanan", ["Semua"] + services)
        with signal_col:
            signal_filter = st.selectbox("Sinyal risiko", ["Semua"] + signals)

    filtered = flagged.copy()
    if claim_search:
        filtered = filtered[filtered["claim_id"].str.lower().str.contains(claim_search, na=False, regex=False)]
    if facility_filter != "Semua":
        filtered = filtered[filtered["facility_id"].astype(str) == facility_filter]
    if tier_filter != "Semua":
        filtered = filtered[filtered["tingkat"].map(TIER_LABELS) == tier_filter]
    if service_filter != "Semua":
        filtered = filtered[filtered["jenis_layanan"].astype(str) == service_filter]
    if signal_filter != "Semua":
        filtered = filtered[filtered["sinyal_utama"].astype(str) == signal_filter]

    if not explain.empty and "claim_id" in explain.columns:
        explain["claim_id"] = explain["claim_id"].astype(str)
        available_explanation = [column for column in EXPLANATION_COLUMNS if column in explain.columns]
        if len(available_explanation) > 1:
            filtered = filtered.merge(explain[available_explanation].drop_duplicates("claim_id"), on="claim_id", how="left")

    st.caption(f"{len(filtered):,} klaim cocok dengan filter · {len(flagged):,} klaim dalam antrean")
    table_columns = [
        "claim_id", "facility_id", "jenis_layanan", "tanggal_klaim", "nilai_klaim",
        "sinyal_utama", "skor_0_100", "tingkat",
    ]
    table = filtered[[column for column in table_columns if column in filtered.columns]].copy()
    if "nilai_klaim" in table:
        table["nilai_klaim"] = table["nilai_klaim"].map(format_currency)
    if "p_value" in table:
        table["p_value"] = table["p_value"].map(lambda value: f"{value:.4f}")
    if "skor_0_100" in table:
        table["skor_0_100"] = table["skor_0_100"].map(lambda value: f"{value:.1f}")
    if "tingkat" in table:
        table["tingkat"] = table["tingkat"].map(TIER_LABELS).fillna(table["tingkat"])
    table = table.rename(columns={
        "claim_id": "ID klaim", "facility_id": "Fasilitas", "jenis_layanan": "Layanan",
        "tanggal_klaim": "Tanggal klaim", "nilai_klaim": "Nilai klaim simulasi",
        "sinyal_utama": "Sinyal utama", "skor_0_100": "Skor / 100", "tingkat": "Prioritas",
    })
    st.dataframe(table, use_container_width=True, hide_index=True, height=340)

    if filtered.empty:
        st.info("Tidak ada klaim yang cocok dengan filter saat ini.")
    else:
        claim_ids = filtered["claim_id"].astype(str).tolist()
        selected_id = st.selectbox("Buka detail klaim", claim_ids)
        selected = filtered.loc[filtered["claim_id"].astype(str) == selected_id].iloc[0]
        st.markdown("---")
        st.markdown(f"### {selected_id}")
        st.markdown('<div class="cg-risk-note"><strong>Perlu verifikasi manusia.</strong> '
                'Penandaan ini menunjukkan pola yang tidak biasa pada data simulasi, bukan bukti kecurangan.</div>',
                unsafe_allow_html=True)
        detail_metrics = st.columns(4)
        detail_metrics[0].metric("Skor risiko / 100", f"{selected['skor_0_100']:.1f}", help="Skor untuk mengurutkan prioritas; bukan probabilitas kecurangan.")
        detail_metrics[1].metric("Prioritas", TIER_LABELS.get(selected.get("tingkat"), "Tidak tersedia"))
        detail_metrics[2].metric("Sinyal utama", str(selected.get("sinyal_utama", "Belum tersedia")))
        detail_metrics[3].metric("Nilai klaim simulasi", format_currency(selected.get("nilai_klaim")))

        info_col, reason_col = st.columns([1, 1.2])
        with info_col:
            st.markdown("#### Informasi klaim")
            left, right = st.columns(2)
            show_claim_field(left, "Fasilitas simulasi", selected.get("facility_id"))
            show_claim_field(right, "Jenis layanan", selected.get("jenis_layanan"))
            show_claim_field(left, "Tanggal klaim", selected.get("tanggal_klaim"))
            show_claim_field(right, "ID pasien anonim", selected.get("patient_id"))
            show_claim_field(left, "Diagnosis utama", selected.get("diagnosis_utama"))
            show_claim_field(right, "Diagnosis sekunder", selected.get("diagnosis_sekunder"))
            show_claim_field(left, "Tindakan", selected.get("tindakan"))
            show_claim_field(right, "Lama rawat", selected.get("lama_rawat"))
            with st.expander("Komponen biaya obat/alkes"):
                st.write(selected.get("obat_alkes", "Tidak tersedia"))
        with reason_col:
            st.markdown("#### Mengapa masuk antrean?")
            show_claim_field(reason_col, "Sinyal utama model", selected.get("sinyal_utama"))
            show_claim_field(reason_col, "Kategori pola", selected.get("sinyal_utama", "Belum tersedia"))
            st.info(DETECTOR_GUIDANCE.get(selected.get("sinyal_utama"), DETECTOR_GUIDANCE["Sinyal gabungan"]))
            explanation_text = selected.get("explanation")
            if pd.notna(explanation_text) and explanation_text:
                with st.expander("Rincian teknis model"):
                    show_claim_field(reason_col, "Fitur teratas", selected.get("top_feature", "Belum tersedia"))
                    if pd.notna(selected.get("top_feature_value")):
                        show_claim_field(reason_col, "Nilai fitur", selected.get("top_feature_value"))
                    st.caption(str(explanation_text))
            elif selected.get("sinyal_utama") == "Belum tersedia":
                st.caption("Rincian model tidak tersedia untuk klaim ini.")

with tab_robustness:
    st.subheader("Seberapa stabil hasilnya?")
    st.markdown('<div class="cg-section-note">Ringkasan pengujian pada beberapa seed dan tingkat anomali buatan. Ini mengukur konsistensi di simulator, bukan validasi dunia nyata.</div>', unsafe_allow_html=True)
    robustness = load_csv(str(ROBUSTNESS_ROOT / "robustness_summary.csv"))
    detail_runs = load_csv(str(ROBUSTNESS_ROOT / "robustness_detail.csv"))
    if robustness.empty:
        st.info("Ringkasan robustness belum ditemukan. Jalankan claimguard_robustness.py terlebih dahulu.")
    else:
        st.caption("Rata-rata ± simpangan baku dari 25 kombinasi prevalensi (1–5%) dan seed.")
        st.caption("Salah tandai (FPR) = klaim normal simulasi yang ikut masuk prioritas. Recall = anomali simulasi yang tertangkap. Keduanya bukan angka untuk klaim JKN nyata.")
        display_rows = []
        for _, row in robustness.iterrows():
            display_rows.append({
                "Pembagian data": "Fasilitas" if row["scheme"] == "faskes" else "Periode waktu",
                "Cara membandingkan": "Saat diajukan" if row["mode"] == "asof" else "Retrospektif",
                "Salah tandai @ target 1%": f"{row['mean_fpr_1']:.2%} ± {row['std_fpr_1']:.2%}",
                "Salah tandai @ target 3%": f"{row['mean_fpr_3']:.2%} ± {row['std_fpr_3']:.2%}",
                "Salah tandai @ target 5%": f"{row['mean_fpr_5']:.2%} ± {row['std_fpr_5']:.2%}",
                "Salah tandai @ target 10%": f"{row['mean_fpr_10']:.2%} ± {row['std_fpr_10']:.2%}",
                "Anomali tertangkap @ review 5%": f"{row['mean_recall_5']:.1%} ± {row['std_recall_5']:.1%}",
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        recall_chart = robustness.assign(
            scenario=robustness["scheme"].map({"faskes": "Fasilitas", "waktu": "Waktu"}) + " / " + robustness["mode"].map({"asof": "Saat diajukan", "retro": "Retrospektif"})
        ).set_index("scenario")[["mean_recall_5"]].rename(columns={"mean_recall_5": "Anomali simulasi tertangkap rata-rata"})
        st.markdown("#### Anomali simulasi tertangkap pada kapasitas review 5%")
        st.bar_chart(recall_chart, color="#07549a")

        if not detail_runs.empty:
            with st.expander("Lihat seluruh 100 hasil pengujian"):
                detail_runs = detail_runs.copy()
                for column in ["prevalence", "fpr_1", "fpr_3", "fpr_5", "fpr_10", "recall_5"]:
                    if column in detail_runs:
                        detail_runs[column] = pd.to_numeric(detail_runs[column], errors="coerce")
                detail_runs["scheme"] = detail_runs["scheme"].map({"faskes": "Fasilitas", "waktu": "Periode waktu"})
                detail_runs["mode"] = detail_runs["mode"].map({"asof": "Saat diajukan", "retro": "Retrospektif"})
                detail_runs = detail_runs.rename(columns={
                    "prevalence": "Prevalensi simulasi", "seed": "Seed", "scheme": "Pembagian data",
                    "mode": "Cara analisis", "fpr_1": "Salah tandai @ 1%", "fpr_3": "Salah tandai @ 3%",
                    "fpr_5": "Salah tandai @ 5%", "fpr_10": "Salah tandai @ 10%", "recall_5": "Tertangkap @ review 5%",
                })
                st.dataframe(
                    detail_runs,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Prevalensi simulasi": st.column_config.NumberColumn("Prevalensi simulasi", format="percent"),
                        "Salah tandai @ 1%": st.column_config.NumberColumn("Salah tandai @ 1%", format="percent"),
                        "Salah tandai @ 3%": st.column_config.NumberColumn("Salah tandai @ 3%", format="percent"),
                        "Salah tandai @ 5%": st.column_config.NumberColumn("Salah tandai @ 5%", format="percent"),
                        "Salah tandai @ 10%": st.column_config.NumberColumn("Salah tandai @ 10%", format="percent"),
                        "Tertangkap @ review 5%": st.column_config.NumberColumn("Tertangkap @ review 5%", format="percent"),
                    },
                )

st.caption("Data berasal dari snapshot simulasi lokal. Hasil bukan keputusan otomatis, bukan bukti kecurangan, dan belum membuktikan performa pada data klaim nyata.")
