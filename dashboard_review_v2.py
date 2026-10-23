#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit demo for human review of the synthetic ClaimGuard V2 queue."""

from pathlib import Path

import pandas as pd
import streamlit as st

from claimguard_review_evidence_v2 import build_claim_comparison
from claimguard_review_log_v2 import (
    EVIDENCE_ITEMS,
    FOLLOW_UP_ACTIONS,
    RECOMMENDATION_FEEDBACK,
    REVIEW_OUTCOMES,
    REVIEWER_ROLES,
    read_review_log,
    record_review_action,
)


ROOT = Path(__file__).resolve().parent
PREP_ROOT = ROOT / "out" / "prep_v2"
QUEUE_FILES = {
    "time": ("split_time", PREP_ROOT / "review_queue_v2_time.csv.gz"),
    "facility": ("split_facility", PREP_ROOT / "review_queue_v2_facility.csv.gz"),
}
CLAIMS_WITH_SPLITS_FILE = PREP_ROOT / "claims_v2_flat_with_splits.csv.gz"
REVIEW_LOG = PREP_ROOT / "review_records_demo_v2.csv"
FEATURE_FILES = {
    "time": PREP_ROOT / "features_v2_time.csv.gz",
    "facility": PREP_ROOT / "features_v2_facility.csv.gz",
}
QUEUE_COLUMNS = [
    "priority_rank",
    "id_klaim",
    "submitted_at",
    "service_type",
    "facility_id",
    "priority_score",
    "top_driver",
    "review_recommended",
]
CLAIM_DETAILS = [
    ("submitted_at", "Tanggal pengajuan"),
    ("service_date", "Tanggal pelayanan"),
    ("service_type", "Jenis layanan"),
    ("facility_id", "ID faskes sintetis"),
    ("ward_class", "Kelas rawat"),
    ("severity", "Severity"),
    ("length_of_stay_days", "Lama rawat (hari)"),
    ("claim_total_idr", "Total klaim (IDR)"),
    ("diagnosis_primary", "Diagnosis utama"),
    ("diagnosis_secondary", "Diagnosis sekunder"),
    ("diagnosis_codes", "Kode diagnosis"),
    ("procedure_codes", "Kode tindakan"),
    ("item_codes", "Kode item"),
    ("procedure_count", "Jumlah tindakan"),
    ("item_count", "Jumlah item"),
]
COST_LABELS = {
    "cost_akomodasi_idr": "Akomodasi",
    "cost_alkes_idr": "Alat kesehatan",
    "cost_konsultasi_idr": "Konsultasi",
    "cost_obat_idr": "Obat",
    "cost_penunjang_idr": "Penunjang",
    "cost_tindakan_idr": "Tindakan",
}
SIGNAL_LABELS = {
    "repeat_rank_percentile": "Pola klaim berulang",
    "cloning_rank_percentile": "Kemiripan dengan klaim lain",
    "inflated_cost_rank_percentile": "Perbandingan biaya",
    "severity_los_unusual_rank_percentile": "Tingkat keparahan dan lama rawat",
}
SPLIT_OPTIONS = {
    "Urut berdasarkan waktu pengajuan": ("time", "Urutan waktu"),
    "Kelompokkan berdasarkan fasilitas": ("facility", "Kelompok fasilitas"),
}
PARTITION_OPTIONS = {
    "Kelompok pembanding": "train",
    "Kelompok validasi": "val",
    "Kelompok pengujian": "test",
}
SERVICE_LABELS = {
    "RJTL": "Rawat jalan (RJTL)",
    "RITL": "Rawat inap (RITL)",
}


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        st.error(f"File wajib tidak ditemukan: {path}")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def _load_claims(path: str) -> pd.DataFrame:
    return _load_required_csv(Path(path))


@st.cache_data
def _load_queue(path: str) -> pd.DataFrame:
    return _load_required_csv(Path(path))


@st.cache_data
def _load_features(path: str) -> pd.DataFrame:
    return _load_required_csv(Path(path))


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} tidak memiliki kolom wajib: {', '.join(missing)}")


def _format_value(value) -> str:
    if pd.isna(value) or value == "":
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value).is_integer():
            return f"{int(value):,}".replace(",", ".")
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)


def _format_currency(value) -> str:
    if pd.isna(value):
        return "-"
    return f"Rp {_format_value(value)}"


def _service_label(value) -> str:
    return SERVICE_LABELS.get(str(value), str(value))


def _review_reason(row, claim) -> tuple[str, str]:
    evidence = str(row["evidence"])
    if row["top_driver"] == "Biaya relatif tinggi terhadap referensi train":
        marker = "rasio biaya terhadap median train="
        ratio = evidence.partition(marker)[2].partition(";")[0]
        try:
            ratio_text = f"{float(ratio):.1f}".replace(".", ",")
        except ValueError:
            ratio_text = "lebih tinggi"
        return (
            "Perbandingan biaya menunjukkan nilai yang lebih tinggi",
            f"Total biaya sekitar {ratio_text} kali nilai tengah pada kelompok pembanding simulasi. "
            "Perbedaan ini menjadi dasar pemeriksaan atas rincian layanan dan biaya.",
        )
    if row["top_driver"] == "Kemiripan dengan klaim pasien lain":
        marker = "pasien lain="
        similarity = evidence.partition(marker)[2]
        try:
            similarity_text = str(round(float(similarity) * 100))
        except ValueError:
            similarity_text = "tinggi"
        return (
            "Ditemukan kemiripan dengan klaim peserta lain",
            f"Sekitar {similarity_text}% kemiripan pada kode diagnosis, tindakan, atau item "
            "dibandingkan klaim terdahulu milik peserta lain. Kemiripan dapat terjadi pada "
            "layanan yang serupa dan bukan bukti kesalahan.",
        )
    if row["top_driver"] == "Pola klaim pasien berulang":
        marker = "klaim terdahulu 30 hari="
        count = evidence.partition(marker)[2].partition(";")[0]
        try:
            count_text = str(int(count))
        except ValueError:
            count_text = "beberapa"
        if "layanan overlap=1" in evidence:
            detail = "Tanggal pelayanan beririsan dengan klaim sebelumnya."
        else:
            detail = "Riwayat ini dapat berkaitan dengan kunjungan lanjutan yang wajar."
        return (
            "Terdapat riwayat klaim dalam waktu berdekatan",
            f"Terdapat {count_text} klaim terdahulu dari peserta yang sama dalam 30 hari. "
            f"{detail} Riwayat berulang bukan bukti kesalahan.",
        )

    severity = str(claim["severity"])
    stay = _format_value(claim["length_of_stay_days"])
    return (
        "Kombinasi tingkat keparahan dan lama rawat perlu ditinjau",
        f"Catatan menunjukkan tingkat keparahan {severity} dan lama rawat {stay} hari. "
        "Kesesuaiannya menjadi bagian dari penelaahan ringkasan pelayanan.",
    )


st.set_page_config(page_title="ClaimGuard | Review Demo V2", page_icon="CG", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f7f8; color: #18364b; }
    [data-testid="stMainBlockContainer"] { max-width: 1320px; padding-top: 1.5rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dce5e8; }
    [data-testid="stSidebar"] * { color: #18364b; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #ffffff !important; border: 1px solid #c8d6dc !important;
        color: #18364b !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div:has(> input[role="combobox"]) {
        background: #ffffff !important; border: 1px solid #c8d6dc !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input[role="combobox"] {
        color: #18364b !important; -webkit-text-fill-color: #18364b !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] input { accent-color: #087a75; }
    h1, h2, h3 { color: #18364b; }
    h1 { font-size: 1.9rem; }
    .cg-subtitle { color: #536b78; font-size: 1.03rem; margin-top: -.7rem; }
    .cg-callout { background: #eaf3f8; border-left: 4px solid #087a75;
        border-radius: 5px; padding: 1rem 1.1rem; margin: .8rem 0 1.4rem; }
    .cg-callout strong { color: #18364b; }
    .cg-reason { background: #ffffff; border: 1px solid #dce5e8;
        border-left: 5px solid #df9b35; border-radius: 7px;
        padding: 1rem 1.2rem; margin: .6rem 0 1rem; }
    .cg-reason-title { color: #18364b; font-size: 1.1rem; font-weight: 700; }
    .cg-reason-detail { color: #536b78; margin-top: .35rem; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce5e8;
        border-radius: 7px; padding: .8rem 1rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #dce5e8; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Daftar Pemeriksaan Klaim")
st.markdown(
    '<p class="cg-subtitle">Daftar ini membantu petugas menentukan urutan pemeriksaan klaim.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="cg-callout"><strong>Keterangan dashboard</strong><br>'
    "Dashboard ini merupakan demonstrasi menggunakan data sintetis. Urutan menunjukkan klaim "
    "yang dapat diperiksa lebih dahulu, bukan penetapan kesalahan atau fraud. Klaim di luar "
    "daftar tetap mengikuti pemeriksaan yang berlaku. Keputusan akhir berada pada petugas "
    "yang berwenang.</div>",
    unsafe_allow_html=True,
)

scheme_label = st.sidebar.selectbox("Pengelompokan daftar", list(SPLIT_OPTIONS))
scheme, scheme_description = SPLIT_OPTIONS[scheme_label]
split_column, queue_path = QUEUE_FILES[scheme]
queue = _load_queue(queue_path)
claims = _load_claims(CLAIMS_WITH_SPLITS_FILE)
features = _load_features(str(FEATURE_FILES[scheme]))

_require_columns(
    queue,
    QUEUE_COLUMNS
    + [split_column, "evidence", "interpretation"]
    + list(SIGNAL_LABELS),
    "File antrean",
)
_require_columns(claims, ["id_klaim"] + [key for key, _ in CLAIM_DETAILS] + list(COST_LABELS), "Data klaim")
if queue["id_klaim"].duplicated().any():
    raise ValueError("Antrean memuat id_klaim duplikat.")
if claims["id_klaim"].duplicated().any():
    raise ValueError("Data klaim memuat id_klaim duplikat.")

partition_label = st.sidebar.selectbox("Jenis kelompok data", list(PARTITION_OPTIONS), index=2)
partition = PARTITION_OPTIONS[partition_label]
st.sidebar.caption(
    "Kelompok pengujian berisi contoh hasil. Kelompok pembanding dan validasi digunakan "
    "untuk menguji prototipe."
)
recommended_only = st.sidebar.checkbox("Tampilkan prioritas pemeriksaan saja", value=True)
search = st.sidebar.text_input(
    "Nomor klaim atau fasilitas",
    placeholder="Nomor klaim atau kode fasilitas",
)

visible = queue.loc[queue[split_column].eq(partition)].copy()
if recommended_only:
    visible = visible.loc[visible["review_recommended"].astype(bool)]
if search.strip():
    query = search.strip().casefold()
    visible = visible.loc[
        visible["id_klaim"].astype(str).str.casefold().str.contains(query, regex=False)
        | visible["facility_id"].astype(str).str.casefold().str.contains(query, regex=False)
    ]
visible = visible.sort_values(["priority_rank", "id_klaim"], kind="stable")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Klaim dalam daftar", f"{len(visible):,}".replace(",", "."))
col_b.metric(
    "Prioritas pemeriksaan",
    f"{int(queue.loc[queue[split_column].eq(partition), 'review_recommended'].astype(bool).sum()):,}".replace(",", "."),
)
col_c.metric("Kelompok data", partition_label)
st.caption(
    f"Pengelompokan: {scheme_description}. Urutan didasarkan pada petunjuk yang muncul "
    "dalam simulasi, bukan kepastian adanya kesalahan."
)

display_queue = visible[
    ["priority_rank", "id_klaim", "submitted_at", "service_type", "facility_id", "top_driver"]
].rename(
    columns={
        "priority_rank": "Urutan",
        "id_klaim": "Nomor klaim",
        "submitted_at": "Tanggal pengajuan",
        "service_type": "Jenis layanan",
        "facility_id": "Fasilitas",
        "top_driver": "Petunjuk pemeriksaan utama",
    }
)
display_queue["Jenis layanan"] = display_queue["Jenis layanan"].map(_service_label)
display_queue["Petunjuk pemeriksaan utama"] = display_queue[
    "Petunjuk pemeriksaan utama"
].replace(
    {
        "Pola klaim pasien berulang": "Riwayat klaim berdekatan",
        "Kemiripan dengan klaim pasien lain": "Rincian mirip dengan klaim lain",
        "Biaya relatif tinggi terhadap referensi train": "Biaya berbeda dari pembanding",
        "Kombinasi severity dan lama rawat perlu ditinjau": "Tingkat keparahan dan lama rawat",
    }
)
st.subheader("Daftar klaim")
st.dataframe(display_queue, width="stretch", hide_index=True)

if visible.empty:
    st.info("Tidak ditemukan klaim yang sesuai dengan filter yang diterapkan.")
    st.stop()

claim_options = visible["id_klaim"].astype(str).tolist()
selected_id = st.selectbox(
    "Klaim yang ditampilkan",
    claim_options,
    format_func=lambda claim_id: (
        f"Urutan {int(visible.loc[visible['id_klaim'].astype(str).eq(claim_id), 'priority_rank'].iloc[0])} "
        f"- {claim_id}"
    ),
)
queue_row = visible.loc[visible["id_klaim"].astype(str).eq(selected_id)].iloc[0]
claim_matches = claims.loc[claims["id_klaim"].astype(str).eq(selected_id)]
if claim_matches.empty:
    raise ValueError(f"Klaim {selected_id} pada antrean tidak ditemukan di data klaim.")
claim = claim_matches.iloc[0]

st.subheader(f"Ringkasan klaim {selected_id}")
reason_title, reason_detail = _review_reason(queue_row, claim)
comparison = build_claim_comparison(claims, features, selected_id, split_column)
st.markdown(
    f'<div class="cg-reason"><div class="cg-reason-title">{reason_title}</div>'
    f'<div class="cg-reason-detail">{reason_detail}</div></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Petunjuk ini bukan kesimpulan tentang benar atau salahnya klaim. "
    "Penilaian tetap memerlukan dokumen pendukung dan konteks pelayanan."
)

details_tab, comparison_tab, signals_tab, action_tab = st.tabs(
    ["Rincian klaim", "Bukti pembanding", "Dasar pengurutan", "Catatan pemeriksaan"]
)
with details_tab:
    st.markdown("**Ringkasan klaim**")
    primary_details = [
        ("submitted_at", "Tanggal pengajuan"),
        ("service_date", "Tanggal pelayanan"),
        ("service_type", "Jenis layanan"),
        ("facility_id", "Kode fasilitas"),
        ("claim_total_idr", "Total biaya klaim"),
        ("length_of_stay_days", "Lama rawat (hari)"),
    ]
    detail_cols = st.columns(3)
    for index, (key, label) in enumerate(primary_details):
        value = claim[key]
        if key == "service_type":
            value = _service_label(value)
        if key == "claim_total_idr":
            value = _format_currency(value)
        detail_cols[index % 3].markdown(f"**{label}**  \n{_format_value(value)}")

    costs = pd.DataFrame(
        [
            {"Komponen biaya": label, "Nilai": _format_currency(claim[column])}
            for column, label in COST_LABELS.items()
        ]
    )
    with st.expander("Rincian biaya dan kode pelayanan"):
        st.markdown("**Rincian biaya**")
        st.dataframe(costs, width="stretch", hide_index=True)
        code_details = [
            ("diagnosis_primary", "Diagnosis utama"),
            ("diagnosis_secondary", "Diagnosis tambahan"),
            ("procedure_codes", "Kode tindakan"),
            ("item_codes", "Kode obat atau alat kesehatan"),
            ("procedure_count", "Jumlah tindakan"),
            ("item_count", "Jumlah item"),
        ]
        code_cols = st.columns(2)
        for index, (key, label) in enumerate(code_details):
            code_cols[index % 2].markdown(f"**{label}**  \n{_format_value(claim[key])}")

with comparison_tab:
    st.markdown(
        "Pembanding di bawah menunjukkan data yang digunakan untuk menjelaskan petunjuk. "
        "Nilai pembanding berasal dari data latih pada kelompok yang tercantum; klaim serupa "
        "hanya dicari di klaim yang diajukan lebih awal dan sesuai urutan partisi."
    )
    cost_comparison = comparison["cost"]
    st.markdown("**Perbandingan biaya**")
    cost_metrics = st.columns(3)
    cost_metrics[0].metric("Total klaim", _format_currency(cost_comparison["current_total_idr"]))
    cost_metrics[1].metric(
        "Nilai tengah pembanding",
        _format_currency(cost_comparison["train_median_total_idr"]),
    )
    cost_metrics[2].metric(
        "Perbandingan dengan nilai tengah",
        f"{cost_comparison['ratio_to_train_median']:.2f} kali".replace(".", ","),
    )
    st.caption(
        f"Kelompok pembanding: {cost_comparison['group_label']}; "
        f"{cost_comparison['reference_count']} klaim data latih. "
        "Nilai tengah setiap komponen dihitung terpisah sehingga jumlahnya tidak selalu "
        "sama dengan nilai tengah total klaim."
    )
    component_labels = {
        "cost_akomodasi_idr": "Akomodasi",
        "cost_alkes_idr": "Alat kesehatan",
        "cost_konsultasi_idr": "Konsultasi",
        "cost_obat_idr": "Obat",
        "cost_penunjang_idr": "Penunjang",
        "cost_tindakan_idr": "Tindakan",
    }
    component_frame = pd.DataFrame(
        [
            {
                "Komponen": component_labels[item["component"]],
                "Klaim terpilih": _format_currency(item["current_idr"]),
                "Nilai tengah data latih": _format_currency(item["train_median_idr"]),
                "Selisih": _format_currency(item["difference_idr"]),
            }
            for item in cost_comparison["components"]
        ]
    )
    st.dataframe(component_frame, width="stretch", hide_index=True)

    repeat_comparison = comparison["repeat"]
    st.markdown("**Riwayat klaim peserta yang sama**")
    if repeat_comparison["previous_claim"] is None:
        st.caption("Tidak ditemukan klaim terdahulu untuk peserta ini dalam konteks data yang digunakan.")
    else:
        previous = repeat_comparison["previous_claim"]
        st.write(
            f"Klaim terdahulu: **{previous['claim_id']}**, diajukan {previous['submitted_at']}, "
            f"jenis layanan {_service_label(previous['service_type'])}, "
            f"total {_format_currency(previous['claim_total_idr'])}. "
            f"Jumlah klaim terdahulu dalam 30 hari: {repeat_comparison['prior_count_30d']}."
        )

    st.markdown("**Klaim terdahulu dengan rincian paling mirip**")
    similar = comparison["similar_claim"]
    if similar is None:
        st.caption("Tidak tersedia klaim terdahulu dari peserta lain untuk dibandingkan.")
    else:
        similarity_metrics = st.columns(3)
        similarity_metrics[0].metric("Klaim pembanding", similar["claim_id"])
        similarity_metrics[1].metric("Tanggal pengajuan", similar["submitted_at"])
        similarity_metrics[2].metric("Kemiripan kode", f"{similar['similarity']:.0%}")
        st.caption(
            "Perbandingan memakai gabungan kode diagnosis, tindakan, dan item. "
            "Kemiripan bukan bukti bahwa klaim tidak sah."
        )
        code_comparison = pd.DataFrame(
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
        )
        st.dataframe(code_comparison, width="stretch", hide_index=True)

    severity_comparison = comparison["severity"]
    if severity_comparison is not None:
        st.markdown("**Konteks tingkat keparahan dan lama rawat**")
        severity_counts = ", ".join(
            f"{level}: {count}"
            for level, count in sorted(severity_comparison["train_severity_counts"].items())
        )
        st.caption(
            f"Kelompok data latih: {severity_comparison['group_label']} "
            f"({severity_comparison['reference_count']} klaim). "
            f"Nilai tengah lama rawat: {severity_comparison['train_median_stay_days']:.1f} hari. "
            f"Sebaran tingkat keparahan di kelompok: {severity_counts}."
        )
        st.caption(
            f"Klaim terpilih mencatat tingkat keparahan {severity_comparison['current_severity']} "
            f"dan lama rawat {severity_comparison['current_stay_days']:g} hari. "
            "Perbandingan ini merupakan konteks statistik, bukan penilaian klinis."
        )

with signals_tab:
    st.markdown(
        "Urutan dibuat dengan membandingkan beberapa petunjuk pada setiap klaim dengan "
        "data pembanding simulasi. Nilai yang lebih tinggi menunjukkan posisi yang lebih "
        "menonjol dalam perbandingan, bukan persentase kemungkinan fraud."
    )
    with st.expander("Rincian perbandingan"):
        signal_values = pd.DataFrame(
            [
                {
                    "Petunjuk": label,
                    "Posisi dibanding data pembanding (0-100)": round(float(queue_row[column])),
                }
                for column, label in SIGNAL_LABELS.items()
            ]
        )
        st.dataframe(signal_values, width="stretch", hide_index=True)
    st.info(
        "Prototipe ini belum dilatih menggunakan data fraud yang telah diverifikasi. "
        "Pengurutan saat ini menggunakan perbandingan pola dan aturan statistik."
    )

with action_tab:
    st.markdown(
        "Formulir ini membantu mencatat dasar pemeriksaan secara terstruktur. Catatan hanya "
        "untuk demonstrasi data sintetis; bukan penetapan fraud dan tidak digunakan untuk "
        "melatih atau menilai sistem analisis."
    )
    st.warning(
        "Jangan masukkan nama, nomor identitas, atau informasi pribadi ke dalam catatan. "
        "Pada penggunaan nyata, definisi hasil review, akses, dan perlindungan data harus "
        "disepakati terlebih dahulu."
    )
    with st.form("review_action_form"):
        reviewer_role = st.selectbox("Peran pemeriksa", REVIEWER_ROLES)
        evidence_reviewed = st.multiselect(
            "Bukti yang telah ditinjau",
            EVIDENCE_ITEMS,
            help="Pilih hanya dokumen atau informasi yang benar-benar diperiksa.",
        )
        review_outcome = st.selectbox("Hasil pemeriksaan saat ini", REVIEW_OUTCOMES)
        follow_up_action = st.selectbox("Tindak lanjut simulasi", FOLLOW_UP_ACTIONS)
        review_notes = st.text_area(
            "Alasan atau catatan pemeriksaan",
            max_chars=2000,
            help="Tuliskan alasan singkat berdasarkan bukti yang dipilih; jangan tuliskan identitas pribadi.",
        )
        recommendation_feedback = st.selectbox(
            "Apakah urutan ClaimGuard membantu pemeriksaan?",
            RECOMMENDATION_FEEDBACK,
        )
        submitted = st.form_submit_button("Simpan catatan review simulasi")
    if submitted:
        if not evidence_reviewed:
            st.error("Pilih setidaknya satu bukti yang telah ditinjau.")
        elif len(review_notes.strip()) < 10:
            st.error("Catatan pemeriksaan harus berisi sedikitnya 10 karakter.")
        else:
            record_review_action(
                REVIEW_LOG,
                selected_id,
                scheme,
                partition,
                reviewer_role,
                evidence_reviewed,
                review_outcome,
                follow_up_action,
                review_notes,
                recommendation_feedback,
            )
            st.success("Catatan review simulasi berhasil disimpan secara lokal.")

    review_log = read_review_log(REVIEW_LOG)
    current_actions = review_log.loc[
        review_log["id_klaim"].eq(selected_id)
        & review_log["split_scheme"].eq(scheme)
        & review_log["split_name"].eq(partition)
    ]
    if current_actions.empty:
        st.caption("Belum terdapat catatan review simulasi untuk klaim ini.")
    else:
        display_actions = current_actions.tail(5).rename(
            columns={
                "recorded_at_utc": "Waktu pencatatan (UTC)",
                "reviewer_role": "Peran pemeriksa",
                "evidence_reviewed": "Bukti yang ditinjau",
                "review_outcome": "Hasil pemeriksaan",
                "follow_up_action": "Tindak lanjut",
                "review_notes": "Alasan pemeriksaan",
                "recommendation_feedback": "Manfaat urutan",
            }
        )
        display_actions["Bukti yang ditinjau"] = display_actions[
            "Bukti yang ditinjau"
        ].str.replace("|", ", ", regex=False)
        st.dataframe(
            display_actions[
                [
                    "Waktu pencatatan (UTC)",
                    "Peran pemeriksa",
                    "Bukti yang ditinjau",
                    "Hasil pemeriksaan",
                    "Tindak lanjut",
                    "Alasan pemeriksaan",
                    "Manfaat urutan",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Hasil yang tercatat menjelaskan pemeriksaan pada simulasi ini. "
            "Hasil tersebut bukan label fraud atau kesimpulan terhadap kebenaran seluruh klaim."
        )
    if not review_log.empty:
        st.download_button(
            "Unduh catatan review simulasi",
            data=review_log.to_csv(index=False).encode("utf-8"),
            file_name="review_records_demo_v2.csv",
            mime="text/csv",
        )
        st.caption("Catatan tersimpan secara lokal dan hanya digunakan untuk demonstrasi.")
