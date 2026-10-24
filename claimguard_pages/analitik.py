"""Menu Analitik."""
from __future__ import annotations

import html
import io
import json
import platform
import shutil
import sys
from datetime import datetime, timedelta, timezone

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
from claimguard_pages.common import *  # noqa: F401,F403


# ------------------------------ analitik ---------------------------------
def _render_analytics() -> None:
    _page_header(
        "Analitik",
        "Gambaran antrean pemeriksaan",
        "Pola pada klaim yang masuk antrean: alasan utama, sebaran skor, perkembangan menurut waktu, "
        "fasilitas, dan kekuatan sinyal. Semua angka berasal dari data sintetis dan bukan ukuran kecurangan.",
    )
    scope = _scope_controls("an")
    try:
        data = _load_scope(scope["scheme"], scope["partition"])
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return
    if data.empty:
        st.info("Tidak ada klaim pada kumpulan data ini.")
        return
    view = data.loc[data["review_recommended"]] if scope["recommended_only"] else data
    if view.empty:
        st.info("Tidak ada klaim yang disarankan untuk ditinjau pada kumpulan data ini.")
        return
    recommended_total = int(data["review_recommended"].sum())
    share = recommended_total / len(data) * 100
    columns = st.columns(5)
    columns[0].metric("Klaim dalam cakupan", _format_number(len(view)))
    columns[1].metric(
        "Disarankan ditinjau",
        _format_number(recommended_total),
        help=f"{_format_percent(share)} dari {_format_number(len(data))} klaim pada kumpulan data ini.",
    )
    columns[2].metric("Total nilai klaim", _format_money(view["claim_total_idr"].sum()))
    columns[3].metric("Nilai tengah klaim", _format_money(view["claim_total_idr"].median()))
    columns[4].metric("Rata-rata skor prioritas", f"{view['priority_score'].mean():.2f}".replace(".", ","))

    summary_tab, trend_tab, facility_tab, signal_tab = st.tabs(
        ["Ringkasan", "Tren waktu", "Fasilitas", "Sinyal"]
    )
    with summary_tab:
        left, right = st.columns(2, gap="large")
        with left:
            _section("Alasan utama untuk ditinjau", "Jumlah klaim menurut sinyal yang paling menonjol.")
            drivers = view["top_driver"].value_counts().rename_axis("Alasan utama").reset_index(name="Jumlah klaim")
            _show_chart(_bar_chart(drivers, "Alasan utama", "Jumlah klaim", color=BRAND["teal"], horizontal=True))
        with right:
            _section("Jenis layanan", "Porsi rawat jalan dan rawat inap pada cakupan ini.")
            services = view["jenis_layanan"].value_counts().rename_axis("Jenis layanan").reset_index(name="Jumlah klaim")
            _show_chart(_donut_chart(services, "Jenis layanan", "Jumlah klaim", [BRAND["teal"], BRAND["blue"], BRAND["amber"]]))
        _section("Sebaran skor prioritas", "Skor membandingkan klaim satu sama lain pada data buatan; bukan kemungkinan kecurangan.")
        scores = view["priority_score"].dropna()
        if scores.nunique() > 1:
            buckets = pd.cut(scores, bins=10).value_counts(sort=False)
            histogram = pd.DataFrame(
                {
                    "Rentang skor": [f"{item.left:.2f}–{item.right:.2f}" for item in buckets.index],
                    "Jumlah klaim": buckets.to_numpy(),
                }
            )
            _show_chart(_bar_chart(histogram, "Rentang skor", "Jumlah klaim", color=BRAND["blue"]))
        else:
            st.caption("Skor prioritas tidak bervariasi pada cakupan ini.")
    with trend_tab:
        dated = data.dropna(subset=["submitted_at"]).copy()
        if dated.empty:
            st.info("Tanggal pengajuan tidak tersedia untuk membuat tren.")
        else:
            dated["Bulan"] = dated["submitted_at"].dt.to_period("M").dt.to_timestamp()
            monthly = (
                dated.groupby("Bulan")
                .agg(
                    **{
                        "Klaim masuk": ("id_klaim", "count"),
                        "Disarankan ditinjau": ("review_recommended", "sum"),
                        "Total nilai (juta Rp)": ("claim_total_idr", lambda values: values.sum() / 1_000_000),
                    }
                )
                .reset_index()
            )
            _section("Klaim menurut bulan pengajuan", "Seluruh klaim pada kumpulan data, tanpa memandang penyaringan di atas.")
            long = monthly.melt(
                id_vars="Bulan", value_vars=["Klaim masuk", "Disarankan ditinjau"],
                var_name="Seri", value_name="Jumlah klaim",
            )
            _show_chart(_line_chart(long, "Bulan", "Jumlah klaim", "Seri",
                                    {"Klaim masuk": BRAND["blue"], "Disarankan ditinjau": BRAND["amber"]}))
            _section("Nilai klaim menurut bulan")
            value_long = monthly[["Bulan", "Total nilai (juta Rp)"]].assign(Seri="Total nilai (juta Rp)")
            _show_chart(_line_chart(value_long, "Bulan", "Total nilai (juta Rp)", "Seri",
                                    {"Total nilai (juta Rp)": BRAND["teal"]}))
            st.dataframe(
                monthly.assign(Bulan=monthly["Bulan"].dt.strftime("%b %Y")).round(1),
                width="stretch", hide_index=True,
            )
    with facility_tab:
        facilities = (
            data.groupby("facility_id")
            .agg(
                Klaim=("id_klaim", "count"),
                Disarankan=("review_recommended", "sum"),
                Total_nilai=("claim_total_idr", "sum"),
                Rata_skor=("priority_score", "mean"),
            )
            .reset_index()
        )
        facilities["Porsi disarankan (%)"] = facilities["Disarankan"] / facilities["Klaim"] * 100
        facilities = facilities.sort_values(["Disarankan", "Klaim"], ascending=False, kind="stable")
        _section(
            "Fasilitas dengan klaim terbanyak dalam antrean",
            "Jumlah klaim yang disarankan ditinjau bukan ukuran tingkat kecurangan fasilitas; "
            "fasilitas besar wajar memiliki lebih banyak klaim.",
        )
        top = facilities.head(15).rename(columns={"facility_id": "Fasilitas", "Disarankan": "Disarankan ditinjau"})
        _show_chart(_bar_chart(top, "Fasilitas", "Disarankan ditinjau", color=BRAND["blue"], horizontal=True))
        table = facilities.rename(
            columns={
                "facility_id": "Fasilitas", "Disarankan": "Disarankan ditinjau",
                "Total_nilai": "Total nilai klaim (Rp)", "Rata_skor": "Rata-rata skor",
            }
        ).round({"Rata-rata skor": 2, "Porsi disarankan (%)": 1})
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Porsi disarankan (%)": st.column_config.ProgressColumn(
                    "Porsi disarankan (%)", format="%.0f", min_value=0, max_value=100
                ),
                "Total nilai klaim (Rp)": st.column_config.NumberColumn("Total nilai klaim (Rp)", format="%d"),
            },
        )
        st.download_button(
            "Unduh tabel fasilitas (CSV)",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"analitik_fasilitas_{scope['scheme']}_{scope['partition']}.csv",
            mime="text/csv",
            key="an_dl_fac",
        )
    with signal_tab:
        signal_columns = list(V2_SIGNAL_LABELS)
        signals = data[signal_columns].apply(pd.to_numeric, errors="coerce")
        scale = 100.0 if signals.max().max() > 1.5 else 1.0
        _section(
            "Kekuatan sinyal",
            "Rata-rata persentil tiap sinyal pada klaim yang disarankan ditinjau dibandingkan klaim lainnya.",
        )
        grouped = signals.groupby(data["review_recommended"]).mean()
        comparison = pd.DataFrame(
            {
                "Sinyal": [V2_SIGNAL_LABELS[column] for column in signal_columns],
                "Disarankan ditinjau": [grouped[column].get(True, float("nan")) for column in signal_columns],
                "Klaim lainnya": [grouped[column].get(False, float("nan")) for column in signal_columns],
            }
        )
        comparison["Selisih"] = comparison["Disarankan ditinjau"] - comparison["Klaim lainnya"]
        _show_chart(_bar_chart(comparison.round(3), "Sinyal", "Disarankan ditinjau", color=BRAND["teal"], horizontal=True))
        st.dataframe(comparison.round(3), width="stretch", hide_index=True)
        _section("Klaim dengan beberapa sinyal tinggi", "Berapa banyak sinyal yang sama-sama berada di atas ambang.")
        threshold_percent = st.slider("Ambang sinyal tinggi (persentil)", 50, 99, 90, key="an_threshold")
        high = signals.loc[view.index.intersection(signals.index)] >= threshold_percent / 100 * scale
        count_high = high.sum(axis=1).value_counts().reindex(range(len(signal_columns) + 1), fill_value=0)
        distribution = pd.DataFrame(
            {"Jumlah sinyal tinggi": [str(number) for number in count_high.index], "Jumlah klaim": count_high.to_numpy()}
        )
        _show_chart(_bar_chart(distribution, "Jumlah sinyal tinggi", "Jumlah klaim", color=BRAND["amber"]))
        st.caption("Banyaknya sinyal tinggi hanya membantu menentukan urutan pemeriksaan, bukan bukti kesalahan.")


