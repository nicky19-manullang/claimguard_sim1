"""Menu Laporan."""
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


# ------------------------------ laporan ----------------------------------
def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Tidak ada data._"
    lines = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for row in frame.itertuples(index=False):
        cells = [str(value).replace("|", "/").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_report_markdown(filtered: pd.DataFrame, period_text: str) -> str:
    kinds = _status_kinds()
    latest = _latest_per_claim(filtered)
    generated = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=7)
    staff = filtered.loc[filtered["actor"].eq("Petugas")]
    outcomes = (
        staff["review_outcome"].replace("", pd.NA).dropna().value_counts()
        .rename_axis("Hasil pemeriksaan").reset_index(name="Jumlah")
    )
    final_status = (
        latest["status_after_action"].replace("", pd.NA).dropna().value_counts()
        .rename_axis("Status akhir klaim").reset_index(name="Jumlah klaim")
    )
    actors = filtered["actor"].replace("", pd.NA).dropna().value_counts().rename_axis("Pihak simulasi").reset_index(name="Jumlah langkah")
    feedback = (
        staff["recommendation_feedback"].replace("", pd.NA).dropna().value_counts()
        .rename_axis("Penilaian petunjuk ClaimGuard").reset_index(name="Jumlah")
    )
    finished = int(sum(1 for status in latest["status_after_action"] if kinds.get(status) == "Selesai (diteruskan)"))
    lines = [
        "# Laporan Tindak Lanjut Simulasi ClaimGuard",
        "",
        f"Dibuat: {generated:%d-%m-%Y %H:%M} WIB",
        f"Periode: {period_text}",
        "",
        "> Prototipe dengan data sintetis. Laporan ini bukan bukti kecurangan dan bukan keputusan klaim.",
        "",
        "## Ringkasan",
        "",
        f"- Langkah tercatat: {len(filtered)}",
        f"- Klaim yang ditangani: {latest.shape[0]}",
        f"- Klaim yang telah diteruskan ke proses resmi: {finished}",
        "",
        "## Hasil pemeriksaan petugas",
        "",
        _md_table(outcomes),
        "",
        "## Status akhir klaim",
        "",
        _md_table(final_status),
        "",
        "## Langkah menurut pihak simulasi",
        "",
        _md_table(actors),
        "",
        "## Penilaian petunjuk ClaimGuard",
        "",
        _md_table(feedback),
        "",
    ]
    return "\n".join(lines)


def _history_display(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.sort_values("recorded_at", ascending=False, kind="stable")[
        [
            "waktu_wib", "claim_id", "analysis_path", "split_name", "actor", "reviewer_role",
            "evidence_reviewed", "review_outcome", "action", "status_after_action", "notes",
        ]
    ].copy()
    table["waktu_wib"] = table["waktu_wib"].dt.strftime("%d-%m-%Y %H:%M")
    return table.rename(
        columns={
            "waktu_wib": "Waktu (WIB)",
            "claim_id": "Nomor klaim",
            "analysis_path": "Pengelompokan",
            "split_name": "Kumpulan data",
            "actor": "Pihak simulasi",
            "reviewer_role": "Peran pemeriksa",
            "evidence_reviewed": "Bukti yang ditinjau",
            "review_outcome": "Hasil pemeriksaan",
            "action": "Langkah",
            "status_after_action": "Status setelah langkah",
            "notes": "Catatan",
        }
    )


def _render_timeline(steps: pd.DataFrame) -> None:
    items = []
    for _, row in steps.sort_values("recorded_at", kind="stable").iterrows():
        stamp = row["waktu_wib"].strftime("%d-%m-%Y %H:%M WIB") if pd.notna(row["waktu_wib"]) else "Waktu tidak tercatat"
        meta = [f"Pihak: {html.escape(str(row['actor']))}"]
        if row["reviewer_role"]:
            meta.append(f"Peran: {html.escape(str(row['reviewer_role']))}")
        if row["review_outcome"]:
            meta.append(f"Hasil pemeriksaan: {html.escape(str(row['review_outcome']))}")
        if row["status_after_action"]:
            meta.append(f"Status: {html.escape(str(row['status_after_action']))}")
        if row["evidence_reviewed"]:
            meta.append(f"Bukti ditinjau: {html.escape(str(row['evidence_reviewed']))}")
        if row["notes"]:
            meta.append(f"Catatan: {html.escape(str(row['notes']))}")
        items.append(
            '<div class="cg-tl-item">'
            f'<div class="cg-tl-time">{stamp}</div>'
            f'<div class="cg-tl-title">{html.escape(str(row["action"]))}</div>'
            f'<div class="cg-tl-meta">{"<br>".join(meta)}</div></div>'
        )
    st.markdown(f'<div class="cg-tl">{"".join(items)}</div>', unsafe_allow_html=True)


def _render_reports() -> None:
    _page_header(
        "Laporan",
        "Laporan pemeriksaan dan tindak lanjut",
        "Riwayat langkah simulasi, ringkasan hasil pemeriksaan petugas, dan ekspor data untuk "
        "dokumentasi. Semua isi berasal dari catatan simulasi pada data sintetis.",
    )
    frame = _workflow_frame()
    if frame.empty:
        st.info(
            "Belum ada langkah simulasi yang tercatat. Catat hasil pemeriksaan pada menu "
            "Pemeriksaan klaim, lalu riwayatnya akan muncul di sini."
        )
        st.button("Buka Pemeriksaan klaim", type="primary", key="rp_empty_open",
                  on_click=_go_page, args=("Pemeriksaan klaim",))
    filtered = frame
    period_text = "Semua periode"
    if not frame.empty:
        _section("Saring laporan")
        dates = frame["waktu_wib"].dropna()
        filter_columns = st.columns([1.2, 1, 1.3, 1])
        if dates.empty:
            date_range = ()
        else:
            first_day, last_day = dates.min().date(), dates.max().date()
            with filter_columns[0]:
                date_range = st.date_input(
                    "Periode (WIB)", value=(first_day, last_day),
                    min_value=first_day, max_value=last_day, key="rp_period",
                )
        with filter_columns[1]:
            actors = sorted(value for value in frame["actor"].unique() if value)
            chosen_actors = st.multiselect("Pihak simulasi", actors, key="rp_actors")
        with filter_columns[2]:
            outcomes = sorted(value for value in frame["review_outcome"].unique() if value)
            chosen_outcomes = st.multiselect("Hasil pemeriksaan", outcomes, key="rp_outcomes")
        with filter_columns[3]:
            search = st.text_input("Cari nomor klaim", key="rp_search").strip().casefold()
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start, end = date_range
            in_range = frame["waktu_wib"].dt.date.between(start, end)
            filtered = filtered.loc[in_range | frame["waktu_wib"].isna()]
            period_text = f"{start:%d-%m-%Y} sampai {end:%d-%m-%Y}"
        if chosen_actors:
            filtered = filtered.loc[filtered["actor"].isin(chosen_actors)]
        if chosen_outcomes:
            filtered = filtered.loc[filtered["review_outcome"].isin(chosen_outcomes)]
        if search:
            filtered = filtered.loc[filtered["claim_id"].str.casefold().str.contains(search, regex=False)]

    history_tab, summary_tab, export_tab = st.tabs(
        ["Riwayat langkah simulasi", "Ringkasan pemeriksaan", "Ekspor"]
    )
    kinds = _status_kinds()
    with history_tab:
        if filtered.empty:
            st.info("Tidak ada langkah yang sesuai dengan penyaringan.")
        else:
            _section(
                "Riwayat langkah simulasi",
                "Seluruh langkah petugas dan tanggapan faskes (simulasi), terbaru di atas.",
            )
            st.dataframe(_history_display(filtered), width="stretch", hide_index=True, height=380)
            _section("Alur satu klaim", "Pilih klaim untuk melihat urutan langkahnya dari awal.")
            claim_ids = sorted(filtered["claim_id"].unique().tolist())
            focus = st.session_state.pop("report_focus_claim", None)
            if focus in claim_ids:
                st.session_state["rp_claim"] = focus
            selected_claim = st.selectbox("Nomor klaim", claim_ids, key="rp_claim")
            _render_timeline(filtered.loc[filtered["claim_id"].eq(selected_claim)])
    with summary_tab:
        if filtered.empty:
            st.info("Belum ada data untuk diringkas.")
        else:
            latest = _latest_per_claim(filtered)
            staff = filtered.loc[filtered["actor"].eq("Petugas")]
            finished = int(sum(1 for status in latest["status_after_action"] if kinds.get(status) == "Selesai (diteruskan)"))
            evidence_counts = [
                len([item for item in EVIDENCE_ITEMS if item in str(text)]) for text in staff["evidence_reviewed"]
            ]
            average_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else float("nan")
            columns = st.columns(4)
            columns[0].metric("Langkah tercatat", _format_number(len(filtered)))
            columns[1].metric("Klaim yang ditangani", _format_number(len(latest)))
            columns[2].metric("Diteruskan ke proses resmi", _format_number(finished))
            columns[3].metric(
                "Bukti ditinjau per pemeriksaan",
                "Tidak tersedia" if pd.isna(average_evidence) else f"{average_evidence:.1f}".replace(".", ","),
            )
            left, right = st.columns(2, gap="large")
            with left:
                _section("Langkah per hari")
                daily = filtered.dropna(subset=["waktu_wib"]).copy()
                if daily.empty:
                    st.caption("Waktu pencatatan tidak tersedia.")
                else:
                    daily["Tanggal"] = daily["waktu_wib"].dt.strftime("%d-%m")
                    daily = daily.groupby("Tanggal", sort=False).size().reset_index(name="Jumlah langkah")
                    _show_chart(_bar_chart(daily, "Tanggal", "Jumlah langkah", color=BRAND["teal"]))
            with right:
                _section("Status akhir klaim")
                final_kinds = latest["status_after_action"].map(lambda status: _kind_of(status, kinds))
                kind_counts = final_kinds.value_counts().reindex(KIND_ORDER, fill_value=0)
                kind_counts = kind_counts.loc[kind_counts > 0]
                kind_frame = kind_counts.rename_axis("Kelompok").reset_index(name="Jumlah klaim")
                _show_chart(_donut_chart(kind_frame, "Kelompok", "Jumlah klaim",
                                         [KIND_COLORS[name] for name in kind_frame["Kelompok"]]))
            left, right = st.columns(2, gap="large")
            with left:
                _section("Hasil pemeriksaan petugas")
                outcome_counts = (
                    staff["review_outcome"].replace("", pd.NA).dropna().value_counts()
                    .rename_axis("Hasil pemeriksaan").reset_index(name="Jumlah")
                )
                if outcome_counts.empty:
                    st.caption("Belum ada hasil pemeriksaan petugas.")
                else:
                    _show_chart(_bar_chart(outcome_counts, "Hasil pemeriksaan", "Jumlah",
                                           color=BRAND["blue"], horizontal=True))
            with right:
                _section("Bukti yang paling sering ditinjau")
                evidence_frame = _count_evidence(staff["evidence_reviewed"])
                if evidence_frame.empty:
                    st.caption("Belum ada bukti yang tercatat.")
                else:
                    _show_chart(_bar_chart(evidence_frame, "Bukti", "Jumlah pemeriksaan",
                                           color=BRAND["teal"], horizontal=True))
            left, right = st.columns(2, gap="large")
            with left:
                _section("Peran pemeriksa")
                roles = (
                    staff["reviewer_role"].replace("", pd.NA).dropna().value_counts()
                    .rename_axis("Peran").reset_index(name="Jumlah pemeriksaan")
                )
                if roles.empty:
                    st.caption("Belum ada peran yang tercatat.")
                else:
                    _show_chart(_bar_chart(roles, "Peran", "Jumlah pemeriksaan",
                                           color=BRAND["amber"], horizontal=True))
            with right:
                _section("Apakah petunjuk ClaimGuard membantu?")
                feedback = (
                    staff["recommendation_feedback"].replace("", pd.NA).dropna().value_counts()
                    .rename_axis("Penilaian").reset_index(name="Jumlah")
                )
                if feedback.empty:
                    st.caption("Belum ada penilaian dari petugas.")
                else:
                    _show_chart(_bar_chart(feedback, "Penilaian", "Jumlah",
                                           color=BRAND["rose"], horizontal=True))
    with export_tab:
        _section("Unduh laporan", "Berkas dibuat dari penyaringan yang sedang aktif di atas.")
        stamp = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=7)).strftime("%Y%m%d_%H%M")
        if filtered.empty:
            st.caption("Belum ada riwayat untuk diunduh.")
        else:
            columns = st.columns(2)
            with columns[0]:
                st.download_button(
                    "Unduh riwayat langkah (CSV)",
                    data=_history_display(filtered).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"riwayat_langkah_{stamp}.csv",
                    mime="text/csv",
                    type="primary",
                    key="rp_dl_csv",
                    width="stretch",
                )
            with columns[1]:
                st.download_button(
                    "Unduh ringkasan laporan (Markdown)",
                    data=_build_report_markdown(filtered, period_text).encode("utf-8"),
                    file_name=f"laporan_tindak_lanjut_{stamp}.md",
                    mime="text/markdown",
                    key="rp_dl_md",
                    width="stretch",
                )
        _section("Ekspor antrean pemeriksaan", "Daftar klaim pada antrean, lengkap dengan alasan utama dan skor prioritas.")
        scope = _scope_controls("rp_ex")
        try:
            data = _load_scope(scope["scheme"], scope["partition"])
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
        else:
            if scope["recommended_only"]:
                data = data.loc[data["review_recommended"]]
            queue_table = data.sort_values("priority_rank", kind="stable")[
                ["priority_rank", "id_klaim", "submitted_at", "jenis_layanan", "facility_id",
                 "top_driver", "priority_score", "claim_total_idr", "review_recommended"]
            ].rename(
                columns={
                    "priority_rank": "Urutan", "id_klaim": "Nomor klaim", "submitted_at": "Tanggal pengajuan",
                    "jenis_layanan": "Jenis layanan", "facility_id": "Fasilitas",
                    "top_driver": "Alasan utama untuk ditinjau", "priority_score": "Skor prioritas",
                    "claim_total_idr": "Total klaim (Rp)", "review_recommended": "Disarankan ditinjau",
                }
            )
            st.caption(f"{_format_number(len(queue_table))} klaim siap diunduh.")
            st.download_button(
                "Unduh antrean (CSV)",
                data=queue_table.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"antrean_pemeriksaan_{scope['scheme']}_{scope['partition']}_{stamp}.csv",
                mime="text/csv",
                key="rp_dl_queue",
            )


