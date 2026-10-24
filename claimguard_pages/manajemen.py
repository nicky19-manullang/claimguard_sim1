"""Menu Manajemen."""
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


# ------------------------------ manajemen --------------------------------
def _render_management() -> None:
    _page_header(
        "Manajemen",
        "Pemantauan tindak lanjut",
        "Pantau klaim yang masih menunggu petugas atau faskes, lihat beban kerja pemeriksa, dan buka "
        "klaim yang perlu ditindaklanjuti. Seluruh proses ini simulasi pada data sintetis.",
    )
    settings = _load_settings()
    scope = _scope_controls("mg")
    try:
        data = _load_scope(scope["scheme"], scope["partition"])
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return
    universe = data.loc[data["review_recommended"]] if scope["recommended_only"] else data
    log = _workflow_frame()
    scoped_log = log.loc[
        log["analysis_source"].eq("v2")
        & log["analysis_path"].eq(scope["scheme"])
        & log["split_name"].eq(scope["partition"])
    ]
    latest = scoped_log.drop_duplicates("claim_id", keep="last")[
        ["claim_id", "status_after_action", "review_outcome", "action", "actor", "recorded_at"]
    ]
    board = universe.merge(latest, left_on="id_klaim", right_on="claim_id", how="left")
    kinds = _status_kinds()
    board["Kelompok"] = board["status_after_action"].map(lambda status: _kind_of(status, kinds))
    now = pd.Timestamp.now(tz="UTC")
    board["Menunggu (hari)"] = ((now - board["recorded_at"]).dt.total_seconds() / 86400).round(1)
    waiting = board["Kelompok"].isin(["Menunggu petugas", "Menunggu faskes"])
    overdue = waiting & (board["Menunggu (hari)"] > settings["overdue_days"])
    counts = board["Kelompok"].value_counts().reindex(KIND_ORDER, fill_value=0)

    columns = st.columns(5)
    columns[0].metric("Belum ditinjau", _format_number(counts["Belum ditinjau"]))
    columns[1].metric("Menunggu petugas", _format_number(counts["Menunggu petugas"]))
    columns[2].metric("Menunggu faskes", _format_number(counts["Menunggu faskes"]))
    columns[3].metric("Selesai (diteruskan)", _format_number(counts["Selesai (diteruskan)"]))
    columns[4].metric(
        f"Lewat {settings['overdue_days']} hari",
        _format_number(int(overdue.sum())),
        help="Batas hari dapat diubah di menu Pengaturan.",
    )
    board_counts = counts.loc[counts > 0].rename_axis("Kelompok").reset_index(name="Jumlah klaim")
    if not board_counts.empty:
        _section("Posisi klaim saat ini", f"{_format_number(len(board))} klaim pada cakupan yang dipilih.")
        _show_chart(_donut_chart(board_counts, "Kelompok", "Jumlah klaim",
                                 [KIND_COLORS[name] for name in board_counts["Kelompok"]]))

    action_tab, all_tab, workload_tab, health_tab = st.tabs(
        ["Perlu ditindaklanjuti", "Semua klaim", "Beban kerja", "Kesehatan data"]
    )
    display_columns = {
        "priority_rank": "Urutan", "id_klaim": "Nomor klaim", "facility_id": "Fasilitas",
        "top_driver": "Alasan utama untuk ditinjau", "Kelompok": "Posisi", "status_after_action": "Status",
        "Menunggu (hari)": "Menunggu (hari)",
    }

    def _table(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values("priority_rank", kind="stable")[list(display_columns)].rename(columns=display_columns)

    with action_tab:
        _section(
            f"Menunggu lebih dari {settings['overdue_days']} hari",
            "Klaim yang sudah lama menunggu langkah berikutnya dari petugas atau faskes.",
        )
        if overdue.any():
            st.dataframe(_table(board.loc[overdue]), width="stretch", hide_index=True)
        else:
            st.caption("Tidak ada klaim yang melewati batas waktu.")
        _section("Sedang menunggu tindak lanjut")
        if waiting.any():
            st.dataframe(_table(board.loc[waiting]), width="stretch", hide_index=True)
        else:
            st.caption("Tidak ada klaim yang sedang menunggu.")
        _section("Belum ditinjau, urut prioritas", "Dua puluh klaim dengan urutan prioritas tertinggi yang belum dicatat.")
        pending = board.loc[board["Kelompok"].eq("Belum ditinjau")]
        if pending.empty:
            st.caption("Semua klaim pada cakupan ini sudah memiliki catatan.")
        else:
            st.dataframe(_table(pending).head(20), width="stretch", hide_index=True)
        candidates = pd.concat([board.loc[overdue], board.loc[waiting & ~overdue], pending.sort_values("priority_rank").head(20)])
        candidate_ids = candidates["id_klaim"].astype(str).drop_duplicates().tolist()
        if candidate_ids:
            _section("Buka klaim")
            open_columns = st.columns([2, 1], vertical_alignment="bottom")
            with open_columns[0]:
                chosen = st.selectbox("Nomor klaim", candidate_ids, key="mg_open_claim")
            with open_columns[1]:
                st.button(
                    "Buka di Pemeriksaan klaim",
                    type="primary",
                    width="stretch",
                    key="mg_open_button",
                    on_click=_open_claim,
                    args=(chosen, scope["scheme_label"], scope["partition_label"]),
                )
    with all_tab:
        chosen_kinds = st.multiselect("Posisi klaim", KIND_ORDER, default=[], key="mg_kinds")
        shown = board if not chosen_kinds else board.loc[board["Kelompok"].isin(chosen_kinds)]
        st.dataframe(_table(shown), width="stretch", hide_index=True, height=420)
        st.download_button(
            "Unduh daftar (CSV)",
            data=_table(shown).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"posisi_klaim_{scope['scheme']}_{scope['partition']}.csv",
            mime="text/csv",
            key="mg_dl",
        )
    with workload_tab:
        staff = scoped_log.loc[scoped_log["actor"].eq("Petugas")]
        if staff.empty:
            st.info("Belum ada pemeriksaan petugas yang tercatat pada cakupan ini.")
        else:
            workload = (
                staff.assign(Peran=staff["reviewer_role"].replace("", "Tidak dicatat"))
                .groupby("Peran")
                .agg(**{"Jumlah pemeriksaan": ("claim_id", "count"), "Klaim berbeda": ("claim_id", "nunique")})
                .reset_index()
                .sort_values("Jumlah pemeriksaan", ascending=False)
            )
            _section("Pemeriksaan per peran", "Jumlah catatan pemeriksaan yang dibuat tiap peran.")
            _show_chart(_bar_chart(workload, "Peran", "Jumlah pemeriksaan", color=BRAND["amber"], horizontal=True))
            st.dataframe(workload, width="stretch", hide_index=True)
        responses = scoped_log.loc[~scoped_log["actor"].eq("Petugas") & scoped_log["actor"].ne("")]
        _section("Tanggapan faskes (simulasi)")
        if responses.empty:
            st.caption("Belum ada tanggapan faskes yang tercatat.")
        else:
            st.dataframe(
                responses["action"].value_counts().rename_axis("Tanggapan").reset_index(name="Jumlah"),
                width="stretch", hide_index=True,
            )
    with health_tab:
        _section("Sumber data", "Pastikan berkas yang dipakai dashboard tersedia dan terbaru.")
        count_rows = st.checkbox("Hitung jumlah baris (lebih lambat)", key="mg_count_rows")
        st.dataframe(_data_source_table(count_rows), width="stretch", hide_index=True)
        st.caption("Pengelolaan cache dan arsip riwayat tersedia di menu Pengaturan.")


