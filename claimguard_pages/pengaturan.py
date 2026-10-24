"""Menu Pengaturan."""
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


# ------------------------------ pengaturan -------------------------------
def _render_settings() -> None:
    _page_header(
        "Pengaturan",
        "Preferensi dan pengelolaan data",
        "Atur tampilan bawaan dashboard, batas waktu tindak lanjut, cadangan riwayat simulasi, "
        "dan cek lingkungan aplikasi.",
    )
    preferences_tab, data_tab, backup_tab, about_tab = st.tabs(
        ["Preferensi", "Data dan cache", "Cadangan dan arsip", "Tentang"]
    )
    settings = _load_settings()
    with preferences_tab:
        _section("Tampilan bawaan", "Berlaku untuk halaman Pemeriksaan klaim, Analitik, dan Manajemen.")
        scheme_options = list(V2_SPLIT_OPTIONS)
        partition_options = list(V2_PARTITIONS)
        role_options = [""] + list(REVIEWER_ROLES)
        with st.form("settings_form"):
            scheme_label = st.selectbox(
                "Cara mengelompokkan klaim", scheme_options,
                index=scheme_options.index(settings["default_scheme_label"]),
            )
            partition_label = st.selectbox(
                "Kumpulan data", partition_options,
                index=partition_options.index(settings["default_partition_label"]),
            )
            recommended_only = st.checkbox(
                "Tampilkan hanya klaim yang disarankan ditinjau", value=settings["default_recommended_only"]
            )
            overdue_days = st.number_input(
                "Batas hari menunggu tindak lanjut", min_value=1, max_value=60,
                value=int(settings["overdue_days"]), step=1,
                help="Klaim yang menunggu lebih lama dari batas ini ditandai di menu Manajemen.",
            )
            default_role = st.selectbox(
                "Peran pemeriksa bawaan", role_options,
                index=role_options.index(settings["default_reviewer_role"]),
                format_func=lambda value: value or "(tidak diatur)",
            )
            assistant_expanded = st.checkbox(
                "Buka ringkasan saran ClaimGuard secara otomatis", value=settings["assistant_expanded"]
            )
            saved = st.form_submit_button("Simpan pengaturan", type="primary")
        if saved:
            try:
                _save_settings(
                    {
                        "default_scheme_label": scheme_label,
                        "default_partition_label": partition_label,
                        "default_recommended_only": bool(recommended_only),
                        "overdue_days": int(overdue_days),
                        "default_reviewer_role": default_role,
                        "assistant_expanded": bool(assistant_expanded),
                    }
                )
            except OSError as error:
                st.error(f"Pengaturan tidak dapat disimpan: {error}")
            else:
                st.success("Pengaturan disimpan. Berlaku saat halaman terkait dibuka kembali.")
        if st.button("Kembalikan ke bawaan pabrik", key="settings_reset"):
            try:
                _save_settings(dict(DEFAULT_SETTINGS))
            except OSError as error:
                st.error(f"Pengaturan tidak dapat dikembalikan: {error}")
            else:
                st.success("Pengaturan dikembalikan. Muat ulang halaman untuk melihat nilainya di formulir.")
        st.caption(f"Pengaturan disimpan di {SETTINGS_FILE.relative_to(ROOT) if SETTINGS_FILE.is_relative_to(ROOT) else SETTINGS_FILE}.")
    with data_tab:
        _section("Status berkas", "Daftar berkas yang dibaca dashboard.")
        count_rows = st.checkbox("Hitung jumlah baris (lebih lambat)", key="st_count_rows")
        st.dataframe(_data_source_table(count_rows), width="stretch", hide_index=True)
        _section("Cache data", "Muat ulang bila berkas data di folder out sudah diganti saat dashboard berjalan.")
        if st.button("Bersihkan cache data", key="st_clear_cache"):
            st.cache_data.clear()
            st.success("Cache dibersihkan. Data akan dibaca ulang dari berkas.")
    with backup_tab:
        _section("Cadangkan riwayat langkah simulasi")
        log = _workflow_frame()
        stamp = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=7)).strftime("%Y%m%d_%H%M%S")
        if log.empty:
            st.caption("Belum ada riwayat untuk dicadangkan.")
        else:
            export = log[WORKFLOW_COLUMNS]
            st.download_button(
                f"Unduh seluruh riwayat ({_format_number(len(export))} langkah, CSV)",
                data=export.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"cadangan_riwayat_{stamp}.csv",
                mime="text/csv",
                type="primary",
                key="st_dl_backup",
            )
        _section(
            "Arsipkan dan kosongkan riwayat",
            "Berkas riwayat dipindahkan ke folder cadangan, bukan dihapus. Riwayat di dashboard menjadi kosong "
            "dan alur klaim kembali ke awal. Untuk memulihkan, salin berkas cadangan kembali ke folder out.",
        )
        confirm_text = st.text_input("Ketik ARSIPKAN untuk mengonfirmasi", key="st_archive_confirm")
        if st.button("Arsipkan riwayat sekarang", key="st_archive"):
            if confirm_text.strip() != "ARSIPKAN":
                st.error("Konfirmasi tidak sesuai. Ketik ARSIPKAN persis seperti tertulis.")
            else:
                moved = []
                try:
                    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                    for path in (WORKFLOW_LOG, LEGACY_WORKFLOW_LOG):
                        if path.is_file():
                            target = BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}"
                            shutil.move(str(path), str(target))
                            moved.append(target.name)
                except OSError as error:
                    st.error(f"Riwayat tidak dapat diarsipkan: {error}")
                else:
                    if moved:
                        st.success("Riwayat diarsipkan ke folder backup_riwayat: " + ", ".join(moved))
                    else:
                        st.info("Tidak ada berkas riwayat yang perlu diarsipkan.")
        backups = sorted(BACKUP_DIR.glob("*.csv"), reverse=True) if BACKUP_DIR.is_dir() else []
        if backups:
            _section("Arsip tersimpan")
            for path in backups[:10]:
                columns = st.columns([3, 1], vertical_alignment="center")
                columns[0].write(f"{path.name} · {path.stat().st_size / 1024:.1f} KB")
                columns[1].download_button(
                    "Unduh", data=path.read_bytes(), file_name=path.name, mime="text/csv",
                    key=f"st_dl_{path.name}",
                )
    with about_tab:
        _section("Lingkungan aplikasi")
        info = pd.DataFrame(
            [
                {"Komponen": "Python", "Nilai": sys.version.split()[0]},
                {"Komponen": "Streamlit", "Nilai": st.__version__},
                {"Komponen": "pandas", "Nilai": pd.__version__},
                {"Komponen": "Altair", "Nilai": alt.__version__},
                {"Komponen": "Sistem operasi", "Nilai": platform.platform()},
                {"Komponen": "Folder proyek", "Nilai": str(ROOT)},
                {"Komponen": "Folder data", "Nilai": str(OUT)},
            ]
        )
        st.dataframe(info, width="stretch", hide_index=True)
        st.markdown(
            "ClaimGuard adalah prototipe demonstrasi dengan data sintetis. Tidak terhubung dengan "
            "BPJS Kesehatan atau sistem klaim resmi, dan hasilnya hanya petunjuk untuk pemeriksaan petugas."
        )

