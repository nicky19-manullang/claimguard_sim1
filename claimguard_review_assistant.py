"""Create editable, traceable review and clarification drafts from claim evidence."""


def _format_idr(value):
    return f"Rp {float(value):,.0f}".replace(",", ".")


def _format_ratio(value):
    return f"{float(value):.2f}".replace(".", ",")


def build_review_assistant_content(claim, analysis, comparisons):
    findings = [
        f"Sinyal utama: {analysis['top_driver']}. Rincian sinyal: {analysis['evidence']}."
    ]
    suggested_evidence = ["Ringkasan pelayanan"]

    cost = comparisons["cost"]
    cost_ratio = float(cost["ratio_to_train_median"])
    findings.append(
        f"Total klaim {_format_idr(cost['current_total_idr'])}; nilai tengah pembanding "
        f"{_format_idr(cost['train_median_total_idr'])} dari {cost['group_label']} "
        f"({cost['reference_count']} klaim acuan), rasio {_format_ratio(cost_ratio)} kali."
    )
    suggested_evidence.append("Rincian biaya")

    repeat = comparisons["repeat"]
    previous_claim = repeat["previous_claim"]
    if previous_claim is not None:
        findings.append(
            f"Ada {repeat['prior_count_30d']} klaim terdahulu peserta yang sama dalam 30 hari; "
            f"klaim sebelumnya yang tercatat adalah {previous_claim['claim_id']} "
            f"({previous_claim['submitted_at']})."
        )
        suggested_evidence.append("Riwayat klaim")

    similar = comparisons["similar_claim"]
    if similar is not None:
        findings.append(
            f"Klaim terdahulu {similar['claim_id']} ({similar['submitted_at']}) memiliki "
            f"kemiripan kode {float(similar['similarity']):.0%}; kemiripan tidak membuktikan "
            "duplikasi atau kesalahan."
        )
        suggested_evidence.extend(["Diagnosis dan kode", "Tindakan dan item"])

    severity = comparisons["severity"]
    if severity is not None:
        findings.append(
            f"Konteks rawat inap: lama rawat {float(severity['current_stay_days']):g} hari; "
            f"median kelompok acuan {float(severity['train_median_stay_days']):.1f} hari "
            f"({severity['group_label']}). Ini pembanding statistik, bukan penilaian klinis."
        )
        suggested_evidence.extend(["Diagnosis dan kode", "Ringkasan pelayanan"])

    if cost_ratio > 1:
        clarification_request = (
            "Mohon klarifikasi rincian biaya klaim ini dengan melampirkan rincian atau dokumen "
            "pendukung yang relevan. Nilai total tercatat "
            f"{_format_idr(cost['current_total_idr'])}, sedangkan nilai tengah kelompok "
            f"pembanding adalah {_format_idr(cost['train_median_total_idr'])}. "
        )
    else:
        clarification_request = (
            "Mohon tinjau kembali rincian klaim dan lengkapi dokumen pendukung yang relevan "
            "agar petugas dapat memastikan kesesuaian data. "
        )

    if previous_claim is not None:
        clarification_request += (
            f"Terdapat {repeat['prior_count_30d']} klaim terdahulu peserta yang sama dalam "
            "30 hari; mohon konfirmasi hubungan pelayanan dan klaim tersebut. "
        )
    if similar is not None:
        clarification_request += (
            f"Sejumlah kode pada klaim ini mirip dengan klaim terdahulu {similar['claim_id']}; "
            "mohon konfirmasi rincian diagnosis, tindakan, dan item yang diajukan. "
        )
    if severity is not None:
        clarification_request += (
            "Mohon sertakan ringkasan pelayanan yang menjelaskan konteks diagnosis dan lama rawat. "
        )
    clarification_request += (
        "Indikator ini merupakan petunjuk untuk ditinjau, bukan kesimpulan kesalahan. "
        "Silakan periksa dan sesuaikan draf ini sebelum menyampaikannya melalui kanal resmi."
    )

    ordered_suggestions = tuple(dict.fromkeys(suggested_evidence))
    return {
        "findings": tuple(findings),
        "suggested_evidence": ordered_suggestions,
        "clarification_request": clarification_request,
    }
