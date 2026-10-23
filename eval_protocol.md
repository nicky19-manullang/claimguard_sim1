# ClaimGuard JKN: Protokol Evaluasi (Tahap 0)

Dikunci sebelum membuat fitur. Kalau ada yang diubah, ubah di sini dulu, jangan setelah melihat hasil.

## 1. Unit analisis

**Klaim** (satu baris `claims.csv`). Skor diberikan per klaim.

- Repeat billing dan cloning: yang berlabel anomali adalah klaim **salinan/ulangannya**; klaim sumber tetap normal. Detektor wajib bisa menunjuk salinannya (`related_claim_id` dipakai untuk fitur klaim pembanding di tahap 7, bukan untuk skor).
- Level faskes dan episode dipakai sebagai fitur dan analisis tambahan, bukan unit evaluasi utama.

## 2. Skema split

| Skema | Train | Val | Test | Mengukur |
|---|---|---|---|---|
| `faskes` | ~60% faskes | ~20% faskes | ~20% faskes | generalisasi ke faskes yang belum pernah dilihat |
| `waktu` | s.d. 31 Agu 2025 | Sep-Okt | Nov-Des | stabilitas ke periode berikutnya |

Faskes berisiko tersebar di ketiga split. Skema `faskes` adalah laporan utama; `waktu` sebagai uji pendamping.

## 3. Aturan anti-kebocoran

1. Fitur hanya dari `data/`. `assert_no_label_columns()` dijalankan pada tabel fitur.
2. Semua langkah yang mem-fit (imputasi, statistik per kelompok, model expected cost, bobot ensemble) di-fit **hanya pada train**; bobot dan ambang dipilih di **val**; **test dibuka sekali** di akhir tiap skenario.
3. EDA hanya pada train.
4. `eval_only/` (label, profil faskes) tidak boleh dibaca oleh kode fitur/model.
5. Ties pada skor dipecah acak dengan seed tetap.

## 4. Metrik

Utama (dilaporkan selalu, dengan CI 95% bootstrap klaster per faskes):

- **PR-AUC** (average precision) dan lift terhadap prevalensi. Prevalensi hanya ~3%, jadi ROC-AUC tidak dipakai sebagai metrik utama.
- **Precision@K** (K = 50, 100, 200, 500) dan **Recall@K**.
- **Precision, Recall, FPR pada anggaran review** (default top 5% klaim). Anggaran bisa juga ditentukan dari ambang di val (`calibrate_on_val=True`).
- **Beban review**: persentase klaim yang harus diperiksa untuk menangkap 50% dan 80% anomali, dan lift terhadap pemeriksaan acak.

Rinci:

- **Recall per modus** dan **per tingkat kesulitan** (easy/medium/hard), plus tabel modus x kesulitan.
- **PR-AUC per modus** (positif = satu modus, negatif = semua klaim normal).
- **Tingkat ditandai pada hard negatives** per tipe (`legit_recurring`, `legit_revisit`, `legit_similar`, `legit_high_cost`). Ini FPR yang paling jujur karena kasus ini sengaja mirip anomali.

Dikerjakan di tahap berikutnya (bukan modul ini): konsistensi skor antar seed, explanation hit-rate (tahap 7), audit disparitas per tipe faskes, simulasi waktu review.

## 5. Uji ketahanan dan pelaporan

- Ulangi pada beberapa seed (`aggregate()` memberi rata-rata dan simpangan baku) dan prevalensi 1-5% (`--prevalence`).
- Bandingkan selalu terhadap baseline: acak, aturan saja, dan hibrida (ablation).
- Laporkan angka test beserta CI, bukan angka val.
- Karena data simulasi, setiap klaim hasil harus dibingkai sebagai "terhadap anomali simulasi", bukan bukti di dunia nyata.

## 6. Hasil uji harness (baseline sederhana, split `faskes/test`, anggaran 5%)

| Baseline | PR-AUC | Catatan |
|---|---|---|
| Acak | 0,030 (prevalensi 2,5%) | harness benar: acak ≈ prevalensi |
| Z-score biaya (fit di train) | 0,133 | recall inflated 0,83, tapi menandai 59% `legit_high_cost` |
| Aturan repeat naif | 0,040 | recall repeat 0,31, tapi menandai 32% `legit_recurring` |

Artinya: baseline satu-sinyal hanya menangkap satu modus dan tertipu hard negatives. Itu yang harus diperbaiki oleh fitur per modus dan detektor hibrida di tahap 4-5.

## 7. Tambahan setelah tahap 4-5

- **Dua mode fitur:** `retro` (batch retrospektif) dan `asof` (online: klaim hanya dibandingkan dengan klaim yang sudah diajukan lebih dulu). Laporkan keduanya; `asof` adalah angka yang lebih realistis untuk penyaringan harian.
- **Recall tingkat pasangan (repeat/cloning):** pada mode `asof`, kalau salinan diajukan lebih dulu daripada klaim sumbernya, klaim yang akhirnya ditandai adalah sumbernya. Karena itu dilaporkan juga recall pasangan (terdeteksi jika salinan ATAU sumbernya ditandai).
- **Pemilihan penggabung detektor:** bobot dipelajari di val, tapi dibandingkan dengan penggabung tanpa pelatihan (maks) lewat cross-validation per faskes di val. Test tidak dipakai untuk memilih.
- **Fitur level faskes (`uc_fac_*`)** tidak dipakai di detektor karena tidak stabil antar split.