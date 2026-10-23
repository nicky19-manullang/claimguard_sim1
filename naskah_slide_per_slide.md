# Naskah Presentasi ClaimGuard JKN

Draft ini mengikuti 20 slide pada deck saat ini. Teks layar dibuat singkat untuk dibaca audiens; narasi presenter memberi konteks tambahan. Seluruh hasil model di bawah merujuk ke data sintetis, bukan klaim peserta JKN nyata.

## Slide 1 — ClaimGuard JKN

**Teks layar**

ClaimGuard JKN  
Dari tumpukan klaim ke daftar prioritas yang bisa dijelaskan  
Deteksi · Jelaskan · Prioritaskan

Prototype analisis risiko klaim JKN berbasis data sintetis

**Narasi presenter**  
ClaimGuard mengubah kumpulan klaim menjadi daftar yang dapat ditinjau berdasarkan tingkat risiko dan alasan yang menyertainya. Sistem ini membantu prioritas pemeriksaan; bukan alat untuk memutuskan bahwa klaim adalah fraud.

**Visual/data**  
Gunakan satu visual sederhana: alur klaim → skor risiko → penjelasan → pemeriksaan manusia. Hapus label kategori dan statistik dari sampul agar judul menjadi fokus.

## Slide 2 — Masalah dan solusi

**Teks layar**

Pemeriksaan klaim perlu tahu tiga hal:

- Klaim mana yang perlu ditinjau lebih dulu?
- Sinyal apa yang membuatnya berisiko?
- Seberapa besar beban pemeriksaan pada tiap ambang?

ClaimGuard menyajikan skor, alasan, dan urutan prioritas. Keputusan akhir tetap pada verifikator.

**Narasi presenter**  
Empat pola yang diuji adalah repeat billing, claim cloning, inflated bills, dan upcoding. Masing-masing menghasilkan sinyal berbeda; skor digunakan untuk membantu menyaring, bukan menggantikan verifikasi klinis atau administrasi.

**Visual/data**  
Empat ikon atau contoh mini untuk tiap modus. Hindari menyebut klaim sebagai fraud sebelum verifikasi.

## Slide 3 — Mengapa prioritas risiko?

**Teks layar**

Tidak semua klaim dapat ditinjau dengan kedalaman yang sama.

Pemeriksaan perlu mengarahkan perhatian ke pola yang tidak biasa, sambil menjaga agar klaim wajar tidak otomatis dianggap bermasalah.

**Narasi presenter**  
Fokus proyek ini adalah membantu penyaringan awal pada skenario sintetis. Klaim tentang total biaya JKN, proporsi biaya rumah sakit, atau besaran fraud hanya ditambahkan bila tersedia sumber resmi, definisi metrik, dan tahun rujukan yang tepat.

**Visual/data**  
Tampilkan diagram sederhana alur klaim menuju antrean pemeriksaan. Jangan gunakan angka Rp191,33 T, 87,1%, atau estimasi fraud pada versi ini tanpa sitasi sumber primer yang dapat diperiksa.

## Slide 4 — Celah yang ditangani

**Teks layar**

Sinyal risiko dapat muncul pada satu klaim maupun pola lintas klaim.

- Aturan menangkap pola yang terdefinisi jelas.
- Similarity membandingkan struktur klaim.
- Deteksi anomali mencari kombinasi yang tidak lazim.
- Verifikator menilai konteks dan mengambil keputusan.

**Narasi presenter**  
ClaimGuard menggabungkan pendekatan tersebut agar penyaringan tidak bergantung pada satu indikator saja. Hasilnya tetap perlu ditafsirkan bersama konteks klaim.

**Visual/data**  
Gunakan empat tahap horizontal: sinyal → pemeringkatan → alasan → tinjauan manusia. Hindari pernyataan tentang proses Vedika, batas waktu verifikasi, atau survei ARSSI jika sumbernya belum dicantumkan.

## Slide 5 — Apa yang diterima verifikator?

**Teks layar**

Satu klaim, satu profil risiko

- Skor risiko 0–100
- Tingkat prioritas
- P-value kalibrasi
- Driver dan fitur yang paling berpengaruh
- Informasi klaim untuk pemeriksaan lanjutan

Skor adalah prioritas pemeriksaan, bukan probabilitas fraud.

**Narasi presenter**  
Contoh klaim pada dashboard berasal dari data simulasi. Detail dan alasan dapat ditelusuri, tetapi keputusan apakah klaim benar-benar bermasalah tidak dibuat otomatis oleh model.

**Visual/data**  
Ganti mockup bertumpuk dengan screenshot aktual dashboard: satu kartu skor dan satu bagian alasan. Gunakan satu klaim dari `out/explain/explain_faskes_retro.csv.gz` dan beri label “contoh sintetis”.

## Slide 6 — Alur kerja sistem

**Teks layar**

Data sintetis → prapemrosesan → fitur → deteksi hibrida → kalibrasi → penjelasan dan audit → dashboard

Evaluasi robustness berjalan pada beberapa seed dan prevalensi.

**Narasi presenter**  
Prapemrosesan dan fitur menyiapkan sinyal per klaim. Detektor hibrida menggabungkan aturan dan model anomali. Kalibrasi mengubah skor mentah menjadi p-value dan skor 0–100. Tahap penjelasan, audit, dan dashboard membuat hasil lebih mudah ditinjau.

**Visual/data**  
Gambarkan pipeline aktual, bukan lima tahap dengan contoh skor buatan. Jika contoh antrean 10.000 klaim tetap dipakai, tandai jelas sebagai ilustrasi dan jangan campur dengan hasil uji.

## Slide 7 — Empat modus dan sinyal

**Teks layar**

| Modus | Sinyal utama |
|---|---|
| Repeat billing | Jarak waktu, pengulangan item, dan pola klaim sebelumnya |
| Claim cloning | Kemiripan diagnosis, tindakan, obat, dan nilai |
| Inflated bills | Residual biaya terhadap kelompok pembanding dan komponen obat/alkes |
| Upcoding | Ketidakwajaran severity dan kombinasi diagnosis/tindakan |

**Narasi presenter**  
Fitur-fitur ini merupakan indikator statistik pada data sintetis. Skor tinggi berarti perlu ditinjau, bukan bahwa penyimpangan telah terbukti.

**Visual/data**  
Gunakan satu contoh mini per modus dengan data sintetis. Cantumkan bahwa mode `retro` melihat batch retrospektif, sedangkan `asof` hanya membandingkan dengan klaim yang sudah diajukan sebelumnya.

## Slide 8 — Arsitektur teknis

**Teks layar**

**Input:** klaim sintetis dan referensi kode  
**Fitur:** sinyal per modus  
**Deteksi:** aturan, similarity, Isolation Forest, ECOD, COPOD  
**Kalibrasi:** p-value konformal dan skor 0–100  
**Output:** prioritas, penjelasan, audit, dashboard

Stack prototype: Python, pandas, NumPy, scikit-learn, PyOD, Streamlit.

**Narasi presenter**  
Label anomali hanya dipakai untuk evaluasi dan pemilihan bobot di validation split sesuai protokol. Label tidak menjadi fitur deteksi. Dashboard membaca snapshot hasil ekspor lokal.

**Visual/data**  
Gunakan diagram arsitektur yang bersih. Sebut pseudonimisasi, role-based access, dan audit log hanya sebagai kebutuhan tahap lanjutan, bukan fitur yang sudah tersedia.

## Slide 9 — Dataset simulasi

**Teks layar**

- 53.954 klaim sintetis
- 60 fasilitas dan 12.000 pasien sintetis
- Periode simulasi: 1 Jan–31 Des 2025
- Target prevalensi anomali: 3%; aktual: 3,04% (1.639 klaim)
- Empat modus anomali dan hard negatives

Label ground truth hanya digunakan untuk evaluasi.

**Narasi presenter**  
Dataset dibangkitkan oleh simulator proyek. Nilai biaya bersifat ilustratif, bukan tarif resmi INA-CBG. Hard negatives mewakili klaim wajar yang sengaja dibuat mirip dengan pola anomali.

**Visual/data**  
Tampilkan alur pembangkitan data dan komposisi label dari `out/metadata.json`. Tabel baris klaim harus diberi label “contoh sintetis”.

## Slide 10 — Cara skor dibentuk

**Teks layar**

Sinyal detektor → skor hibrida → p-value konformal → skor risiko

**Skor risiko = 100 × (1 − p-value)**

Critical: p ≤ 0,01 · High: p ≤ 0,03 · Medium: p ≤ 0,05 · selainnya: Low

Skor bukan probabilitas kecurangan.

**Narasi presenter**  
Komponen detektor dipelajari pada data latih dan validation sesuai rancangan. Kalibrasi menggunakan klaim normal di validation. P-value mengukur seberapa ekstrem skor klaim dibanding distribusi pembanding kalibrasi, dengan asumsi pertukaran data yang perlu diuji.

**Visual/data**  
Ganti diagram lima bobot tetap dan contoh jumlah poinnya dengan alur skor di atas. Jelaskan bahwa kontribusi driver adalah penjelasan komponen dominan, bukan dekomposisi eksak SHAP.

## Slide 11 — Mengapa pendekatan hibrida?

**Teks layar**

Pada test `faskes/retro`, anggaran review 5%:

| Varian | PR-AUC |
|---|---:|
| Hibrida terpilih | 0,604 |
| Aturan saja (maksimum) | 0,565 |
| Ensemble tak terawasi | 0,297 |

**Narasi presenter**  
Pada skenario uji ini, hibrida memberi PR-AUC lebih tinggi daripada dua pembanding tersebut. Hasil ini hanya berlaku pada konfigurasi dan data sintetis yang diuji; bukan bukti bahwa hibrida selalu lebih baik pada data nyata.

**Visual/data**  
Buat bar chart sederhana, sertakan split test, skema `faskes`, mode `retro`, dan prevalensi aktual. Sumber: `out/detect/deteksi_report.md`.

## Slide 12 — Status prototype

**Teks layar**

**Berfungsi pada data simulasi:**

- Pembangkitan data, fitur, deteksi, dan kalibrasi
- Penjelasan klaim dan audit operasional
- Dashboard monitoring dan detail klaim
- Uji robustness 100 run

**Belum dilakukan:** pilot dengan klaim nyata atau evaluasi oleh pengguna operasional.

**Narasi presenter**  
Dashboard adalah prototype yang membaca snapshot lokal. Status ini menunjukkan komponen dapat didemonstrasikan; bukan bahwa sistem siap dipakai dalam proses klaim operasional.

**Visual/data**  
Ganti dua kotak `[MOCKUP]` dengan screenshot aktual dashboard monitoring dan detail klaim. Tampilkan “Prototype · data 100% sintetis”.

## Slide 13 — Protokol evaluasi

**Teks layar**

- Train: fit fitur dan detektor
- Validation: pilih penggabung dan kalibrasi
- Test: evaluasi akhir
- Dua skema: fasilitas yang ditahan dan pembagian waktu
- Label tidak masuk sebagai fitur

Metrik utama: PR-AUC, Precision@K, Recall@K, FPR, dan beban review.

**Narasi presenter**  
Skema `faskes` menguji generalisasi ke fasilitas yang tidak terlihat saat training dan menjadi laporan utama. Skema `waktu` menjadi uji pendamping. Hasil test dipisahkan dari proses pemilihan.

**Visual/data**  
Tampilkan diagram split train/validation/test dan tabel ringkas metrik. Jangan isi confusion matrix dengan angka perkiraan.

## Slide 14 — Hasil utama pada test

**Teks layar**

**Detektor hibrida · `faskes/retro` · test**

- PR-AUC: **0,604** (CI 95%: 0,506–0,681)
- Precision@5%: **36,0%**
- Recall@5%: **73,6%**

Pada ambang kalibrasi p ≤ 0,05: FPR **4,6%** (CI 95%: 3,9–5,3%), recall **72,5%**, dan **6,3%** klaim masuk review.

**Narasi presenter**  
Baris pertama mengukur pemeringkatan pada kapasitas review top 5%. Baris kedua memakai ambang p-value kalibrasi; karena klaim anomali juga masuk antrean, persentase semua klaim yang ditinjau dapat lebih besar daripada FPR.

**Visual/data**  
Pisahkan dua panel metrik agar top-5 review tidak tertukar dengan threshold p ≤ 0,05. Sumber: laporan deteksi dan kalibrasi pada folder `out/`.

## Slide 15 — Robustness kalibrasi

**Teks layar**

25 kombinasi seed × prevalensi per skema/mode (5 seed × prevalensi 1–5%); 100 run keseluruhan.

| Skema/mode | FPR target 5% | Recall@5% |
|---|---:|---:|
| Faskes / retro | 4,54% ± 1,17% | 66,8% ± 5,9% |
| Faskes / asof | 4,85% ± 1,36% | 56,5% ± 5,5% |
| Waktu / retro | 4,88% ± 0,25% | 69,9% ± 3,3% |
| Waktu / asof | 5,00% ± 0,29% | 56,2% ± 4,0% |

**Narasi presenter**  
FPR rata-rata pada target 5% berada dekat target, tetapi variasi berbeda antar skema. Simpangan baku merangkum perubahan seed dan prevalensi sekaligus; jangan menafsirkannya sebagai variasi seed saja.

**Visual/data**  
Tampilkan mean ± SD dan label unit persen. Tambahkan catatan bahwa ini adalah robustness pada data sintetis. Sumber: `robustness/robustness_summary.csv`; seluruh run: `robustness/robustness_detail.csv`.

## Slide 16 — Batasan, risiko, dan tata kelola

**Teks layar**

- Data seluruhnya sintetis; pola nyata bisa berbeda.
- P-value bergantung pada kesesuaian data baru dengan data kalibrasi.
- False positive dapat membebani fasilitas atau verifikator.
- Skor tidak membuktikan fraud dan tidak menggantikan keputusan manusia.
- Penggunaan data nyata memerlukan izin dan kontrol privasi yang sesuai.

**Narasi presenter**  
Kalibrasi tidak menghilangkan kebutuhan menguji pergeseran distribusi dan perbedaan antar fasilitas. Karena itu sistem perlu validasi, pengawasan, dan otorisasi sebelum penggunaan pada data nyata.

**Visual/data**  
Gunakan tabel risiko → mitigasi → status. Bedakan kontrol yang sudah ada dari rekomendasi yang belum diimplementasikan.

## Slide 17 — Nilai yang diharapkan dan tahap berikutnya

**Teks layar**

**Yang telah diuji:** deteksi, kalibrasi, penjelasan, audit, dashboard, dan robustness pada simulasi.

**Langkah berikutnya:** validasi sumber eksternal, perbaikan materi presentasi, evaluasi pengguna, lalu studi pilot hanya dengan izin dan tata kelola yang memadai.

**Narasi presenter**  
Manfaat yang diharapkan adalah membantu petugas memulai dari klaim prioritas dan melihat alasan yang dapat diperiksa. Penghematan waktu atau biaya belum diukur, sehingga belum boleh diklaim sebagai hasil.

**Visual/data**  
Ganti roadmap M1–M10 yang tidak berurutan dengan status “selesai / berikutnya / prasyarat”. Jangan sajikan dampak target sebagai dampak terukur.

## Slide 18 — Tim

**Teks layar**

Cantumkan nama, peran, dan kontribusi yang benar-benar dapat dikonfirmasi oleh anggota tim.

**Narasi presenter**  
Jelaskan pembagian kerja secara singkat: data dan machine learning, riset domain JKN, serta dashboard dan visualisasi. Tambahkan pengalaman hanya jika faktual dan disepakati oleh anggota yang bersangkutan.

**Visual/data**  
Gunakan tiga profil ringkas dengan kontribusi proyek yang konkret. Periksa ejaan nama, istilah, tautan portofolio, dan persetujuan penggunaan foto.

## Slide 19 — Kesimpulan

**Teks layar**

ClaimGuard JKN menunjukkan bahwa sinyal risiko dapat digabung, dikalibrasi, dan dijelaskan untuk membantu menyusun prioritas pemeriksaan pada data simulasi.

Hasil mendukung kelayakan prototype untuk demonstrasi dan penelitian lanjutan, bukan keputusan fraud otomatis atau kesiapan produksi.

**Narasi presenter**  
Kontribusi utama proyek adalah menghubungkan deteksi dengan kalibrasi, penjelasan, audit operasional, dan evaluasi robustness. Langkah selanjutnya adalah memperkuat validasi dan menguji kebutuhan pengguna secara terkontrol.

**Visual/data**  
Ringkas tiga keluaran: skor, alasan, antrean review. Hapus statistik biaya JKN yang berbeda dari slide sebelumnya atau samakan definisi, tahun, dan sumbernya.

## Slide 20 — Tanya jawab

**Teks layar**

Terima kasih  
ClaimGuard JKN  
Deteksi · Jelaskan · Prioritaskan

Kontak tim: [alamat kontak yang disetujui]

**Narasi presenter**  
Kami terbuka untuk pertanyaan mengenai metode, hasil simulasi, keterbatasan, dan rencana validasi selanjutnya.

**Visual/data**  
Jadikan slide penutup yang bersih. Isi kontak hanya setelah dikonfirmasi; hapus slide yang sekarang hanya berisi judul “Kematangan dan prototype”.

## Sumber angka proyek

- Dataset dan prevalensi aktual: `out/metadata.json`
- Deteksi dan metrik test: `out/detect/deteksi_report.md`
- Kalibrasi konformal: `out/calibrated/kalibrasi_report.md`
- Penjelasan klaim: `out/explain/explain_report.md`
- Audit operasional: `out/audit/audit_report.md`
- Robustness 100 run: `robustness/robustness_summary.csv` dan `robustness/robustness_detail.csv`
- Definisi split dan aturan anti-leakage: `eval_protocol.md`

Angka eksternal tentang belanja JKN, fraud, Vedika, ARSSI, dan volume layanan sengaja tidak dimasukkan sebagai fakta final karena deck belum menyertakan sumber primer yang dapat diverifikasi. Tambahkan hanya setelah sumber, definisi, dan tahun datanya dikonfirmasi.