# Laporan Proses ClaimGuard

Dokumen ini merangkum bagian sistem yang sudah tersedia agar tim memiliki pemahaman yang sama tentang alur, fungsi, dan batas prototipe ClaimGuard.

## Ringkasan

ClaimGuard adalah prototipe pendamping pemeriksaan klaim. Tujuannya membantu petugas menemukan klaim yang perlu ditinjau, memahami alasan dan bukti pembanding, lalu mencatat hasil pemeriksaan dan tindak lanjut.

Dashboard final dirancang sebagai **satu alur kerja**. Petugas tidak diminta memilih antara Model ML lama dan Analisis V2. Antrean yang tampil pada dashboard final menggunakan analisis V2 berbasis sinyal statistik dan bukti pembanding. Simulator ML lama masih tersedia sebagai eksperimen terpisah, tetapi tidak ditampilkan sebagai pilihan analisis pada dashboard final dan skornya tidak dicampurkan dengan antrean utama.

> **Status saat ini:** yang tersedia adalah demonstrasi menggunakan data sintetis. ClaimGuard belum terhubung ke sistem klaim resmi dan belum membuktikan kinerja pada data klaim nyata.

## Alur pemeriksaan yang sudah tersedia

```text
Peserta menerima pelayanan
        ↓
Faskes mencatat pelayanan dan menyiapkan klaim
        ↓
Faskes memeriksa kelengkapan dan mengajukan klaim
        ↓
ClaimGuard mengurutkan klaim simulasi dan menampilkan alasan serta pembanding
        ↓
Petugas meninjau dokumen, konteks, dan petunjuk ClaimGuard
        ├── Tidak perlu klarifikasi
        │       ↓
        │   Catat bahwa klaim diteruskan ke proses resmi
        │
        └── Perlu klarifikasi
                ↓
            Petugas meminta klarifikasi melalui kanal resmi
                ↓
            Faskes melengkapi dokumen, memberi penjelasan,
            atau mengoreksi klaim
                ↓
            Petugas meninjau ulang tanggapan
                ├── Informasi cukup → catat diteruskan ke proses resmi
                └── Belum cukup → minta klarifikasi tambahan
```

Dashboard saat ini hanya **mensimulasikan pencatatan** langkah-langkah setelah pemeriksaan. Permintaan klarifikasi dan tanggapan faskes yang sesungguhnya tetap harus berlangsung melalui kanal resmi yang berlaku.

## Fungsi yang sudah tersedia

### 1. Beranda dan navigasi

- Beranda menjelaskan tujuan dan batas prototipe serta menyediakan satu tombol **Mulai pemeriksaan klaim**.
- Menu **Pemeriksaan klaim** membuka satu antrean.
- Menu Analitik, Laporan, Manajemen, dan Pengaturan masih berupa penanda bahwa fitur tersebut belum tersedia.
- Tampilan menggunakan logo BPJS Kesehatan dan JKN sebagai konteks mockup, bukan sebagai tanda afiliasi atau integrasi resmi.

### 2. Antrean pemeriksaan

Antrean dibangun dari data sintetis dan memberikan urutan relatif untuk membantu petugas menentukan klaim yang ditinjau. Di dalam halaman, pengguna dapat:

- mengelompokkan data berdasarkan periode waktu atau fasilitas kesehatan;
- melihat kelompok data yang tersedia;
- memfilter klaim yang disarankan untuk ditinjau;
- mencari nomor klaim atau kode fasilitas;
- membuka klaim tertentu untuk melihat rincian, alasan, dan pembanding.

Skor atau urutan bukan probabilitas kecurangan dan bukan keputusan atas klaim.

### 3. Rincian dan bukti pembanding

Untuk klaim terpilih, dashboard menampilkan data simulasi, alasan utama untuk ditinjau, dan pembanding yang tersedia, seperti:

- nilai total serta komponen biaya dibandingkan dengan nilai tengah kelompok acuan;
- riwayat klaim terdahulu peserta yang sama;
- klaim terdahulu lain yang memiliki kemiripan;
- konteks klinis simulasi yang tersedia, misalnya tingkat keparahan atau lama rawat.

Pembanding biaya bukan tarif resmi atau standar klinis. Kemiripan dan sinyal hanya membantu pemeriksaan; keduanya bukan bukti kesalahan, fraud, atau duplikasi.

### 4. Pencatatan pemeriksaan dan tindak lanjut

Bagian pemeriksaan menampilkan ringkasan otomatis dari sinyal dan bukti pembanding yang tersedia, termasuk perbandingan biaya, klaim terdahulu, kemiripan kode, dan konteks rawat inap bila tersedia. ClaimGuard juga menyarankan jenis dokumen atau informasi yang sebaiknya diperiksa dan menyiapkan draf permintaan klarifikasi yang dapat diedit petugas.

Saran otomatis **tidak** mengisi daftar “bukti yang benar-benar telah ditinjau” dan tidak memutuskan perlu klarifikasi. Petugas memilih perannya, memeriksa dokumen, memilih bukti yang benar-benar telah ditinjau, menentukan perlu klarifikasi atau tidak, lalu menyunting draf sebelum digunakan. Draf dibuat dari data terstruktur yang tersedia; ini belum menggunakan model AI generatif, dan tidak ada pesan yang dikirim otomatis.

Alur demo mencakup:

1. Petugas mencatat peran, bukti yang ditinjau, hasil pemeriksaan, catatan, dan penilaian apakah petunjuk ClaimGuard membantu.
2. Jika klarifikasi tidak diperlukan, status demo berubah menjadi **Diteruskan ke proses resmi**. Proses pembayaran, bila berlaku, tetap melalui sistem dan kewenangan resmi; dashboard tidak menyetujui atau melakukan pembayaran.
3. Jika klarifikasi diperlukan, status demo berubah menjadi **Menunggu tanggapan faskes**.
4. Faskes simulasi mencatat bahwa dokumen dilengkapi, penjelasan diberikan, atau klaim telah dikoreksi.
5. Status kembali ke petugas untuk pemeriksaan ulang. Petugas dapat meneruskan ke proses resmi atau meminta klarifikasi tambahan.

Catatan baru disimpan secara append-only pada `out/workflow_records_demo_v2.csv`. Log lama `out/workflow_records_demo.csv` dapat dibaca sebagai riwayat lama dan tidak diubah. Catatan demo bukan label pelatihan model. Status diteruskan ke proses resmi tidak berarti klaim disetujui atau dibayar.

## Komponen eksperimen yang tidak menjadi pilihan dashboard final

Simulator ML lama masih dapat dipelajari sebagai eksperimen terpisah. Hasilnya mencakup skor, pola, dan penjelasan yang dihasilkan pada simulator. Karena pendekatan, skala, dan interpretasinya berbeda dari antrean V2, hasilnya tidak dicampurkan ke antrean utama.

Dashboard review V2 mandiri juga masih tersedia untuk demonstrasi pengembangan. Keduanya tidak mengubah prinsip produk final: pengguna pemeriksaan klaim memakai satu antrean dan satu alur tindak lanjut.

## Menjalankan dan menguji

Jalankan dashboard final dari direktori proyek:

```bash
streamlit run dashboard_final.py
```

Jalankan seluruh tes proyek:

```bash
python -m unittest discover -s tests -v
```

Tes navigasi khusus memastikan halaman awal hanya menawarkan satu pintu masuk dan tombol tersebut membuka antrean tanpa pilihan metode analisis. Pada validasi terakhir, **36 tes lulus**.

## Batas dan hal yang belum tersedia

- Data dan hasil analisis yang ditampilkan adalah sintetis; kinerja pada klaim nyata belum dibuktikan.
- Belum ada integrasi dengan BPJS, faskes, sistem klaim, atau kanal komunikasi resmi.
- Dashboard tidak mengirim permintaan klarifikasi, tidak menerima unggahan dokumen nyata, dan tidak menyimpan versi klaim yang telah dikoreksi.
- Belum tersedia akun, autentikasi, otorisasi, atau verifikasi identitas petugas dan faskes.
- Dashboard tidak memutuskan klaim disetujui, ditolak, ditahan, atau dibayar.
- Draf temuan/klarifikasi adalah bantuan berbasis sinyal dan pembanding, bukan diagnosis kesalahan atau keluaran model AI generatif; petugas harus memeriksa dan menyuntingnya.
- Menu Analitik, Laporan, Manajemen, dan Pengaturan belum berfungsi.
- Memasukkan data baru belum melatih ulang model, membangun ulang pembanding, memperbarui kalibrasi, atau menjamin akurasi.
- Logo BPJS Kesehatan dan JKN hanya digunakan sebagai konteks mockup. ClaimGuard bukan portal resmi dan tidak menyatakan afiliasi atau integrasi.

## Prinsip yang disepakati

**ClaimGuard membantu petugas memeriksa, bukan menggantikan petugas.** Sistem menampilkan urutan, alasan, dan pembanding agar pemeriksaan lebih terarah. Petugas yang berwenang tetap meninjau dokumen dan menentukan tindak lanjut; komunikasi nyata dengan faskes mengikuti kanal resmi yang berlaku.
