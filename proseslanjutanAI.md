# Konsep Proses Lanjutan AI ClaimGuard

## 1. Gagasan utama

ClaimGuard dirancang sebagai sistem analisis berbantuan AI yang menerima data klaim setelah diajukan oleh fasilitas kesehatan (faskes). Sistem menganalisis pola pada klaim, menyusun prioritas pemeriksaan, dan menyajikan alasan serta bukti pendukung agar petugas dapat meninjau klaim dengan lebih terarah.

Tujuannya bukan menggantikan verifikator, menetapkan bahwa suatu klaim adalah fraud, atau membuat keputusan persetujuan dan pembayaran secara otomatis. AI memberikan **sinyal dan rekomendasi pemeriksaan**; keputusan tetap berada pada petugas yang berwenang melalui proses resmi.

## 2. Posisi dalam alur klaim

Alur konseptual yang diusulkan:

```text
Peserta menerima pelayanan
        ↓
Faskes mencatat pelayanan dan mengajukan klaim
        ↓
Pemeriksaan kelengkapan dan validasi awal
        ↓
AI ClaimGuard menganalisis klaim dan menyusun prioritas
        ↓
Petugas meninjau klaim dengan bantuan alasan dari AI
        ↓
Klarifikasi atau pemeriksaan lanjutan bila diperlukan
        ↓
Petugas berwenang melakukan registrasi dan persetujuan sesuai proses
        ↓
Klaim yang disetujui diproses melalui alur keuangan
```

Analisis AI ditempatkan setelah klaim diajukan dan data minimum tersedia, sebelum atau sebagai masukan untuk pemeriksaan petugas. Pemeriksaan administratif yang wajib tetap dilakukan. AI tidak menggantikan pemeriksaan tersebut.

### Bagaimana prioritas digunakan

- **Prioritas lebih tinggi:** diarahkan lebih awal untuk ditinjau karena memiliki sinyal yang lebih kuat atau kombinasi sinyal yang perlu diperiksa.
- **Prioritas menengah:** ditinjau sesuai antrean dan kapasitas petugas, dengan memperhatikan alasan yang ditampilkan.
- **Prioritas lebih rendah:** tetap mengikuti pemeriksaan rutin yang berlaku; skor rendah bukan jaminan bahwa klaim pasti benar.

Tingkat prioritas mengatur urutan perhatian, bukan hasil keputusan. Klaim tidak otomatis ditolak, disetujui, atau ditahan hanya karena skor AI.

## 3. Apa yang dianalisis AI

ClaimGuard dapat menggabungkan beberapa jenis sinyal, bergantung pada data yang sah, tersedia, dan tervalidasi:

1. **Pola klaim berulang**  
   Memeriksa jarak waktu, kesamaan layanan, nilai, item, serta riwayat klaim terkait. Pengulangan yang wajar—misalnya terapi berkala atau kunjungan lanjutan—perlu dibedakan dari pola yang patut ditinjau.

2. **Kemiripan antar-klaim**  
   Membandingkan kombinasi informasi klaim untuk menemukan klaim yang sangat mirip, termasuk kemungkinan salinan. Kemiripan sendiri bukan bukti pelanggaran; layanan dengan protokol serupa dapat menghasilkan klaim yang memang mirip.

3. **Biaya yang tidak biasa**  
   Membandingkan nilai dan komponen biaya dengan kelompok pembanding yang relevan, misalnya jenis layanan dan karakteristik kasus. Perbedaan biaya menjadi sinyal untuk memeriksa rincian serta bukti pendukung, bukan kesimpulan bahwa biaya tidak sah.

4. **Ketidakselarasan tingkat keparahan atau koding**  
   Mencari kombinasi diagnosis, tindakan, tingkat keparahan, lama rawat, dan informasi lain yang tampak tidak selaras. Penilaian klinis dan aturan pengodean tetap perlu dilakukan oleh pihak berwenang.

5. **Pola tidak biasa lintas beberapa dimensi**  
   Detektor anomali dapat menandai kombinasi yang tidak tercakup aturan sederhana. Sinyal ini perlu dijelaskan sejauh mungkin dan ditinjau hati-hati karena pola yang jarang belum tentu salah.

Analisis tersebut hanya layak dijalankan dengan data dan dasar akses yang sesuai. Data klinis maupun identitas peserta perlu dilindungi, dibatasi sesuai kewenangan, dan tidak ditampilkan lebih luas dari kebutuhan pemeriksaan.

## 4. Informasi yang diterima petugas

Untuk setiap klaim yang dianalisis, layar atau keluaran ClaimGuard sebaiknya menyediakan:

- identitas referensi klaim yang diperlukan untuk penelusuran;
- skor atau tingkat prioritas beserta arti batasannya;
- sinyal atau modus yang paling berpengaruh;
- fakta klaim yang memicu sinyal, seperti selisih waktu, kemiripan, atau komponen biaya;
- pembanding yang digunakan dan keterbatasannya, bila tersedia;
- data yang hilang atau kualitas data yang dapat memengaruhi hasil;
- tindakan lanjutan yang disarankan, misalnya tinjau dokumen atau minta klarifikasi;
- ruang bagi petugas untuk mencatat hasil peninjauan dan alasan keputusan sesuai proses resmi.

Penjelasan harus dapat ditelusuri ke data yang mendasarinya. Hindari menyajikan skor sebagai “probabilitas fraud” kecuali makna probabilitas tersebut benar-benar dikalibrasi dan divalidasi untuk tujuan itu. Istilah yang lebih aman adalah **skor risiko** atau **prioritas pemeriksaan**.

## 5. Alur pemrosesan yang diharapkan

1. **Penerimaan data:** klaim yang diajukan diterima melalui mekanisme integrasi resmi setelah persyaratan akses, keamanan, dan kualitas data dipenuhi.
2. **Validasi awal:** format, kelengkapan, konsistensi, serta duplikasi teknis diperiksa. Kesalahan data ditampilkan secara jelas dan tidak disamarkan sebagai hasil analisis.
3. **Pembentukan fitur:** informasi yang diizinkan diolah menjadi sinyal klaim dan, bila diperbolehkan, sinyal lintas klaim. Label evaluasi atau informasi yang baru diketahui setelah keputusan tidak boleh bocor ke fitur.
4. **Analisis dan pemeringkatan:** detektor berbasis aturan/domain dan metode statistik atau machine learning menghasilkan sinyal. Penggabungan dan kalibrasi skor harus ditetapkan melalui data validasi, bukan dipilih berdasarkan hasil test.
5. **Penyajian hasil:** ClaimGuard mengirimkan prioritas, alasan, keterbatasan, dan referensi ke antrean pemeriksaan petugas.
6. **Tinjauan manusia:** petugas memeriksa bukti dan konteks, kemudian memutuskan langkah sesuai kewenangan. Jika dibutuhkan, petugas melakukan klarifikasi atau eskalasi.
7. **Pencatatan dan pembelajaran:** hasil pemeriksaan dicatat dengan kontrol akses. Data hasil tindak lanjut baru boleh dipakai untuk evaluasi atau perbaikan model setelah definisi label, kualitas, izin, dan tata kelolanya jelas.

Jika analisis belum selesai, layanan gagal, atau data tidak memadai, status tersebut harus terlihat. Sistem tidak boleh memberikan keluaran sukses palsu atau membiarkan kegagalan AI dipahami sebagai persetujuan maupun penolakan klaim. Mekanisme antrean, percobaan ulang, batas waktu, serta proses cadangan manual perlu dirancang sebelum integrasi operasional.

## 6. Makna “AI yang kuat”

AI yang kuat bukan berarti AI yang selalu benar atau selalu memberi jawaban pasti. Untuk ClaimGuard, kemampuan yang kuat berarti sistem:

- mampu mendeteksi beberapa pola dengan pendekatan yang saling melengkapi;
- dapat menjelaskan sinyal secara relevan dan dapat ditelusuri;
- tidak terlalu mudah menandai variasi layanan yang sah;
- tetap diuji pada kasus sulit, data tidak lengkap, perubahan waktu, dan perbedaan antar-faskes;
- menunjukkan ketidakpastian atau keterbatasan saat data tidak mendukung kesimpulan;
- dapat dipantau, diaudit, dan diperbaiki tanpa menghapus tanggung jawab manusia.

Performa harus dilihat sebagai pertukaran antara klaim bermasalah yang berhasil diprioritaskan, klaim wajar yang ikut tertandai, jumlah klaim yang perlu ditinjau, serta beban kerja petugas. Tidak ada satu angka akurasi yang cukup untuk menggambarkan keseluruhan mutu sistem.

## 7. Pengawasan, keselamatan, dan tata kelola

- **Keputusan manusia:** AI tidak menjadi satu-satunya dasar penolakan, penahanan, atau persetujuan klaim.
- **Penjelasan yang jujur:** alasan yang ditampilkan harus sesuai dengan sinyal yang benar-benar dihitung; skor bukan bukti dan bukan tuduhan.
- **Perlindungan data:** akses mengikuti kewenangan, data diminimalkan, dan penggunaan data mengikuti persetujuan serta kebijakan yang berlaku.
- **Jejak audit:** versi model, waktu analisis, data masukan yang relevan, keluaran, perubahan hasil, dan tindak lanjut dicatat sesuai kebijakan retensi.
- **Pemeriksaan bias dan dampak:** performa diperiksa lintas jenis layanan dan kelompok yang relevan agar kesalahan tidak terkonsentrasi pada kelompok tertentu.
- **Pemantauan perubahan:** perubahan pola data, kualitas data, dan tingkat penandaan dipantau. Model perlu dievaluasi ulang bila kondisi berubah.
- **Jalur koreksi:** petugas harus dapat menandai sinyal yang tidak relevan, meminta peninjauan, dan mencatat konteks yang belum ditangkap sistem.

## 8. Evaluasi sebelum digunakan secara nyata

Sebelum ClaimGuard dihubungkan ke alur klaim nyata, tahap evaluasi minimum meliputi:

1. validasi dengan data yang penggunaannya sah dan berizin, serta definisi label yang ditinjau ahli;
2. pemisahan data latih, validasi, dan test yang mencegah kebocoran antar-waktu maupun antar-faskes;
3. pengukuran precision dan recall pada kapasitas pemeriksaan yang benar-benar tersedia;
4. pengukuran tingkat klaim wajar yang tertandai, termasuk hard negatives dan kasus klinis yang sah;
5. pemeriksaan performa per modus, tingkat kesulitan, periode, dan kelompok layanan;
6. uji kalibrasi dan kestabilan skor; asumsi statistiknya harus diuji, bukan dianggap otomatis berlaku;
7. uji operasional atas keterlambatan, data tidak lengkap, kegagalan integrasi, dan mekanisme cadangan;
8. uji coba terbatas dengan hasil AI sebagai rekomendasi saja, lalu tinjauan bersama petugas dan pemangku kepentingan;
9. dokumentasi batas penggunaan, pemilik keputusan, proses eskalasi, serta kriteria penghentian atau evaluasi ulang.

Penggunaan pada data sintetis berguna untuk menguji pipeline dan mendemonstrasikan konsep, tetapi tidak membuktikan bahwa model sudah akurat pada klaim JKN nyata. Parameter biaya pada simulator proyek juga bersifat ilustratif, bukan tarif resmi.

## 9. Batas kondisi saat ini

Prototipe ClaimGuard saat ini adalah pipeline analisis pada snapshot data sintetis: deteksi, kalibrasi, penjelasan, evaluasi, dan dashboard. Prototipe tersebut belum terintegrasi dengan sistem pengajuan atau verifikasi klaim nyata dan belum menjadi bagian dari keputusan operasional.

Karena itu, alur AI setelah pengajuan klaim dalam dokumen ini adalah **konsep pengembangan**, bukan pernyataan bahwa integrasi atau kemampuan tersebut sudah tersedia. Tahap implementasi nyata memerlukan data yang berizin, validasi ahli, persetujuan pemangku kepentingan, rancangan keamanan, integrasi resmi, dan uji coba operasional.

## 10. Ringkasan konsep

> Setelah faskes mengajukan klaim, AI ClaimGuard menganalisis pola dan menyusun prioritas pemeriksaan beserta alasan yang dapat ditelusuri. Petugas menggunakan informasi itu untuk meninjau klaim, meminta klarifikasi, atau meneruskan proses sesuai kewenangannya. AI membantu mengarahkan perhatian; keputusan klaim tetap mengikuti proses resmi dan tanggung jawab manusia.

Peserta mendapat pelayanan
        ↓
Faskes mencatat pelayanan dan menyiapkan klaim
        ↓
Faskes memeriksa kelengkapan dan mengajukan klaim
        ↓
ClaimGuard membantu mengurutkan klaim dan menunjukkan hal yang perlu diperiksa
        ↓
Petugas berwenang memeriksa dokumen serta alasan yang ditampilkan
        ↓
 ┌────────────────────────────────────────────────┐
 │ Tidak ada hal yang perlu diklarifikasi         │
 │ → Klaim lanjut mengikuti proses resmi yang ada │
 └────────────────────────────────────────────────┘
        atau
 ┌────────────────────────────────────────────────┐
 │ Ada kekurangan atau hal yang perlu dijelaskan  │
 │ → Petugas meminta klarifikasi melalui kanal    │
 │   resmi yang berlaku                           │
 │ → Faskes melengkapi atau memberi penjelasan    │
 │ → Petugas meninjau kembali dan menentukan      │
 │   tindak lanjut sesuai kewenangannya           │
 └────────────────────────────────────────────────┘


 PESERTA MENERIMA PELAYANAN
        ↓
FASKES MENCATAT PELAYANAN DAN MENYIAPKAN KLAIM
        ↓
FASKES MEMERIKSA KELENGKAPAN DAN MENGAJUKAN KLAIM
        ↓
CLAIMGUARD MENGANALISIS DAN MENGURUTKAN KLAIM
        ↓
PETUGAS MEMERIKSA DOKUMEN, KONTEKS, DAN ALASAN
        ├── Tidak perlu klarifikasi
        │       ↓
        │   Dicatat: “Diteruskan ke proses resmi”
        │
        └── Perlu klarifikasi
                ↓
            Petugas mencatat permintaan klarifikasi
                ↓
            Faskes simulasi mencatat dokumen/penjelasan
                ↓
            Petugas meninjau ulang
                ├── Klarifikasi cukup
                │       ↓
                │   Dicatat: “Diteruskan ke proses resmi”
                └── Belum cukup
                        ↓
                    Petugas dapat meminta klarifikasi tambahan
                        ↺ kembali ke faskes

## 11. Penerapan alur bersama pada dashboard final

Dashboard final menyediakan satu antrean pemeriksaan berbasis sinyal statistik dan bukti pembanding, diikuti satu proses pemeriksaan manusia. Petugas tidak perlu memilih metode analisis. Simulator ML lama tetap menjadi artefak eksperimen terpisah; skornya tidak dicampur dengan analisis utama atau ditampilkan sebagai pilihan kepada petugas. Setelah ClaimGuard menampilkan alasan dan pembanding, klaim masuk ke formulir **Pemeriksaan dan tindak lanjut**:

1. Petugas memilih bukti yang ditinjau dan mencatat hasil pemeriksaan.
2. Jika klarifikasi tidak diperlukan, klaim dicatat untuk diteruskan ke proses resmi.
3. Jika klarifikasi diperlukan, petugas mencatat permintaan; pada simulasi, faskes dapat mencatat dokumen, penjelasan, atau bahwa klaim telah dikoreksi.
4. Petugas meninjau tanggapan dan dapat meneruskan klaim ke proses resmi atau meminta klarifikasi tambahan.

Catatan review dan status tindak lanjut disimpan sebagai satu rangkaian per klaim. ClaimGuard memberi petunjuk untuk membantu pemeriksaan; petugas tetap menilai dokumen dan menentukan tindak lanjut sesuai kewenangannya.

Pada formulir pemeriksaan, ClaimGuard menyusun ringkasan temuan dan saran bukti berdasarkan sinyal serta pembanding yang tersedia. Jika petugas memilih perlu klarifikasi, sistem menyiapkan draf permintaan yang dapat disunting. Draf ini bukan model AI generatif atau kesimpulan kesalahan. Saran bukti tidak dianggap sudah ditinjau: petugas harus memeriksa dokumen, memilih bukti yang benar-benar ditinjau, dan menentukan tindak lanjut. Jika tidak perlu klarifikasi, status hanya mencatat bahwa klaim diteruskan ke proses resmi; dashboard tidak menyetujui klaim atau memproses pembayaran.

**Batas implementasi saat ini:** alur ini hanya demonstrasi dengan data sintetis. Belum ada pengiriman permintaan klarifikasi, akun/otorisasi petugas atau faskes, unggah dokumen/tanggapan nyata, penyimpanan versi klaim koreksi, atau integrasi kanal resmi. Pilihan faskes untuk “mencatat klaim yang telah dikoreksi” hanya merekam tanggapan simulasi, bukan mengubah data klaim. “Diteruskan ke proses resmi” tidak berarti klaim disetujui atau dibayar; keputusan tetap mengikuti proses dan kewenangan resmi.

## 12. Halaman awal dashboard final

Dashboard final dibuka pada halaman beranda yang menjelaskan tujuan dan batas prototipe, memperlihatkan fungsi yang tersedia, serta menyediakan satu tombol untuk memulai pemeriksaan klaim. Pengguna tidak diminta memilih jalur analisis. Sidebar juga menampilkan menu Analitik, Laporan, Manajemen, dan Pengaturan sebagai tujuan navigasi; halaman tersebut menampilkan pemberitahuan belum tersedia, bukan fitur operasional.

Logo BPJS Kesehatan dan JKN yang tersedia pada aset lokal dapat ditampilkan sebagai konteks visual pada mockup. Halaman harus menyatakan dengan jelas bahwa prototipe ini bukan portal resmi, tidak berafiliasi/terintegrasi dengan BPJS, menggunakan data sintetis, dan tidak melakukan pemantauan real-time atau verifikasi otomatis terhadap klaim nyata. Keputusan tetap berada pada petugas berwenang.