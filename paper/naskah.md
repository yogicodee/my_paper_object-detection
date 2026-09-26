# Evaluasi Komparatif Arsitektur YOLOv8, YOLO11, dan YOLO26 untuk Deteksi Alat Pelindung Diri pada Perangkat Tepi dengan Uji Signifikansi Statistik

> **CATATAN PENGGUNAAN — HAPUS SEBELUM SUBMIT.**
> Setiap penanda `⟦ISI: …⟧` menandai angka atau kalimat yang **hanya boleh diisi
> dari hasil eksperimen yang benar-benar dijalankan**. Tidak ada satu pun angka
> hasil yang sudah diisi di berkas ini — semuanya harus berasal dari
> `results/analysis/tabel_naskah.md` yang dihasilkan `scripts/06_figures.py`.
> Mengisi penanda ini dengan tebakan adalah fabrikasi data dan dapat berujung
> pada penarikan artikel.

**Penulis:** ⟦ISI: Nama Lengkap⟧¹, ⟦ISI: Nama Penulis Kedua⟧²
**Afiliasi:** ¹⟦ISI: Program Studi, Universitas, Kota, Indonesia⟧
**Korespondensi:** ⟦ISI: alamat surel⟧

---

## Abstrak

Kepatuhan penggunaan alat pelindung diri (APD) merupakan faktor penentu keselamatan
kerja di sektor manufaktur dan konstruksi, namun pengawasannya masih banyak
mengandalkan inspeksi manual yang terbatas jangkauan dan konsistensinya. Deteksi
objek berbasis pembelajaran mendalam menawarkan pengawasan otomatis, tetapi
pemilihan arsitektur untuk penerapan nyata masih sulit karena dua alasan: keluarga
model YOLO berkembang sangat cepat sehingga varian terbaru belum teruji pada domain
APD, dan sebagian besar studi pembanding melaporkan selisih akurasi tanpa menguji
apakah selisih tersebut bermakna secara statistik. Penelitian ini mengevaluasi tiga
generasi arsitektur — YOLOv8, YOLO11, dan YOLO26 — pada skala nano, yaitu kelas
model yang relevan bagi penerapan pada perangkat tepi di lokasi kerja, menggunakan
dataset SH17 yang memuat 8.099 citra dan 75.994 instance dari 17 kelas APD dan
bagian tubuh.
Seluruh model dilatih dengan protokol identik, dan himpunan uji resmi dipertahankan
sebagai data tertahan murni yang tidak pernah digunakan untuk penghentian dini
maupun pemilihan model. Perbedaan antar-model diuji menggunakan bootstrap
berpasangan sebanyak ⟦ISI: jumlah resample⟧ resample dengan koreksi Holm terhadap
perbandingan ganda. Hasil menunjukkan ⟦ISI: model terbaik⟧ mencapai mAP@0,5:0,95
sebesar ⟦ISI: nilai⟧ (IK 95%: ⟦ISI: batas bawah⟧–⟦ISI: batas atas⟧), sementara
⟦ISI: ringkasan temuan signifikansi — sebutkan secara eksplisit pasangan mana yang
berbeda signifikan dan mana yang tidak⟧. Analisis per-ukuran objek memperlihatkan
⟦ISI: temuan pada objek kecil⟧, yang menjadi kendala utama pada kelas APD berukuran
kecil seperti kacamata dan penutup telinga. Temuan ini memberikan dasar empiris
bagi pemilihan arsitektur deteksi APD yang sesuai dengan anggaran komputasi di
lapangan.

**Kata kunci:** deteksi objek; alat pelindung diri; keselamatan kerja; YOLO;
uji signifikansi statistik

---

## Abstract

Compliance with personal protective equipment (PPE) requirements is a decisive
factor in occupational safety across manufacturing and construction, yet its
enforcement still relies largely on manual inspection that is limited in both
coverage and consistency. Deep-learning object detection offers automated
monitoring, but selecting an architecture for real deployment remains difficult
for two reasons: the YOLO family evolves rapidly so that the newest variants
remain untested on the PPE domain, and most comparative studies report accuracy
differences without testing whether those differences are statistically
meaningful. This study evaluates three architectural generations — YOLOv8, YOLO11,
and YOLO26 — at the nano scale, the model class relevant to edge deployment at work
sites, using the SH17 dataset containing 8,099
images and 75,994 instances across 17 PPE and body-part classes. All models are
trained under an identical protocol, and the official test split is preserved as
a pure held-out set never used for early stopping or model selection. Differences
between models are tested using a paired bootstrap over ⟦ISI: jumlah resample⟧
resamples with Holm correction for multiple comparisons. Results show that
⟦ISI: model terbaik⟧ attains an mAP@0.5:0.95 of ⟦ISI: nilai⟧ (95% CI:
⟦ISI: batas bawah⟧–⟦ISI: batas atas⟧), while ⟦ISI: ringkasan temuan
signifikansi⟧. Size-stratified analysis reveals ⟦ISI: temuan pada objek kecil⟧,
which constitutes the principal bottleneck for small PPE classes such as glasses
and earmuffs. These findings provide an empirical basis for selecting a PPE
detection architecture that fits field computational budgets.

**Keywords:** object detection; personal protective equipment; occupational
safety; YOLO; statistical significance testing

---

## I. PENDAHULUAN

Kecelakaan kerja masih menjadi persoalan serius pada sektor manufaktur dan
konstruksi, dan sebagian besar di antaranya berkaitan dengan ketidakpatuhan
penggunaan alat pelindung diri (APD). Pengawasan kepatuhan APD secara konvensional
dilakukan melalui inspeksi manual oleh petugas keselamatan. Pendekatan ini memiliki
keterbatasan mendasar: jangkauannya terbatas pada area dan waktu yang diawasi,
konsistensinya bergantung pada kondisi petugas, dan biayanya meningkat sebanding
dengan luas area yang harus dipantau.

Deteksi objek berbasis pembelajaran mendalam menawarkan alternatif pengawasan yang
berjalan terus-menerus melalui kamera yang sudah terpasang. Sejak diperkenalkannya
pendekatan satu tahap oleh Redmon dkk. [4], keluarga model *You Only Look Once*
(YOLO) menjadi pilihan dominan untuk aplikasi waktu nyata karena menyeimbangkan
akurasi dan kecepatan inferensi. Perkembangan keluarga ini berlangsung cepat:
YOLOv9 memperkenalkan *programmable gradient information* [8], YOLOv10 menghapus
kebutuhan *non-maximum suppression* melalui pelatihan konsisten ganda [9], YOLO11
menyempurnakan blok ekstraksi fitur [3], dan YOLO26 yang dirilis pada awal 2026
menghadirkan inferensi *end-to-end* tanpa NMS, penghapusan *Distribution Focal
Loss*, serta penugasan label sadar objek kecil (*small-target-aware label
assignment*) [2]. Secara paralel, pendekatan berbasis transformer seperti DETR [7]
dan penyempurnaannya RT-DETR [6] menawarkan jalur arsitektur yang berbeda.

### A. Celah Penelitian

Penelitian mengenai deteksi APD telah banyak dilakukan, namun tiga celah berikut
masih terbuka.

**Pertama, varian arsitektur terbaru belum tervalidasi pada domain APD.** Dataset
SH17 [1] merupakan tolok ukur terbuka terbesar untuk deteksi APD di lingkungan
manufaktur, dengan 8.099 citra dan 75.994 instance dari 17 kelas. Namun tolok ukur
yang disediakan penulis dataset hanya mencakup YOLOv8, YOLOv9, dan YOLOv10. Dua
generasi arsitektur sesudahnya — YOLO11 dan YOLO26 — belum pernah dievaluasi pada
dataset ini, padahal keduanya memuat perubahan yang secara teoretis relevan bagi
domain APD. Secara khusus, mekanisme penugasan label sadar objek kecil pada YOLO26
menyasar persoalan yang justru dominan pada SH17: sebagian besar anotasinya adalah
objek berukuran sangat kecil seperti telinga, penutup telinga, dan kacamata.

**Kedua, pembandingan antar-model jarang disertai uji signifikansi.** Praktik yang
lazim adalah melaporkan selisih mAP antar-model dan menyimpulkan satu model lebih
unggul. Padahal selisih kecil dapat timbul semata-mata dari variasi pengambilan
sampel citra uji. Tanpa kuantifikasi ketidakpastian, kesimpulan semacam itu tidak
dapat diandalkan sebagai dasar keputusan penerapan. Dalam literatur pembelajaran
mesin, kebutuhan pengujian statistik untuk pembandingan model telah lama
ditegaskan [12], namun penerapannya pada studi deteksi objek masih jarang.

**Ketiga, protokol evaluasi sering mencampur pemilihan model dan pelaporan.** Banyak
studi menggunakan satu himpunan yang sama untuk penghentian dini sekaligus
pelaporan akhir. Praktik ini menghasilkan estimasi yang bias optimistis karena
himpunan pelaporan ikut memandu pemilihan model.

### B. Kontribusi

Penelitian ini memberikan empat kontribusi berikut.

1. **Tolok ukur pertama YOLO11 dan YOLO26 pada dataset SH17**, dilaksanakan dengan
   protokol pelatihan identik sehingga perbandingan antar-arsitektur bersifat adil
   dan hasilnya dapat langsung disandingkan dengan tolok ukur asli dataset.
2. **Kuantifikasi ketidakpastian melalui bootstrap berpasangan** dengan koreksi
   Holm [10], sehingga setiap klaim keunggulan disertai selang kepercayaan dan
   nilai p, bukan sekadar selisih titik.
3. **Protokol evaluasi yang memisahkan pemilihan model dari pelaporan**, dengan
   himpunan uji resmi dipertahankan sebagai data tertahan murni.
4. **Analisis kegagalan berstrata ukuran objek dan per kelas**, yang
   mengidentifikasi kelas APD mana yang menjadi hambatan utama dan mengapa.

Seluruh kode, konfigurasi, dan prosedur analisis dipublikasikan agar penelitian ini
dapat direproduksi.

---

## II. METODE PENELITIAN

### A. Dataset

Penelitian menggunakan dataset *Safety-PPE Unified v1* [1], gabungan tiga himpunan
citra APD publik berlisensi CC BY 4.0 yang telah disatukan ke dalam satu taksonomi
dan dibersihkan dari duplikasi. Dataset memuat **17.951 citra** dengan **51.179
kotak pembatas** pada **12 kelas**. Seluruh citra berukuran seragam 640×640 piksel.

Ciri yang paling menentukan bagi penelitian ini adalah struktur kelasnya: kedua
belas kelas tersusun sebagai **enam pasang kepatuhan dan pelanggaran** —
`helmet`/`no_helmet`, `vest`/`no_vest`, `gloves`/`no_gloves`,
`goggles`/`no_goggles`, `boots`/`no_boots`, dan `mask`/`no_mask`. Struktur inilah
yang memungkinkan analisis asimetri yang menjadi kontribusi utama penelitian ini.

**TABEL I. KOMPOSISI DATASET**

| Bagian | Citra | Instance | Instance/citra |
|---|---:|---:|---:|
| Latih | 12.566 | 35.832 | 2,85 |
| Validasi | 3.590 | 10.233 | 2,85 |
| Uji | 1.795 | 5.114 | 2,85 |
| **Total** | **17.951** | **51.179** | **2,85** |

**TABEL II. KESEIMBANGAN PASANGAN KEPATUHAN DAN PELANGGARAN**

| Pasangan | Kepatuhan | Pelanggaran | Porsi pelanggaran |
|---|---:|---:|---:|
| helmet | 16.482 | 8.279 | 33,4% |
| gloves | 5.020 | 6.162 | 55,1% |
| goggles | 3.686 | 3.868 | 51,2% |
| vest | 2.904 | 1.123 | 27,9% |
| mask | 1.185 | 1.324 | 52,8% |
| boots | 1.109 | 37 | 3,2% |

Tabel II memperlihatkan bahwa keseimbangan antar-pasangan sangat beragam. Pada
`gloves`, `goggles`, dan `mask`, contoh pelanggaran justru lebih banyak daripada
contoh kepatuhan. Sebaliknya `helmet` dan `vest` didominasi contoh kepatuhan,
sedangkan `boots` nyaris tidak memiliki contoh pelanggaran sama sekali. Rasio
ketidakseimbangan antara kelas terbanyak (`helmet`, 16.482 instance) dan paling
langka (`no_boots`, 37 instance) mencapai **445 kali**.

Karena hanya tersedia 37 kotak `no_boots` pada keseluruhan data — empat di
antaranya pada himpunan uji — kelas tersebut tidak dapat menghasilkan estimasi AP
yang bermakna. Kelas ini tetap dilaporkan secara terpisah namun **dikecualikan dari
perhitungan rerata agregat**, dan keputusan tersebut dinyatakan terbuka agar
pembaca dapat menilai pengaruhnya.

Sebaran ukuran objek juga timpang: **27.833 instance (54,4%) tergolong kecil**
(luas < 1% luas citra), 18.244 (35,6%) sedang, dan hanya 5.102 (10,0%) besar.
Median lebar kotak adalah 0,076 dan median tinggi 0,115 dari dimensi citra.
Dominasi objek kecil ini relevan secara langsung dengan mekanisme penugasan label
sadar objek kecil pada YOLO26.

**Gambar 1.** Komposisi dataset: (a) sebaran instance per kelas pada skala
logaritmik, (b) sebaran ukuran objek relatif terhadap luas citra.
*(berkas: `results/figures/gambar1_dataset.png`)*

### B. Verifikasi Mutu Anotasi

Karena dataset merupakan gabungan beberapa sumber, mutu anotasinya diperiksa
secara menyeluruh sebelum digunakan. Dari 51.179 kotak, ditemukan **4 kotak yang
sedikit melampaui batas citra** (0,008%) dan **1 kotak berukuran sangat kecil**
(luas < 0,01% luas citra). Tidak ditemukan indeks kelas di luar rentang maupun
kotak berdimensi nol atau negatif. Pemindaian bawaan pustaka Ultralytics juga
melaporkan nol citra rusak serta satu label duplikat yang dihapus otomatis.

Mutu anotasi dengan demikian tergolong bersih. Meski begitu, pemeriksaan ini
bersifat statistik dan tidak menggantikan penilaian visual menyeluruh terhadap
ketepatan penempatan kotak; hal tersebut dinyatakan sebagai keterbatasan pada
Subbagian III-E.

### C. Pembagian Data

Dataset sudah menyediakan pembagian latih, validasi, dan uji dengan proporsi
70/20/10 yang disusun secara **terstratifikasi per kelas** serta **dikelompokkan
menurut citra asal**, sehingga tidak ada citra hasil augmentasi dari sumber yang
sama muncul pada dua bagian berbeda. Pembagian bawaan ini dipakai apa adanya.

Pemisahan validasi dari uji merupakan syarat metodologis yang penting. Bila satu
himpunan dipakai sekaligus untuk penghentian dini dan pelaporan akhir, angka yang
dilaporkan menjadi bias optimistis karena himpunan tersebut ikut memandu pemilihan
model. Dalam penelitian ini, himpunan validasi dipakai untuk penghentian dini dan
pemantauan konvergensi, sedangkan **himpunan uji hanya disentuh satu kali** pada
tahap evaluasi akhir.

Karena seluruh citra sudah berukuran seragam 640×640 piksel — sama dengan resolusi
pelatihan — tidak diperlukan tahap penyesuaian ukuran maupun prapemrosesan lain di
luar augmentasi baku yang dijalankan selama pelatihan.

### D. Arsitektur yang Dibandingkan

Tiga generasi arsitektur dibandingkan pada skala *nano* (n). Skala ini dipilih
karena dua pertimbangan yang saling menguatkan. Pertama, pengawasan APD di lapangan
dijalankan pada perangkat tepi berdaya rendah yang terpasang dekat kamera, sehingga
kelas model nano justru paling relevan secara praktis. Kedua, keseluruhan pelatihan
berlangsung pada CPU, dan pembatasan pada skala nano membuat perbandingan yang adil
antar-generasi dapat diselesaikan dalam waktu yang wajar.

| Keluarga | Varian | Parameter | GFLOPs | Karakteristik arsitektur utama |
|---|---|---:|---:|---|
| YOLOv8 | `yolov8n` | 3,16 jt | 8,86 | Dasar CNN satu tahap; pembanding terhadap tolok ukur asli SH17 |
| YOLO11 | `yolo11n` | 2,62 jt | 6,67 | Penyempurnaan blok ekstraksi fitur dan kepala deteksi [3] |
| YOLO26 | `yolo26n` | 2,57 jt | 6,24 | Inferensi tanpa NMS, penghapusan DFL, penugasan label sadar objek kecil, pengoptimal MuSGD [2] |

Jumlah parameter dan GFLOPs pada tabel di atas diukur langsung dari bobot pralatih
yang digunakan, pada 80 kelas keluaran COCO sebelum kepala deteksi disesuaikan ke
17 kelas SH17.

Perlu dicatat bahwa ketiga model berada pada rentang kompleksitas yang berdekatan,
sehingga perbandingan ini menguji perbedaan rancangan arsitektur, bukan sekadar
perbedaan kapasitas model.

### E. Protokol Pelatihan

Seluruh model dilatih dengan hiperparameter identik agar perbedaan yang teramati
dapat diatribusikan pada arsitektur, bukan pada penyetelan. Nilai bawaan pustaka
Ultralytics [14] dipertahankan kecuali yang disebutkan pada Tabel II, sehingga
penelitian ini dapat direproduksi tanpa penyetelan tersembunyi.

**TABEL II. KONFIGURASI PELATIHAN**

| Parameter | Nilai |
|---|---|
| Jumlah epoch | 100 |
| Resolusi masukan | 640 × 640 |
| Penghentian dini (*patience*) | 25 epoch |
| Pengoptimal | otomatis (bawaan Ultralytics) |
| Penjadwalan laju pembelajaran | kosinus |
| Penonaktifan mosaik | 10 epoch terakhir |
| *Seed* acak | 0 |
| Mode deterministik | aktif |
| Ukuran *batch* | 16 (sama untuk ketiga model) |
| Bobot awal | pralatih COCO [5] |

Ukuran *batch* dibuat sama untuk ketiga model, sehingga tidak ada satu pun faktor
selain rancangan arsitektur yang membedakan kondisi pelatihan.

Pelatihan dijalankan pada prosesor AMD Ryzen 5 7535HS (6 inti fisik, 12 utas) dengan
memori 13,7 GB, menggunakan PyTorch 2.14 varian CPU. Ketiadaan akselerator CUDA pada
perangkat ini merupakan alasan kedua pembatasan cakupan pada skala nano, sebagaimana
dijelaskan pada Subbagian II-D.

⟦ISI: laporkan total waktu pelatihan nyata tiap model dari `results/runs/<id>/meta.json`
setelah eksperimen selesai⟧

### F. Metrik Evaluasi

Metrik utama adalah *mean Average Precision* (mAP) mengikuti definisi COCO [5],
yaitu rata-rata AP pada sepuluh ambang IoU dari 0,50 hingga 0,95 dengan langkah
0,05, menggunakan interpolasi 101 titik *recall*. Selain mAP@0,5:0,95, dilaporkan
pula mAP@0,5 mengikuti konvensi PASCAL VOC [13] agar dapat dibandingkan dengan
literatur terdahulu.

Karena sistem pengawasan APD di lapangan beroperasi pada satu ambang keyakinan
tetap dan bukan pada seluruh kurva, dilaporkan pula presisi, *recall*, dan skor F1
pada ambang keyakinan 0,25. Efisiensi komputasi diukur melalui jumlah parameter,
GFLOPs, serta latensi inferensi pada ukuran *batch* 1 — kondisi yang mencerminkan
pemrosesan satu bingkai kamera pada satu waktu.

Analisis berstrata ukuran objek menggunakan kategori luas relatif terhadap luas
citra: kecil (< 1%), sedang (1–5%), dan besar (> 5%). Luas relatif dipilih
menggantikan luas piksel absolut ala COCO karena resolusi citra SH17 sangat
beragam, sehingga ambang piksel absolut akan memperlakukan objek yang secara visual
setara sebagai kategori berbeda.

### G. Uji Signifikansi Statistik

Selisih mAP antar-model diuji menggunakan **bootstrap berpasangan** pada tataran
citra. Prosedurnya sebagai berikut.

1. Dari 1.620 citra uji, diambil ulang 1.620 citra dengan pengembalian sebanyak
   *B* = ⟦ISI: nilai B yang dipakai⟧ kali.
2. Untuk setiap resample, mAP dihitung ulang bagi **seluruh model pada citra yang
   sama persis**. Pemasangan ini menghilangkan ragam yang berasal dari citra mana
   yang kebetulan terambil, sehingga yang tersisa adalah perbedaan antar-model.
3. Selang kepercayaan 95% diperoleh dari persentil ke-2,5 dan ke-97,5 distribusi
   selisih.
4. Nilai p dua sisi dihitung dari proporsi resample yang selisihnya berlawanan
   tanda terhadap selisih teramati.
5. Karena terdapat banyak pasangan yang diuji sekaligus, nilai p dikoreksi
   menggunakan prosedur Holm [10] yang mengendalikan *family-wise error rate*
   dengan daya uji lebih tinggi daripada koreksi Bonferroni.

Agar prosedur ini dapat dijalankan, hasil pencocokan deteksi terhadap kebenaran
dasar disimpan sekali sebagai tabel per-deteksi (citra, kelas, skor keyakinan,
status *true positive* pada sepuluh ambang IoU). Dengan tabel ini, mAP dapat
dihitung ulang untuk sembarang gugus citra tanpa mengulang inferensi. Perlu
dicatat bahwa implementasi `COCOeval` pada pustaka `pycocotools` melakukan
de-duplikasi indeks citra, sehingga pengambilan sampel dengan pengembalian akan
berubah menjadi sub-sampling tanpa bobot dan menghasilkan selang kepercayaan yang
bias. Karena itu perhitungan AP diimplementasikan ulang mengikuti definisi COCO,
dan kebenarannya diverifikasi terhadap kasus uji yang jawabannya dapat dihitung
secara analitis (`scripts/test_detection_eval.py`).

**Batas tafsir yang perlu ditegaskan:** bootstrap ini mengukur ketidakpastian yang
berasal dari sampel citra uji, bukan ketidakpastian akibat inisialisasi bobot acak.
Untuk yang terakhir diperlukan pelatihan ulang dengan beberapa *seed*
⟦ISI: laporkan di sini bila eksperimen multi-seed opsional benar-benar
dijalankan; bila tidak, pertahankan kalimat ini sebagai keterbatasan⟧.

---

## III. HASIL DAN PEMBAHASAN

### A. Konvergensi Pelatihan

Gambar 2 menyajikan kurva mAP@0,5:0,95 pada himpunan validasi sepanjang proses
pelatihan untuk ketiga model.

**Gambar 2.** Konvergensi pelatihan pada himpunan validasi.
*(berkas: `results/figures/gambar2_konvergensi.png`)*

⟦ISI: Deskripsikan pola konvergensi yang benar-benar teramati. Perhatikan:
(a) model mana yang konvergen paling cepat; (b) apakah ada model yang terhenti
lebih awal karena penghentian dini, dan pada epoch berapa; (c) apakah ada tanda
*overfitting* berupa penurunan metrik validasi; (d) efek penonaktifan mosaik pada
10 epoch terakhir.⟧

### B. Hasil Utama pada Himpunan Uji

Tabel III merangkum kinerja ketiga model pada himpunan uji tertahan, lengkap dengan
selang kepercayaan hasil bootstrap.

**TABEL III. HASIL PADA HIMPUNAN UJI**

⟦ISI: salin Tabel 2 dari results/analysis/tabel_naskah.md⟧

**Gambar 3.** Akurasi pada himpunan uji dengan selang kepercayaan 95%.
*(berkas: `results/figures/gambar3_map_ci.png`)*

⟦ISI: Nyatakan model dengan mAP tertinggi beserta nilainya dan selang
kepercayaannya. Sebutkan pula apakah urutan peringkat antar-keluarga konsisten
pada kedua skala (n dan s), atau justru berbalik.⟧

### C. Uji Signifikansi Antar-Model

Tabel IV menyajikan hasil uji bootstrap berpasangan untuk seluruh pasangan model.

**TABEL IV. UJI SIGNIFIKANSI BERPASANGAN (KOREKSI HOLM)**

⟦ISI: salin Tabel 3 dari results/analysis/tabel_naskah.md⟧

⟦ISI: Laporkan pasangan mana yang berbeda signifikan dan mana yang tidak. **Bila
tidak ada pasangan yang signifikan, laporkan itu apa adanya** — temuan bahwa
generasi arsitektur baru tidak memberikan peningkatan yang dapat dibedakan dari
derau adalah kontribusi ilmiah yang sah dan justru menjelaskan mengapa uji
signifikansi diperlukan. Jangan melunakkan atau menyembunyikan hasil negatif.⟧

Perlu ditekankan bahwa selisih mAP yang tampak besar pada tabel peringkat tidak
otomatis bermakna. Selang kepercayaan yang saling tumpang tindih menunjukkan bahwa
urutan peringkat dapat berubah bila himpunan uji yang berbeda digunakan.

### D. Analisis Kegagalan

#### 1) Kinerja per kelas

**Gambar 5.** AP@0,5:0,95 per kelas untuk seluruh model.
*(berkas: `results/figures/gambar5_ap_per_kelas.png`)*

⟦ISI: Identifikasi tiga kelas dengan AP tertinggi dan tiga kelas dengan AP
terendah. Hubungkan dengan dua faktor: jumlah instance pelatihan (dari Tabel I)
dan ukuran objek tipikal. Bahas apakah kelas berkinerja buruk disebabkan
kelangkaan data, ukuran objek yang kecil, atau keduanya — dan jelaskan dasar
pembedaannya.⟧

Kelas yang secara langsung menentukan kepatuhan APD — *helmet*, *safety-vest*,
*gloves*, dan *earmuffs* — memerlukan perhatian khusus karena kesalahan deteksi
pada kelas inilah yang berdampak pada keselamatan.
⟦ISI: Laporkan AP keempat kelas tersebut dan bandingkan dengan rata-rata
keseluruhan.⟧

#### 2) Kinerja menurut ukuran objek

**Gambar 6.** Kinerja terhadap kategori ukuran objek.
*(berkas: `results/figures/gambar6_ukuran_objek.png`)*

⟦ISI: Bandingkan mAP pada objek kecil, sedang, dan besar. Ujilah secara khusus
hipotesis berikut: apakah mekanisme penugasan label sadar objek kecil pada YOLO26
menghasilkan keunggulan pada kategori objek kecil dibandingkan YOLOv8 dan YOLO11?
Bila ya, sebutkan besar selisihnya. **Bila tidak, nyatakan secara eksplisit bahwa
klaim arsitektural tersebut tidak terkonfirmasi pada domain ini** — ini justru
temuan yang bernilai bagi pembaca.⟧

#### 3) Imbangan akurasi dan kecepatan

**Gambar 4.** Imbangan akurasi terhadap latensi inferensi.
*(berkas: `results/figures/gambar4_tradeoff.png`)*

⟦ISI: Identifikasi model yang berada pada garis depan Pareto. Berikan rekomendasi
praktis: model mana yang sebaiknya dipilih bila prioritasnya akurasi, dan model
mana bila prioritasnya kecepatan pada perangkat tepi. Dasarkan rekomendasi pada
angka latensi yang terukur, bukan pada asumsi umum.⟧

### E. Keterbatasan Penelitian

Beberapa keterbatasan berikut perlu dinyatakan secara terbuka.

1. **Ragam akibat inisialisasi tidak terukur penuh.** Setiap konfigurasi dilatih
   dengan satu *seed*. Selang kepercayaan yang dilaporkan mencerminkan
   ketidakpastian dari sampel citra uji, bukan dari inisialisasi bobot.
   ⟦ISI: Perbarui bagian ini bila eksperimen multi-seed opsional dijalankan.⟧
2. **Cakupan terbatas pada skala nano.** Penelitian ini tidak menguji skala
   *small* maupun yang lebih besar, sehingga temuan di sini tidak dapat
   digeneralisasi ke seluruh rentang kapasitas model. Ada kemungkinan urutan
   peringkat antar-arsitektur berubah pada kapasitas yang lebih besar, karena
   sebagian perbaikan rancangan baru menampakkan pengaruhnya ketika kapasitas
   model memadai. Pengujian pada skala *small* dan *medium* merupakan lanjutan
   yang paling langsung dari penelitian ini.

3. **Arsitektur transformer tidak disertakan.** RT-DETR [6] dan turunan DETR
   lainnya memerlukan memori GPU yang tidak tersedia pada perangkat penelitian
   ini. Perbandingan lintas paradigma CNN dan transformer karenanya berada di
   luar cakupan.
4. **Cakupan domain dataset.** SH17 dihimpun dari citra Pexels yang umumnya
   beresolusi tinggi dan berpencahayaan baik. Kondisi lapangan nyata mencakup
   pencahayaan rendah, oklusi berat, dan kekaburan gerak yang tidak terwakili
   secara memadai. Kinerja pada penerapan nyata dapat lebih rendah daripada yang
   dilaporkan di sini.
5. **Sudut pandang kamera.** Dataset tidak memuat informasi sudut pandang kamera
   pengawas yang biasanya terpasang tinggi, padahal perspektif ini memengaruhi
   tampakan APD secara signifikan.
6. **Tanpa penyetelan hiperparameter.** Nilai bawaan dipertahankan demi keadilan
   perbandingan. Penyetelan khusus per arsitektur berpotensi mengubah peringkat.

---

## IV. KESIMPULAN

⟦ISI: Tuliskan kesimpulan yang **hanya** memuat klaim yang didukung hasil. Ikuti
kerangka berikut:

1. Model dengan kinerja tertinggi beserta nilai mAP dan selang kepercayaannya.
2. Pernyataan eksplisit mengenai pasangan mana yang berbeda signifikan secara
   statistik dan mana yang tidak. Bila sebagian besar selisih tidak signifikan,
   nyatakan demikian — jangan menonjolkan peringkat yang tidak dapat dibedakan
   secara statistik.
3. Temuan analisis kegagalan: kelas dan kategori ukuran objek yang menjadi
   hambatan utama.
4. Rekomendasi praktis pemilihan arsitektur berdasarkan anggaran komputasi.
5. Arah penelitian lanjutan, misalnya penanganan ketidakseimbangan kelas, augmentasi
   untuk kondisi pencahayaan rendah dan oklusi, atau validasi lintas domain pada
   data lapangan nyata.

Hindari kata "terbaik" tanpa kualifikasi statistik, dan hindari klaim yang melampaui
cakupan dataset yang digunakan.⟧

---

## UCAPAN TERIMA KASIH

⟦ISI: opsional — sebutkan sumber pendanaan atau dukungan institusional bila ada⟧

## KETERSEDIAAN DATA DAN KODE

Dataset SH17 tersedia secara publik melalui Kaggle di bawah lisensi CC BY-NC-SA 4.0.
Seluruh kode, konfigurasi eksperimen, dan prosedur analisis statistik yang digunakan
dalam penelitian ini tersedia di ⟦ISI: tautan repositori⟧.

---

## DAFTAR PUSTAKA

[1] H. M. Ahmad dan A. Rahimi, "SH17: A dataset for human safety and personal
protective equipment detection in manufacturing industry," *Journal of Safety
Science and Resilience*, 2024, doi: 10.1016/j.jnlssr.2024.09.002.

[2] R. Sapkota dan M. Karkee, "Ultralytics YOLO evolution: An overview of YOLO27,
YOLO26, YOLO11, YOLOv8, and YOLOv5 object detectors for computer vision and
pattern recognition," *arXiv preprint arXiv:2510.09653*, 2025.

[3] R. Khanam dan M. Hussain, "YOLOv11: An overview of the key architectural
enhancements," *arXiv preprint arXiv:2410.17725*, 2024.

[4] J. Redmon, S. Divvala, R. Girshick, dan A. Farhadi, "You only look once:
Unified, real-time object detection," dalam *Proc. IEEE Conf. Computer Vision and
Pattern Recognition (CVPR)*, 2016, hlm. 779–788.

[5] T.-Y. Lin dkk., "Microsoft COCO: Common objects in context," dalam *European
Conference on Computer Vision (ECCV)*, 2014, hlm. 740–755.

[6] Y. Zhao dkk., "DETRs beat YOLOs on real-time object detection," dalam *Proc.
IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024.

[7] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, dan S. Zagoruyko,
"End-to-end object detection with transformers," dalam *European Conference on
Computer Vision (ECCV)*, 2020, hlm. 213–229.

[8] C.-Y. Wang, I.-H. Yeh, dan H.-Y. M. Liao, "YOLOv9: Learning what you want to
learn using programmable gradient information," dalam *European Conference on
Computer Vision (ECCV)*, 2024.

[9] A. Wang dkk., "YOLOv10: Real-time end-to-end object detection," dalam
*Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[10] S. Holm, "A simple sequentially rejective multiple test procedure,"
*Scandinavian Journal of Statistics*, vol. 6, no. 2, hlm. 65–70, 1979.

[11] B. Efron dan R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman &
Hall/CRC, 1993.

[12] J. Demšar, "Statistical comparisons of classifiers over multiple data sets,"
*Journal of Machine Learning Research*, vol. 7, hlm. 1–30, 2006.

[13] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, dan A. Zisserman,
"The PASCAL visual object classes (VOC) challenge," *International Journal of
Computer Vision*, vol. 88, no. 2, hlm. 303–338, 2010.

[14] G. Jocher, A. Chaurasia, dan J. Qiu, "Ultralytics YOLO," perangkat lunak
sumber terbuka, 2023. [Daring]. Tersedia: https://github.com/ultralytics/ultralytics

⟦ISI: Tambahkan 3–5 rujukan dari jurnal nasional terakreditasi yang relevan dengan
deteksi APD, keselamatan kerja, atau penerapan YOLO di Indonesia. Jurnal SINTA
umumnya menilai positif keterhubungan dengan literatur nasional, dan sebagian
mensyaratkan minimal 80% rujukan dari sepuluh tahun terakhir.⟧
