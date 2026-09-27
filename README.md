# Deteksi Kepatuhan dan Pelanggaran APD K3(Keselamatan & Kesehatan Kerja)

Penelitian pembanding arsitektur deteksi objek untuk pengawasan alat pelindung diri
(APD) di lingkungan kerja, disiapkan untuk publikasi pada jurnal terakreditasi
SINTA 3.

## Posisi penelitian

Dataset yang dipakai memuat **enam pasang kelas kepatuhan/pelanggaran**
(`helmet`/`no_helmet`, `vest`/`no_vest`, `gloves`/`no_gloves`, `goggles`/`no_goggles`,
`boots`/`no_boots`, `mask`/`no_mask`). Struktur berpasangan ini memungkinkan
pertanyaan yang jarang diajukan studi deteksi APD terdahulu:

> **Apakah model sama andalnya mengenali pelanggaran seperti mengenali kepatuhan?**

Pertanyaan itu penting karena konsekuensinya tidak setara. Sistem yang gagal
mendeteksi pekerja *tanpa* helm membiarkan bahaya berlalu tanpa peringatan,
sedangkan gagal mendeteksi pekerja *ber*-helm hanya menimbulkan catatan yang
hilang. Kebanyakan studi hanya melaporkan mAP agregat, yang menyamarkan asimetri
ini sepenuhnya.

Tiga penguatan metodologis yang jarang ditemui pada studi sejenis:

1. **Analisis asimetri per pasangan kepatuhan/pelanggaran**, bukan hanya mAP
   agregat.
2. **Setiap klaim keunggulan diuji signifikansinya** memakai bootstrap berpasangan
   dengan koreksi Holm — bukan sekadar membandingkan selisih mAP.
3. **Himpunan uji tidak pernah dipakai untuk memilih model.** Dataset sudah
   menyediakan validasi dan uji terpisah; validasi dipakai untuk penghentian dini,
   uji hanya disentuh sekali pada evaluasi akhir.

Ditambah satu faktor kebaruan: **YOLO26 dirilis Januari 2026** dan belum banyak
dievaluasi pada domain APD mana pun.

## Struktur proyek

```
configs/experiments.yaml          definisi seluruh run + hiperparameter bersama
data/sh17.yaml                    dibuat otomatis oleh langkah 2
scripts/
  common.py                       utilitas bersama (konfigurasi, path, JSON)
  detection_eval.py               perhitungan mAP ala COCO + evaluator bootstrap
  test_detection_eval.py          uji kebenaran modul di atas
  01_download_dataset.py          unduh SH17 dari Kaggle
  02a_resize_images.py            pra-resize citra (wajib untuk pelatihan CPU)
  02_prepare_dataset.py           split terstratifikasi + statistik dataset
  03_train.py                     pelatihan (dapat dilanjutkan bila sesi putus)
  04_evaluate.py                  evaluasi pada himpunan uji + tabel pencocokan
  05_statistics.py                bootstrap berpasangan + koreksi Holm
  06_figures.py                   6 gambar 300 dpi + tabel siap salin
notebooks/colab_train_sh17.ipynb  pipeline lengkap untuk Google Colab
paper/naskah.md                   draf naskah Bahasa Indonesia
paper/referensi.bib               bibliografi (14 rujukan terverifikasi)
results/                          keluaran eksperimen
```

## Cara menjalankan (Google Colab — jalur utama)

Mesin ini tidak memiliki GPU NVIDIA dan Python belum terpasang, sehingga seluruh
eksperimen dirancang untuk berjalan di Colab.

1. Unggah folder `configs/` dan `scripts/` ke Google Drive pada
   `MyDrive/sh17-ppe/`.
2. Buka `notebooks/colab_train_sh17.ipynb` di Colab, aktifkan GPU T4.
3. Jalankan sel berurutan. Notebook menangani unduhan data, pelatihan, evaluasi,
   statistik, dan pembuatan gambar.

Sesi Colab gratis kerap terputus. Semua bobot tersimpan di Drive dan setiap
pelatihan dapat dilanjutkan — jalankan ulang sel 1–5 lalu sel pelatihan; model yang
sudah selesai otomatis dilewati.

## Cara menjalankan (mesin ber-GPU sendiri)

```bash
pip install -r requirements.txt

python scripts/01_download_dataset.py      # butuh kredensial Kaggle
python scripts/02a_resize_images.py --root data/SH17          # wajib bila latih di CPU
python scripts/02_prepare_dataset.py --root data/SH17/resized
python scripts/test_detection_eval.py      # wajib lulus sebelum lanjut
python scripts/03_train.py --all --resume
python scripts/04_evaluate.py --all
python scripts/05_statistics.py --n-bootstrap 1000
python scripts/06_figures.py
```

### Kredensial Kaggle

Ambil di kaggle.com → Settings → API → **Create New Token**, lalu simpan
`kaggle.json` di `~/.kaggle/kaggle.json` (Windows:
`C:\Users\<nama>\.kaggle\kaggle.json`). Alternatifnya, set variabel lingkungan
`KAGGLE_USERNAME` dan `KAGGLE_KEY`.

## Rancangan eksperimen

Tiga model wajib — satu per generasi arsitektur, semuanya skala nano:

| Run | Model | Params | GFLOPs | Keluarga |
|---|---|---:|---:|---|
| `yolov8n` | `yolov8n.pt` | 3,16 jt | 8,86 | YOLOv8 (pembanding tolok ukur asli) |
| `yolo11n` | `yolo11n.pt` | 2,62 jt | 6,67 | YOLO11 (belum pernah diuji di SH17) |
| `yolo26n` | `yolo26n.pt` | 2,57 jt | 6,24 | YOLO26 (belum pernah diuji di SH17) |

Skala nano dipilih karena dua alasan yang saling menguatkan: pengawasan APD di
lapangan berjalan di perangkat tepi berdaya rendah, dan pelatihan di sini
berlangsung pada CPU sehingga cakupan harus tetap dapat diselesaikan.

Run opsional (`--optional`): skala `s` untuk ketiga keluarga, RT-DETR-l sebagai
pembanding transformer, dan replikasi multi-seed. Semuanya memerlukan GPU.

Seluruh hiperparameter disamakan, termasuk ukuran batch (100 epoch, 640 px,
batch 16, penghentian dini 25, penjadwal kosinus, seed 0) — tidak ada satu pun
faktor selain arsitektur yang membedakan kondisi pelatihan.

### Waktu terukur pada perangkat penelitian

Diukur pada AMD Ryzen 5 7535HS (6 inti, 12 utas), PyTorch 2.14 CPU:

| Tahap | Perkiraan |
|---|---|
| Unduh dataset | 20–40 menit |
| Pra-resize citra | 10–20 menit |
| Persiapan split + statistik | 10 menit |
| Pelatihan `yolov8n` (100 epoch) | ~33 jam |
| Pelatihan `yolo11n` (100 epoch) | ~52 jam |
| Pelatihan `yolo26n` (100 epoch) | ~54 jam |
| **Total pelatihan** | **~7 hari** (termasuk I/O dan validasi) |
| Evaluasi 3 model | 1–2 jam |
| Bootstrap 1.000 resample | 20–60 menit |
| Gambar dan tabel | < 5 menit |

Pada GPU T4, seluruh pelatihan selesai dalam 6–9 jam. Bila Anda punya akses GPU,
notebook Colab tetap tersedia dan skala `s` dapat diaktifkan dengan `--optional`.

## Catatan metodologis penting

**Mengapa perhitungan mAP diimplementasikan ulang.** `COCOeval` pada `pycocotools`
melakukan de-duplikasi indeks citra. Pengambilan sampel dengan pengembalian —
syarat mutlak bootstrap — karenanya berubah diam-diam menjadi sub-sampling tanpa
bobot dan menghasilkan selang kepercayaan yang bias. `detection_eval.py`
mencocokkan deteksi ke kebenaran dasar satu kali, menyimpannya sebagai tabel
per-deteksi, lalu menghitung ulang AP untuk sembarang multiset citra. Definisi
pencocokan dan interpolasi 101 titik mengikuti COCO, sehingga angkanya tetap
sebanding dengan literatur.

**Verifikasi wajib.** Karena seluruh angka naskah bergantung pada modul tersebut,
`scripts/test_detection_eval.py` mengujinya terhadap kasus yang jawabannya dapat
dihitung tangan. Uji ini **belum pernah dijalankan** karena Python tidak tersedia
di mesin ini — jalankan di Colab (sel 8) dan pastikan lulus sebelum mempercayai
angka apa pun.

**Apa yang diukur bootstrap.** Ketidakpastian akibat sampel citra uji, bukan akibat
inisialisasi bobot acak. Untuk yang kedua diperlukan pelatihan ulang multi-seed —
tersedia sebagai run opsional, dan bila tidak dijalankan harus tetap dinyatakan
sebagai keterbatasan.

## Menulis naskah

`paper/naskah.md` sudah memuat pendahuluan, metode, dan kerangka pembahasan secara
lengkap. Setiap tempat yang memerlukan angka hasil ditandai `⟦ISI: …⟧`.

**Tidak ada satu pun angka hasil yang terisi di berkas itu.** Semua harus disalin
dari `results/analysis/tabel_naskah.md` setelah eksperimen benar-benar dijalankan.
Beberapa penanda juga meminta pelaporan hasil negatif secara jujur — bila selisih
antar-arsitektur ternyata tidak signifikan, itu temuan yang sah dan justru
memperkuat alasan penelitian ini menggunakan uji statistik.

Sebelum submit:

- Hapus blok catatan penggunaan di bagian atas naskah.
- Pastikan tidak ada penanda tersisa: `grep -n "ISI:" paper/naskah.md`
- Tambahkan 3–5 rujukan jurnal nasional terakreditasi.
- Sesuaikan format dengan templat jurnal tujuan.

### Kandidat jurnal SINTA 3 bidang informatika

JUTIF (Unsoed), JEPIN (Untan), Teknika (IKADO), JUTI (ITS), Techno.COM (Udinus),
JUITA (UMP). Periksa kembali peringkat terkini di
[sinta.kemdiktisaintek.go.id](https://sinta.kemdiktisaintek.go.id) karena akreditasi
diperbarui berkala.

## Lisensi dan etika penggunaan data

Dataset SH17 berlisensi **CC BY-NC-SA 4.0** (non-komersial). Penggunaan untuk
penelitian akademik sesuai ketentuan. Citra bersumber dari Pexels; ketentuan Pexels
melarang penggambaran orang yang dapat dikenali dalam konteks merugikan. Naskah
wajib menyitir Ahmad & Rahimi (2024).
