"""
Pra-resize citra SH17 agar pelatihan pada CPU menjadi praktis.

Mengapa langkah ini ada
-----------------------
Citra SH17 beresolusi hingga 8192x5462 piksel. Pada setiap epoch, pemuat data
harus mendekode ulang JPEG sebesar itu lalu menyusutkannya ke 640 piksel. Pada
GPU, biaya dekode tersebut tersembunyi di balik komputasi model. Pada CPU,
justru dekode inilah yang mendominasi waktu - kerap melampaui waktu komputasi
model itu sendiri.

Menyusutkan citra satu kali di awal menghilangkan biaya berulang tersebut.
Karena pelatihan tetap berlangsung pada 640 piksel, dan sisi terpanjang hasil
penyusutan (baku 1024 piksel) masih di atas resolusi pelatihan, mutu masukan
model secara praktis tidak berubah.

Anotasi TIDAK perlu diubah: format YOLO menyimpan koordinat ternormalisasi
terhadap dimensi citra, sehingga nilainya tetap sahih setelah penyusutan.

Pemakaian:
  python scripts/02a_resize_images.py --root data/SH17
  python scripts/02a_resize_images.py --root data/SH17 --max-side 1024 --quality 90

Selanjutnya arahkan tahap persiapan ke folder hasil:
  python scripts/02_prepare_dataset.py --root data/SH17/resized
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import human_bytes, resolve, setup_logging  # noqa: E402

log = setup_logging("resize")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def locate_dirs(root: Path) -> tuple[Path, Path]:
    """Cari folder images/ dan labels/, melewati folder hasil resize."""
    candidates = [root, *[p for p in root.rglob("*") if p.is_dir()]]
    for base in candidates:
        if "resized" in base.parts:
            continue
        img, lbl = base / "images", base / "labels"
        if img.is_dir() and lbl.is_dir():
            n_img = sum(1 for p in img.iterdir() if p.suffix.lower() in IMG_EXT)
            if n_img > 500:
                return img, lbl
    raise SystemExit(
        f"Folder images/ + labels/ tidak ditemukan di {root}.\n"
        "Jalankan dulu: python scripts/01_download_dataset.py"
    )


def resize_one(args: tuple[str, str, int, int]) -> tuple[str, int, int, str]:
    """
    Susutkan satu citra. Dijalankan pada proses terpisah.

    Return: (nama, byte_sumber, byte_hasil, status)
      status: 'ok' | 'copied' | 'skipped' | 'error:<pesan>'
    """
    src_s, dst_s, max_side, quality = args
    src, dst = Path(src_s), Path(dst_s)
    try:
        src_bytes = src.stat().st_size
        if dst.is_file():                      # sudah dikerjakan pada jalannya sebelumnya
            return src.name, src_bytes, dst.stat().st_size, "skipped"

        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None          # SH17 memuat citra sangat besar

        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest <= max_side:
                # Sudah cukup kecil - salin apa adanya agar piksel tidak berubah.
                shutil.copy2(src, dst)
                return src.name, src_bytes, dst.stat().st_size, "copied"

            scale = max_side / longest
            new_size = (max(round(w * scale), 1), max(round(h * scale), 1))
            im = im.resize(new_size, Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality, optimize=True)

        return src.name, src_bytes, dst.stat().st_size, "ok"
    except Exception as e:                     # satu citra rusak tidak boleh menghentikan semua
        return src.name, 0, 0, f"error:{e}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="Akar dataset SH17")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="Panjang sisi terpanjang setelah penyusutan (baku 1024)")
    ap.add_argument("--quality", type=int, default=90, help="Mutu JPEG (baku 90)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Jumlah proses paralel (0 = otomatis)")
    args = ap.parse_args()

    root = resolve(args.root)
    if not root.is_dir():
        raise SystemExit(f"{root} tidak ada.")

    if args.max_side < 640:
        log.warning("max-side %d berada di bawah resolusi pelatihan 640 - "
                    "ini akan menurunkan mutu masukan model.", args.max_side)

    src_img, src_lbl = locate_dirs(root)
    out = root / "resized"
    out_img, out_lbl = out / "images", out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"Tidak ada citra di {src_img}")
    log.info("Sumber : %s (%d citra)", src_img, len(images))
    log.info("Tujuan : %s", out_img)
    log.info("Sisi terpanjang -> %d px, mutu JPEG %d", args.max_side, args.quality)

    # Semua keluaran berekstensi .jpg agar nama berkas konsisten dengan label.
    tasks = [(str(p), str(out_img / f"{p.stem}.jpg"), args.max_side, args.quality)
             for p in images]

    workers = args.workers or 4
    n_ok = n_copied = n_skipped = n_err = 0
    total_src = total_dst = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resize_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            name, sb, db, status = fut.result()
            total_src += sb
            total_dst += db
            if status == "ok":
                n_ok += 1
            elif status == "copied":
                n_copied += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_err += 1
                if n_err <= 5:
                    log.warning("  gagal %s -> %s", name, status)

            if i % 500 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1e-9)
                sisa = (len(futures) - i) / max(rate, 1e-9)
                log.info("  %d/%d citra (%.1f/dtk, sisa ~%.0f dtk)",
                         i, len(futures), rate, sisa)

    # Label disalin apa adanya - koordinat YOLO ternormalisasi, jadi tetap sahih.
    n_lbl = 0
    for p in src_lbl.glob("*.txt"):
        dst = out_lbl / p.name
        if not dst.is_file():
            shutil.copy2(p, dst)
        n_lbl += 1

    # Berkas daftar split resmi ikut disalin agar 02_prepare menemukannya.
    for pattern in ("train_files.txt", "test_files.txt", "val_files.txt",
                    "train.txt", "test.txt"):
        for p in root.rglob(pattern):
            if "resized" in p.parts:
                continue
            shutil.copy2(p, out / p.name)
            break

    elapsed = time.time() - t0
    info = {
        "max_side": args.max_side,
        "quality": args.quality,
        "images_processed": n_ok,
        "images_copied_unchanged": n_copied,
        "images_skipped_existing": n_skipped,
        "images_failed": n_err,
        "labels_copied": n_lbl,
        "source_bytes": total_src,
        "output_bytes": total_dst,
        "elapsed_seconds": round(elapsed, 1),
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "resize_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    log.info("")
    log.info("Selesai dalam %.1f menit", elapsed / 60)
    log.info("  disusutkan          : %d", n_ok)
    log.info("  disalin apa adanya  : %d", n_copied)
    log.info("  dilewati (sudah ada): %d", n_skipped)
    log.info("  gagal               : %d", n_err)
    log.info("  label disalin       : %d", n_lbl)
    if total_src:
        log.info("  ukuran: %s -> %s (%.1f%% dari semula)",
                 human_bytes(total_src), human_bytes(total_dst),
                 100 * total_dst / total_src)
    if n_err:
        log.warning("%d citra gagal diproses - periksa pesan di atas.", n_err)

    log.info("")
    log.info("Langkah berikutnya:")
    log.info("  python scripts/02_prepare_dataset.py --root %s", out)


if __name__ == "__main__":
    main()
