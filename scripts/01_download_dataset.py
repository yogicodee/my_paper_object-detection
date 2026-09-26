r"""
Unduh dataset SH17 dari Kaggle.

Prasyarat kredensial (salah satu, diperiksa berurutan):
  1. Variabel lingkungan KAGGLE_API_TOKEN berisi token berawalan KGAT_
  2. Berkas ~/.kaggle/access_token berisi token tersebut
     (Windows: C:\Users\<nama>\.kaggle\access_token)
  3. Berkas ~/.kaggle/kaggle.json bentuk lama berisi username dan key
  4. Variabel lingkungan KAGGLE_USERNAME dan KAGGLE_KEY

Cara memperoleh token:
  kaggle.com -> Settings -> bagian API -> "Create New Token"

Pemakaian:
  python scripts/01_download_dataset.py
  python scripts/01_download_dataset.py --dest /content/data/SH17   # Colab
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import human_bytes, load_config, resolve, setup_logging  # noqa: E402

log = setup_logging("download")


def check_credentials() -> None:
    """
    Pastikan kredensial Kaggle tersedia.

    Kaggle mendukung dua bentuk kredensial. Bentuk yang lebih baru berupa token
    tunggal (berawalan `KGAT_`) yang disalin dari laman pengaturan, sedangkan
    bentuk lama berupa berkas `kaggle.json` berisi pasangan username dan key.
    Keduanya diterima di sini.
    """
    # --- Bentuk baru: token tunggal -----------------------------------------
    if os.getenv("KAGGLE_API_TOKEN"):
        log.info("Kredensial ditemukan pada variabel lingkungan KAGGLE_API_TOKEN.")
        return

    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.is_file() and token_file.stat().st_size > 0:
        log.info("Kredensial ditemukan: %s", token_file)
        if os.name == "posix":
            try:
                os.chmod(token_file, 0o600)
            except OSError:
                pass
        return

    # --- Bentuk lama: username + key ----------------------------------------
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        log.info("Kredensial ditemukan pada variabel lingkungan KAGGLE_USERNAME/KAGGLE_KEY.")
        return

    for c in (Path.home() / ".kaggle" / "kaggle.json",
              Path(os.getenv("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"):
        if c.is_file():
            log.info("Kredensial ditemukan: %s", c)
            # Kaggle CLI menolak berkas yang terlalu terbuka di sistem POSIX.
            if os.name == "posix":
                try:
                    os.chmod(c, 0o600)
                except OSError:
                    pass
            return

    raise SystemExit(
        "\nKredensial Kaggle tidak ditemukan.\n"
        "Buka kaggle.com/settings -> bagian API -> Create New Token, lalu:\n"
        f"  simpan token yang muncul ke: {token_file}\n"
        "atau set variabel lingkungan KAGGLE_API_TOKEN dengan token tersebut.\n"
    )


def already_downloaded(dest: Path) -> bool:
    """Dataset dianggap lengkap bila folder gambar & label ada dan tidak kosong."""
    for images_dir in (dest / "images", dest):
        labels_dir = images_dir.parent / "labels" if images_dir.name == "images" else dest / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            n_img = sum(1 for _ in images_dir.glob("*.*"))
            n_lbl = sum(1 for _ in labels_dir.glob("*.txt"))
            if n_img > 1000 and n_lbl > 1000:
                log.info("Dataset sudah ada: %d gambar, %d label di %s", n_img, n_lbl, dest)
                return True
    return False


def download(slug: str, dest: Path, force: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not force and already_downloaded(dest):
        log.info("Lewati unduhan (gunakan --force untuk mengunduh ulang).")
        return

    free = shutil.disk_usage(dest).free
    log.info("Ruang disk tersedia di %s: %s", dest, human_bytes(free))
    if free < 25 * 1024**3:
        log.warning(
            "Ruang bebas < 25 GB. SH17 berisi gambar resolusi tinggi "
            "(hingga 8192x5462) dan butuh ruang lega saat ekstraksi."
        )

    log.info("Mengunduh %s -> %s (bisa memakan waktu lama, ~10 GB)", slug, dest)

    # API Python dipakai lebih dulu karena tidak bergantung pada keberadaan
    # perintah `kaggle` di PATH - hal yang kerap meleset di dalam virtualenv.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    except Exception as e:
        log.warning("Unduhan via API Python gagal (%s). Mencoba perintah CLI.", e)
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
                check=True,
            )
        except FileNotFoundError:
            raise SystemExit("Perintah `kaggle` tidak ditemukan. Jalankan: pip install kaggle")
        except subprocess.CalledProcessError as e2:
            # --unzip kadang gagal pada arsip besar; unduh lalu ekstrak manual.
            log.warning("Unduhan dengan --unzip gagal (kode %s). Mencoba tanpa --unzip.",
                        e2.returncode)
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)], check=True
            )

    # Bila arsip tersisa (mode tanpa --unzip atau ekstraksi parsial), buka di sini.
    for zf in sorted(dest.glob("*.zip")):
        log.info("Mengekstrak %s ...", zf.name)
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()

    log.info("Unduhan selesai.")


def summarise(dest: Path) -> None:
    log.info("--- Isi %s ---", dest)
    entries = sorted(dest.iterdir())[:25]
    for p in entries:
        if p.is_dir():
            n = sum(1 for _ in p.iterdir())
            log.info("  [dir ] %-24s %d berkas", p.name, n)
        else:
            log.info("  [file] %-24s %s", p.name, human_bytes(p.stat().st_size))
    if not entries:
        log.warning("Folder kosong — unduhan kemungkinan gagal.")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=None, help="Folder tujuan (default dari configs/experiments.yaml)")
    ap.add_argument("--force", action="store_true", help="Unduh ulang walau data sudah ada")
    args = ap.parse_args()

    dest = Path(args.dest) if args.dest else resolve(cfg["dataset"]["root"])

    check_credentials()
    download(cfg["dataset"]["kaggle_slug"], dest, args.force)
    summarise(dest)

    log.info("")
    log.info("Langkah berikutnya: python scripts/02_prepare_dataset.py")


if __name__ == "__main__":
    main()
