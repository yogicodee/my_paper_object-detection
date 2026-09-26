"""
Siapkan dataset untuk pelatihan Ultralytics + hitung statistik deskriptif.

Skrip ini mengenali dua tata letak dataset.

  A. SUDAH TERBAGI - folder train/, val/, test/, masing-masing berisi
     images/ dan labels/. Pembagian bawaan dataset dipakai apa adanya.

  B. DATAR - satu folder images/ dan labels/, disertai daftar pembagian
     resmi (train_files.txt, test_files.txt). Karena tata letak ini umumnya
     hanya menyediakan latih dan uji, himpunan latih dipecah menjadi
     latih/validasi secara terstratifikasi agar himpunan uji tetap murni.

Mengapa pemisahan validasi dan uji itu penting
----------------------------------------------
Bila satu himpunan dipakai sekaligus untuk penghentian dini DAN pelaporan
akhir, angka yang dilaporkan menjadi bias optimistis karena himpunan itu ikut
memandu pemilihan model. Pada tata letak A, dataset sudah menyediakan validasi
dan uji terpisah sehingga syarat ini otomatis terpenuhi.

Pemakaian:
  python scripts/02_prepare_dataset.py
  python scripts/02_prepare_dataset.py --root data/PPE
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve, setup_logging, write_json  # noqa: E402

log = setup_logging("prepare")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


# --------------------------------------------------------------------------- #
# Penemuan tata letak
# --------------------------------------------------------------------------- #
def _count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def find_presplit(root: Path) -> dict[str, Path] | None:
    """Cari folder train/val/test yang masing-masing punya images/ + labels/."""
    for base in [root, *[p for p in root.iterdir() if p.is_dir()]]:
        found: dict[str, Path] = {}
        for split, aliases in SPLIT_ALIASES.items():
            for a in aliases:
                cand = base / a
                if (cand / "images").is_dir() and (cand / "labels").is_dir():
                    if _count_images(cand / "images") > 0:
                        found[split] = cand
                        break
        if "train" in found and "test" in found:
            return found
    return None


def find_flat(root: Path) -> tuple[Path, Path] | None:
    """Cari satu pasang folder images/ + labels/ yang datar."""
    for base in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        img, lbl = base / "images", base / "labels"
        if img.is_dir() and lbl.is_dir() and _count_images(img) > 500:
            return img, lbl
    return None


def find_split_file(root: Path, *names: str) -> Path | None:
    for name in names:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


def read_split_file(path: Path, images_dir: Path) -> list[Path]:
    """Baca daftar pembagian; entri bisa nama berkas atau path penuh."""
    out, missing = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if not line:
            continue
        p = images_dir / Path(line).name
        if p.is_file():
            out.append(p)
        else:
            missing += 1
    if missing:
        log.warning("%d entri di %s tidak punya berkas citra padanan.", missing, path.name)
    return out


# --------------------------------------------------------------------------- #
# Label
# --------------------------------------------------------------------------- #
def label_for(img: Path, fallback: Path | None = None) -> Path:
    """
    Tentukan berkas label milik sebuah citra.

    Konvensi Ultralytics menaruh label pada folder `labels/` yang sejajar
    dengan `images/`, sehingga penelusuran relatif ini berlaku untuk kedua
    tata letak tanpa perlu tahu mana yang sedang dipakai.
    """
    sibling = img.parent.parent / "labels" / f"{img.stem}.txt"
    if sibling.is_file():
        return sibling
    if fallback is not None:
        return fallback / f"{img.stem}.txt"
    return sibling


def read_boxes(lbl: Path) -> list[tuple[int, float, float, float, float]]:
    """Baca anotasi format YOLO: cls cx cy w h (ternormalisasi)."""
    if not lbl.is_file():
        return []
    boxes = []
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
        except ValueError:
            continue
    return boxes


# --------------------------------------------------------------------------- #
# Pemecahan latih/validasi terstratifikasi (hanya untuk tata letak datar)
# --------------------------------------------------------------------------- #
def stratified_val_split(
    train_imgs: list[Path], labels_dir: Path | None, val_frac: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """
    Stratifikasi berdasarkan kelas paling langka pada tiap citra, agar kelas
    minoritas tetap terwakili di validasi. Pembagian acak murni berisiko
    mengosongkan kelas langka sehingga penghentian dini dipandu metrik yang
    tidak mewakili seluruh kelas.
    """
    class_freq: Counter[int] = Counter()
    per_image: dict[Path, set[int]] = {}
    for img in train_imgs:
        cls = {b[0] for b in read_boxes(label_for(img, labels_dir))}
        per_image[img] = cls
        class_freq.update(cls)

    buckets: dict[int, list[Path]] = defaultdict(list)
    for img, cls in per_image.items():
        key = min(cls, key=lambda c: class_freq[c]) if cls else -1
        buckets[key].append(img)

    rng = random.Random(seed)
    train_out, val_out = [], []
    for key in sorted(buckets):
        group = sorted(buckets[key])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac)) if len(group) > 1 else 0
        val_out.extend(group[:n_val])
        train_out.extend(group[n_val:])

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


# --------------------------------------------------------------------------- #
# Statistik deskriptif
# --------------------------------------------------------------------------- #
def describe(
    splits: dict[str, list[Path]],
    labels_dir: Path | None,
    names: dict[int, str],
    pairs: dict[str, str] | None,
) -> dict:
    """
    Statistik per-split, per-kelas, dan per-pasangan kepatuhan.

    Kategori ukuran memakai luas relatif terhadap luas citra
    (kecil < 1%, sedang 1-5%, besar > 5%) agar tetap adil lintas resolusi.
    """
    report: dict = {"splits": {}, "per_class": {}}
    global_cls: Counter[int] = Counter()
    global_size: Counter[str] = Counter()

    for split, imgs in splits.items():
        cls_count: Counter[int] = Counter()
        size_count: Counter[str] = Counter()
        n_boxes = empty = 0
        for img in imgs:
            boxes = read_boxes(label_for(img, labels_dir))
            if not boxes:
                empty += 1
            n_boxes += len(boxes)
            for c, _cx, _cy, w, h in boxes:
                cls_count[c] += 1
                area = w * h
                size_count["small" if area < 0.01 else
                            ("medium" if area < 0.05 else "large")] += 1
        global_cls.update(cls_count)
        global_size.update(size_count)
        report["splits"][split] = {
            "images": len(imgs),
            "instances": n_boxes,
            "instances_per_image": round(n_boxes / max(len(imgs), 1), 2),
            "images_without_labels": empty,
            "size_distribution": dict(size_count),
            "class_counts": {names[c]: cls_count[c] for c in sorted(names) if cls_count[c]},
        }

    total = sum(global_cls.values())
    for c in sorted(names):
        n = global_cls[c]
        report["per_class"][names[c]] = {
            "id": c,
            "instances": n,
            "share_pct": round(100 * n / total, 3) if total else 0.0,
        }

    report["totals"] = {
        "images": sum(len(v) for v in splits.values()),
        "instances": total,
        "size_distribution": dict(global_size),
    }
    counts = [v["instances"] for v in report["per_class"].values() if v["instances"] > 0]
    if counts:
        report["totals"]["imbalance_ratio"] = round(max(counts) / min(counts), 1)

    # Keseimbangan tiap pasangan kepatuhan/pelanggaran. Rasio yang jauh dari 1
    # menandakan model akan melihat contoh pelanggaran jauh lebih sedikit
    # daripada contoh kepatuhan - dasar bagi analisis asimetri di naskah.
    if pairs:
        report["compliance_pairs"] = {}
        for pos, neg in pairs.items():
            p = report["per_class"].get(pos, {}).get("instances", 0)
            n = report["per_class"].get(neg, {}).get("instances", 0)
            report["compliance_pairs"][pos] = {
                "positive": p,
                "negative": n,
                "negative_share_pct": round(100 * n / (p + n), 2) if (p + n) else 0.0,
                "ratio_pos_to_neg": round(p / n, 2) if n else None,
            }
    return report


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    ds = cfg["dataset"]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=None, help="Akar dataset")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Porsi latih yang dijadikan validasi (tata letak datar saja)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root) if args.root else resolve(ds["root"])
    if not root.is_dir():
        raise SystemExit(f"{root} tidak ada. Jalankan dulu scripts/01_download_dataset.py")

    names = {int(k): v for k, v in ds["names"].items()}
    pairs = ds.get("compliance_pairs")
    labels_dir: Path | None = None

    # --- Tata letak A: sudah terbagi ----------------------------------------
    presplit = find_presplit(root)
    if presplit:
        log.info("Tata letak terdeteksi: SUDAH TERBAGI")
        splits: dict[str, list[Path]] = {}
        for split in ("train", "val", "test"):
            if split not in presplit:
                continue
            d = presplit[split] / "images"
            splits[split] = sorted(p for p in d.iterdir()
                                   if p.is_file() and p.suffix.lower() in IMG_EXT)
            log.info("  %-5s %s (%d citra)", split, presplit[split], len(splits[split]))

        if "val" not in splits:
            log.warning("Tidak ada himpunan validasi bawaan - memecah dari latih.")
            tr, va = stratified_val_split(splits["train"], None, args.val_frac, args.seed)
            splits["train"], splits["val"] = tr, va
        else:
            log.info("Validasi dan uji sudah terpisah pada dataset - "
                     "himpunan uji tidak akan tersentuh sampai evaluasi akhir.")

    # --- Tata letak B: datar -------------------------------------------------
    else:
        flat = find_flat(root)
        if not flat:
            raise SystemExit(
                f"Tidak menemukan tata letak dataset yang dikenali di {root}.\n"
                "Diharapkan salah satu:\n"
                "  train/images + train/labels (dan val/, test/)\n"
                "  images/ + labels/ beserta train_files.txt dan test_files.txt"
            )
        images_dir, labels_dir = flat
        log.info("Tata letak terdeteksi: DATAR (%s)", images_dir.parent)

        train_file = find_split_file(root, "train_files.txt", "train.txt")
        test_file = find_split_file(root, "test_files.txt", "test.txt", "val_files.txt")
        if train_file and test_file:
            log.info("Memakai pembagian resmi: %s / %s", train_file.name, test_file.name)
            train_all = read_split_file(train_file, images_dir)
            test_imgs = read_split_file(test_file, images_dir)
        else:
            log.warning("Berkas pembagian resmi tidak ada - membuat pembagian 80/20.")
            every = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            rng = random.Random(args.seed)
            rng.shuffle(every)
            cut = int(0.8 * len(every))
            train_all, test_imgs = every[:cut], every[cut:]

        if not train_all or not test_imgs:
            raise SystemExit("Pembagian kosong - periksa hasil unduhan dataset.")

        train_imgs, val_imgs = stratified_val_split(
            train_all, labels_dir, args.val_frac, args.seed
        )
        splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    log.info("Pembagian akhir  latih=%d  validasi=%d  uji=%d",
             len(splits["train"]), len(splits.get("val", [])), len(splits["test"]))

    # --- Daftar berkas (path absolut, syarat Ultralytics) -------------------
    split_dir = root / "splits"
    split_dir.mkdir(exist_ok=True)
    for name, imgs in splits.items():
        out = split_dir / f"{name}.txt"
        out.write_text("\n".join(str(p.resolve()).replace("\\", "/") for p in imgs) + "\n",
                       encoding="utf-8")
        log.info("  ditulis %s (%d baris)", out, len(imgs))

    # --- Berkas konfigurasi dataset -----------------------------------------
    yaml_path = resolve(ds["yaml"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join([
        "# Dibuat otomatis oleh scripts/02_prepare_dataset.py - jangan diedit manual.",
        f"path: {str(root.resolve()).replace(chr(92), '/')}",
        "train: splits/train.txt",
        "val: splits/val.txt",
        "test: splits/test.txt",
        "",
        "names:",
        *[f"  {i}: {names[i]}" for i in sorted(names)],
        "",
    ]), encoding="utf-8")
    log.info("Ditulis %s", yaml_path)

    # --- Statistik -----------------------------------------------------------
    log.info("Menghitung statistik dataset ...")
    stats = describe(splits, labels_dir, names, pairs)
    write_json(stats, resolve("results/analysis/dataset_stats.json"))

    t = stats["totals"]
    log.info("")
    log.info("Total: %d citra, %d instance", t["images"], t["instances"])
    log.info("Sebaran ukuran objek: %s", t["size_distribution"])
    log.info("Rasio ketidakseimbangan kelas: %sx", t.get("imbalance_ratio", "n/a"))

    if stats.get("compliance_pairs"):
        log.info("")
        log.info("%-10s %10s %10s %14s", "PASANGAN", "PATUH", "LANGGAR", "%LANGGAR")
        for pos, v in stats["compliance_pairs"].items():
            log.info("%-10s %10d %10d %13.1f%%",
                     pos, v["positive"], v["negative"], v["negative_share_pct"])

    log.info("")
    log.info("5 kelas paling langka:")
    for name, v in sorted(stats["per_class"].items(), key=lambda kv: kv[1]["instances"])[:5]:
        log.info("  %-14s %6d instance (%.2f%%)", name, v["instances"], v["share_pct"])

    log.info("")
    log.info("Langkah berikutnya: python scripts/03_train.py --run yolo11n")


if __name__ == "__main__":
    main()
