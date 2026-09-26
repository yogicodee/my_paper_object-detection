"""
Evaluasi model terlatih pada split TEST (held-out) dan simpan tabel pencocokan.

Split test hanya disentuh di sini - tidak pernah dipakai untuk early stopping
maupun pemilihan model - sehingga angka yang dilaporkan bebas dari bias seleksi.

Keluaran per model:
  results/runs/<id>/matches_test.npz   tabel pencocokan (bahan uji bootstrap)
  results/analysis/eval_<id>.json      metrik agregat, per-kelas, per-ukuran

Pemakaian:
  python scripts/04_evaluate.py --run yolo11n
  python scripts/04_evaluate.py --all
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    all_runs, find_run, load_config, resolve, setup_logging, write_json,
)
from detection_eval import (  # noqa: E402
    AREA_RANGES, MatchTable, compute_ap, match_image, precision_recall_f1,
)

log = setup_logging("eval")


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def read_image_list(data_yaml: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    listing = root / cfg[split]
    if not listing.is_file():
        raise SystemExit(f"Daftar split tidak ditemukan: {listing}")
    return [Path(line.strip()) for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]


def gt_for_image(img: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Baca GT satu gambar sebagai xyxy ternormalisasi [0,1] + kelas.

    Label YOLO sudah ternormalisasi, jadi tidak perlu ukuran gambar asli di sini.
    Prediksi akan dinormalisasi dengan cara yang sama agar keduanya sebanding.
    """
    lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
    if not lbl.is_file():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int16)

    boxes, classes = [], []
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
        except ValueError:
            continue
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        classes.append(c)

    return (np.array(boxes, np.float32).reshape(-1, 4),
            np.array(classes, np.int16))


def area_bucket_index(area: float) -> int:
    for i, (name, (lo, hi)) in enumerate(AREA_RANGES.items()):
        if name == "all":
            continue
        if lo <= area < hi:
            return i
    return len(AREA_RANGES) - 1        # fallback: large


# --------------------------------------------------------------------------- #
def build_match_table(model, images: list[Path], n_classes: int, device, conf: float,
                      batch: int) -> tuple[MatchTable, float]:
    """Jalankan inferensi pada seluruh test set lalu cocokkan ke GT."""
    all_img_idx, all_cls, all_score, all_tp, all_area = [], [], [], [], []
    n_gt = np.zeros((len(images), n_classes), np.int32)
    n_gt_area = np.zeros((len(images), n_classes, len(AREA_RANGES)), np.int32)
    gt_areas: list[float] = []

    infer_seconds = 0.0

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        t0 = time.time()
        results = model.predict(
            [str(p) for p in chunk],
            conf=conf,          # ambang rendah: mAP perlu ekor skor yang panjang
            iou=0.7,            # NMS default Ultralytics
            max_det=300,        # konvensi COCO
            device=device,
            verbose=False,
            stream=False,
        )
        infer_seconds += time.time() - t0

        for offset, (img, res) in enumerate(zip(chunk, results)):
            i = start + offset
            gt_boxes, gt_cls = gt_for_image(img)

            for c, box in zip(gt_cls, gt_boxes):
                if 0 <= c < n_classes:
                    a = float((box[2] - box[0]) * (box[3] - box[1]))
                    n_gt[i, c] += 1
                    n_gt_area[i, c, 0] += 1                      # "all"
                    n_gt_area[i, c, area_bucket_index(a)] += 1
                    gt_areas.append(a)

            b = res.boxes
            if b is None or len(b) == 0:
                continue

            # Normalisasi prediksi ke [0,1] memakai ukuran gambar asli.
            h, w = res.orig_shape
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] /= max(w, 1)
            xyxy[:, [1, 3]] /= max(h, 1)
            pcls = b.cls.cpu().numpy().astype(np.int16)
            pscore = b.conf.cpu().numpy().astype(np.float32)

            tp = match_image(xyxy, pcls, pscore, gt_boxes, gt_cls)

            all_img_idx.append(np.full(len(pcls), i, np.int32))
            all_cls.append(pcls)
            all_score.append(pscore)
            all_tp.append(tp)
            all_area.append(((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])).astype(np.float32))

        if (start // max(batch, 1)) % 20 == 0:
            log.info("  %d/%d gambar", min(start + batch, len(images)), len(images))

    def cat(parts, shape, dtype):
        return np.concatenate(parts) if parts else np.zeros(shape, dtype)

    mt = MatchTable(
        image_idx=cat(all_img_idx, (0,), np.int32),
        cls=cat(all_cls, (0,), np.int16),
        score=cat(all_score, (0,), np.float32),
        tp=cat(all_tp, (0, 10), bool),
        area=cat(all_area, (0,), np.float32),
        n_gt=n_gt,
        n_gt_area=n_gt_area,
        gt_area=np.array(gt_areas, np.float32),
        n_images=len(images),
        n_classes=n_classes,
    )
    return mt, infer_seconds


def measure_latency(model, images: list[Path], device, n: int = 100, warmup: int = 10) -> float:
    """
    Latensi inferensi batch=1 dalam milidetik per gambar.

    Batch=1 dipilih karena mencerminkan penerapan nyata pemantauan APD (satu
    frame kamera pada satu waktu), bukan pemrosesan berkelompok.
    """
    sample = images[: n + warmup]
    for p in sample[:warmup]:
        model.predict(str(p), device=device, verbose=False)

    t0 = time.time()
    for p in sample[warmup:]:
        model.predict(str(p), device=device, verbose=False)
    elapsed = time.time() - t0

    used = max(len(sample) - warmup, 1)
    return round(1000 * elapsed / used, 3)


# --------------------------------------------------------------------------- #
def evaluate_run(run: dict, cfg: dict, args) -> dict | None:
    from ultralytics import RTDETR, YOLO

    run_id = run["id"]
    weights = resolve(f"results/runs/{run_id}/weights/best.pt")
    if not weights.is_file():
        log.warning("[%s] best.pt belum ada - lewati (latih dulu).", run_id)
        return None

    names = {int(k): v for k, v in cfg["dataset"]["names"].items()}
    n_classes = int(cfg["dataset"]["nc"])
    data_yaml = resolve(cfg["dataset"]["yaml"])
    images = read_image_list(data_yaml, args.split)
    log.info("[%s] mengevaluasi %d gambar split '%s'", run_id, len(images), args.split)

    Loader = RTDETR if str(run["model"]).lower().startswith("rtdetr") else YOLO
    model = Loader(str(weights))

    mt, infer_seconds = build_match_table(
        model, images, n_classes, args.device, args.conf, args.batch
    )
    log.info("[%s] %d deteksi terkumpul", run_id, len(mt.score))

    npz_path = resolve(f"results/runs/{run_id}/matches_{args.split}.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    mt.save(npz_path)
    log.info("[%s] tabel pencocokan disimpan: %s", run_id, npz_path)

    overall = compute_ap(mt)
    by_area = {}
    for a in ("small", "medium", "large"):
        r = compute_ap(mt, area=a)
        by_area[a] = {"map50": r["map50"], "map5095": r["map5095"]}

    latency = None if args.skip_latency else measure_latency(model, images, args.device)

    result = {
        "run_id": run_id,
        "family": run.get("family", ""),
        "scale": run.get("scale", ""),
        "split": args.split,
        "n_images": len(images),
        "n_detections": int(len(mt.score)),
        "n_gt_instances": int(mt.n_gt.sum()),
        "map50": round(overall["map50"], 5),
        "map5095": round(overall["map5095"], 5),
        "per_class": {
            names[c]: {
                "ap50": (None if np.isnan(overall["ap_per_class_50"][c])
                         else round(float(overall["ap_per_class_50"][c]), 5)),
                "ap5095": (None if np.isnan(overall["ap_per_class_5095"][c])
                           else round(float(overall["ap_per_class_5095"][c]), 5)),
                "n_gt": int(mt.n_gt[:, c].sum()),
            }
            for c in range(n_classes)
        },
        "by_area": {k: {kk: (None if np.isnan(vv) else round(vv, 5)) for kk, vv in v.items()}
                    for k, v in by_area.items()},
        "operating_point": precision_recall_f1(mt, conf=0.25),
        "latency_ms_per_image": latency,
        "fps": (round(1000 / latency, 2) if latency else None),
        "total_inference_seconds": round(infer_seconds, 2),
        "evaluated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out = resolve(f"results/analysis/eval_{run_id}.json")
    write_json(result, out)
    log.info("[%s] mAP50=%.4f  mAP50-95=%.4f  latensi=%s ms",
             run_id, result["map50"], result["map5095"], latency)
    return result


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", help="ID run tunggal")
    ap.add_argument("--all", action="store_true", help="Evaluasi semua run yang sudah dilatih")
    ap.add_argument("--optional", action="store_true", help="Sertakan run opsional")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--conf", type=float, default=0.001, help="Ambang confidence minimum")
    ap.add_argument("--batch", type=int, default=8, help="Ukuran batch inferensi")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-latency", action="store_true", help="Lewati pengukuran latensi")
    args = ap.parse_args()

    if args.all:
        targets = all_runs(cfg, include_optional=args.optional)
    elif args.run:
        targets = [find_run(cfg, args.run)]
    else:
        raise SystemExit("Tentukan --run <id> atau --all")

    done = []
    for run in targets:
        try:
            r = evaluate_run(run, cfg, args)
            if r:
                done.append(r)
        except Exception as e:
            log.error("[%s] GAGAL: %s", run["id"], e, exc_info=True)
            if not args.all:
                raise

    if done:
        log.info("")
        log.info("%-14s %8s %10s %10s %8s", "RUN", "mAP50", "mAP50-95", "F1@0.25", "ms/img")
        for r in sorted(done, key=lambda x: -x["map5095"]):
            log.info("%-14s %8.4f %10.4f %10.4f %8s",
                     r["run_id"], r["map50"], r["map5095"],
                     r["operating_point"]["f1"], r["latency_ms_per_image"])
        log.info("")
        log.info("Langkah berikutnya: python scripts/05_statistics.py")


if __name__ == "__main__":
    main()
