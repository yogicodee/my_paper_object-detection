"""
Latih satu model sesuai definisi run di configs/experiments.yaml.

Dirancang untuk Colab gratis: sesi sering terputus, jadi setiap run dapat
dilanjutkan (`--resume`) dari checkpoint `last.pt` tanpa kehilangan progres.

Pemakaian:
  python scripts/03_train.py --run yolo11n
  python scripts/03_train.py --run yolo26s --resume
  python scripts/03_train.py --all                 # semua run wajib, berurutan
  python scripts/03_train.py --list                # tampilkan daftar run
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    all_runs, find_run, load_config, read_json, resolve, setup_logging, write_json,
)

log = setup_logging("train")


def model_complexity(model) -> tuple[float, float]:
    """
    Ambil jumlah parameter (juta) dan GFLOPs untuk tabel efisiensi di naskah.

    `model.info()` mengembalikan None pada Ultralytics 8.4.x, jadi nilai diambil
    langsung dari utilitas torch_utils. Jalur `info()` dipertahankan sebagai
    cadangan untuk versi lama yang mengembalikan tuple.
    """
    params = gflops = 0.0
    try:
        from ultralytics.utils import torch_utils as tu
        params = float(tu.get_num_params(model.model)) / 1e6
        gflops = float(tu.get_flops(model.model))
    except Exception as e:  # pragma: no cover - bergantung versi ultralytics
        log.warning("torch_utils tidak terbaca (%s), mencoba model.info()", e)
        try:
            info = model.info(detailed=False, verbose=False)
            if isinstance(info, (tuple, list)) and len(info) >= 4:
                params = float(info[1]) / 1e6
                gflops = float(info[3])
        except Exception as e2:
            log.warning("Gagal membaca kompleksitas model: %s", e2)
    return round(params, 2), round(gflops, 2)


def train_one(run: dict, cfg: dict, args) -> Path:
    from ultralytics import RTDETR, YOLO

    run_id = run["id"]
    out_dir = resolve("results/runs")
    run_dir = out_dir / run_id
    meta_path = run_dir / "meta.json"

    # Lewati bila sudah selesai, kecuali dipaksa ulang.
    if meta_path.is_file() and not args.force and not args.resume:
        meta = read_json(meta_path)
        if meta.get("completed"):
            log.info("[%s] sudah selesai (val mAP50-95=%.4f) - dilewati.",
                     run_id, meta.get("val_map5095", float("nan")))
            return run_dir

    data_yaml = resolve(cfg["dataset"]["yaml"])
    if not data_yaml.is_file():
        raise SystemExit(f"{data_yaml} tidak ada. Jalankan dulu scripts/02_prepare_dataset.py")

    # RT-DETR memakai kelas loader berbeda di Ultralytics.
    weights = run["model"]
    Loader = RTDETR if str(weights).lower().startswith("rtdetr") else YOLO

    last_ckpt = run_dir / "weights" / "last.pt"
    resuming = args.resume and last_ckpt.is_file()
    if resuming:
        log.info("[%s] melanjutkan dari %s", run_id, last_ckpt)
        model = Loader(str(last_ckpt))
    else:
        log.info("[%s] memulai dari bobot pralatih %s", run_id, weights)
        model = Loader(weights)

    params_m, gflops = model_complexity(model)
    log.info("[%s] %.2f juta parameter, %.2f GFLOPs", run_id, params_m, gflops)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=run["epochs"],
        imgsz=run["imgsz"],
        batch=run["batch"],
        seed=run["seed"],
        patience=run["patience"],
        optimizer=run["optimizer"],
        deterministic=run["deterministic"],
        cos_lr=run["cos_lr"],
        close_mosaic=run["close_mosaic"],
        workers=run["workers"],
        cache=run["cache"],
        plots=run["plots"],
        project=str(out_dir),
        name=run_id,
        exist_ok=True,
        resume=resuming,
        device=args.device,
        val=True,
    )
    if args.epochs:                      # override cepat untuk uji coba pipeline
        train_kwargs["epochs"] = args.epochs

    t0 = time.time()
    model.train(**train_kwargs)
    elapsed = time.time() - t0

    # Validasi pada split val (bukan test) untuk mencatat metrik seleksi model.
    metrics = model.val(data=str(data_yaml), split="val", device=args.device, verbose=False)

    meta = {
        "run_id": run_id,
        "model": run["model"],
        "family": run.get("family", ""),
        "scale": run.get("scale", ""),
        "note": run.get("note", ""),
        "params_m": params_m,
        "gflops": gflops,
        "epochs_requested": train_kwargs["epochs"],
        "imgsz": run["imgsz"],
        "batch": run["batch"],
        "seed": run["seed"],
        "train_seconds": round(elapsed, 1),
        "train_hours": round(elapsed / 3600, 3),
        "val_map50": round(float(metrics.box.map50), 5),
        "val_map5095": round(float(metrics.box.map), 5),
        "completed": True,
        "finished_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(meta, meta_path)
    log.info("[%s] selesai dalam %.2f jam | val mAP50=%.4f mAP50-95=%.4f",
             run_id, elapsed / 3600, meta["val_map50"], meta["val_map5095"])
    return run_dir


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", help="ID run, mis. yolo11n")
    ap.add_argument("--all", action="store_true", help="Jalankan semua run wajib")
    ap.add_argument("--optional", action="store_true", help="Sertakan run opsional pada --all")
    ap.add_argument("--list", action="store_true", help="Tampilkan daftar run lalu keluar")
    ap.add_argument("--resume", action="store_true", help="Lanjutkan dari last.pt")
    ap.add_argument("--force", action="store_true", help="Latih ulang walau sudah selesai")
    ap.add_argument("--device", default=None, help="mis. 0, cpu (default: deteksi otomatis)")
    ap.add_argument("--epochs", type=int, default=None, help="Override epoch (uji pipeline)")
    args = ap.parse_args()

    if args.list:
        log.info("%-14s %-14s %-8s %-6s %s", "ID", "MODEL", "FAMILY", "BATCH", "CATATAN")
        for r in all_runs(cfg, include_optional=True):
            log.info("%-14s %-14s %-8s %-6s %s",
                     r["id"], r["model"], r.get("family", ""), r["batch"], r.get("note", ""))
        return

    if args.all:
        targets = all_runs(cfg, include_optional=args.optional)
    elif args.run:
        targets = [find_run(cfg, args.run)]
    else:
        raise SystemExit("Tentukan --run <id>, atau --all, atau --list")

    log.info("Akan melatih %d model: %s", len(targets), ", ".join(t["id"] for t in targets))
    for i, run in enumerate(targets, 1):
        log.info("")
        log.info("=" * 70)
        log.info("[%d/%d] %s", i, len(targets), run["id"])
        log.info("=" * 70)
        try:
            train_one(run, cfg, args)
        except KeyboardInterrupt:
            log.warning("Dihentikan pengguna. Lanjutkan nanti dengan --resume.")
            raise
        except Exception as e:
            # Satu model gagal tidak boleh membatalkan seluruh antrean.
            log.error("[%s] GAGAL: %s", run["id"], e, exc_info=True)
            if not args.all:
                raise

    log.info("")
    log.info("Langkah berikutnya: python scripts/04_evaluate.py --all")


if __name__ == "__main__":
    main()
