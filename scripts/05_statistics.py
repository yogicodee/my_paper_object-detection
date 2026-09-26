"""
Uji signifikansi antar-model dengan bootstrap berpasangan pada himpunan uji.

Mengapa bootstrap berpasangan
-----------------------------
Membandingkan dua model hanya dari selisih mAP tunggal tidak cukup: selisih 0,5
poin bisa saja sekadar derau pengambilan sampel gambar uji. Reviewer jurnal
terakreditasi umumnya menuntut bukti bahwa selisih itu tidak terjadi kebetulan.

Prosedur:
  1. Ambil ulang 1.620 gambar uji dengan pengembalian sebanyak B kali.
  2. Untuk tiap resample, hitung mAP SEMUA model pada gambar yang sama persis
     (berpasangan). Pemasangan ini membuang variasi akibat "gambar mana yang
     kebetulan terambil" dan hanya menyisakan perbedaan antar-model.
  3. Selang kepercayaan 95% = persentil 2,5 dan 97,5 dari distribusi selisih.
  4. Nilai p dua sisi dari proporsi resample yang berlawanan tanda, lalu
     dikoreksi Holm-Bonferroni karena ada banyak pasangan yang diuji.

Catatan kejujuran metodologis: bootstrap ini mengukur ketidakpastian akibat
sampel gambar uji, BUKAN akibat inisialisasi bobot. Untuk yang kedua diperlukan
pelatihan ulang multi-seed (tersedia sebagai run opsional di experiments.yaml).

Pemakaian:
  python scripts/05_statistics.py
  python scripts/05_statistics.py --n-bootstrap 2000 --optional
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    all_runs, load_config, read_json, resolve, setup_logging, write_json,
)
from detection_eval import BootstrapEvaluator, MatchTable  # noqa: E402

log = setup_logging("stats")


def holm_correction(pvals: list[float]) -> list[float]:
    """
    Koreksi Holm-Bonferroni (step-down), lebih berdaya daripada Bonferroni polos
    namun tetap mengendalikan family-wise error rate.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, min(val, 1.0))   # jaga sifat monoton
        adjusted[idx] = running
    return adjusted


def load_evaluators(run_ids: list[str], iou_only_50: bool) -> dict[str, BootstrapEvaluator]:
    evals: dict[str, BootstrapEvaluator] = {}
    n_images = None
    for rid in run_ids:
        npz = resolve(f"results/runs/{rid}/matches_test.npz")
        if not npz.is_file():
            log.warning("[%s] matches_test.npz tidak ada - lewati (jalankan 04_evaluate).", rid)
            continue
        mt = MatchTable.load(npz)
        if n_images is None:
            n_images = mt.n_images
        elif mt.n_images != n_images:
            raise SystemExit(
                f"[{rid}] jumlah gambar uji ({mt.n_images}) berbeda dari model lain "
                f"({n_images}). Pemasangan bootstrap mensyaratkan himpunan uji identik."
            )
        evals[rid] = BootstrapEvaluator(mt, area="all", iou_only_50=iou_only_50)
        log.info("[%s] dimuat (%d deteksi)", rid, len(mt.score))
    return evals


def run_bootstrap(evals: dict[str, BootstrapEvaluator], n_boot: int, seed: int) -> np.ndarray:
    """Return array [n_model, n_boot] berisi mAP50-95 tiap model per resample."""
    ids = list(evals)
    n_images = evals[ids[0]].n_images
    rng = np.random.default_rng(seed)
    draws = np.zeros((len(ids), n_boot), dtype=np.float64)

    log.info("Menjalankan %d resample bootstrap atas %d gambar uji ...", n_boot, n_images)
    for b in range(n_boot):
        # Satu resample dipakai bersama SEMUA model -> perbandingan berpasangan.
        idx = rng.integers(0, n_images, size=n_images)
        counts = np.bincount(idx, minlength=n_images)
        for k, rid in enumerate(ids):
            draws[k, b] = evals[rid].map_for_counts(counts)[1]
        if (b + 1) % max(n_boot // 20, 1) == 0:
            log.info("  %d/%d resample", b + 1, n_boot)
    return draws


def main() -> None:
    cfg = load_config()
    scfg = cfg.get("statistics", {})
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-bootstrap", type=int, default=int(scfg.get("n_bootstrap", 1000)))
    ap.add_argument("--alpha", type=float, default=float(scfg.get("alpha", 0.05)))
    ap.add_argument("--seed", type=int, default=int(scfg.get("seed", 42)))
    ap.add_argument("--optional", action="store_true", help="Sertakan run opsional")
    ap.add_argument("--fast", action="store_true",
                    help="Hanya IoU 0.50 (10x lebih cepat, untuk uji coba pipeline)")
    args = ap.parse_args()

    runs = all_runs(cfg, include_optional=args.optional)
    evals = load_evaluators([r["id"] for r in runs], iou_only_50=args.fast)
    if len(evals) < 2:
        raise SystemExit("Perlu minimal 2 model terevaluasi untuk perbandingan.")

    ids = list(evals)

    # --- Estimasi titik pada himpunan uji penuh ------------------------------
    point = {rid: evals[rid].point_estimate() for rid in ids}
    for rid in ids:
        log.info("[%s] mAP50=%.4f  mAP50-95=%.4f", rid, *point[rid])

    # --- Distribusi bootstrap ------------------------------------------------
    draws = run_bootstrap(evals, args.n_bootstrap, args.seed)

    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    models_out = {}
    for k, rid in enumerate(ids):
        d = draws[k]
        models_out[rid] = {
            "map50": round(point[rid][0], 5),
            "map5095": round(point[rid][1], 5),
            "ci95_map5095": [round(float(np.percentile(d, lo_q)), 5),
                             round(float(np.percentile(d, hi_q)), 5)],
            "se_bootstrap": round(float(d.std(ddof=1)), 5),
        }

    # --- Perbandingan berpasangan -------------------------------------------
    pairs = list(itertools.combinations(range(len(ids)), 2))
    raw_p, records = [], []
    for i, j in pairs:
        diff = draws[i] - draws[j]
        observed = point[ids[i]][1] - point[ids[j]][1]
        # Nilai p dua sisi: seberapa sering selisih menyeberangi nol.
        prop_le = float(np.mean(diff <= 0))
        prop_ge = float(np.mean(diff >= 0))
        p = min(1.0, 2 * min(prop_le, prop_ge))
        raw_p.append(p)
        records.append({
            "a": ids[i],
            "b": ids[j],
            "delta_map5095": round(observed, 5),
            "ci95": [round(float(np.percentile(diff, lo_q)), 5),
                     round(float(np.percentile(diff, hi_q)), 5)],
            "p_raw": round(p, 6),
        })

    for rec, p_adj in zip(records, holm_correction(raw_p)):
        rec["p_holm"] = round(p_adj, 6)
        rec["significant"] = bool(p_adj < args.alpha)

    out = {
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "correction": "holm",
            "seed": args.seed,
            "iou_range": "0.50" if args.fast else "0.50:0.05:0.95",
            "note": ("Bootstrap berpasangan atas gambar uji; mengukur ketidakpastian "
                     "sampel uji, bukan variasi inisialisasi bobot."),
        },
        "models": models_out,
        "pairwise": records,
    }
    write_json(out, resolve("results/analysis/statistics.json"))

    # --- Tabel gabungan untuk naskah ----------------------------------------
    rows = []
    for r in runs:
        rid = r["id"]
        if rid not in models_out:
            continue
        meta_p = resolve(f"results/runs/{rid}/meta.json")
        ev_p = resolve(f"results/analysis/eval_{rid}.json")
        meta = read_json(meta_p) if meta_p.is_file() else {}
        ev = read_json(ev_p) if ev_p.is_file() else {}
        m = models_out[rid]
        rows.append({
            "run_id": rid,
            "family": r.get("family", ""),
            "scale": r.get("scale", ""),
            "params_m": meta.get("params_m", ""),
            "gflops": meta.get("gflops", ""),
            "map50": m["map50"],
            "map5095": m["map5095"],
            "ci_low": m["ci95_map5095"][0],
            "ci_high": m["ci95_map5095"][1],
            "f1": ev.get("operating_point", {}).get("f1", ""),
            "latency_ms": ev.get("latency_ms_per_image", ""),
            "fps": ev.get("fps", ""),
            "train_hours": meta.get("train_hours", ""),
        })
    rows.sort(key=lambda r: -float(r["map5095"]))

    csv_path = resolve("results/analysis/tabel_hasil.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("Ditulis %s", csv_path)

    # --- Ringkasan ke layar --------------------------------------------------
    log.info("")
    log.info("%-14s %8s %10s %-22s", "RUN", "mAP50", "mAP50-95", "IK 95% (mAP50-95)")
    for r in rows:
        log.info("%-14s %8.4f %10.4f  [%.4f, %.4f]",
                 r["run_id"], r["map50"], r["map5095"], r["ci_low"], r["ci_high"])

    log.info("")
    log.info("Perbandingan berpasangan (signifikan setelah koreksi Holm):")
    sig = [r for r in records if r["significant"]]
    if not sig:
        log.info("  Tidak ada pasangan yang berbeda signifikan pada alpha=%.2f.", args.alpha)
        log.info("  Ini temuan yang sah dan wajib dilaporkan apa adanya di naskah.")
    for r in sorted(sig, key=lambda x: -abs(x["delta_map5095"])):
        log.info("  %-12s vs %-12s  delta=%+.4f  IK[%+.4f, %+.4f]  p_holm=%.4g",
                 r["a"], r["b"], r["delta_map5095"], r["ci95"][0], r["ci95"][1], r["p_holm"])

    log.info("")
    log.info("Langkah berikutnya: python scripts/06_figures.py")


if __name__ == "__main__":
    main()
