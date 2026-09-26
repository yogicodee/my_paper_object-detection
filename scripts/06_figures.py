"""
Hasilkan seluruh gambar dan tabel untuk naskah.

Keputusan visual (mengikuti kaidah visualisasi data yang berlaku umum):
  - Warna kategorikal dipakai dalam urutan tetap per KELUARGA model, tidak pernah
    didaur ulang, sehingga satu keluarga selalu berwarna sama di semua gambar.
  - Palet lolos uji keterbacaan buta warna (Delta E >= 8 pada simulasi protan,
    deutan, dan tritan), dan tiap batang diberi arsiran berbeda supaya tetap
    terbaca bila jurnal mencetak dalam skala abu-abu.
  - Skala magnitudo (heatmap, distribusi kelas) memakai SATU rona terang->gelap,
    bukan pelangi, agar urutan nilainya terbaca tanpa legenda.
  - Tidak ada sumbu ganda: besaran berbeda skala selalu dipisah ke gambar sendiri.

Keluaran: results/figures/*.png dan *.pdf (300 dpi) + results/analysis/tabel_naskah.md

Pemakaian:
  python scripts/06_figures.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import all_runs, load_config, read_json, resolve, setup_logging  # noqa: E402

log = setup_logging("figures")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- Parameter visual -------------------------------------------------------
# Slot kategorikal, urutan tetap. Divalidasi: pasangan terdekat Delta E 9.1
# (protan) dan 22.9 (penglihatan normal).
FAMILY_COLOR = {
    "YOLOv8": "#2a78d6",
    "YOLO11": "#eb6834",
    "YOLO26": "#1baf7a",
    "RT-DETR": "#eda100",
}
# Pembeda kedua untuk cetak skala abu-abu.
FAMILY_HATCH = {"YOLOv8": "", "YOLO11": "///", "YOLO26": "...", "RT-DETR": "xxx"}
FAMILY_MARKER = {"YOLOv8": "o", "YOLO11": "s", "YOLO26": "^", "RT-DETR": "D"}

INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

# Rampa sekuensial satu rona (biru terang -> gelap) untuk besaran.
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "biru", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
})

FIGDIR = resolve("results/figures")


def style(ax) -> None:
    """Buat rangka dan kisi menjadi resesif agar data yang menonjol."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("  %s.png / .pdf", name)


def fam_of(run_id: str, runs: list[dict]) -> str:
    for r in runs:
        if r["id"] == run_id:
            return r.get("family", "")
    return ""


# --------------------------------------------------------------------------- #
# Gambar 1 - komposisi dataset
# --------------------------------------------------------------------------- #
def figure_dataset(stats: dict) -> None:
    per_class = stats["per_class"]
    items = sorted(per_class.items(), key=lambda kv: kv[1]["instances"], reverse=True)
    labels = [k.replace("_", " ") for k, _ in items]
    counts = [v["instances"] for _, v in items]
    if not counts or max(counts) == 0:
        log.warning("  gambar1 dilewati - statistik kelas kosong")
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [2.3, 1]}
    )

    # Magnitudo satu seri -> satu rona, gelap = besar.
    norm = np.array(counts, float) / max(counts)
    colors = [BLUE_RAMP(0.25 + 0.7 * v) for v in norm]

    y = np.arange(len(labels))
    ax1.barh(y, counts, color=colors, height=0.72, edgecolor=SURFACE, linewidth=1.2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlabel("Jumlah instance (skala logaritmik)")
    ax1.set_title("(a) Sebaran instance per kelas", loc="left", color=INK)
    for yi, c in zip(y, counts):
        ax1.text(c * 1.12, yi, f"{c:,}".replace(",", "."), va="center",
                 fontsize=7.5, color=INK_MUTED)
    ax1.set_xlim(right=max(counts) * 3)
    style(ax1)
    ax1.grid(axis="x", visible=True)
    ax1.grid(axis="y", visible=False)

    size = stats["totals"]["size_distribution"]
    order = ["small", "medium", "large"]
    names_id = {"small": "Kecil\n(<1%)", "medium": "Sedang\n(1-5%)", "large": "Besar\n(>5%)"}
    vals = [size.get(k, 0) for k in order]
    total = max(sum(vals), 1)
    ax2.bar(range(3), vals, color=[BLUE_RAMP(0.35), BLUE_RAMP(0.55), BLUE_RAMP(0.8)],
            width=0.62, edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([names_id[k] for k in order])
    ax2.set_ylabel("Jumlah instance")
    ax2.set_title("(b) Sebaran ukuran objek", loc="left", color=INK)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{100 * v / total:.1f}%", ha="center", va="bottom",
                 fontsize=8, color=INK)
    ax2.set_ylim(top=max(max(vals), 1) * 1.15)
    style(ax2)

    fig.tight_layout()
    save(fig, "gambar1_dataset")


# --------------------------------------------------------------------------- #
# Gambar 2 - kurva pelatihan
# --------------------------------------------------------------------------- #
def read_curve(run_id: str):
    path = resolve(f"results/runs/{run_id}/results.csv")
    if not path.is_file():
        return None
    epochs, vals = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clean = {k.strip(): v for k, v in row.items() if k}
            key = next((k for k in clean if "mAP50-95" in k), None)
            if key is None:
                return None
            try:
                epochs.append(float(clean["epoch"]))
                vals.append(float(clean[key]))
            except (KeyError, ValueError, TypeError):
                continue
    return (np.array(epochs), np.array(vals)) if epochs else None


def figure_curves(run_ids: list[str], runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plotted = 0
    for rid in run_ids:
        curve = read_curve(rid)
        if curve is None:
            continue
        ep, v = curve
        fam = fam_of(rid, runs)
        ax.plot(ep, v, linewidth=1.8, color=FAMILY_COLOR.get(fam, INK_MUTED),
                marker=FAMILY_MARKER.get(fam, "o"), markevery=max(len(ep) // 12, 1),
                markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=rid)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        log.warning("  gambar2 dilewati - results.csv belum tersedia")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0,5:0,95 (validasi)")
    ax.set_title("Konvergensi pelatihan pada himpunan validasi", loc="left", color=INK)
    ax.legend(ncol=2, fontsize=8)
    style(ax)
    ax.grid(axis="y", visible=True)
    fig.tight_layout()
    save(fig, "gambar2_konvergensi")


# --------------------------------------------------------------------------- #
# Gambar 3 - mAP dengan selang kepercayaan
# --------------------------------------------------------------------------- #
def figure_map_ci(stats_json: dict, runs: list[dict]) -> None:
    models = stats_json["models"]
    ids = sorted(models, key=lambda r: models[r]["map5095"], reverse=True)

    vals = [models[r]["map5095"] for r in ids]
    lo = [models[r]["map5095"] - models[r]["ci95_map5095"][0] for r in ids]
    hi = [models[r]["ci95_map5095"][1] - models[r]["map5095"] for r in ids]
    fams = [fam_of(r, runs) for r in ids]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(ids))
    for xi, v, f in zip(x, vals, fams):
        ax.bar(xi, v, width=0.62, color=FAMILY_COLOR.get(f, INK_MUTED),
               hatch=FAMILY_HATCH.get(f, ""), edgecolor=SURFACE, linewidth=1.4)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK,
                elinewidth=1.1, capsize=3.5, capthick=1.1)

    for xi, v, h in zip(x, vals, hi):
        ax.text(xi, v + h + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=15, ha="right")
    ax.set_ylabel("mAP@0,5:0,95 (uji)")
    ax.set_title("Akurasi pada himpunan uji dengan selang kepercayaan 95%",
                 loc="left", color=INK)
    ax.set_ylim(0, max(v + h for v, h in zip(vals, hi)) * 1.18)

    names = [f for f in dict.fromkeys(fams) if f in FAMILY_COLOR]
    if len(names) >= 2:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLOR[f],
                                 hatch=FAMILY_HATCH[f], edgecolor=SURFACE)
                   for f in names]
        ax.legend(handles, names, ncol=len(names), fontsize=8, loc="upper right")
    style(ax)
    ax.grid(axis="y", visible=True)
    fig.tight_layout()
    save(fig, "gambar3_map_ci")


# --------------------------------------------------------------------------- #
# Gambar 4 - trade-off akurasi vs kecepatan
# --------------------------------------------------------------------------- #
def figure_tradeoff(rows: list[dict], runs: list[dict]) -> None:
    pts = [r for r in rows if r.get("latency_ms") not in ("", None)]
    if not pts:
        log.warning("  gambar4 dilewati - latensi belum terukur")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for r in pts:
        fam = r.get("family", "")
        try:
            params = float(r.get("params_m") or 1)
        except ValueError:
            params = 1.0
        ax.scatter(float(r["latency_ms"]), float(r["map5095"]),
                   s=40 + 14 * params, color=FAMILY_COLOR.get(fam, INK_MUTED),
                   marker=FAMILY_MARKER.get(fam, "o"),
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.annotate(r["run_id"], (float(r["latency_ms"]), float(r["map5095"])),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color=INK_MUTED)

    ax.set_xlabel("Latensi inferensi (ms/gambar, batch = 1)")
    ax.set_ylabel("mAP@0,5:0,95 (uji)")
    ax.set_title("Imbangan akurasi terhadap kecepatan\n"
                 "(luas penanda sebanding dengan jumlah parameter)",
                 loc="left", color=INK)
    fams = [f for f in dict.fromkeys(r.get("family", "") for r in pts) if f in FAMILY_COLOR]
    if len(fams) >= 2:
        handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], marker=FAMILY_MARKER[f],
                              linestyle="", markersize=7, markeredgecolor=SURFACE)
                   for f in fams]
        ax.legend(handles, fams, fontsize=8, loc="lower right")
    style(ax)
    ax.grid(visible=True)
    fig.tight_layout()
    save(fig, "gambar4_tradeoff")


# --------------------------------------------------------------------------- #
# Gambar 5 - peta panas AP per kelas
# --------------------------------------------------------------------------- #
def figure_heatmap(run_ids: list[str]) -> None:
    evals = {}
    for rid in run_ids:
        p = resolve(f"results/analysis/eval_{rid}.json")
        if p.is_file():
            evals[rid] = read_json(p)
    if len(evals) < 2:
        log.warning("  gambar5 dilewati - perlu minimal 2 hasil evaluasi")
        return

    ids = list(evals)
    class_names = list(next(iter(evals.values()))["per_class"])

    def mean_ap(c: str) -> float:
        vals = [evals[r]["per_class"][c]["ap5095"] for r in ids
                if evals[r]["per_class"][c]["ap5095"] is not None]
        return float(np.mean(vals)) if vals else -1.0

    # Urutkan kelas menurut kesulitan rata-rata agar polanya terbaca.
    class_names.sort(key=mean_ap, reverse=True)
    mat = np.array([[(evals[r]["per_class"][c]["ap5095"] or 0.0) for r in ids]
                    for c in class_names])

    fig, ax = plt.subplots(figsize=(1.35 * len(ids) + 2.6, 5.4))
    im = ax.imshow(mat, cmap=BLUE_RAMP, aspect="auto", vmin=0, vmax=max(mat.max(), 0.01))

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([c.replace("_", " ") for c in class_names])

    # Nilai ditulis di tiap sel: identitas tidak pernah bergantung warna saja.
    thresh = mat.max() * 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=("#ffffff" if mat[i, j] > thresh else INK))

    ax.set_title("AP@0,5:0,95 per kelas", loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("AP@0,5:0,95")
    cb.outline.set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, "gambar5_ap_per_kelas")


# --------------------------------------------------------------------------- #
# Gambar 6 - kinerja menurut ukuran objek
# --------------------------------------------------------------------------- #
def figure_by_size(run_ids: list[str], runs: list[dict]) -> None:
    evals = {}
    for rid in run_ids:
        p = resolve(f"results/analysis/eval_{rid}.json")
        if p.is_file():
            evals[rid] = read_json(p)
    if not evals:
        log.warning("  gambar6 dilewati - hasil evaluasi belum ada")
        return

    buckets = ["small", "medium", "large"]
    label_id = {"small": "Kecil (<1%)", "medium": "Sedang (1-5%)", "large": "Besar (>5%)"}
    ids = list(evals)

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    width = 0.8 / max(len(ids), 1)
    xbase = np.arange(len(buckets))

    for k, rid in enumerate(ids):
        fam = fam_of(rid, runs)
        vals = [(evals[rid]["by_area"].get(b, {}).get("map5095") or 0.0) for b in buckets]
        ax.bar(xbase + k * width - 0.4 + width / 2, vals, width=width * 0.88,
               color=FAMILY_COLOR.get(fam, INK_MUTED), hatch=FAMILY_HATCH.get(fam, ""),
               edgecolor=SURFACE, linewidth=1.1, label=rid)

    ax.set_xticks(xbase)
    ax.set_xticklabels([label_id[b] for b in buckets])
    ax.set_xlabel("Kategori ukuran objek (fraksi luas gambar)")
    ax.set_ylabel("mAP@0,5:0,95 (uji)")
    ax.set_title("Kinerja terhadap ukuran objek", loc="left", color=INK)
    ax.legend(ncol=min(len(ids), 3), fontsize=8)
    style(ax)
    ax.grid(axis="y", visible=True)
    fig.tight_layout()
    save(fig, "gambar6_ukuran_objek")


# --------------------------------------------------------------------------- #
# Tabel Markdown untuk disalin ke naskah
# --------------------------------------------------------------------------- #
def write_tables(rows: list[dict], stats_json: dict, stats_ds: dict) -> None:
    def id_num(n) -> str:
        return f"{n:,}".replace(",", ".")

    out = ["# Tabel untuk naskah", "",
           "Dihasilkan otomatis oleh scripts/06_figures.py.", ""]

    out += ["## Tabel 1. Komposisi dataset SH17", "",
            "| Bagian | Gambar | Instance | Instance/gambar |",
            "|---|---:|---:|---:|"]
    for split, v in stats_ds["splits"].items():
        out.append(f"| {split} | {id_num(v['images'])} | {id_num(v['instances'])} | "
                   f"{v['instances_per_image']:.2f} |")
    t = stats_ds["totals"]
    out += [f"| **Total** | **{id_num(t['images'])}** | **{id_num(t['instances'])}** | - |", ""]

    out += ["## Tabel 2. Hasil utama pada himpunan uji", "",
            "| Model | Params (jt) | GFLOPs | mAP@0,5 | mAP@0,5:0,95 | IK 95% | "
            "F1@0,25 | ms/gambar | FPS |",
            "|---|---:|---:|---:|---:|:---:|---:|---:|---:|"]
    for r in rows:
        out.append(
            f"| {r['run_id']} | {r['params_m']} | {r['gflops']} | "
            f"{float(r['map50']):.4f} | {float(r['map5095']):.4f} | "
            f"[{float(r['ci_low']):.4f}; {float(r['ci_high']):.4f}] | "
            f"{r['f1']} | {r['latency_ms']} | {r['fps']} |"
        )
    out.append("")

    out += ["## Tabel 3. Uji signifikansi berpasangan (bootstrap, koreksi Holm)", "",
            "| Model A | Model B | Selisih mAP | IK 95% selisih | p (Holm) | Signifikan |",
            "|---|---|---:|:---:|---:|:---:|"]
    for p in sorted(stats_json["pairwise"], key=lambda x: -abs(x["delta_map5095"])):
        out.append(
            f"| {p['a']} | {p['b']} | {p['delta_map5095']:+.4f} | "
            f"[{p['ci95'][0]:+.4f}; {p['ci95'][1]:+.4f}] | {p['p_holm']:.4f} | "
            f"{'ya' if p['significant'] else 'tidak'} |"
        )
    out.append("")

    path = resolve("results/analysis/tabel_naskah.md")
    path.write_text("\n".join(out), encoding="utf-8")
    log.info("Ditulis %s", path)


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optional", action="store_true")
    args = ap.parse_args()

    runs = all_runs(cfg, include_optional=args.optional)
    run_ids = [r["id"] for r in runs]

    log.info("Membuat gambar ...")

    ds_path = resolve("results/analysis/dataset_stats.json")
    stats_ds = read_json(ds_path) if ds_path.is_file() else None
    if stats_ds:
        figure_dataset(stats_ds)
    else:
        log.warning("  gambar1 dilewati - jalankan 02_prepare_dataset.py dulu")

    figure_curves(run_ids, runs)

    st_path = resolve("results/analysis/statistics.json")
    stats_json = read_json(st_path) if st_path.is_file() else None
    if stats_json:
        figure_map_ci(stats_json, runs)
    else:
        log.warning("  gambar3 dilewati - jalankan 05_statistics.py dulu")

    csv_path = resolve("results/analysis/tabel_hasil.csv")
    rows = []
    if csv_path.is_file():
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if rows:
        figure_tradeoff(rows, runs)

    figure_heatmap(run_ids)
    figure_by_size(run_ids, runs)

    if rows and stats_json and stats_ds:
        write_tables(rows, stats_json, stats_ds)

    log.info("")
    log.info("Gambar tersimpan di %s", FIGDIR)
    log.info("Tabel siap salin: results/analysis/tabel_naskah.md")


if __name__ == "__main__":
    main()
