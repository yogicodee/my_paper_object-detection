"""
Uji kebenaran modul detection_eval.

Seluruh angka pada naskah bersandar pada modul ini, jadi perhitungannya
diverifikasi terhadap kasus yang jawabannya dapat dihitung tangan.

Jalankan:  python scripts/test_detection_eval.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_eval import (  # noqa: E402
    IOU_THRS, BootstrapEvaluator, MatchTable, compute_ap, iou_matrix, match_image,
)

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# --------------------------------------------------------------------------- #
def test_iou() -> None:
    print("\niou_matrix")
    a = np.array([[0.0, 0.0, 1.0, 1.0]], np.float32)
    check("kotak identik -> IoU 1", close(float(iou_matrix(a, a)[0, 0]), 1.0))

    b = np.array([[2.0, 2.0, 3.0, 3.0]], np.float32)
    check("kotak terpisah -> IoU 0", close(float(iou_matrix(a, b)[0, 0]), 0.0))

    # Tumpang tindih separuh: irisan 0.5, gabungan 1.5 -> 1/3
    c = np.array([[0.5, 0.0, 1.5, 1.0]], np.float32)
    check("tumpang tindih separuh -> IoU 1/3", close(float(iou_matrix(a, c)[0, 0]), 1 / 3, 1e-5))

    check("prediksi kosong -> matriks kosong",
          iou_matrix(np.zeros((0, 4), np.float32), a).shape == (0, 1))


def test_match() -> None:
    print("\nmatch_image")
    gt_b = np.array([[0.0, 0.0, 1.0, 1.0]], np.float32)
    gt_c = np.array([0], np.int16)

    tp = match_image(gt_b.copy(), np.array([0], np.int16), np.array([0.9], np.float32), gt_b, gt_c)
    check("deteksi sempurna -> TP di semua ambang", bool(tp.all()))

    # Kelas salah -> tidak pernah TP
    tp = match_image(gt_b.copy(), np.array([1], np.int16), np.array([0.9], np.float32), gt_b, gt_c)
    check("kelas keliru -> tak pernah TP", not tp.any())

    # Dua deteksi pada satu GT: hanya yang berskor tertinggi jadi TP
    pred = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], np.float32)
    tp = match_image(pred, np.array([0, 0], np.int16), np.array([0.9, 0.8], np.float32), gt_b, gt_c)
    check("duplikat -> hanya skor tertinggi TP", bool(tp[0].all()) and not tp[1].any())

    # IoU 0.6: TP pada ambang <=0.6, FP di atasnya.
    # [0,0,1,1] vs [0.25,0,1.25,1] -> irisan 0.75, gabungan 1.25 -> IoU 0.6
    p = np.array([[0.25, 0.0, 1.25, 1.0]], np.float32)
    tp = match_image(p, np.array([0], np.int16), np.array([0.9], np.float32), gt_b, gt_c)
    expected = IOU_THRS <= 0.6 + 1e-9
    check("ambang IoU dihormati", bool((tp[0] == expected).all()),
          f"dapat {tp[0].astype(int)}, harap {expected.astype(int)}")


def make_table(n_images: int, n_classes: int, dets, n_gt) -> MatchTable:
    """dets: list (image_idx, cls, score, tp_semua_ambang, area)."""
    if dets:
        img = np.array([d[0] for d in dets], np.int32)
        cls = np.array([d[1] for d in dets], np.int16)
        sc = np.array([d[2] for d in dets], np.float32)
        tp = np.array([[d[3]] * len(IOU_THRS) for d in dets], bool)
        area = np.array([d[4] for d in dets], np.float32)
    else:
        img = np.zeros(0, np.int32)
        cls = np.zeros(0, np.int16)
        sc = np.zeros(0, np.float32)
        tp = np.zeros((0, len(IOU_THRS)), bool)
        area = np.zeros(0, np.float32)

    n_gt_area = np.zeros((n_images, n_classes, 4), np.int32)
    n_gt_area[:, :, 0] = n_gt
    n_gt_area[:, :, 3] = n_gt          # anggap semua GT berukuran besar
    return MatchTable(img, cls, sc, tp, area, n_gt, n_gt_area,
                      np.zeros(0, np.float32), n_images, n_classes)


def test_ap() -> None:
    print("\ncompute_ap")
    n_gt = np.array([[1]], np.int32)          # 1 gambar, 1 kelas, 1 GT

    mt = make_table(1, 1, [(0, 0, 0.9, True, 0.5)], n_gt)
    check("1 GT, 1 deteksi benar -> AP 1", close(compute_ap(mt)["map5095"], 1.0))

    mt = make_table(1, 1, [(0, 0, 0.9, False, 0.5)], n_gt)
    check("1 GT, 1 deteksi salah -> AP 0", close(compute_ap(mt)["map5095"], 0.0))

    # TP berskor tinggi lalu FP: presisi tetap 1 sampai recall penuh -> AP 1
    mt = make_table(1, 1, [(0, 0, 0.9, True, 0.5), (0, 0, 0.5, False, 0.5)], n_gt)
    check("TP lalu FP -> AP tetap 1", close(compute_ap(mt)["map5095"], 1.0))

    # FP berskor tinggi lalu TP: presisi di recall=1 adalah 0.5 -> AP sekitar 0.5
    mt = make_table(1, 1, [(0, 0, 0.9, False, 0.5), (0, 0, 0.5, True, 0.5)], n_gt)
    got = compute_ap(mt)["map5095"]
    check("FP lalu TP -> AP sekitar 0.5", 0.45 < got < 0.55, f"dapat {got:.4f}")

    # Setengah GT terdeteksi -> recall 0.5, AP sekitar 0.5
    n_gt2 = np.array([[2]], np.int32)
    mt = make_table(1, 1, [(0, 0, 0.9, True, 0.5)], n_gt2)
    got = compute_ap(mt)["map5095"]
    check("recall 0.5 dengan presisi 1 -> AP sekitar 0.5", 0.45 < got < 0.55, f"dapat {got:.4f}")

    # Kelas tanpa GT tidak boleh ikut rata-rata
    n_gt3 = np.array([[1, 0]], np.int32)
    mt = make_table(1, 2, [(0, 0, 0.9, True, 0.5)], n_gt3)
    check("kelas tanpa GT diabaikan dari mAP", close(compute_ap(mt)["map5095"], 1.0))


def test_bootstrap_evaluator() -> None:
    print("\nBootstrapEvaluator")
    rng = np.random.default_rng(0)
    n_images, n_classes = 40, 3
    n_gt = rng.integers(0, 3, size=(n_images, n_classes)).astype(np.int32)

    dets = []
    for i in range(n_images):
        for c in range(n_classes):
            for _ in range(int(rng.integers(0, 4))):
                dets.append((i, c, float(rng.random()), bool(rng.random() > 0.4), 0.5))
    mt = make_table(n_images, n_classes, dets, n_gt)

    be = BootstrapEvaluator(mt)
    ref = compute_ap(mt)
    got50, got5095 = be.point_estimate()
    check("point_estimate cocok dengan compute_ap (mAP50)", close(got50, ref["map50"], 1e-9),
          f"{got50:.6f} vs {ref['map50']:.6f}")
    check("point_estimate cocok dengan compute_ap (mAP50-95)", close(got5095, ref["map5095"], 1e-9),
          f"{got5095:.6f} vs {ref['map5095']:.6f}")

    # Menggandakan seluruh dataset tidak boleh mengubah mAP.
    single = be.map_for_counts(np.ones(n_images, np.int64))
    double = be.map_for_counts(np.full(n_images, 2, np.int64))
    check("mAP invarian terhadap penggandaan seragam", close(single[1], double[1], 1e-9),
          f"{single[1]:.6f} vs {double[1]:.6f}")

    # Subset lewat counts harus sama dengan compute_ap pada subset yang sama.
    counts = np.zeros(n_images, np.int64)
    counts[:20] = 1
    sub_be = be.map_for_counts(counts)[1]
    sub_ref = compute_ap(mt, image_subset=np.arange(20))["map5095"]
    check("konsisten dengan compute_ap pada subset", close(sub_be, sub_ref, 1e-9),
          f"{sub_be:.6f} vs {sub_ref:.6f}")


def test_holm() -> None:
    print("\nkoreksi Holm")
    spec = importlib.util.spec_from_file_location(
        "stats05", Path(__file__).resolve().parent / "05_statistics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # m=3, p terkecil 0.01 -> 3*0.01 = 0.03
    adj = mod.holm_correction([0.01, 0.04, 0.03])
    check("p terkecil dikali m", close(adj[0], 0.03), f"dapat {adj[0]}")
    check("hasil tidak pernah menurun menurut peringkat", adj[0] <= adj[2] <= adj[1],
          f"dapat {adj}")
    check("dibatasi 1.0", all(v <= 1.0 for v in mod.holm_correction([0.5, 0.6, 0.9])))


if __name__ == "__main__":
    test_iou()
    test_match()
    test_ap()
    test_bootstrap_evaluator()
    test_holm()
    print(f"\n{PASS} lulus, {FAIL} gagal")
    sys.exit(1 if FAIL else 0)
