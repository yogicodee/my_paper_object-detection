"""
Evaluasi deteksi objek gaya COCO dengan hasil pencocokan yang dapat di-*resample*.

Mengapa tidak memakai pycocotools apa adanya
--------------------------------------------
Uji signifikansi kami memerlukan perhitungan ulang mAP pada ribuan *resample*
bootstrap dari himpunan gambar uji. COCOeval melakukan de-duplikasi `imgIds`,
sehingga pengambilan sampel dengan pengembalian (yang wajib pada bootstrap)
diam-diam berubah menjadi sub-sampling tanpa bobot dan menghasilkan selang
kepercayaan yang bias.

Solusinya: cocokkan deteksi ke ground-truth SATU KALI, simpan hasilnya sebagai
tabel per-deteksi (gambar, kelas, skor, TP/FP pada 10 ambang IoU), lalu hitung
ulang AP dari tabel itu untuk sembarang multiset gambar. Protokol pencocokan dan
interpolasi 101 titik mengikuti definisi COCO, sehingga angkanya tetap sebanding
dengan literatur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ambang IoU COCO: 0.50, 0.55, ..., 0.95
IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
# Titik recall untuk interpolasi AP ala COCO
RECALL_THRS = np.linspace(0.0, 1.0, 101)

# Batas area relatif (fraksi luas gambar) untuk analisis per-ukuran objek.
# COCO memakai piksel absolut; SH17 beresolusi sangat beragam (1920x1002 s.d.
# 8192x5462) sehingga area relatif lebih adil antar-gambar.
AREA_RANGES = {
    "all": (0.0, 1.0),
    "small": (0.0, 0.01),
    "medium": (0.01, 0.05),
    "large": (0.05, 1.0),
}


@dataclass
class MatchTable:
    """Hasil pencocokan deteksi-vs-GT, siap di-*resample* per gambar."""

    image_idx: np.ndarray   # int32 [N]      indeks gambar tiap deteksi
    cls: np.ndarray         # int16 [N]      kelas prediksi
    score: np.ndarray       # float32 [N]    confidence
    tp: np.ndarray          # bool [N, T]    true-positive per ambang IoU
    area: np.ndarray        # float32 [N]    area relatif kotak prediksi
    n_gt: np.ndarray        # int32 [n_img, n_cls]          jumlah GT
    n_gt_area: np.ndarray   # int32 [n_img, n_cls, n_area]  GT per kategori ukuran
    gt_area: np.ndarray     # float32 [M]    area relatif tiap GT (untuk statistik)
    n_images: int
    n_classes: int

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            image_idx=self.image_idx, cls=self.cls, score=self.score, tp=self.tp,
            area=self.area, n_gt=self.n_gt, n_gt_area=self.n_gt_area,
            gt_area=self.gt_area,
            n_images=self.n_images, n_classes=self.n_classes,
        )

    @staticmethod
    def load(path) -> "MatchTable":
        d = np.load(path)
        return MatchTable(
            image_idx=d["image_idx"], cls=d["cls"], score=d["score"], tp=d["tp"],
            area=d["area"], n_gt=d["n_gt"], n_gt_area=d["n_gt_area"],
            gt_area=d["gt_area"],
            n_images=int(d["n_images"]), n_classes=int(d["n_classes"]),
        )


# --------------------------------------------------------------------------- #
# Pencocokan
# --------------------------------------------------------------------------- #
def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU antara kotak prediksi [P,4] dan GT [G,4], format xyxy."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)

    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_p = np.clip(pred[:, 2] - pred[:, 0], 0, None) * np.clip(pred[:, 3] - pred[:, 1], 0, None)
    area_g = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)


def match_image(
    pred_boxes: np.ndarray, pred_cls: np.ndarray, pred_scores: np.ndarray,
    gt_boxes: np.ndarray, gt_cls: np.ndarray,
) -> np.ndarray:
    """
    Cocokkan deteksi satu gambar ke GT-nya, mengikuti aturan COCO.

    Dalam tiap kelas, deteksi diproses berdasarkan skor menurun; tiap deteksi
    mengambil GT tersedia dengan IoU tertinggi di atas ambang. Satu GT hanya
    boleh dipakai satu kali per ambang IoU.

    Return: bool [P, T] - status TP tiap deteksi pada tiap ambang IoU.
    """
    n_pred, n_thr = len(pred_boxes), len(IOU_THRS)
    tp = np.zeros((n_pred, n_thr), dtype=bool)
    if n_pred == 0:
        return tp

    for c in np.unique(pred_cls):
        p_idx = np.flatnonzero(pred_cls == c)
        g_idx = np.flatnonzero(gt_cls == c)
        if len(g_idx) == 0:
            continue                       # semua deteksi kelas ini adalah FP

        # Urutkan deteksi menurut skor menurun (stabil, agar deterministik).
        order = p_idx[np.argsort(-pred_scores[p_idx], kind="stable")]
        ious = iou_matrix(pred_boxes[order], gt_boxes[g_idx])

        for t, thr in enumerate(IOU_THRS):
            taken = np.zeros(len(g_idx), dtype=bool)
            for r, det in enumerate(order):
                cand = ious[r].copy()
                cand[taken] = -1.0
                best = int(np.argmax(cand))
                if cand[best] >= thr:
                    taken[best] = True
                    tp[det, t] = True
    return tp


# --------------------------------------------------------------------------- #
# Perhitungan AP
# --------------------------------------------------------------------------- #
def _nanmean_rows(ap: np.ndarray) -> np.ndarray:
    """
    Rata-rata tiap baris dengan mengabaikan NaN, tanpa memicu peringatan.

    `np.nanmean` memperingatkan pada baris yang seluruhnya NaN (kelas yang tidak
    hadir pada resample tertentu). Kasus itu normal dan jumlahnya ribuan selama
    bootstrap, sehingga peringatannya hanya membanjiri log.
    """
    valid = ~np.isnan(ap)
    counts = valid.sum(axis=1)
    sums = np.where(valid, ap, 0.0).sum(axis=1)
    out = np.full(ap.shape[0], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _ap_from_pr(tp_cum: np.ndarray, fp_cum: np.ndarray, n_pos: int) -> float:
    """AP dengan interpolasi 101 titik recall (definisi COCO)."""
    if n_pos == 0:
        return float("nan")             # kelas tak hadir -> tidak dirata-ratakan
    recall = tp_cum / n_pos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Buat precision monoton tidak naik (envelope), dari kanan ke kiri.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    idx = np.searchsorted(recall, RECALL_THRS, side="left")
    q = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(q.mean())


def compute_ap(
    mt: MatchTable,
    image_subset: np.ndarray | None = None,
    area: str = "all",
) -> dict:
    """
    Hitung AP per kelas untuk suatu multiset gambar.

    `image_subset` boleh mengandung duplikat (itulah inti bootstrap): jumlah GT
    dan deteksi dari gambar yang terpilih berkali-kali ikut dihitung berulang,
    sehingga estimatornya konsisten dengan resampling dengan pengembalian.
    """
    n_cls = mt.n_classes
    a_idx = list(AREA_RANGES).index(area)
    lo, hi = AREA_RANGES[area]

    if image_subset is None:
        # Jalur cepat: seluruh gambar, tiap gambar sekali.
        sel = np.ones(len(mt.score), dtype=bool)
        npig = mt.n_gt.sum(0) if area == "all" else mt.n_gt_area[:, :, a_idx].sum(0)
        rep_of_det = None
    else:
        image_subset = np.asarray(image_subset, dtype=np.int64)
        counts = np.bincount(image_subset, minlength=mt.n_images)
        npig = ((mt.n_gt * counts[:, None]).sum(0) if area == "all"
                else (mt.n_gt_area[:, :, a_idx] * counts[:, None]).sum(0))
        rep_of_det = counts[mt.image_idx]        # berapa kali tiap deteksi ikut
        sel = rep_of_det > 0

    if area != "all":
        # Batasi deteksi pada rentang ukuran yang ditinjau.
        sel = sel & (mt.area >= lo) & (mt.area < hi)

    cls_sel = mt.cls[sel]
    score_sel = mt.score[sel]
    tp_sel = mt.tp[sel]
    rep_sel = rep_of_det[sel] if rep_of_det is not None else None

    n_thr = len(IOU_THRS)
    ap = np.full((n_cls, n_thr), np.nan, dtype=np.float64)

    for c in range(n_cls):
        m = cls_sel == c
        pos = int(npig[c])
        if pos == 0:
            continue
        if not m.any():
            ap[c, :] = 0.0                # ada GT tapi nol deteksi -> AP 0
            continue

        order = np.argsort(-score_sel[m], kind="stable")
        tp_c = tp_sel[m][order]                     # [n_det, T]
        w = (rep_sel[m][order] if rep_sel is not None
             else np.ones(len(order), dtype=np.int64))

        for t in range(n_thr):
            hit = tp_c[:, t].astype(np.int64) * w
            miss = (~tp_c[:, t]).astype(np.int64) * w
            ap[c, t] = _ap_from_pr(np.cumsum(hit), np.cumsum(miss), pos)

    ap_per_class_5095 = _nanmean_rows(ap)
    ap_per_class_50 = ap[:, 0]
    valid50 = ~np.isnan(ap_per_class_50)
    valid5095 = ~np.isnan(ap_per_class_5095)
    return {
        "ap": ap,
        "ap_per_class_50": ap_per_class_50,
        "ap_per_class_5095": ap_per_class_5095,
        "map50": float(ap_per_class_50[valid50].mean()) if valid50.any() else float("nan"),
        "map5095": float(ap_per_class_5095[valid5095].mean()) if valid5095.any() else float("nan"),
    }


class BootstrapEvaluator:
    """
    Penghitung mAP cepat untuk ribuan *resample* bootstrap.

    Urutan skor deteksi tidak berubah antar-resample - yang berubah hanya berapa
    kali tiap gambar ikut terpilih. Karena itu pengurutan (bagian termahal)
    dilakukan sekali di konstruktor, dan tiap resample hanya perlu satu cumsum
    berbobot. Tanpa pra-pengurutan ini, satu model butuh berjam-jam.
    """

    def __init__(self, mt: MatchTable, area: str = "all", iou_only_50: bool = False):
        self.n_images = mt.n_images
        self.n_classes = mt.n_classes
        self.n_thr = 1 if iou_only_50 else len(IOU_THRS)

        a_idx = list(AREA_RANGES).index(area)
        lo, hi = AREA_RANGES[area]

        keep = np.ones(len(mt.score), dtype=bool)
        if area != "all":
            keep = (mt.area >= lo) & (mt.area < hi)

        self.n_gt_per_image = (mt.n_gt if area == "all" else mt.n_gt_area[:, :, a_idx])

        # Pra-urutkan deteksi tiap kelas menurut skor menurun.
        self._cls_img: list[np.ndarray] = []
        self._cls_tp: list[np.ndarray] = []
        cls_all = mt.cls[keep]
        score_all = mt.score[keep]
        img_all = mt.image_idx[keep]
        tp_all = mt.tp[keep][:, : self.n_thr]

        for c in range(mt.n_classes):
            m = np.flatnonzero(cls_all == c)
            if len(m) == 0:
                self._cls_img.append(np.zeros(0, np.int64))
                self._cls_tp.append(np.zeros((0, self.n_thr), bool))
                continue
            order = m[np.argsort(-score_all[m], kind="stable")]
            self._cls_img.append(img_all[order].astype(np.int64))
            self._cls_tp.append(tp_all[order])

    def map_for_counts(self, counts: np.ndarray) -> tuple[float, float]:
        """
        mAP50 dan mAP50-95 untuk multiset gambar yang dinyatakan sebagai
        `counts[i]` = berapa kali gambar ke-i terpilih.
        """
        npig = (self.n_gt_per_image * counts[:, None]).sum(0)
        ap = np.full((self.n_classes, self.n_thr), np.nan, dtype=np.float64)

        for c in range(self.n_classes):
            pos = int(npig[c])
            if pos == 0:
                continue
            img_c, tp_c = self._cls_img[c], self._cls_tp[c]
            if len(img_c) == 0:
                ap[c, :] = 0.0
                continue

            w = counts[img_c]
            nz = w > 0
            if not nz.any():
                ap[c, :] = 0.0
                continue
            w = w[nz].astype(np.float64)
            tp_w = tp_c[nz]

            # Satu cumsum untuk seluruh ambang IoU sekaligus.
            hit = np.cumsum(tp_w * w[:, None], axis=0)
            total = np.cumsum(np.broadcast_to(w[:, None], tp_w.shape), axis=0)
            miss = total - hit

            for t in range(self.n_thr):
                ap[c, t] = _ap_from_pr(hit[:, t], miss[:, t], pos)

        per_class_5095 = _nanmean_rows(ap)
        v50 = ~np.isnan(ap[:, 0])
        v95 = ~np.isnan(per_class_5095)
        map50 = float(ap[v50, 0].mean()) if v50.any() else float("nan")
        map5095 = float(per_class_5095[v95].mean()) if v95.any() else float("nan")
        return map50, map5095

    def point_estimate(self) -> tuple[float, float]:
        """mAP pada himpunan uji penuh (tiap gambar tepat sekali)."""
        return self.map_for_counts(np.ones(self.n_images, dtype=np.int64))


def precision_recall_f1(mt: MatchTable, conf: float = 0.25, iou_idx: int = 0) -> dict:
    """
    Precision / recall / F1 pada ambang confidence operasional.

    Dilaporkan di naskah karena sistem pemantauan APD di lapangan berjalan pada
    satu ambang tetap, bukan pada seluruh kurva seperti mAP.
    """
    sel = mt.score >= conf
    tp = int(mt.tp[sel, iou_idx].sum())
    fp = int((~mt.tp[sel, iou_idx]).sum())
    total_gt = int(mt.n_gt.sum())
    fn = max(total_gt - tp, 0)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(total_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "conf": conf, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 5), "recall": round(rec, 5), "f1": round(f1, 5),
    }
