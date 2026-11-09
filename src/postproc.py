"""
SolarFilNet-V4: Physics-Informed Solar Image Preprocessing & Adaptive Post-Processing
====================================================================================
Provides:
  1. Contrast-adaptive chromospheric enhancement (adaptive_solar_preprocess & compute_adaptive_channels)
  2. Multi-threshold instance decoding & hole filling (probs_to_instances)
  3. Official Panoptic Quality (PQ = SQ x RQ) evaluator matching host specifications.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils


def disk_mask(img_u8: np.ndarray, r_solar: float = 903.0) -> np.ndarray:
    """Masker disk matahari sirkular untuk memotong artefak di luar piringan matahari."""
    h, w = img_u8.shape
    cx, cy = w // 2, h // 2
    r_px = int(min(h, w) * (r_solar / 2048.0))
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r_px ** 2)


def adaptive_solar_preprocess(img_u8: np.ndarray, r_solar: float = 903.0) -> Tuple[np.ndarray, bool, float]:
    """
    Menganalisis aktivitas matahari (Quiet vs Active Sun) berbasis proporsi plage/flare
    di dalam piringan matahari.
    """
    h, w = img_u8.shape
    cx, cy = w // 2, h // 2
    r_px = int(min(h, w) * (r_solar / 2048.0))

    y, x = np.ogrid[:h, :w]
    dmask = (x - cx) ** 2 + (y - cy) ** 2 <= r_px ** 2
    disc_pixels = img_u8[dmask]

    if len(disc_pixels) == 0:
        return img_u8, False, 0.0

    mean_val = float(np.mean(disc_pixels))
    std_val = float(np.std(disc_pixels))
    plage_thr = mean_val + 1.8 * std_val
    plage_pct = float(np.sum(disc_pixels > plage_thr) / len(disc_pixels) * 100.0)

    is_active = (plage_pct >= 0.65)
    if not is_active:
        return img_u8, False, plage_pct

    # Pada kondisi matahari aktif (plage tinggi), kompresi highlight ekstrem
    p98 = float(np.percentile(disc_pixels, 98.5))
    processed = img_u8.copy()
    disc_active = processed[dmask]
    over = disc_active > p98
    disc_active[over] = np.clip(p98 + (disc_active[over] - p98) * 0.25, 0, 255).astype(np.uint8)
    processed[dmask] = disc_active

    smoothed = cv2.bilateralFilter(processed, d=5, sigmaColor=25, sigmaSpace=5)
    return smoothed, True, plage_pct


def compute_adaptive_channels(img_u8: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra monokrom H-Alpha (1 channel) menjadi tensor fitur 3 channel:
      - Channel 0: Kontras disk ternormalisasi
      - Channel 1: Residu absorpsi kromosferik lokal (background estimation - raw)
      - Channel 2: CLAHE absorpsi untuk mempertajam filamen tipis dan barbs
    """
    pre_img, _, _ = adaptive_solar_preprocess(img_u8)
    h, w = pre_img.shape
    x = pre_img.astype(np.float32) / 255.0

    s = cv2.resize(pre_img, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    d = (s > 0.04).astype(np.float32)
    num = cv2.GaussianBlur(s * d, (0, 0), 5)
    den = cv2.GaussianBlur(d, (0, 0), 5)
    bg = cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)

    c0 = np.clip((x - 0.33) / 0.23, -3.0, 3.0)
    c1 = np.clip((bg - x) * 10.0, -3.0, 3.0)
    absorption_u8 = np.clip((bg - x) * 255.0 * 2.5, 0, 255).astype(np.uint8)
    clahe_u8 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(absorption_u8)

    ch0_u8 = np.clip((c0 + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    ch1_u8 = np.clip((c1 + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    return np.stack([ch0_u8, ch1_u8, clahe_u8], axis=-1)


def probs_to_instance_dicts(prob_u8: np.ndarray, pp: dict) -> List[Dict]:
    """
    Mengekstrak instansi filamen sebagai list dict [{'mask': uint8(H,W), 'score': float}]
    dari peta probabilitas (0-255).
    """
    h, w = prob_u8.shape
    thr_val = int(pp["thr"] * 255)
    m = (prob_u8 > thr_val).astype(np.uint8)
    if not m.any():
        return []

    j = int(pp["join_px"])
    groups = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (j, j))) if j > 1 else m
    n, lab, st, _ = cv2.connectedComponentsWithStats(groups, connectivity=8)

    taken = np.zeros((h, w), dtype=bool)
    out = []
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        sub = (lab[y:y + bh, x:x + bw] == i) & (m[y:y + bh, x:x + bw] > 0)
        if sub.sum() < pp["min_area"]:
            continue
        if int(pp.get("fill_holes", 1)):
            pad = np.pad(sub.astype(np.uint8), 1)
            cnts, _ = cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fl = np.zeros_like(pad)
            cv2.drawContours(fl, cnts, -1, 1, thickness=-1)
            sub = fl[1:-1, 1:-1].astype(bool)

        sub = sub & ~taken[y:y + bh, x:x + bw]
        if sub.sum() < pp["min_area"]:
            continue
        score = float(prob_u8[y:y + bh, x:x + bw][sub].mean()) / 255.0
        if score < pp["min_score"]:
            continue

        taken[y:y + bh, x:x + bw] |= sub
        full = np.zeros((h, w), dtype=np.uint8)
        full[y:y + bh, x:x + bw] = sub
        out.append({"mask": full, "score": score})

    return out


def encode_instance_dicts(inst_dicts: List[Dict]) -> List[Dict]:
    """Mengonversi mask biner ke format COCO RLE string menggunakan pycocotools."""
    return [mask_utils.encode(np.asfortranarray(d["mask"].astype(np.uint8))) for d in inst_dicts]


def probs_to_instances(prob_u8: np.ndarray, pp: dict) -> List[Dict]:
    """Pipeline terintegrasi dari probabilitas uint8 ke COCO RLE instances."""
    return encode_instance_dicts(probs_to_instance_dicts(prob_u8, pp))


def pq_eval_official(pred_by_obs: dict, gt_by_obs: dict, obs_list: list) -> Tuple[float, float, float, int, int, int]:
    """
    Kalkulasi Panoptic Quality (PQ = SQ x RQ) persis sesuai formulasi resmi host
    (Self_Evaluation_Notebook.pdf).
    """
    tp_ious = []
    fp_total = 0
    fn_total = 0

    for obs in obs_list:
        preds = pred_by_obs.get(obs, [])
        gts = gt_by_obs.get(obs, [])

        if not gts and not preds:
            continue
        if not gts:
            fp_total += len(preds)
            continue
        if not preds:
            fn_total += len(gts)
            continue

        # Hitung pairwise IoU antar lapisan
        ious = mask_utils.iou(preds, gts, [False] * len(gts))
        n_p, n_g = ious.shape

        matched_p = set()
        matched_g = set()

        for p_idx in range(n_p):
            for g_idx in range(n_g):
                iou_val = float(ious[p_idx, g_idx])
                if iou_val > 0.50:
                    tp_ious.append(iou_val)
                    matched_p.add(p_idx)
                    matched_g.add(g_idx)

        fp_total += (n_p - len(matched_p))
        fn_total += (n_g - len(matched_g))

    tp_count = len(tp_ious)
    denom = tp_count + 0.5 * fp_total + 0.5 * fn_total
    sq = (sum(tp_ious) / tp_count) if tp_count > 0 else 0.0
    rq = (tp_count / denom) if denom > 0 else 0.0
    pq = (sum(tp_ious) / denom) if denom > 0 else 0.0

    return pq, sq, rq, tp_count, fp_total, fn_total
