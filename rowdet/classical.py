"""Classical vegetation segmentation (excess-green index + Otsu), no learning involved."""
import cv2
import numpy as np


def excess_green(bgr):
    """Chromatic excess-green index 2g - r - b, robust to overall brightness."""
    f = bgr.astype(np.float32)
    s = f.sum(2)
    exg = (3.0 * f[..., 1] - s) / np.maximum(s, 1.0)
    exg[s < 60] = 0.0  # chromaticity is noise in near-black pixels
    return exg


def vegetation_map(bgr, width=320):
    """Returns (smoothed vegetation likelihood in [0, 1], binary vegetation mask)."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (width, max(8, round(width * h / w))), interpolation=cv2.INTER_AREA)
    exg = excess_green(small)
    e8 = np.clip((exg + 0.2) / 0.8 * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(e8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t / 255.0 * 0.8 - 0.2, 0.04)  # floor: soil-only frames must not become "vegetation"
    mask = (exg > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    prob = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    return prob, mask
