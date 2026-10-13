"""Row-pattern geometry: the 4-parameter vanishing-point model used by the HORSCH labels.

Coordinates are normalised: x by image width, y by image height (y = 1 is the bottom edge).
A pattern p = (Vx, Vy, c, d) describes straight rows through the vanishing point V = (Vx, Vy)
that cross the bottom edge at x = Vx + c + k*d for every integer k.
"""
import math

import cv2
import numpy as np

EVAL_YS = np.linspace(0.1, 1.0, 10)  # image heights at which row positions are compared
SIGMA_ROW = 0.08  # soft row half-width, in units of the local row spacing
CRDA_SIGMA = 0.1  # tolerance of the crop-row detection accuracy score (CRBD convention)


def vy_to_r(vy):
    """r = row spacing at the top edge / row spacing at the bottom edge."""
    return -vy / (1.0 - vy)


def r_to_vy(r):
    return -r / (1.0 - r)


def scale(p, y):
    """Row spacing at height y relative to the spacing at the bottom edge."""
    return (y - p[1]) / (1.0 - p[1])


def canon(p):
    vx, vy, c, d = (float(v) for v in p)
    d = max(abs(d), 1e-4)
    return np.array([vx, vy, c % d, d])


def pattern_from_points(pts):
    """Pattern from four normalised points: two on one row, two on the neighbouring row
    (the base_points convention of the HORSCH labels)."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = (tuple(map(float, p)) for p in pts)
    a, b = (x2 - x1) / (y2 - y1), (x4 - x3) / (y4 - y3)  # dx/dy of each row
    if abs(a - b) < 1e-6:
        b = a + 1e-6  # parallel rows: vanishing point far above the image
    vy = (x3 - x1 + a * y1 - b * y3) / (a - b)
    vx = x1 + a * (vy - y1)
    xa, xb = x1 + a * (1.0 - y1), x3 + b * (1.0 - y3)  # where the rows cross the bottom edge
    return canon([vx, vy, min(xa, xb) - vx, abs(xb - xa)])


def rows_at(p, y, lo=0.0, hi=1.0):
    """x positions of all rows crossing height y inside [lo, hi]."""
    step = p[3] * scale(p, y)
    if step <= 1e-6:
        return np.empty(0)
    x0 = p[0] + p[2] * scale(p, y)
    k = np.arange(math.ceil((lo - x0) / step), math.floor((hi - x0) / step) + 1)
    return x0 + k * step


def lattice_dist(x, x0, step):
    """Distance from x to the nearest point of the lattice x0 + k*step."""
    return np.abs((x - x0 + 0.5 * step) % step - 0.5 * step)


def row_error(pred, gt, ys=EVAL_YS):
    """Symmetric row-position error between two patterns.

    Returns (mean error in units of the ground-truth row spacing, CRDA score in [0, 1]).
    Every visible ground-truth row is matched to the nearest predicted row and vice versa,
    so both shifted rows and a wrong row spacing (missing / extra rows) are penalised.
    """
    pred, gt = canon(pred), canon(gt)
    errs = []
    for y in ys:
        sg, sp = gt[3] * scale(gt, y), pred[3] * scale(pred, y)
        if sg <= 1e-6:
            continue
        xg = rows_at(gt, y)
        if sp <= 1e-6:
            errs.append(np.full(max(len(xg), 1), 0.5))
            continue
        e1 = lattice_dist(xg, pred[0] + pred[2] * scale(pred, y), sp) / sg
        e2 = lattice_dist(rows_at(pred, y), gt[0] + gt[2] * scale(gt, y), sg) / sg
        errs.append(np.minimum(np.concatenate([e1, e2]), 0.5))
    if not errs:
        return 0.5, 0.0
    e = np.concatenate(errs)
    return float(e.mean()), float(np.maximum(0.0, 1.0 - (e / CRDA_SIGMA) ** 2).mean())


def to_reg(p):
    """Bounded regression target [Vx, r, d, sin(phase), cos(phase)] with phase = 2*pi*c/d."""
    vx, vy, c, d = canon(p)
    ph = 2 * math.pi * c / d
    return np.array([vx, vy_to_r(vy), d, math.sin(ph), math.cos(ph)], dtype=np.float32)


def from_reg(t):
    vx, r, d, s, co = (float(v) for v in t)
    r = min(max(r, -0.5), 0.95)
    d = max(d, 0.02)
    ph = math.atan2(s, co) % (2 * math.pi)
    return np.array([vx, r_to_vy(r), ph / (2 * math.pi) * d, d])


def to_fit(p):
    """(Vx, Vy, c, d) -> (Vx, r, c, d), the well-conditioned form used by the fitter."""
    return np.array([p[0], vy_to_r(p[1]), p[2], p[3]], dtype=np.float64)


def from_fit(t):
    return canon([t[0], r_to_vy(min(max(t[1], -0.5), 0.95)), t[2], t[3]])


def flip(p):
    vx, vy, c, d = canon(p)
    return canon([1.0 - vx, vy, -c, d])


def crop(p, x0, y0, sx, sy):
    """Pattern seen through the crop window [x0, x0+sx] x [y0, y0+sy] (normalised)."""
    vx, vy, c, d = canon(p)
    sb = scale(p, y0 + sy)  # spacing at the new bottom edge
    return canon([(vx - x0) / sx, (vy - y0) / sy, c * sb / sx, d * sb / sx])


def phase_dist(p, h, w):
    """Per-pixel distance to the nearest row, in units of the local row spacing (0..0.5)."""
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    s = np.maximum((ys - p[1]) / (1.0 - p[1]), 1e-4)[:, None]
    u = (xs[None, :] - p[0]) / s  # position on the bottom edge, relative to Vx
    return np.abs(((u - p[2]) / p[3] + 0.5) % 1.0 - 0.5)


def soft_mask(p, h, w, sigma=SIGMA_ROW):
    """Row-likelihood target: 1 on a row, falling off as a Gaussian of the phase distance."""
    return np.exp(-0.5 * (phase_dist(p, h, w) / sigma) ** 2).astype(np.float32)


def row_mask(p, h, w, half_width=0.1):
    """Binary row segmentation: pixels within `half_width` row spacings of a row."""
    return phase_dist(p, h, w) <= half_width


def lateral_offset(p, track="row"):
    """Offset of the image centre from the nearest target line at the bottom edge.

    In units of row spacing, range [-0.5, 0.5); positive = target is left of the centre.
    track="row" follows a crop row, track="gap" follows the middle between two rows.
    """
    e = (0.5 - (p[0] + p[2])) / p[3] + (0.5 if track == "gap" else 0.0)
    return (e + 0.5) % 1.0 - 0.5


def wrap(e):
    return (e + 0.5) % 1.0 - 0.5


def draw_pattern(img, p, color=(0, 255, 255), thickness=3):
    """Draw every row of pattern p on a BGR image (in place)."""
    h, w = img.shape[:2]
    y_top = max(0.0, p[1] + 0.02)
    s_top, s_bot = scale(p, y_top), 1.0
    for k in range(-60, 61):
        o = p[2] + k * p[3]
        p1 = (int(round((p[0] + o * s_top) * w)), int(round(y_top * h)))
        p2 = (int(round((p[0] + o * s_bot) * w)), h)
        ok, a, b = cv2.clipLine((0, 0, w, h), p1, p2)
        if ok:
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    return img
