"""Fit the vanishing-point row pattern to a row-likelihood map.

The map can come from classical vegetation segmentation or from the CNN. A pattern scores
high when the map is bright along its rows and dark half-way between them, which also
rejects half- and double-spacing harmonics. Fit parameters are theta = (Vx, r, c, d).
"""
import itertools

import numpy as np
from scipy.optimize import minimize

FIT_YS = np.linspace(0.04, 0.98, 16)
_K = np.arange(-12, 13)

# (lo, hi) search bounds for Vx, r, d; replaced by training-label statistics when available
DEFAULT_BOUNDS = {"vx": (0.2, 0.8), "r": (0.1, 0.75), "d": (0.1, 0.5)}


def label_bounds(thetas):
    """Search bounds from labelled patterns (N x [Vx, r, ...]) with d last: 1-99 % range + padding.

    Percentiles keep a few extreme headland frames from inflating the search space.
    """
    t = np.asarray(thetas, dtype=np.float64)
    out = {}
    for k, col in (("vx", t[:, 0]), ("r", t[:, 1]), ("d", t[:, -1])):
        lo, hi = np.percentile(col, [1, 99])
        pad = 0.1 * (hi - lo) + 0.01
        out[k] = (float(lo - pad), float(hi + pad))
    return out


def score(P, thetas, ys=FIT_YS):
    """Row contrast for N patterns. Returns (contrast, on, off), each of shape (N,)."""
    th = np.atleast_2d(np.asarray(thetas, dtype=np.float64))
    H, W = P.shape
    s = th[:, 1:2] + (1.0 - th[:, 1:2]) * ys[None, :]  # (N, J) local spacing scale
    off0 = th[:, 2:3] + _K[None, :] * th[:, 3:4]  # (N, M) bottom-edge offsets
    yi = np.clip(np.round(ys * (H - 1)).astype(int), 0, H - 1)

    def sample(shift):
        x = th[:, 0, None, None] + (off0[:, :, None] + shift[:, None, None]) * s[:, None, :]
        inside = (x >= 0.0) & (x <= 1.0)
        xi = np.clip(np.round(x * (W - 1)).astype(int), 0, W - 1)
        v = P[yi[None, None, :], xi] * inside
        n = inside.sum(1)  # (N, J) rows visible per height
        per_y = v.sum(1) / np.maximum(n, 1)
        valid = n > 0
        return per_y.sum(1) / np.maximum(valid.sum(1), 1)

    on = sample(np.zeros(len(th)))
    off = sample(0.5 * th[:, 3])
    return on - off, on, off


def _grid(bounds, n=(9, 6, 9, 8)):
    vx, r, d = (np.linspace(bounds[k][0], bounds[k][1], m) for k, m in zip(("vx", "r", "d"), n))
    ph = np.arange(n[3]) / n[3]
    g = np.array(list(itertools.product(vx, r, d, ph)))  # columns: Vx, r, d, phase
    return np.column_stack([g[:, 0], g[:, 1], g[:, 3] * g[:, 2], g[:, 2]])  # -> Vx, r, c, d


def _refine(P, theta, bounds):
    lo = np.array([bounds["vx"][0], bounds["r"][0], -np.inf, bounds["d"][0]])
    hi = np.array([bounds["vx"][1], bounds["r"][1], np.inf, bounds["d"][1]])

    def cost(t):
        if np.any(t < lo) or np.any(t > hi):
            return 1.0
        return -score(P, t)[0][0]

    res = minimize(cost, theta, method="Nelder-Mead",
                   options={"xatol": 5e-4, "fatol": 1e-4, "maxiter": 160,
                            "initial_simplex": theta + np.vstack([np.zeros(4), np.diag([0.02, 0.03, 0.015, 0.015])])})
    return res.x, -res.fun


def fit(P, bounds=DEFAULT_BOUNDS, inits=(), n_global=3):
    """Best pattern for map P.

    inits: optional theta candidates (e.g. the CNN regression) that are refined alongside
    the n_global best patterns of a coarse global search.
    Returns (theta, contrast, on, off, index of the winning candidate; inits come first).
    """
    cands = [np.clip(np.asarray(t, dtype=np.float64),
                     [bounds["vx"][0], bounds["r"][0], -np.inf, bounds["d"][0]],
                     [bounds["vx"][1], bounds["r"][1], np.inf, bounds["d"][1]]) for t in inits]
    if n_global > 0:
        g = _grid(bounds)
        sc = score(P, g)[0]
        cands += [g[i] for i in np.argsort(-sc)[:n_global]]
    best, best_sc, best_i = None, -np.inf, -1
    for i, t in enumerate(cands):
        t, sc = _refine(P, t, bounds)
        # a global candidate must clearly beat a refined init to replace it (stability)
        margin = 0.02 if 0 <= best_i < len(inits) <= i else 0.0
        if sc > best_sc + margin:
            best, best_sc, best_i = t, sc, i
    c, on, off = score(P, best)
    return best, float(c[0]), float(on[0]), float(off[0]), best_i
