"""End-to-end inference: image -> row pattern (+ confidence) with three interchangeable methods.

classical    excess-green vegetation map        -> geometric pattern fit   (no learning)
regression   RowNet regression head             -> pattern                 (end-to-end CNN)
hybrid       RowNet row-likelihood map, fit initialised by the regression head,
             global search as fallback when that fit is weak               (CNN + geometry)
"""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .classical import vegetation_map
from .fit import DEFAULT_BOUNDS, fit, score
from .steering import guidance

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "models" / "rownet.pt"
METHODS = ("classical", "regression", "hybrid")
COLORS = {"classical": (0, 165, 255), "regression": (255, 120, 0), "hybrid": (0, 255, 255), "gt": (255, 255, 255)}


@dataclass
class Result:
    method: str
    params: np.ndarray  # (Vx, Vy, c, d), normalised
    confidence: float  # heuristic, 0..1 (see README for how it was validated)
    prob: np.ndarray  # row-likelihood map the pattern was derived from
    ms: float
    extra: dict = field(default_factory=dict)


class RowDetector:
    def __init__(self, ckpt=DEFAULT_CKPT):
        self.net, self.meta, self.bounds = None, {}, dict(DEFAULT_BOUNDS)
        if Path(ckpt).exists():
            import torch
            from .model import RowNet
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.net = RowNet(pretrained=False)
            self.net.load_state_dict(state["model"])
            self.net.eval()
            self.meta = state.get("meta", {})
            self.bounds = {k: tuple(v) for k, v in state.get("bounds", DEFAULT_BOUNDS).items()}

    @property
    def has_model(self):
        return self.net is not None

    def available_methods(self):
        return METHODS if self.has_model else ("classical",)

    def _forward(self, bgr):
        import torch
        from .model import IN_H, IN_W, preprocess
        rgb = cv2.cvtColor(cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if self.net is None:
            raise RuntimeError("No trained model loaded (models/rownet.pt is missing).")
        with torch.inference_mode():
            seg, reg = self.net(preprocess(rgb))
        return torch.sigmoid(seg)[0, 0].numpy(), G.from_reg(reg[0].numpy())

    def _classical(self, bgr):
        t0 = time.perf_counter()
        prob, mask = vegetation_map(bgr)
        th, _, on, off, _ = fit(prob, self.bounds)
        veg = float(mask.mean())
        # contrast of the fit, discounted when there is almost no vegetation to fit to
        conf = max(0.0, (on - off) / (on + off + 1e-6)) * (1.0 - math.exp(-veg / 0.01))
        return Result("classical", G.from_fit(th), float(min(conf, 1.0)), prob,
                      (time.perf_counter() - t0) * 1e3, {"on": on, "off": off, "vegetation": veg})

    def predict(self, bgr, methods=("hybrid",), prior=None):
        """Run the requested methods on a BGR image. Returns {method: Result}.

        prior: pattern of the previous video frame. The hybrid fit then starts from it and
        only falls back to the single-image path when that fit is weak.
        """
        out = {}
        if "classical" in methods:
            out["classical"] = self._classical(bgr)
        if not self.has_model or not ({"regression", "hybrid"} & set(methods)):
            return out
        t0 = time.perf_counter()
        prob, reg = self._forward(bgr)
        t_net = (time.perf_counter() - t0) * 1e3
        reg_fit = G.to_fit(reg)
        if "regression" in methods:
            c = float(score(prob, reg_fit)[0][0])
            out["regression"] = Result("regression", G.canon(reg), min(max(c, 0.0), 1.0), prob, t_net,
                                       {"contrast": c})
        if "hybrid" in methods:
            t1 = time.perf_counter()
            init = "previous frame"
            best = fit(prob, self.bounds, inits=[G.to_fit(prior)], n_global=0) if prior is not None else None
            if best is None or best[1] < 0.45:
                init, best = "regression head", fit(prob, self.bounds, inits=[reg_fit], n_global=0)
            if best[1] < 0.45:  # weak fit around the regression: fall back to a global search
                alt = fit(prob, self.bounds, n_global=2)
                if alt[1] > best[1] + 0.02:
                    init, best = "global search", alt
            th, c, on, off, _ = best
            p = G.from_fit(th)
            # Confidence is the row contrast alone. Weighting it by head agreement was tried and
            # dropped: on the HRSD validation split it tracked the error no better (Spearman
            # -0.58 vs -0.59) and flagged 40 % of correct HRDED frames as low confidence.
            disagree = G.row_error(reg, p)[0]  # regression head vs. geometric fit (diagnostic)
            out["hybrid"] = Result("hybrid", p, min(max(c, 0.0), 1.0), prob, t_net + (time.perf_counter() - t1) * 1e3,
                                   {"contrast": c, "on": on, "off": off, "head_disagreement": disagree,
                                    "fit_started_from": init})
        return out


def accuracy(pred, gt, track="row", row_spacing_cm=50.0, hfov_deg=60.0):
    """Scores of a detected pattern against a labelled one.

    iou / pixel_acc   row segmentation: the detected rows and the labelled rows, both drawn as
                      bands of +-0.1 row spacings, compared pixel by pixel
    shift_rows / _cm  error of the steering shift (lateral offset to the tracked line)
    heading_deg       error of the row heading (uses the assumed field of view)
    """
    err, crda = G.row_error(pred, gt)
    a, b = G.row_mask(G.canon(pred), 180, 320), G.row_mask(G.canon(gt), 180, 320)
    union = int((a | b).sum())
    sp, sg = (guidance(p, row_spacing_cm, hfov_deg, track) for p in (pred, gt))
    shift = abs(float(G.wrap(sp.offset_rows - sg.offset_rows)))
    return {"row_err": err, "crda": crda, "iou": float((a & b).sum()) / union if union else 1.0,
            "pixel_acc": float((a == b).mean()), "shift_rows": shift, "shift_cm": shift * row_spacing_cm,
            "heading_deg": abs(sp.heading_deg - sg.heading_deg)}


def interrow_vegetation(bgr, params, tol=0.25):
    """Vegetation split by distance to the nearest detected row (heuristic weed indicator).

    Returns (share of vegetation pixels farther than `tol` row spacings from a row,
    in-row mask, inter-row mask). Not validated: the dataset has no weed labels.
    """
    veg = np.asarray(vegetation_map(bgr)[1], dtype=bool)
    far = G.phase_dist(params, *veg.shape) > tol
    inter, inrow = veg & far, veg & ~far
    n = int(veg.sum())
    return (float(inter.sum()) / n if n else 0.0), inrow, inter


def overlay(bgr, params, color=(0, 255, 255), gt=None, weeds=None, track="row"):
    """Draw rows (and optionally ground truth, in/inter-row vegetation, target line)."""
    img = bgr.copy()
    h, w = img.shape[:2]
    if weeds is not None:
        inrow, inter = (np.asarray(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)) > 0
                        for m in weeds)
        img[inrow] = (0.4 * img[inrow] + 0.6 * np.array([60, 220, 60])).astype(np.uint8)
        img[inter] = (0.4 * img[inter] + 0.6 * np.array([40, 40, 240])).astype(np.uint8)
    drawn = img.copy()
    if gt is not None:
        G.draw_pattern(drawn, gt, COLORS["gt"], max(2, w // 500))
    G.draw_pattern(drawn, params, color, max(3, w // 320))
    img = cv2.addWeighted(drawn, 0.65, img, 0.35, 0)  # lines at 65 % opacity, rest unchanged
    # image centre (machine axis, if the camera is centred) and the tracked target line
    cv2.line(img, (w // 2, h), (w // 2, int(h * 0.82)), (255, 255, 255), max(2, w // 500), cv2.LINE_AA)
    tx = int(round((0.5 - G.lateral_offset(params, track) * params[3]) * w))
    cv2.drawMarker(img, (tx, h - max(8, h // 40)), (255, 0, 255), cv2.MARKER_TRIANGLE_UP, max(14, w // 50), max(2, w // 400))
    return img
