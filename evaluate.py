"""Evaluate all methods on held-out data and write outputs/metrics.json (read by the app).

    python evaluate.py [--stride 1] [--ckpt models/rownet.pt] [--out outputs/metrics.json]

Held-out sets
  hrsd_test  HRSD test split (random frame split by the data provider: same videos as training)
  hrded      9 HRDED clips; training frames inside their time windows were removed
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from rowdet import data as D
from rowdet import geometry as G
from rowdet.pipeline import DEFAULT_CKPT, RowDetector

THRESH = np.linspace(0.0, 0.5, 51)


def constant_baseline(train):
    """Pattern that ignores the image: mean training pattern (circular mean for the phase)."""
    return G.from_reg(np.mean([G.to_reg(s.params) for s in train], axis=0))


def frame_metrics(pred, gt):
    err, crda = G.row_error(pred, gt)
    return {"row_err": err, "crda": crda, "vx_err": abs(pred[0] - gt[0]),
            "d_rel_err": abs(pred[3] - gt[3]) / gt[3],
            "lat_err": abs(G.wrap(G.lateral_offset(pred) - G.lateral_offset(gt)))}


def summarise(rows):
    if not rows:
        return {}
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    e = a["row_err"]
    out = {"n": len(rows), "row_err_mean": e.mean(), "row_err_median": np.median(e), "row_err_p90": np.percentile(e, 90),
           "crda": a["crda"].mean(), "acc_0.10": (e <= 0.10).mean(), "acc_0.25": (e <= 0.25).mean(),
           "vx_mae": a["vx_err"].mean(), "d_rel_mae": a["d_rel_err"].mean(), "lat_mae_rows": a["lat_err"].mean(),
           "ms_mean": a["ms"].mean(), "cdf": [(e <= t).mean() for t in THRESH]}
    conf = a["conf"]
    if np.ptp(conf) > 1e-9:
        hi = np.zeros(len(conf), dtype=bool)
        hi[np.argsort(-conf, kind="stable")[:len(conf) // 2]] = True  # rank split: robust to ties
        acc = conf >= 0.35
        out["confidence"] = {"spearman_conf_vs_err": spearmanr(conf, e)[0],
                             "row_err_top_half": e[hi].mean(), "row_err_bottom_half": e[~hi].mean(),
                             "coverage_at_0.35": acc.mean(),
                             "row_err_accepted": e[acc].mean() if acc.any() else None,
                             "row_err_rejected": e[~acc].mean() if (~acc).any() else None}
    return json.loads(json.dumps(out, default=float))


def run(det, samples, const, sequence):
    """Per-frame records for every method over a list of samples (one clip or one split).

    sequence=True adds "hybrid+video": the hybrid as the video mode runs it, each fit started
    from the previous frame's pattern.
    """
    rec = {m: [] for m in (*det.available_methods(), "constant")}
    prior = None
    if sequence and det.has_model:
        rec["hybrid+video"] = []
    for s in samples:
        img = cv2.imread(str(s.path))
        for m, r in det.predict(img, det.available_methods()).items():
            rec[m].append({**frame_metrics(r.params, s.params), "conf": r.confidence, "ms": r.ms})
        if "hybrid+video" in rec:
            r = det.predict(img, ("hybrid",), prior=prior)["hybrid"]
            prior = r.params
            rec["hybrid+video"].append({**frame_metrics(r.params, s.params), "conf": r.confidence, "ms": r.ms})
        rec["constant"].append({**frame_metrics(const, s.params), "conf": 0.0, "ms": 0.0})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default="outputs/metrics.json")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th HRDED frame")
    args = ap.parse_args()

    det = RowDetector(args.ckpt)
    hrsd, hrded = D.load_hrsd(), D.load_hrded()
    const = constant_baseline(D.drop_overlap(hrsd["train"], D.hrded_windows(hrded))[0])
    held = set(det.meta.get("holdout_videos", []))
    t0 = time.time()
    result = {"meta": {"checkpoint": det.meta, "has_model": det.has_model, "stride": args.stride,
                       "thresholds": THRESH.tolist(), "created": time.strftime("%Y-%m-%d %H:%M")},
              "datasets": {}}

    rec = run(det, hrsd["test"], const, sequence=False)
    result["datasets"]["hrsd_test"] = {m: summarise(r) for m, r in rec.items()}
    print(f"hrsd_test done ({time.time() - t0:.0f}s)", flush=True)

    total, per_clip, unseen = {}, {}, {}
    for clip, samples in hrded.items():
        rec = run(det, samples[::args.stride], const, sequence=True)
        per_clip[clip] = {m: {k: v for k, v in summarise(r).items() if k != "cdf"} for m, r in rec.items()}
        for m, r in rec.items():
            total.setdefault(m, []).extend(r)
            if samples and samples[0].video in held:
                unseen.setdefault(m, []).extend(r)
        print(f"{clip} done ({time.time() - t0:.0f}s)", flush=True)
    result["datasets"]["hrded"] = {m: summarise(r) for m, r in total.items()}
    if unseen:
        result["datasets"]["hrded_unseen_videos"] = {m: summarise(r) for m, r in unseen.items()}
    result["hrded_per_clip"] = per_clip

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    for ds, methods in result["datasets"].items():
        print(f"\n{ds}")
        print(f"{'method':18s} {'n':>5s} {'row_err':>8s} {'median':>7s} {'CRDA':>6s} {'acc.10':>7s} {'acc.25':>7s} {'lat_mae':>8s} {'ms':>6s}")
        for m, s in methods.items():
            print(f"{m:18s} {s['n']:5d} {s['row_err_mean']:8.3f} {s['row_err_median']:7.3f} {s['crda']:6.3f} {s['acc_0.10']:7.3f} {s['acc_0.25']:7.3f} {s['lat_mae_rows']:8.3f} {s['ms_mean']:6.0f}")


if __name__ == "__main__":
    main()
