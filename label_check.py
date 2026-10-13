"""Measure the pseudo-labels against a few hand labels.

    python label_check.py [--n 10]      label the sample by hand (opens a window), then score
    python label_check.py --score       only score what has been labelled so far

One frame per unseen video first, then the rest spread over the remaining pseudo-labels. The
model's rows are not shown while clicking. Click two points on one crop row, then two points on
the row next to it. Keys: Enter/Space = accept, r = redo, s = skip frame, q = quit.
Hand labels go to data/pseudo_labels/check_labels.lbl (same format as the HORSCH labels).
"""
import argparse
import csv
import json

import cv2
import numpy as np

from rowdet import data as D
from rowdet import geometry as G
from rowdet import video as V
from rowdet.pipeline import RowDetector

OUT = D.PSEUDO / "check_labels.lbl"
WIN, W, H = "label check", 1280, 720


def pick(n):
    """n pseudo-labelled samples: one per unseen video (middle frame), then evenly from the rest."""
    pseudo = D.load_pseudo()
    unseen = {v["name"] for v in V.demo_videos() if not v["label_source"]}
    by_video = {}
    for s in pseudo:
        by_video.setdefault(s.video, []).append(s)
    first = [v[len(v) // 2] for k, v in sorted(by_video.items()) if k in unseen]
    rest = [s for s in pseudo if s not in first]
    extra = [rest[i] for i in np.linspace(0, len(rest) - 1, max(n - len(first), 0)).round().astype(int)]
    return (first + extra)[:n]


def click_label(sample, title):
    """Hand pattern for one frame, 'skip' or 'quit'."""
    base = cv2.resize(cv2.imread(str(sample.path)), (W, H), interpolation=cv2.INTER_CUBIC)
    pts = []
    cv2.setMouseCallback(WIN, lambda e, x, y, *_: pts.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4 else None)
    while True:
        img = base.copy()
        if len(pts) == 4 and pts[0][1] != pts[1][1] and pts[2][1] != pts[3][1]:
            G.draw_pattern(img, G.pattern_from_points(np.array(pts, dtype=float) / [W, H]), (0, 255, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 0, 255) if i < 2 else (255, 0, 0), -1)
        hint = "Enter = accept, r = redo" if len(pts) == 4 else f"click point {len(pts) % 2 + 1} of row {len(pts) // 2 + 1}"
        cv2.putText(img, f"{title}  {hint}  (s = skip, q = quit)", (10, 28), 0, 0.7, (255, 255, 255), 2)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return "quit"
        if k == ord("s"):
            return "skip"
        if k == ord("r"):
            pts.clear()
        if k in (13, 32) and len(pts) == 4:
            return pts


def score(samples, hand):
    rows = [(s, G.row_error(s.params, hand[s.path.name])) for s in samples if s.path.name in hand]
    if not rows:
        print("no hand labels yet")
        return
    unseen = {v["name"] for v in V.demo_videos() if not v["label_source"]}
    bounds = RowDetector().bounds
    print(f"\n{'frame':46s} {'video':>8s} {'row error':>10s} {'CRDA':>6s}  note")
    for s, (err, crda) in rows:
        t = G.to_fit(G.canon(s.params))
        pinned = any(abs(t[i] - b) < 1e-3 for k, i in (("vx", 0), ("r", 1), ("d", 3)) for b in bounds[k])
        print(f"{s.path.name:46s} {'unseen' if s.video in unseen else 'seen':>8s} {err:10.3f} {crda:6.2f}  {'at search limit' if pinned else ''}")
    e = np.array([r[1][0] for r in rows])
    print(f"\n{len(e)} frames: mean row error {e.mean():.3f}, median {np.median(e):.3f}, worst {e.max():.3f}, "
          f"within 0.10: {(e <= 0.10).mean() * 100:.0f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    samples = pick(args.n)
    hand, points = {}, {}
    if OUT.exists():
        with open(OUT, newline="") as f:
            for r in csv.DictReader(f):
                points[r["image_name"]] = r["base_points"]
                hand[r["image_name"]] = np.array(json.loads(r["true_pattern_parameter"]))
    if not args.score:
        cv2.namedWindow(WIN)
        todo = [s for s in samples if s.path.name not in hand]
        for i, s in enumerate(todo):
            res = click_label(s, f"{i + 1}/{len(todo)}")
            if res == "quit":
                break
            if res == "skip":
                continue
            points[s.path.name] = json.dumps([list(p) for p in res])
            hand[s.path.name] = G.pattern_from_points(np.array(res, dtype=float) / [W, H])
            with open(OUT, "w", newline="") as f:  # rewritten after every frame, so quitting loses nothing
                w = csv.writer(f)
                w.writerow(["image_name", "base_points", "true_pattern_parameter"])
                for name, p in hand.items():
                    w.writerow([name, points[name], json.dumps([float(x) for x in p])])
        cv2.destroyAllWindows()
    score(samples, hand)


if __name__ == "__main__":
    main()
