"""Pseudo-label the unlabelled iPad videos with the current model (self-training, step 1).

    python pseudo_label.py [--every 0.5] [--min-conf 0.6] [--max-flip-err 0.05] [--max-frames 400]

Frames are sampled from every full video and labelled with the hybrid detector. A frame is kept
when the fit is confident and the detector gives the same rows on the mirrored frame. Kept frames
and their patterns go to data/pseudo_labels (same labels.lbl format as the HORSCH datasets);
`python train.py --pseudo` then trains on them. Frames inside HRDED clip windows or next to HRSD
validation / test frames are skipped here, so the held-out sets stay held out.
"""
import argparse
import csv
import json
import shutil

import cv2
import numpy as np

from rowdet import data as D
from rowdet import geometry as G
from rowdet import video as V
from rowdet.pipeline import DEFAULT_CKPT, RowDetector

SAVE_W = 640  # stored frame width; training reads frames at 480 px


def label_frame(det, bgr):
    """(pattern, confidence, flip error). Flip error is the row error between the pattern found
    on the frame and the one found on its mirror image, mirrored back."""
    r = det.predict(bgr, ("hybrid",))["hybrid"]
    m = det.predict(np.ascontiguousarray(bgr[:, ::-1]), ("hybrid",))["hybrid"]
    return r.params, r.confidence, G.row_error(G.flip(m.params), r.params)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--max-flip-err", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=400, help="cap on kept frames (evenly thinned)")
    args = ap.parse_args()

    det = RowDetector(args.ckpt)
    if not det.has_model:
        raise SystemExit("No trained model: run train.py first.")
    hrsd = D.load_hrsd()
    windows = D.hrded_windows(D.load_hrded())
    held = hrsd["validation"] + hrsd["test"]

    kept, stats = [], {}
    for v in V.demo_videos():
        src = V.VideoSource(v["path"])
        step = max(1, round(args.every * src.fps))
        n = n_free = n_ok = 0
        while True:
            frame, _ = src.read(step - 1)
            if frame is None:
                break
            n += 1
            s = D.Sample(None, None, v["name"], src.index / src.fps, "pseudo")
            if not D.drop_overlap([s], windows)[0] or not D.drop_near([s], held)[0]:
                continue
            n_free += 1
            p, conf, flip_err = label_frame(det, frame)
            if conf >= args.min_conf and flip_err <= args.max_flip_err:
                n_ok += 1
                kept.append((v["name"], s.time_s, cv2.resize(frame, (SAVE_W, round(SAVE_W * frame.shape[0] / frame.shape[1])),
                                                             interpolation=cv2.INTER_AREA), p))
        src.release()
        stats[v["name"]] = (n, n_free, n_ok)
        print(f"{v['name']}  sampled {n:4d}  outside held-out windows {n_free:4d}  kept {n_ok:4d}"
              f"  ({'label source' if v['label_source'] else 'unseen video'})", flush=True)

    if len(kept) > args.max_frames:  # thin evenly, so every video keeps its share
        kept = [kept[i] for i in np.linspace(0, len(kept) - 1, args.max_frames).round().astype(int)]
    if D.PSEUDO.exists():
        shutil.rmtree(D.PSEUDO)
    (D.PSEUDO / "images").mkdir(parents=True)
    with open(D.PSEUDO / "labels.lbl", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "base_points", "true_pattern_parameter"])
        for name, t, img, p in kept:
            fn = f"{name}_at_{t:.2f}.jpg"
            cv2.imwrite(str(D.PSEUDO / "images" / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([fn, "[]", json.dumps([float(x) for x in p])])
    print(f"kept {sum(s[2] for s in stats.values())} of {sum(s[1] for s in stats.values())} eligible frames, "
          f"wrote {len(kept)} to {D.PSEUDO}")


if __name__ == "__main__":
    main()
