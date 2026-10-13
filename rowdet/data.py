"""Readers for the HORSCH datasets (HRSD segmentation set, HRDED evaluation clips)."""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "data" / "labeld_data"
HRSD = LABELED / "hrsd_segmentation_dataset"
HRDED = LABELED / "hrded_evaluation_dataset"
CLIPS = ROOT / "data" / "unlabeld_data" / "raw_data" / "videos_ipad" / "clipped_videos"
PSEUDO = ROOT / "data" / "pseudo_labels"  # written by pseudo_label.py

_AT = re.compile(r"^(IMG_\d+_[\d-]+_[\d-]+)_at_([\d.]+)\.jpg$")
_CLIP = re.compile(r"^(IMG_\d+_[\d-]+_[\d-]+)_clip_(\d+)_start_(\d+)_(\d+)_(\d+)\.jpg$")


@dataclass
class Sample:
    path: Path
    params: np.ndarray  # (Vx, Vy, c, d)
    video: str
    time_s: float  # position in the full video (seconds)
    group: str  # split name (HRSD) or clip name (HRDED)


def read_lbl(path):
    """labels.lbl -> {image_name: params}."""
    with open(path, newline="") as f:
        return {r["image_name"]: np.array(json.loads(r["true_pattern_parameter"]), dtype=np.float64)
                for r in csv.DictReader(f)}


def load_hrsd():
    """{split: [Sample]} for train / validation / test, labels taken from labels.lbl."""
    labels = read_lbl(HRSD / "all_images" / "labels" / "labels.lbl")
    out = {}
    for split in ("train", "validation", "test"):
        items = []
        for img in sorted((HRSD / f"{split}_data" / "image").glob("*.jpg")):
            m = _AT.match(img.name)
            if m and img.name in labels:
                items.append(Sample(img, labels[img.name], m.group(1), float(m.group(2)), split))
        out[split] = items
    return out


def _clip_duration(prefix, n_frames):
    cap = cv2.VideoCapture(str(CLIPS / f"{prefix}.mp4"))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return max(n / fps if fps > 0 else 0.0, n_frames / 30.0)


def load_hrded():
    """{clip: [Sample]} in frame order. time_s is the clip start (frames are consecutive)."""
    out = {}
    for d in sorted(p for p in HRDED.iterdir() if p.is_dir()):
        labels = read_lbl(d / "labels" / "labels.lbl")
        items = []
        for img in sorted((d / "images").glob("*.jpg")):
            m = _CLIP.match(img.name)
            if m and img.name in labels:
                start = int(m.group(3)) * 60 + int(m.group(4))
                items.append(Sample(img, labels[img.name], m.group(1), float(start), d.name))
        out[d.name] = items
    return out


def hrded_windows(hrded, margin=2.0):
    """{video: [(t0, t1)]} time windows of the full videos covered by evaluation clips."""
    win = {}
    for items in hrded.values():
        if not items:
            continue
        s = items[0]
        prefix = s.path.name.rsplit("_", 1)[0]
        dur = _clip_duration(prefix, len(items))
        win.setdefault(s.video, []).append((s.time_s - margin, s.time_s + dur + margin))
    return win


def drop_overlap(samples, windows):
    """Remove samples that fall inside an evaluation-clip time window (leak prevention)."""
    keep = [s for s in samples
            if not any(t0 <= s.time_s <= t1 for t0, t1 in windows.get(s.video, []))]
    return keep, len(samples) - len(keep)


def drop_near(samples, others, margin=1.0):
    """Remove samples within `margin` seconds of any sample in `others` from the same video."""
    keep = [s for s in samples
            if not any(o.video == s.video and abs(o.time_s - s.time_s) <= margin for o in others)]
    return keep, len(samples) - len(keep)


def load_pseudo():
    """[Sample] of model-labelled video frames (empty when pseudo_label.py was not run)."""
    if not (PSEUDO / "labels.lbl").exists():
        return []
    labels = read_lbl(PSEUDO / "labels.lbl")
    items = []
    for img in sorted((PSEUDO / "images").glob("*.jpg")):
        m = _AT.match(img.name)
        if m and img.name in labels:
            items.append(Sample(img, labels[img.name], m.group(1), float(m.group(2)), "pseudo"))
    return items
