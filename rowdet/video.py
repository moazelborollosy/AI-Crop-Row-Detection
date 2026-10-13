"""Sequential video inference for prerecorded files and live cameras (no Streamlit here).

Every yielded frame carries a real prediction and measured timings. Unlabelled video has no
ground truth, so nothing here computes accuracy.
"""
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import data as D
from .pipeline import COLORS, Result, overlay
from .steering import Steering, guidance

VIDEO_DIR = D.ROOT / "data" / "unlabeld_data" / "raw_data" / "videos_ipad" / "full_videos"


def demo_videos():
    """iPad videos shipped with the dataset: [{path, name, fps, frames, seconds, width, height,
    label_source}]. label_source marks videos that frames were labelled (and trained) from;
    the others were never seen by the model. Never-seen videos come first."""
    if not VIDEO_DIR.exists():
        return []
    used = {p.name.split("_at_")[0] for p in (D.HRSD / "train_data" / "image").glob("*.jpg")}
    used |= {p.name.split("_clip_")[0] for p in D.HRDED.glob("*/images/*_000000.jpg")}
    out = []
    for f in sorted(VIDEO_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(f))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = {"path": str(f), "name": f.stem, "fps": fps, "frames": n, "seconds": n / fps if fps > 0 else 0.0,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "label_source": f.stem in used}
        cap.release()
        out.append(info)
    return sorted(out, key=lambda v: (v["label_source"], v["name"]))


def labelled_clips():
    """HRDED evaluation clips as playable sources: [{name, source, frames, fps, seconds, labels}].

    source is an image-sequence pattern OpenCV reads like a video; labels[i] is the hand-labelled
    pattern of frame i. These clips are held out from training, so accuracy measured on them is real.
    """
    if not D.HRDED.exists():
        return []
    out = []
    for clip, items in D.load_hrded().items():
        if items:
            stem = items[0].path.stem.rsplit("_", 1)[0]
            out.append({"name": clip.replace("_Dataset", ""), "source": str(items[0].path.parent / f"{stem}_%06d.jpg"),
                        "frames": len(items), "fps": 30.0, "seconds": len(items) / 30.0,
                        "labels": [s.params for s in items]})
    return out


class VideoSource:
    """A prerecorded file (read frame by frame) or a live camera (always the newest frame)."""

    def __init__(self, source, live=False, fps=None):
        """fps overrides the rate OpenCV reports (image sequences carry none of their own)."""
        self.live, self.source = live, source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            what = f"camera {source}" if isinstance(source, int) else f"video file {Path(str(source)).name}"
            raise RuntimeError(f"Could not open {what}. " + (
                "Is a camera connected, free, and allowed for desktop apps?" if isinstance(source, int)
                else "The file may be missing, damaged or in an unsupported codec."))
        fps = fps or self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self.frames = 0 if live else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = -1  # index of the last frame handed out
        if live:
            self._latest, self._count, self._taken, self._alive = None, 0, 0, True
            self._lock = threading.Lock()
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()

    def _pump(self):
        # drain the device continuously so read() never returns a stale, buffered frame
        while self._alive:
            ok, frame = self.cap.read()
            if not ok:
                self._alive = False
                break
            with self._lock:
                self._latest, self._count = frame, self._count + 1
            if not isinstance(self.source, int):
                time.sleep(1.0 / self.fps)  # a file standing in for a camera: play at its own rate

    def seek(self, seconds):
        if not self.live and seconds > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * self.fps))
            self.index = int(seconds * self.fps) - 1

    def read(self, skip=0):
        """Next frame after skipping `skip` frames. Returns (frame or None, frames skipped)."""
        if self.live:
            deadline = time.perf_counter() + 5.0
            while True:
                with self._lock:
                    if self._latest is not None and self._count > self._taken:
                        frame, dropped = self._latest, self._count - self._taken - 1
                        self._taken, self._latest = self._count, None
                        self.index += dropped + 1
                        return frame, dropped
                if not self._alive or time.perf_counter() > deadline:
                    return None, 0
                time.sleep(0.002)
        for _ in range(skip):
            if not self.cap.grab():
                return None, 0
        ok, frame = self.cap.read()
        if not ok:
            return None, 0
        self.index += skip + 1
        return frame, skip

    def release(self):
        if self.live:
            self._alive = False
            self._thread.join(timeout=2.0)
        self.cap.release()


@dataclass
class Frame:
    index: int  # source frame number
    time_s: float  # position in the file, or seconds since the stream started (live)
    raw: np.ndarray  # BGR frame at processing width
    image: np.ndarray  # same frame with the row overlay
    result: Result
    steering: Steering
    infer_ms: float  # row detection only
    latency_ms: float  # frame available -> overlay ready (resize + detection + drawing)
    skipped: int  # source frames skipped since the previous processed frame
    fps: float  # processed frames per second, measured over the last 30 frames


def save_upload(name, data, previous=None):
    """Write uploaded video bytes to a temp file (OpenCV needs a path); removes the previous one."""
    if previous:
        Path(previous).unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, prefix="rowdet_", suffix=Path(name).suffix) as f:
        f.write(data)
    return f.name


def probe(path):
    """(fps, seconds) of a video file; raises RuntimeError when it cannot be decoded."""
    src = VideoSource(path)
    ok = src.read()[0] is not None
    fps, seconds = src.fps, src.frames / src.fps
    src.release()
    if not ok:
        raise RuntimeError(f"{Path(path).name} opened but no frame could be decoded (unsupported codec?).")
    return fps, seconds


def stream(src, det, method="hybrid", step=1, width=960, realtime=True, warm_start=True, max_frames=None,
           track="row", row_spacing_cm=50.0, hfov_deg=60.0, min_confidence=0.35, color=COLORS["hybrid"]):
    """Yield a Frame for every processed frame of an open VideoSource until it ends.

    step      process every step-th source frame (files; a live camera always uses the newest)
    width     frames are downscaled to this width before detection and drawing
    realtime  files only: follow the video clock, dropping frames when processing is slower
              and waiting when it is faster. Off = process every step-th frame as fast as possible.
    """
    first, t_start = None, time.perf_counter()
    stamps, prior, n = deque(maxlen=30), None, 0
    while max_frames is None or n < max_frames:
        skip = step - 1
        if realtime and not src.live and first is not None:
            due = first + int((time.perf_counter() - t_start) * src.fps)  # frame the clock is at
            ahead = (src.index + step) - due
            if ahead > 0:
                time.sleep(ahead / src.fps)
            skip = max(skip, due - src.index - 1)
        frame, skipped = src.read(skip)
        if frame is None:
            return
        t0 = time.perf_counter()
        if first is None:
            first, t_start = src.index, t0
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        r = det.predict(frame, (method,), prior=prior if warm_start else None)[method]
        prior = r.params
        s = guidance(r.params, row_spacing_cm, hfov_deg, track, r.confidence, min_confidence)
        img = overlay(frame, r.params, color, track=track)
        t1 = time.perf_counter()
        stamps.append(t1)
        n += 1
        yield Frame(src.index, (t1 - t_start) if src.live else src.index / src.fps, frame, img, r, s, r.ms,
                    (t1 - t0) * 1e3, skipped,
                    (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0)


def summarise(frames, seconds, source_fps):
    """Measured run statistics from [(latency_ms, infer_ms, confidence, skipped, below_threshold)]."""
    if not frames:
        return {}
    a = np.array(frames, dtype=float)
    covered = len(a) + a[:, 3].sum()  # source frames consumed
    return {"frames_processed": len(a), "wall_seconds": seconds, "processing_fps": len(a) / max(seconds, 1e-6),
            "latency_ms_median": float(np.median(a[:, 0])), "latency_ms_p95": float(np.percentile(a[:, 0], 95)),
            "inference_ms_median": float(np.median(a[:, 1])), "confidence_mean": float(a[:, 2].mean()),
            "low_confidence_share": float(a[:, 4].mean()), "source_frames_skipped_share": float(a[:, 3].sum() / covered),
            "playback_speed": float(covered / source_fps / max(seconds, 1e-6))}
