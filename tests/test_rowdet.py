"""Run with:  python -m unittest discover -s tests -v"""
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rowdet import data as D
from rowdet import geometry as G
from rowdet import video as V
from rowdet.fit import DEFAULT_BOUNDS, fit
from rowdet.pipeline import DEFAULT_CKPT, RowDetector
from rowdet.steering import guidance

P = np.array([0.51, -0.83, 0.06, 0.25])


class Geometry(unittest.TestCase):
    def test_reg_roundtrip(self):
        np.testing.assert_allclose(G.from_reg(G.to_reg(P)), P, atol=1e-5)

    def test_error_is_zero_and_periodic(self):
        self.assertAlmostEqual(G.row_error(P, P)[0], 0.0, places=6)
        self.assertAlmostEqual(G.row_error(P + [0, 0, P[3], 0], P)[0], 0.0, places=6)
        self.assertAlmostEqual(G.row_error(P, P)[1], 1.0, places=6)

    def test_error_penalises_shift_and_wrong_spacing(self):
        shifted = P + [0, 0, 0.25 * P[3], 0]
        self.assertAlmostEqual(G.row_error(shifted, P)[0], 0.25, places=3)
        half = np.array([P[0], P[1], P[2], P[3] / 2])  # extra row in every gap
        self.assertGreater(G.row_error(half, P)[0], 0.1)
        double = np.array([P[0], P[1], P[2], P[3] * 2])  # every second row missing
        self.assertGreater(G.row_error(double, P)[0], 0.1)

    def test_flip_and_crop_keep_points_on_rows(self):
        y = 0.7
        xs = G.rows_at(P, y)
        f = G.flip(P)
        self.assertTrue(np.all(G.lattice_dist(1 - xs, f[0] + f[2] * G.scale(f, y), f[3] * G.scale(f, y)) < 1e-9))
        x0, y0, sx, sy = 0.1, 0.05, 0.8, 0.9
        c = G.crop(P, x0, y0, sx, sy)
        yc = (y - y0) / sy
        d = G.lattice_dist((xs - x0) / sx, c[0] + c[2] * G.scale(c, yc), c[3] * G.scale(c, yc))
        self.assertTrue(np.all(d < 1e-9))

    def test_soft_mask_peaks_on_rows(self):
        m = G.soft_mask(P, 112, 192)
        x = G.rows_at(P, (100 + 0.5) / 112)
        self.assertTrue(np.all(m[100, np.clip((x * 192).astype(int), 0, 191)] > 0.6))
        self.assertLess(m.mean(), 0.35)


class Fit(unittest.TestCase):
    def test_recovers_pattern_from_rendered_map(self):
        for p in (P, np.array([0.44, -1.3, 0.2, 0.33]), np.array([0.6, -0.4, 0.01, 0.2])):
            th, contrast, *_ = fit(G.soft_mask(p, 112, 192), DEFAULT_BOUNDS)
            self.assertLess(G.row_error(G.from_fit(th), p)[0], 0.05)
            self.assertGreater(contrast, 0.6)


class Steering(unittest.TestCase):
    def test_signs(self):
        left = np.array([0.5, -1.0, 0.2, 0.25])  # nearest row 0.05 left of centre (offset 0.2 - 0.25)
        self.assertAlmostEqual(G.lateral_offset(left), 0.2, places=6)
        self.assertLess(guidance(left).command, 0)  # target on the left -> steer left
        self.assertGreater(guidance(np.array([0.7, -1.0, 0.0, 0.25])).command, 0)  # rows lead right
        self.assertEqual(guidance(left, confidence=0.1).command, 0.0)


@unittest.skipUnless(D.HRDED.exists(), "dataset not present")
class Labels(unittest.TestCase):
    def test_base_points_agree_with_pattern(self):
        """The two hand-clicked lines must be rows of the labelled pattern (1280x720 frames)."""
        lbl = next(D.HRDED.glob("*/labels/labels.lbl"))
        with open(lbl, newline="") as f:
            row = next(csv.DictReader(f))
        pts = np.array(json.loads(row["base_points"]), dtype=float) / [1280, 720]
        p = np.array(json.loads(row["true_pattern_parameter"]))
        for x, y in pts:
            s = G.scale(p, y)
            self.assertLess(G.lattice_dist(x, p[0] + p[2] * s, p[3] * s), 0.01)

    def test_splits_load(self):
        hrsd = D.load_hrsd()
        self.assertTrue(all(len(v) > 0 for v in hrsd.values()))
        self.assertTrue(all(len(v) > 0 for v in D.load_hrded().values()))


@unittest.skipUnless(V.VIDEO_DIR.exists() and DEFAULT_CKPT.exists(), "iPad videos or trained model not present")
class Video(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = RowDetector()
        cls.video = min(V.demo_videos(), key=lambda v: v["frames"])

    def test_demo_videos_listed(self):
        vids = V.demo_videos()
        self.assertTrue(all(v["fps"] > 0 and v["frames"] > 0 for v in vids))
        self.assertFalse(vids[0]["label_source"])  # videos the model never saw come first

    def test_file_stream_yields_real_predictions(self):
        src = V.VideoSource(self.video["path"])
        frames = list(V.stream(src, self.det, step=5, width=640, realtime=False, max_frames=6))
        src.release()
        self.assertEqual([f.index for f in frames], [4, 9, 14, 19, 24, 29])
        self.assertTrue(all(f.skipped == 4 and f.image.shape[1] == 640 for f in frames))
        self.assertTrue(all(0.0 <= f.result.confidence <= 1.0 and f.latency_ms > 0 for f in frames))
        s = V.summarise([(f.latency_ms, f.infer_ms, f.result.confidence, f.skipped, False) for f in frames], 1.0, src.fps)
        self.assertEqual(s["frames_processed"], 6)
        self.assertAlmostEqual(s["source_frames_skipped_share"], 24 / 30)

    def test_seek_then_runs_to_end_of_file(self):
        src = V.VideoSource(self.video["path"])
        src.seek(self.video["seconds"] - 1.0)
        frames = list(V.stream(src, self.det, step=10, realtime=False))
        src.release()
        self.assertTrue(0 < len(frames) <= 4)

    def test_live_path_hands_out_newest_frames(self):
        src = V.VideoSource(self.video["path"], live=True)  # a file standing in for a camera
        frames = list(V.stream(src, self.det, width=480, max_frames=4))
        src.release()
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(b.index > a.index for a, b in zip(frames, frames[1:])))

    def test_bad_sources_and_upload_spooling(self):
        with self.assertRaises(RuntimeError):
            V.VideoSource("missing.mp4")
        junk = V.save_upload("junk.mp4", b"not a video")
        with self.assertRaises(RuntimeError):
            V.probe(junk)
        good = V.save_upload("clip.mp4", Path(self.video["path"]).read_bytes(), previous=junk)
        self.assertFalse(Path(junk).exists())
        fps, seconds = V.probe(good)
        Path(good).unlink()
        self.assertAlmostEqual(seconds, self.video["seconds"], delta=0.5)


if __name__ == "__main__":
    unittest.main()
