"""Steering guidance derived from a row pattern.

What the image supports without calibration: the lateral offset in units of row spacing
and the horizontal position of the vanishing point. Metric values need two assumptions
supplied by the user (real row spacing, camera horizontal field of view) and a camera
mounted on the machine centreline looking forward. The steering command is an
illustrative proportional law; its gains were never tuned or validated on a vehicle.
"""
import math
from dataclasses import dataclass

from . import geometry as G


@dataclass
class Steering:
    offset_rows: float  # image centre vs. target line, in row spacings (+ = target is left)
    offset_cm: float  # same, using the assumed row spacing
    heading_deg: float  # row direction vs. camera axis (+ = rows lead to the right)
    command: float  # -1 (full left) .. +1 (full right)
    advice: str


def guidance(p, row_spacing_cm=50.0, hfov_deg=60.0, track="row", confidence=1.0, min_confidence=0.35):
    e = float(G.lateral_offset(p, track))
    heading = math.degrees(math.atan((p[0] - 0.5) * 2.0 * math.tan(math.radians(hfov_deg) / 2.0)))
    cmd = max(-1.0, min(1.0, 0.6 * heading / 10.0 - 0.8 * 2.0 * e))
    if confidence < min_confidence:
        advice = "LOW CONFIDENCE - hold course / hand over to operator"
        cmd = 0.0
    elif abs(cmd) < 0.08:
        advice = "KEEP STRAIGHT"
    else:
        advice = f"STEER {'RIGHT' if cmd > 0 else 'LEFT'} ({abs(cmd) * 100:.0f} %)"
    return Steering(e, e * row_spacing_cm, heading, cmd, advice)
