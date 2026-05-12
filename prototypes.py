"""
Idealized wrist-centered, scale-normalized landmark layouts (flattened length 63)
for synthetic training and geometric nearest-neighbor fallback.
"""

from __future__ import annotations

import numpy as np

from config import SIGN_CLASSES

RIGHT = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, -1.0, 0.0])


def _chain(pts, i0, i1, i2, i3, start, end):
    a, b = np.asarray(start, float), np.asarray(end, float)
    d = b - a
    pts[i0] = a
    pts[i1] = a + 0.30 * d
    pts[i2] = a + 0.62 * d
    pts[i3] = b


def build_prototype_dict() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    def blank():
        return np.zeros((21, 3))

    # --- A: closed fist, thumb folded across index side ---
    p = blank()
    for finger_base, idx0 in [(0.10 * RIGHT + 0.28 * UP, 5), (0.02 * RIGHT + 0.30 * UP, 9),
                               (-0.06 * RIGHT + 0.28 * UP, 13), (-0.14 * RIGHT + 0.24 * UP, 17)]:
        tip = finger_base + 0.12 * UP
        _chain(p, idx0, idx0 + 1, idx0 + 2, idx0 + 3, finger_base, tip)
    _chain(p, 1, 2, 3, 4, 0.12 * RIGHT + 0.08 * UP, 0.14 * RIGHT + 0.18 * UP)
    out["A"] = p.reshape(-1)

    # --- B: flat palm, fingers up, thumb tucked ---
    p = blank()
    b_up = 0.52 * UP
    for i, base in enumerate([0.12 * RIGHT + 0.32 * UP, 0.03 * RIGHT + 0.34 * UP,
                            -0.05 * RIGHT + 0.33 * UP, -0.14 * RIGHT + 0.30 * UP]):
        idx = 5 + i * 4
        _chain(p, idx, idx + 1, idx + 2, idx + 3, base, base + b_up * 0.9 + 0.01 * i * RIGHT)
    _chain(p, 1, 2, 3, 4, 0.10 * RIGHT + 0.12 * UP, 0.06 * RIGHT + 0.22 * UP)
    out["B"] = p.reshape(-1)

    # --- C: curved gap ---
    p = blank()
    for j, base in enumerate([0.10 * RIGHT + 0.30 * UP, 0.02 * RIGHT + 0.32 * UP,
                            -0.06 * RIGHT + 0.30 * UP, -0.14 * RIGHT + 0.27 * UP]):
        idx = 5 + j * 4
        tip = base + 0.35 * UP + 0.08 * j * RIGHT
        _chain(p, idx, idx + 1, idx + 2, idx + 3, base, tip)
    _chain(p, 1, 2, 3, 4, 0.18 * RIGHT + 0.12 * UP, 0.28 * RIGHT + 0.05 * UP)
    out["C"] = p.reshape(-1)

    # --- I: pinky extended only ---
    p = blank()
    for i, (base, tip) in enumerate([
        (0.12 * RIGHT + 0.30 * UP, 0.12 * RIGHT + 0.30 * UP + 0.18 * UP),
        (0.03 * RIGHT + 0.32 * UP, 0.03 * RIGHT + 0.32 * UP + 0.16 * UP),
        (-0.05 * RIGHT + 0.31 * UP, -0.05 * RIGHT + 0.31 * UP + 0.16 * UP),
        (-0.14 * RIGHT + 0.28 * UP, -0.14 * RIGHT + 0.28 * UP + 0.52 * UP),
    ]):
        idx = 5 + i * 4
        _chain(p, idx, idx + 1, idx + 2, idx + 3, base, tip)
    _chain(p, 1, 2, 3, 4, 0.10 * RIGHT + 0.10 * UP, 0.08 * RIGHT + 0.20 * UP)
    out["I"] = p.reshape(-1)

    # --- L: thumb + index orthogonal ---
    p = blank()
    bases = [0.12 * RIGHT + 0.32 * UP, 0.03 * RIGHT + 0.34 * UP,
             -0.05 * RIGHT + 0.32 * UP, -0.14 * RIGHT + 0.29 * UP]
    tips = [bases[0] + 0.55 * UP, bases[1] + 0.18 * UP, bases[2] + 0.16 * UP, bases[3] + 0.15 * UP]
    for i in range(4):
        idx = 5 + i * 4
        _chain(p, idx, idx + 1, idx + 2, idx + 3, bases[i], tips[i])
    _chain(p, 1, 2, 3, 4, 0.08 * RIGHT + 0.12 * UP, 0.48 * RIGHT + 0.08 * UP)
    out["L"] = p.reshape(-1)

    # --- V: index + middle spread ---
    p = blank()
    bases = [0.12 * RIGHT + 0.32 * UP, 0.03 * RIGHT + 0.34 * UP,
             -0.05 * RIGHT + 0.32 * UP, -0.14 * RIGHT + 0.29 * UP]
    tips = [
        bases[0] + 0.50 * UP + 0.10 * RIGHT,
        bases[1] + 0.52 * UP - 0.10 * RIGHT,
        bases[2] + 0.17 * UP,
        bases[3] + 0.15 * UP,
    ]
    for i in range(4):
        idx = 5 + i * 4
        _chain(p, idx, idx + 1, idx + 2, idx + 3, bases[i], tips[i])
    _chain(p, 1, 2, 3, 4, 0.10 * RIGHT + 0.10 * UP, 0.10 * RIGHT + 0.22 * UP)
    out["V"] = p.reshape(-1)

    # --- Y: thumb + pinky ---
    p = blank()
    bases = [0.12 * RIGHT + 0.30 * UP, 0.03 * RIGHT + 0.32 * UP,
             -0.05 * RIGHT + 0.31 * UP, -0.14 * RIGHT + 0.28 * UP]
    tips = [bases[0] + 0.18 * UP, bases[1] + 0.18 * UP, bases[2] + 0.18 * UP, bases[3] + 0.52 * UP]
    for i in range(4):
        idx = 5 + i * 4
        _chain(p, idx, idx + 1, idx + 2, idx + 3, bases[i], tips[i])
    _chain(p, 1, 2, 3, 4, 0.06 * RIGHT + 0.10 * UP, 0.42 * RIGHT + 0.12 * UP)
    out["Y"] = p.reshape(-1)

    # --- Open palm: all extended spread ---
    p = blank()
    dirs = [0.45 * UP + 0.20 * RIGHT, 0.52 * UP + 0.06 * RIGHT, 0.50 * UP - 0.08 * RIGHT, 0.44 * UP - 0.22 * RIGHT]
    bases = [0.12 * RIGHT + 0.30 * UP, 0.03 * RIGHT + 0.33 * UP, -0.05 * RIGHT + 0.32 * UP, -0.14 * RIGHT + 0.28 * UP]
    for i in range(4):
        idx = 5 + i * 4
        _chain(p, idx, idx + 1, idx + 2, idx + 3, bases[i], bases[i] + dirs[i])
    _chain(p, 1, 2, 3, 4, 0.08 * RIGHT + 0.10 * UP, 0.32 * RIGHT + 0.18 * UP)
    out["OPEN_PALM"] = p.reshape(-1)

    for k in SIGN_CLASSES:
        assert k in out, k
    return out


PROTOTYPES = build_prototype_dict()
