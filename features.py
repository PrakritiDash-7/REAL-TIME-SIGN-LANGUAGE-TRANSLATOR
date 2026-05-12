"""Normalize MediaPipe hand landmarks to a fixed-length vector for ML + distance metrics."""

from __future__ import annotations

import numpy as np


def landmarks_to_vector(landmarks) -> np.ndarray | None:
    """
    landmarks: iterable of 21 items with .x, .y, .z in [0,1] image space.
    Returns shape (63,) wrist-centered and scale-normalized, or None if invalid.
    """
    if landmarks is None:
        return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    if pts.shape != (21, 3):
        return None
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9] - pts[0])  # wrist to middle finger MCP
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.reshape(-1)


def vector_to_landmarks(vec: np.ndarray):
    """Inverse for visualization only: vec (63,) -> (21,3). Scale/wrist unknown."""
    v = np.asarray(vec, dtype=np.float64).reshape(21, 3)
    return v
