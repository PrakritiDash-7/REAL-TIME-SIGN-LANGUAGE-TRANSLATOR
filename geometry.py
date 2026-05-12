"""Geometric similarity: nearest prototype in normalized landmark space."""

from __future__ import annotations

import numpy as np

from config import SIGN_CLASSES
from prototypes import PROTOTYPES


def nearest_prototype(vec: np.ndarray) -> tuple[str, float]:
    """
    Returns (class_name, score) where score is 1 / (1 + euclidean_distance),
    so higher is more similar.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    best_name = SIGN_CLASSES[0]
    best_dist = float("inf")
    for name in SIGN_CLASSES:
        p = PROTOTYPES[name]
        d = float(np.linalg.norm(v - p))
        if d < best_dist:
            best_dist = d
            best_name = name
    score = 1.0 / (1.0 + best_dist)
    return best_name, score


def prototype_distance(vec: np.ndarray, class_name: str) -> float:
    return float(np.linalg.norm(np.asarray(vec).reshape(-1) - PROTOTYPES[class_name]))
