"""
Shared fixtures, and the synthetic castings the suite runs on.

The tests never touch the real dataset. A synthetic impeller is built with
the same structure the pipeline relies on (a dark annular cavity around a
bright hub, a bright machined rim, a light background), so a test can place a
defect at a known position and assert that the detector finds it there. Real
images could not support that assertion, because nobody has labeled where the
defects are.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


def make_casting(size: int = 256, background: int = 200, rim: int = 215,
                 cavity: int = 20, hub: int = 150, seed: int = 0) -> np.ndarray:
    """
    A synthetic impeller with the structure the pipeline anchors on.

    Radii are chosen so the cavity, the rim band and the background land where
    `find_anchor` and `rim_mask` expect them, which is what makes a positive
    detection test meaningful rather than circular.
    """
    rng = np.random.default_rng(seed)
    image = np.full((size, size), background, dtype=np.float64)
    center = size / 2
    rows, cols = np.ogrid[:size, :size]
    radius = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)

    unit = size / 10.0
    image[radius <= 4.2 * unit] = rim          # the machined face
    image[radius <= 2.0 * unit] = cavity       # the dark vane cavity
    image[radius <= 0.7 * unit] = hub          # the bright bore collar

    image += rng.normal(0, 1.2, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def add_defect(image: np.ndarray, center: tuple[int, int], radius: int = 6,
               depth: int = 90) -> np.ndarray:
    """Darken a small disc, the way a blow hole darkens a machined surface."""
    out = image.copy()
    rows, cols = np.ogrid[:image.shape[0], :image.shape[1]]
    hit = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius ** 2
    out[hit] = np.clip(out[hit].astype(int) - depth, 0, 255).astype(np.uint8)
    return out


def add_marks(image: np.ndarray, count: int = 12, seed: int = 0) -> np.ndarray:
    """Scatter small dark marks over the rim, the kind of detail unique to one casting."""
    rng = np.random.default_rng(seed)
    out = image
    for angle in rng.uniform(0, 2 * np.pi, count):
        out = add_defect(out, rim_point(image.shape[0], angle), radius=2, depth=40)
    return out


def rim_point(size: int = 256, angle: float = 0.0) -> tuple[int, int]:
    """A coordinate on the machined rim, where a defect would actually sit."""
    center = size / 2
    offset = 3.1 * (size / 10.0)
    return (int(center + offset * np.sin(angle)), int(center + offset * np.cos(angle)))


@pytest.fixture
def good_casting() -> np.ndarray:
    return make_casting()


@pytest.fixture
def defective_casting() -> np.ndarray:
    return add_defect(make_casting(), rim_point())


def make_feature_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A small feature table with a genuine signal, for the modeling tests."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)]
    shift = labels * 1.5
    frame = pd.DataFrame({
        "tophat_max": rng.normal(0, 1, n) + shift,
        "tophat_p95": rng.normal(0, 1, n) + shift,
        "response_area_10": rng.normal(0, 1, n) + shift,
        "sobel_mean": rng.normal(0, 1, n) + shift,
        "wedge_worst_gap": rng.normal(0, 1, n) + shift,
        "nuisance_corner_mean": rng.normal(200, 10, n) - labels * 25,
    })
    frame["filename"] = [f"img_{i}.jpeg" for i in range(n)]
    frame["label"] = labels
    frame["group"] = np.arange(n) // 2  # consecutive rows share a casting
    return frame
