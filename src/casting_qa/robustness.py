"""
How the shipped classifier holds up when the photograph changes.

The brief asks for robust classification, and the review asks how variation in
image quality and lighting is handled. Normalizing by the band's median cancels
a pure change of exposure in theory; this module measures what happens in
practice under rotation, gain, gamma and JPEG re-encoding, on the validation
split so the sealed test set stays opened once.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage import transform

from casting_qa.config import DEFECT, N_JOBS, OK, PIXEL_MAX, ROBUSTNESS_PERTURBATIONS
from casting_qa.features import describe


def perturb(image: np.ndarray, kind: str, value: float) -> np.ndarray:
    """
    One controlled change to a photograph.

    rotate: degrees about the frame center, edges filled with the nearest
    pixel. gain: every pixel multiplied, as a longer exposure would. gamma: a
    non-linear tone curve, which a median division cannot cancel. jpeg:
    re-encoded at the given quality, adding compression artifacts.
    """
    if kind == "rotate":
        rotated = transform.rotate(image, value, mode="edge", preserve_range=True)
        return np.clip(np.round(rotated), 0, PIXEL_MAX).astype(np.uint8)
    if kind == "gain":
        return np.clip(np.round(image.astype(np.float64) * value), 0, PIXEL_MAX).astype(np.uint8)
    if kind == "gamma":
        scaled = (image.astype(np.float64) / PIXEL_MAX) ** value
        return np.clip(np.round(scaled * PIXEL_MAX), 0, PIXEL_MAX).astype(np.uint8)
    if kind == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=int(value))
        buffer.seek(0)
        with Image.open(buffer) as reloaded:
            return np.asarray(reloaded.convert("L"), dtype=np.uint8)
    raise ValueError(f"unknown perturbation {kind!r}")


def _perturbed_features(image: np.ndarray, perturbations) -> list[dict[str, float]]:
    return [describe(perturb(image, kind, value)) for kind, value in perturbations]


def robustness_table(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
                     model, columns: list[str], baseline: np.ndarray,
                     perturbations=ROBUSTNESS_PERTURBATIONS,
                     n_jobs: int = N_JOBS) -> pd.DataFrame:
    """
    Accuracy and flipped decisions under each perturbation.

    `baseline` is the model's decision on the unmodified images, so "flipped"
    counts decisions that changed because of the perturbation alone, whether
    the change was toward the truth or away from it.
    """
    per_image = Parallel(n_jobs=n_jobs)(
        delayed(_perturbed_features)(loader(p), perturbations) for p in inventory["path"])
    labels = inventory["label"].to_numpy()
    rows = []
    for position, (kind, value) in enumerate(perturbations):
        frame = pd.DataFrame([features[position] for features in per_image])[columns]
        predicted = model.predict(frame)
        rows.append({
            "perturbation": f"{kind} {value:g}",
            "accuracy": round(float((predicted == labels).mean()), 3),
            "flipped": int((predicted != baseline).sum()),
            "missed_defects": int(np.sum((labels == DEFECT) & (predicted == OK))),
            "false_alarms": int(np.sum((labels == OK) & (predicted == DEFECT))),
        })
    return pd.DataFrame(rows)
