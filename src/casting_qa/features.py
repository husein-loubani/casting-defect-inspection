"""
Describing a casting with numbers a classifier can use.

Two rules govern everything here, and both come from measurements.

Every model feature is computed strictly inside the part. The two class
folders were photographed under different lighting (notebook section 5), so any
feature whose support touches background could score well by reading the lamp
rather than the metal. None of them do, and the test suite repaints the
background to prove it.

Everything angular is computed on the polar unwrapping. The impeller sits at an
arbitrary angle on the rig, so a descriptor that changes with orientation would
encode how the part was placed rather than whether it is sound.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma

from casting_qa.config import (
    BLOB_RESPONSE_LEVEL,
    CORNER_PATCH,
    GAUSSIAN_ABLATION_SIGMA,
    LBP_INPUT_CEILING,
    LBP_METHOD,
    LBP_POINTS,
    LBP_RADIUS,
    MEDIAN_KERNEL,
    MIN_BLOB_AREA,
    N_JOBS,
    PIXEL_MAX,
    RESPONSE_AREA_LEVELS,
    TOPHAT_RADII,
    WEDGE_COUNT,
)
from casting_qa.geometry import (
    Anchor,
    background_mask,
    find_anchor,
    rim_mask,
    structuring_disk,
    unwrap,
)


def denoise(image: np.ndarray, kernel: int = MEDIAN_KERNEL) -> np.ndarray:
    """
    Median filter, chosen over Gaussian for what the next stages measure.

    A median replaces each pixel with the middle value of its window. An
    isolated speck, whether sensor grain or a JPEG artifact, is an extreme of
    its window and is discarded outright, while a step edge survives because
    the middle value on either side of it is still that side's level. A
    Gaussian is a weighted average instead: it cannot discard an outlier, only
    spread it, and it rounds every edge. The edge of a chip is the signal here,
    so the median is the filter that keeps it.
    """
    return ndimage.median_filter(image, size=kernel)


def normalize_to_part(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """
    Rescale intensity using the part's own machined surface as the reference.

    Dividing by the median of the inspected band means a feature measures how a
    pixel compares to the metal around it, not how bright the photograph
    happened to be. A pure change of exposure multiplies every pixel by the
    same factor and cancels exactly; the notebook's robustness check measures
    how well that holds for the non-linear changes it cannot cancel.
    """
    scaled = image.astype(np.float64) / PIXEL_MAX
    if not region.any():
        return scaled
    reference = np.median(scaled[region])
    return scaled / max(reference, 1e-6)


def isolate(normalized: np.ndarray, region: np.ndarray) -> np.ndarray:
    """
    Replace everything outside the band with the band's own median.

    Any operator with a spatial footprint (a top-hat, a Sobel kernel, an LBP
    neighborhood) reads pixels around the one it writes. At the edge of the
    band those neighbors are background, and background is the confounded
    quantity on this dataset. Filling first means the footprint sees flat
    neutral surface where the background used to be, so the only thing a
    feature can respond to is metal.
    """
    if not region.any():
        return normalized
    return np.where(region, normalized, np.median(normalized[region]))


def defect_response(normalized: np.ndarray, region: np.ndarray,
                    radii: tuple[int, ...] = TOPHAT_RADII,
                    decompose: bool = True) -> np.ndarray:
    """
    A map that is bright where the surface is locally darker than its surroundings.

    The black top-hat is the closing of the image minus the image. Closing is a
    dilation (local max) followed by an erosion (local min) with the same
    element, and it fills in any dark feature the element cannot fit inside,
    while leaving larger dark regions and all bright structure alone. Subtracting
    the original leaves exactly the filled-in amount: how much darker a pixel is
    than the metal around it, at the scale of the element. A blow hole or a chip
    is exactly that. Unlike a global threshold it ignores the slow illumination
    gradient across the part, because a gradient is not a pit at any scale.

    Three radii are combined because the defects are not one size, and the
    pixelwise maximum keeps whichever scale responded.
    """
    isolated = isolate(normalized, region)
    responses = [morphology.black_tophat(isolated, structuring_disk(r, decompose)) for r in radii]
    return np.where(region, np.maximum.reduce(responses), 0.0)


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Summary statistics of whatever survived a mask."""
    if values.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "std", "p95", "p99", "max")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": float(values.max()),
    }


def wedge_features(polar: np.ndarray, count: int = WEDGE_COUNT) -> dict[str, float]:
    """
    Order statistics over angular wedges of the unwrapped rim.

    A defect occupies a few degrees of arc, so it barely moves the mean over the
    whole ring but dominates its own wedge. The angular profile is smoothed with
    a wrapped sliding window one wedge wide rather than cut at fixed wedge
    boundaries: fixed boundaries are not rotation-invariant, because a defect
    inside one wedge at one orientation straddles two at another. With the
    sliding window a rotation is a roll of the profile, and every order
    statistic of a rolled profile is unchanged.
    """
    if polar.size == 0:
        return {f"wedge_{k}": 0.0 for k in ("min", "max", "range", "std", "worst_gap")}

    profile = polar.mean(axis=1)
    width = max(len(profile) // count, 1)
    means = ndimage.uniform_filter1d(profile, size=width, mode="wrap")
    return {
        "wedge_min": float(means.min()),
        "wedge_max": float(means.max()),
        "wedge_range": float(means.max() - means.min()),
        "wedge_std": float(means.std()),
        "wedge_worst_gap": float(means.mean() - means.min()),
    }


def texture_input(isolated: np.ndarray, ceiling: float = LBP_INPUT_CEILING) -> np.ndarray:
    """
    The normalized band as 8-bit gray levels for LBP.

    The band median sits at 1.0 after normalization, so [0, ceiling] is mapped
    onto [0, 255] and the median lands mid-gray. Scaling by 255 directly would
    send every pixel brighter than the median to 255, and LBP cannot see texture
    on a saturated plateau because every comparison ties.
    """
    return np.clip(isolated / ceiling * PIXEL_MAX, 0, PIXEL_MAX).astype(np.uint8)


def texture_features(image: np.ndarray, region: np.ndarray) -> dict[str, float]:
    """
    Rotation-invariant uniform local binary patterns over the inspected band.

    For each pixel, its 8 neighbors on a radius-1 circle are compared with it
    and each comparison becomes one bit, so the code records which directions
    are brighter than the center. A code is "uniform" when the circular bit
    string changes between 0 and 1 at most twice: those are the codes of
    flat patches, edges, lines and corners, and they are the large majority on
    real surfaces. Taking the minimum over all circular rotations of the bit
    string makes the code independent of orientation, which leaves P + 1 = 9
    uniform classes (numbered by how many neighbors are brighter) plus one bin
    for every non-uniform code. A pitted or scratched surface moves weight out
    of the flat bins into the edge and corner bins, and the histogram over the
    band records that.
    """
    bins = LBP_POINTS + 2
    if not region.any():
        return {f"lbp_{i}": 0.0 for i in range(bins)}

    # Sample only where the whole neighborhood is inside the band. The erosion
    # covers the LBP radius plus the median filter's reach, because the denoise
    # runs before the band is known and its window pulls in background at the edge.
    inner = morphology.erosion(region, structuring_disk(LBP_RADIUS + MEDIAN_KERNEL // 2 + 1))
    if not inner.any():
        inner = region

    codes = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    histogram, _ = np.histogram(codes[inner], bins=bins, range=(0, bins), density=True)
    return {f"lbp_{i}": float(v) for i, v in enumerate(histogram)}


def nuisance_features(image: np.ndarray, anchor: Anchor,
                      patch: int = CORNER_PATCH) -> dict[str, float]:
    """
    The confound, measured on purpose, from workbench pixels only.

    Four corner patches and everything clearly beyond the part's edge. No pixel
    of the casting enters any of these, so a classifier fitted on them alone
    measures how far the lighting difference between the two folders goes
    without any metal. Nothing here is fed to the shipped model.
    """
    corners = np.concatenate([
        image[:patch, :patch].ravel(), image[:patch, -patch:].ravel(),
        image[-patch:, :patch].ravel(), image[-patch:, -patch:].ravel(),
    ]).astype(np.float64)
    background = image[background_mask(image, anchor)].astype(np.float64)
    if background.size == 0:
        background = corners
    return {
        "nuisance_corner_mean": float(corners.mean()),
        "nuisance_background_mean": float(background.mean()),
        "nuisance_background_std": float(background.std()),
    }


class Stages(NamedTuple):
    """The intermediate images of the pipeline, computed once and shared."""

    clean: np.ndarray
    anchor: Anchor
    band: np.ndarray
    normalized: np.ndarray
    isolated: np.ndarray
    response: np.ndarray


IDENTIFIERS = {"filename", "label", "group", "augmented"}


def prefilter(image: np.ndarray, method: str = "median") -> np.ndarray:
    """The first-pass filter: the shipped median, or the Gaussian it was tested against."""
    if method == "median":
        return denoise(image)
    if method == "gaussian":
        blurred = ndimage.gaussian_filter(image.astype(np.float64), GAUSSIAN_ABLATION_SIGMA)
        return np.clip(np.round(blurred), 0, PIXEL_MAX).astype(np.uint8)
    raise ValueError(f"unknown prefilter {method!r}")


def pipeline_stages(image: np.ndarray, method: str = "median") -> Stages:
    """
    Denoise, locate the part, restrict to the band, normalize, and respond.

    Feature extraction and defect localization both start from these, so the
    expensive top-hat runs once per image however many consumers there are.
    """
    clean = prefilter(image, method)
    anchor = find_anchor(clean)
    band = rim_mask(clean, anchor)
    normalized = normalize_to_part(clean, band)
    return Stages(clean, anchor, band, normalized, isolate(normalized, band),
                  defect_response(normalized, band))


def features_from(stages: Stages, image: np.ndarray | None = None,
                  include_nuisance: bool = False) -> dict[str, float]:
    """Every model feature, measured on precomputed pipeline stages."""
    band, response, isolated = stages.band, stages.response, stages.isolated
    # Sobel is a 3x3 derivative: a central difference across the direction of
    # interest, smoothed along the other. Its magnitude is large where intensity
    # changes fast, which is what the rim of a chip and a rough surface do.
    gradient = np.where(band, filters.sobel(isolated), 0.0)

    features: dict[str, float] = {}
    features.update(_stats(response[band], "tophat"))
    features.update(_stats(gradient[band], "sobel"))
    features.update(_stats(stages.normalized[band], "intensity"))
    features.update(wedge_features(unwrap(isolated, stages.anchor)))
    features.update(texture_features(texture_input(isolated), band))

    # How much of the band responded, as a fraction of the band so a larger
    # part does not look worse than a small one.
    area = max(band.sum(), 1)
    for level in RESPONSE_AREA_LEVELS:
        features[f"response_area_{round(level * 100)}"] = float((response[band] > level).sum()) / area

    blobs = measure.label(response > BLOB_RESPONSE_LEVEL)
    regions = [r for r in measure.regionprops(blobs) if r.area >= MIN_BLOB_AREA]
    features["blob_count"] = float(len(regions))
    features["blob_max_area"] = float(max((r.area for r in regions), default=0.0))
    features["blob_total_area"] = float(sum(r.area for r in regions))

    if include_nuisance:
        if image is None:
            raise ValueError("the nuisance probes read the raw image; pass it")
        features.update(nuisance_features(image, stages.anchor))
    return features


def describe(image: np.ndarray, include_nuisance: bool = False,
             method: str = "median") -> dict[str, float]:
    """
    Every feature for one image.

    `include_nuisance` adds the confound probes, which belong in the EDA and the
    control experiments and never in the model. `method` swaps the prefilter for
    the ablation and is otherwise left alone.
    """
    return features_from(pipeline_stages(image, method), image, include_nuisance)


def attach_identifiers(rows: list[dict[str, float]], inventory: pd.DataFrame) -> pd.DataFrame:
    """A feature table with the inventory's filename, label and group (if any) attached."""
    frame = pd.DataFrame(rows)
    frame.insert(0, "filename", inventory["filename"].to_numpy())
    frame["label"] = inventory["label"].to_numpy()
    if "group" in inventory.columns:
        frame["group"] = inventory["group"].to_numpy()
    return frame


def describe_batch(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
                   include_nuisance: bool = False, method: str = "median",
                   n_jobs: int = N_JOBS) -> pd.DataFrame:
    """
    Features for a whole partition, one image per worker.

    `loader` is injected rather than imported so this module stays free of
    file-system concerns and can be tested on synthetic images.
    """
    rows = Parallel(n_jobs=n_jobs)(
        delayed(describe)(loader(path), include_nuisance, method) for path in inventory["path"])
    return attach_identifiers(rows, inventory)


def feature_columns(frame: pd.DataFrame, include_nuisance: bool = False) -> list[str]:
    """The model's input columns: everything numeric that is not an identifier."""
    columns = [c for c in frame.columns if c not in IDENTIFIERS]
    if not include_nuisance:
        columns = [c for c in columns if not c.startswith("nuisance_")]
    return columns


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    """
    Image-quality measurements for the EDA, taken on the rim band.

    Sharpness is the variance of the Laplacian: a focused image has strong
    second derivatives at its edges and a blurred one does not. Noise is the
    wavelet-based estimate of the Gaussian noise standard deviation. Contrast is
    the band's standard deviation. All three are measured inside the part, so
    they describe the photograph of the metal rather than the workbench.
    """
    anchor = find_anchor(denoise(image))
    band = rim_mask(image, anchor)
    values = image.astype(np.float64) / PIXEL_MAX
    laplacian = filters.laplace(values)
    return {
        "sharpness": float(laplacian[band].var()) if band.any() else 0.0,
        "noise_sigma": float(estimate_sigma(values)) * PIXEL_MAX,
        "band_brightness": float(values[band].mean()) * PIXEL_MAX if band.any() else 0.0,
        "band_contrast": float(values[band].std()) * PIXEL_MAX if band.any() else 0.0,
        "anchor_found": float(anchor.found),
        "anchor_row": float(anchor.center[0]),
        "anchor_col": float(anchor.center[1]),
        "anchor_radius": float(anchor.radius),
        "anchor_axis_ratio": float(anchor.axis_ratio),
    }


def quality_batch(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
                  n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Quality metrics for every image of a partition."""
    rows = Parallel(n_jobs=n_jobs)(delayed(quality_metrics)(loader(p)) for p in inventory["path"])
    return attach_identifiers(rows, inventory)


def decomposition_check(paths, loader: Callable[[str], np.ndarray]) -> dict[str, float]:
    """
    Time the three-radius top-hat with plain and decomposed disks, and measure
    how far the decomposition moves the response.
    """
    timings = {True: [], False: []}
    peak_change, pixel_change = [], []
    for path in paths:
        stages = pipeline_stages(loader(path))
        outputs = {}
        for decompose in (True, False):
            started = time.perf_counter()
            outputs[decompose] = defect_response(stages.normalized, stages.band, decompose=decompose)
            timings[decompose].append(time.perf_counter() - started)
        plain, octagon = outputs[False][stages.band], outputs[True][stages.band]
        peak_change.append(abs(octagon.max() - plain.max()) / plain.max())
        pixel_change.append(np.abs(octagon - plain).mean() / plain.mean())
    return {
        "images": len(peak_change),
        "plain_ms": round(1000 * float(np.mean(timings[False])), 1),
        "decomposed_ms": round(1000 * float(np.mean(timings[True])), 1),
        "speedup": round(float(np.mean(timings[False]) / np.mean(timings[True])), 1),
        "max_peak_change_%": round(100 * float(np.max(peak_change)), 2),
        "mean_pixel_change_%": round(100 * float(np.mean(pixel_change)), 2),
    }
