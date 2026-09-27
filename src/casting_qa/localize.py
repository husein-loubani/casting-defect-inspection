"""
Putting a box on the defect, and the end-to-end inspection of one casting.

The top-hat response is bright wherever the surface is locally darker than its
surroundings, and on this part that includes structure every casting has: the
step down into the vane cavity and the chamfer at the outer edge both read as
"darker than the metal beside it" at some radius. Thresholding the raw response
therefore boxes those rings on sound and defective castings alike.

The fix is a reference for what sound metal produces at each radius. The
response of the sound castings in the exploration split is pooled per radial
ring (in units of the cavity radius, so it transfers between images), and a
high quantile of it is the level ordinary structure reaches there. A defect is
whatever exceeds that level. The rings subtract out because they are present
in the reference; a chip is not, so it remains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage
from skimage import measure

from casting_qa.config import (
    DEFECT,
    LOCALIZER_AREA_GRID,
    LOCALIZER_FALSE_BOX_TARGET,
    LOCALIZER_QUANTILES,
    LOCALIZER_RADIAL_BINS,
    LOCALIZER_SMOOTHING_SIGMA,
    LOCALIZER_SPOT_PERCENTILE,
    MAX_BOXES,
    N_JOBS,
    OK,
    RESPONSE_HISTOGRAM_BINS,
    RESPONSE_HISTOGRAM_MAX,
    RIM_INNER_FACTOR,
    RIM_OUTER_FACTOR,
)
from casting_qa.features import Stages, features_from, pipeline_stages
from casting_qa.geometry import radius_map

Box = tuple[int, int, int, int]
Region = tuple[Box, float]
OUTSIDE_BAND = -1.0


@dataclass
class Localizer:
    """The sound-metal reference per ring, and the settings chosen on explore."""

    edges: np.ndarray
    curve: np.ndarray
    quantile: float
    min_area: int
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)


def ring_edges(bins: int = LOCALIZER_RADIAL_BINS) -> np.ndarray:
    """Ring boundaries across the inspected band, in units of the cavity radius."""
    return np.linspace(RIM_INNER_FACTOR, RIM_OUTER_FACTOR, bins + 1)


def ring_index(stages: Stages, edges: np.ndarray) -> np.ndarray:
    """Which ring each pixel falls in."""
    scaled = radius_map(stages.band.shape, stages.anchor)
    return np.clip(np.digitize(scaled, edges) - 1, 0, len(edges) - 2)


def smoothed_response(stages: Stages, sigma: float = LOCALIZER_SMOOTHING_SIGMA) -> np.ndarray:
    """A light blur so a pitted patch reads as one region rather than speckle."""
    return ndimage.gaussian_filter(stages.response, sigma)


def excess_map(stages: Stages, curve: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """How far each band pixel's response sits above the sound-metal reference for its ring."""
    excess = smoothed_response(stages) - curve[ring_index(stages, edges)]
    return np.where(stages.band, excess, OUTSIDE_BAND)


def ring_histograms(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts of smoothed response values per ring, for one image."""
    stages = pipeline_stages(image)
    rings = ring_index(stages, edges)[stages.band]
    values = np.clip(smoothed_response(stages)[stages.band], 0, RESPONSE_HISTOGRAM_MAX - 1e-9)
    value_bins = (values / RESPONSE_HISTOGRAM_MAX * RESPONSE_HISTOGRAM_BINS).astype(int)
    counts = np.bincount(rings * RESPONSE_HISTOGRAM_BINS + value_bins,
                         minlength=(len(edges) - 1) * RESPONSE_HISTOGRAM_BINS)
    return counts.reshape(len(edges) - 1, RESPONSE_HISTOGRAM_BINS)


def quantile_curve(histograms: np.ndarray, quantile: float) -> np.ndarray:
    """The response level below which `quantile` of sound pixels fall, ring by ring."""
    cumulative = np.cumsum(histograms, axis=1)
    totals = np.maximum(cumulative[:, -1:], 1)
    first_above = (cumulative / totals >= quantile).argmax(axis=1)
    return (first_above + 1) * RESPONSE_HISTOGRAM_MAX / RESPONSE_HISTOGRAM_BINS


def find_regions(excess: np.ndarray, min_area: int, max_boxes: int = MAX_BOXES) -> list[Region]:
    """Connected regions of positive excess, strongest peak first."""
    labeled = measure.label(excess > 0)
    regions = [r for r in measure.regionprops(labeled, intensity_image=excess) if r.area >= min_area]
    regions.sort(key=lambda r: r.intensity_max, reverse=True)
    return [(tuple(int(v) for v in r.bbox), float(r.intensity_max)) for r in regions[:max_boxes]]


def most_anomalous_spot(excess: np.ndarray, band: np.ndarray,
                        percentile: float = LOCALIZER_SPOT_PERCENTILE) -> list[Region]:
    """
    The single region around the band's highest excess, however small.

    Used only when a flagged casting has nothing that clears the reference, so
    the operator is still shown where the evidence is strongest, marked weak.
    """
    if not band.any():
        return []
    level = np.percentile(excess[band], percentile)
    labeled = measure.label((excess >= level) & band)
    peak = np.unravel_index(np.argmax(np.where(band, excess, -np.inf)), excess.shape)
    region = next(r for r in measure.regionprops(labeled, intensity_image=excess)
                  if labeled[peak] == r.label)
    return [(tuple(int(v) for v in region.bbox), float(region.intensity_max))]


def _region_areas(image: np.ndarray, curves: dict[float, np.ndarray],
                  edges: np.ndarray) -> dict[float, list[int]]:
    """Pixel area of every region exceeding each candidate reference, for one image."""
    stages = pipeline_stages(image)
    areas = {}
    for quantile, curve in curves.items():
        labeled = measure.label(excess_map(stages, curve, edges) > 0)
        areas[quantile] = [int(r.area) for r in measure.regionprops(labeled)]
    return areas


def fit_localizer(explore: pd.DataFrame, loader: Callable[[str], np.ndarray],
                  quantiles: tuple[float, ...] = LOCALIZER_QUANTILES,
                  area_grid: tuple[int, ...] = LOCALIZER_AREA_GRID,
                  target: float = LOCALIZER_FALSE_BOX_TARGET,
                  n_jobs: int = N_JOBS) -> Localizer:
    """
    Fit the references on sound explore castings and choose the settings.

    Every (quantile, minimum area) pair is scored on the exploration split by
    the share of sound castings it boxes and the share of defective castings it
    boxes. The pair chosen boxes the most defective castings among those that
    box at most `target` of the sound ones. Validation and test judge it.
    """
    edges = ring_edges()
    sound_paths = explore.loc[explore["label"] == OK, "path"]
    histograms = sum(Parallel(n_jobs=n_jobs)(delayed(ring_histograms)(loader(p), edges)
                                             for p in sound_paths))
    curves = {q: quantile_curve(histograms, q) for q in quantiles}

    areas = Parallel(n_jobs=n_jobs)(delayed(_region_areas)(loader(p), curves, edges)
                                    for p in explore["path"])
    labels = explore["label"].to_numpy()
    rows = []
    for quantile in quantiles:
        for min_area in area_grid:
            boxed = np.array([any(a >= min_area for a in image_areas[quantile]) for image_areas in areas])
            rows.append({"quantile": quantile, "min_area": min_area,
                         "sound_boxed": round(float(boxed[labels == OK].mean()), 3),
                         "defective_boxed": round(float(boxed[labels == DEFECT].mean()), 3)})
    selection = pd.DataFrame(rows)
    allowed = selection[selection["sound_boxed"] <= target]
    if allowed.empty:
        allowed = selection[selection["sound_boxed"] == selection["sound_boxed"].min()]
    chosen = allowed.sort_values(["defective_boxed", "quantile", "min_area"],
                                 ascending=[False, False, False]).iloc[0]
    return Localizer(edges, curves[chosen["quantile"]], float(chosen["quantile"]),
                     int(chosen["min_area"]), selection)


def locate(stages: Stages, localizer: Localizer) -> dict[str, list[Region]]:
    """
    Regions that clear the sound-metal reference, or else the most anomalous spot.

    A casting the classifier flags always gets a box, and the caller can tell
    whether it is backed by the fitted reference ("strong") or is only the best
    candidate available ("weak").
    """
    excess = excess_map(stages, localizer.curve, localizer.edges)
    strong = find_regions(excess, localizer.min_area)
    if strong:
        return {"strong": strong, "weak": []}
    return {"strong": [], "weak": most_anomalous_spot(excess, stages.band)}


def _box_summary(image: np.ndarray, localizer: Localizer) -> dict[str, float]:
    found = locate(pipeline_stages(image), localizer)
    sides = [max(b[2] - b[0], b[3] - b[1]) for b, _ in found["strong"]]
    return {"strong_boxes": len(found["strong"]), "largest_strong_side": max(sides, default=0)}


def box_rates(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
              localizer: Localizer, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """
    For each class, how often the fitted reference produces a box, and how big.

    Without defect-location labels this is the measurable part of localization:
    the false-box rate on sound castings, the share of defective castings that
    yield a region, and the box size, where small means local.
    """
    summaries = pd.DataFrame(Parallel(n_jobs=n_jobs)(
        delayed(_box_summary)(loader(p), localizer) for p in inventory["path"]))
    summaries["label"] = inventory["label"].to_numpy()
    rows = []
    for label, part in summaries.groupby("label"):
        boxed = part[part["strong_boxes"] > 0]
        rows.append({
            "class": "defective" if label == DEFECT else "ok",
            "images": len(part),
            "boxed_%": round(100 * len(boxed) / len(part), 1),
            "boxes_per_boxed_image": round(float(boxed["strong_boxes"].mean()), 2) if len(boxed) else 0.0,
            "median_largest_side_px": float(boxed["largest_strong_side"].median()) if len(boxed) else 0.0,
        })
    return pd.DataFrame(rows)


def inspect_casting(image: np.ndarray, model, columns: list[str], localizer: Localizer) -> dict:
    """
    The whole product for one photograph: decision and boxes.

    Features and localization share one pass of the pipeline. Boxes are drawn
    only when the shipped classifier calls the part defective, as the brief
    asks, so a sound decision costs no localization at all.
    """
    stages = pipeline_stages(image)
    decision = int(model.predict(pd.DataFrame([features_from(stages)])[columns])[0])
    boxes = locate(stages, localizer) if decision == DEFECT else {"strong": [], "weak": []}
    return {"decision": decision, "boxes": boxes}


def inspection_gallery(names, path_of: pd.Series, loader: Callable[[str], np.ndarray],
                       model, columns: list[str], localizer: Localizer) -> list[tuple]:
    """Image, boxes and caption for each named casting, as the shipped product sees it."""
    items = []
    for name in names:
        image = loader(path_of[name])
        result = inspect_casting(image, model, columns, localizer)
        if result["decision"] == DEFECT:
            evidence = "strong" if result["boxes"]["strong"] else "weak"
            caption = f"{name}\nflagged, {evidence} evidence"
        else:
            caption = f"{name}\npassed, no boxes"
        items.append((image, result["boxes"], caption))
    return items
