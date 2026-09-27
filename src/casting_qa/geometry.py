"""
Locating the casting and putting it in a coordinate system that ignores rotation.

Everything downstream depends on one measurement: where the part is. The
approach is built on what the images contain rather than on what a casting
photograph is usually assumed to contain.

The usual reading is "bright part on a dark background, so threshold it".
These images are the other way round. The background is a light workbench, the
machined faces are brighter still, and the vane cavity in the middle is nearly
black. A global threshold therefore joins the rim to the background, which the
notebook measures with `otsu_largest_share`.

The vane cavity is the reliable anchor instead. It is the darkest large object
in every image and sits at the middle of the part by construction. Radii are
then measured in units of the cavity radius, so the pipeline follows the part
if it is photographed nearer to or further from the camera.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from skimage import filters, measure, morphology, transform

from casting_qa.config import (
    ANCHOR_MIN_AREA,
    ANCHOR_OPEN_RADIUS,
    ANCHOR_THRESHOLD,
    BACKGROUND_INNER_FACTOR,
    DEFECT,
    OK,
    POLAR_ANGULAR_BINS,
    POLAR_RADIAL_BINS,
    RADIAL_PROFILE_BINS,
    RADIAL_PROFILE_EXTENT,
    RIM_INNER_FACTOR,
    RIM_OUTER_FACTOR,
)


def structuring_disk(radius: int, decompose: bool = True) -> np.ndarray | tuple:
    """
    A disk structuring element, by default decomposed into smaller ones.

    Grayscale dilation and erosion take a max or a min over every pixel under
    the element, so a plain radius-21 disk costs 1,373 comparisons per output
    pixel. Dilating by A and then by B equals dilating once by their Minkowski
    sum, so a disk can be replaced by a short sequence of small elements whose
    sum approximates it as an octagon, and the cost falls from the disk's area
    to roughly its perimeter. The notebook measures both the speed-up and how
    far the values move.
    """
    return morphology.disk(radius, decomposition="sequence" if decompose else None)


class Anchor:
    """
    Where the part is: the cavity center, its radius, and the rim band.

    `axis_ratio` is the cavity's minor axis over its major axis, 1.0 for a
    circle. A casting photographed at a tilt appears foreshortened, so this is
    the natural place to look for tilt; the notebook tests whether it works.
    """

    def __init__(self, center: tuple[float, float], radius: float, found: bool,
                 axis_ratio: float = 1.0):
        self.center = center
        self.radius = radius
        self.found = found
        self.axis_ratio = axis_ratio

    @property
    def rim_inner(self) -> float:
        return RIM_INNER_FACTOR * self.radius

    @property
    def rim_outer(self) -> float:
        return RIM_OUTER_FACTOR * self.radius

    def __repr__(self) -> str:
        row, col = self.center
        return f"Anchor(center=({row:.0f}, {col:.0f}), radius={self.radius:.0f}, found={self.found})"


def find_anchor(image: np.ndarray) -> Anchor:
    """
    Locate the vane cavity, and through it the whole part.

    Opening before labeling removes the scattered dark speckle that would
    otherwise fragment the cavity or win the area contest outright. If nothing
    large enough is dark enough, the frame center is returned with `found`
    false, so a failure is visible to the caller instead of silently producing
    a plausible-looking wrong answer.
    """
    fallback = Anchor((image.shape[0] / 2, image.shape[1] / 2), min(image.shape) / 5, found=False)
    dark = morphology.opening(image < ANCHOR_THRESHOLD, structuring_disk(ANCHOR_OPEN_RADIUS))
    labeled = measure.label(dark)
    if labeled.max() == 0:
        return fallback

    region = max(measure.regionprops(labeled), key=lambda r: r.area)
    if region.area < ANCHOR_MIN_AREA:
        return fallback
    ratio = region.axis_minor_length / max(region.axis_major_length, 1e-6)
    return Anchor(region.centroid, float(np.sqrt(region.area / np.pi)), found=True,
                  axis_ratio=float(ratio))


def radius_map(shape: tuple[int, int], anchor: Anchor) -> np.ndarray:
    """Distance from the anchor center, in units of the cavity radius."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    center_row, center_col = anchor.center
    distance = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    return distance / max(anchor.radius, 1e-6)


def rim_mask(image: np.ndarray, anchor: Anchor | None = None) -> np.ndarray:
    """
    The annular band of machined surface where defects appear.

    Bounded inside by the cavity and outside by the part's edge, both in units
    of the cavity radius. This is the only region any model feature is allowed
    to look at, because the background carries a photographic confound that the
    notebook measures in section 6.
    """
    if anchor is None:
        anchor = find_anchor(image)
    scaled = radius_map(image.shape, anchor)
    return (scaled >= RIM_INNER_FACTOR) & (scaled <= RIM_OUTER_FACTOR)


def background_mask(image: np.ndarray, anchor: Anchor | None = None) -> np.ndarray:
    """Workbench only: everything clearly beyond the part's edge."""
    if anchor is None:
        anchor = find_anchor(image)
    return radius_map(image.shape, anchor) >= BACKGROUND_INNER_FACTOR


def otsu_largest_share(image: np.ndarray) -> float:
    """
    Share of the frame taken by the largest bright component under Otsu.

    Otsu picks the threshold that maximizes the between-class variance of the
    intensity histogram, which is exact when the histogram has one mode for the
    part and one for the background. The notebook uses this to show the
    assumption fails here: the component that should be "the part" swallows the
    workbench as well.
    """
    bright = image > filters.threshold_otsu(image)
    labeled = measure.label(bright)
    if labeled.max() == 0:
        return 0.0
    largest = max(measure.regionprops(labeled), key=lambda r: r.area).area
    return float(largest / image.size)


def unwrap(image: np.ndarray, anchor: Anchor | None = None,
           angular_bins: int = POLAR_ANGULAR_BINS,
           radial_bins: int = POLAR_RADIAL_BINS) -> np.ndarray:
    """
    Resample the rim band into polar coordinates.

    The impeller can sit at any angle on the rig, so a defect at the top of one
    photograph and the same defect at the side of another are the same event
    described two ways. Unwrapping turns rotation into a cyclic shift along the
    angle axis, and a statistic that ignores that shift is rotation-invariant by
    construction rather than by hoping the descriptor is.

    Rows are angle over the full turn, columns are radius from the inner to the
    outer rim bound.
    """
    if anchor is None:
        anchor = find_anchor(image)

    polar = transform.warp_polar(
        image.astype(np.float64),
        center=anchor.center,
        radius=anchor.rim_outer,
        output_shape=(angular_bins, int(anchor.rim_outer)),
    )
    band = polar[:, int(anchor.rim_inner):]
    if band.shape[1] < 2:
        return np.zeros((angular_bins, radial_bins))
    return transform.resize(band, (angular_bins, radial_bins), anti_aliasing=True)


def radial_profile(image: np.ndarray, anchor: Anchor | None = None,
                   bins: int = RADIAL_PROFILE_BINS,
                   extent: float = RADIAL_PROFILE_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean intensity in concentric rings, in units of the cavity radius.

    This is the measurement that placed the inspection band. Averaged over a
    class it shows where the rim sits, where the part ends, and, past the part's
    edge, where the two classes diverge for a reason that has nothing to do with
    metal. Returns the ring means and the ring edges.
    """
    if anchor is None:
        anchor = find_anchor(image)
    scaled = radius_map(image.shape, anchor)
    edges = np.linspace(0.0, extent, bins + 1)
    ring_of = np.digitize(scaled, edges) - 1
    inside = (ring_of >= 0) & (ring_of < bins)
    sums = np.bincount(ring_of[inside], weights=image[inside], minlength=bins)
    counts = np.bincount(ring_of[inside], minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(counts > 0, sums / counts, np.nan)
    return values, edges


def class_profiles(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
                   per_class: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mean radial profile per class over the first `per_class` images of each."""
    profiles = {}
    edges = np.linspace(0.0, RADIAL_PROFILE_EXTENT, RADIAL_PROFILE_BINS + 1)
    for label, name in ((DEFECT, "defective"), (OK, "ok")):
        paths = inventory.loc[inventory["label"] == label, "path"].head(per_class)
        profiles[name] = np.nanmean([radial_profile(loader(path))[0] for path in paths], axis=0)
    return profiles, edges
