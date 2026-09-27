"""
Every constant the project uses.

A reviewer asking "where does this number come from" should find the answer
here rather than scattered through the notebook. Nothing downstream hardcodes a
threshold, a kernel size, a split fraction, a sample size or a color.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

DEFECT_DIR = "def_front"
OK_DIR = "ok_front"
IMAGE_GLOB = "*.jpeg"

# 1 means "defective". The primary metric is F1 on this class, because it
# charges for missed defects and for false alarms at once and the brief gives
# no cost ratio between them. Recall on this class is reported beside it as the
# guardrail a factory would watch.
DEFECT = 1
OK = 0
CLASS_NAMES = {OK: "ok", DEFECT: "defective"}

# The brief fixes these: 30% to understand the images, 30% to choose between
# approaches, 40% opened exactly once at the end. The split is built from ten
# group-aware folds, so each fraction has to be a whole number of tenths.
EXPLORE_FRACTION = 0.30
VALIDATION_FRACTION = 0.30
TEST_FRACTION = 0.40
SPLIT_FOLDS = 10

RANDOM_SEED = 42

# Featurization is independent per image, so it runs on every core. Timing is
# measured separately on one core, image by image.
N_JOBS = -1

IMAGE_SIZE = 512
PIXEL_MAX = 255

# Median rather than Gaussian for the first pass: it removes isolated grain and
# JPEG speckle while keeping a step edge a step, and the edge of a chip is the
# signal. The kernel stays at 3 because a pin hole is only a few pixels across.
MEDIAN_KERNEL = 3

# Blow holes run roughly 10-40 px across at 512, pin holes 5-15. A top-hat
# structuring element has to be larger than the defect it is meant to find and
# smaller than the surface curvature it is meant to ignore.
TOPHAT_RADII = (5, 11, 21)

# How much of the band responded, at three severities of the top-hat response,
# and the severity at which responding pixels are grouped into blobs.
RESPONSE_AREA_LEVELS = (0.05, 0.10, 0.20)
BLOB_RESPONSE_LEVEL = 0.10
MIN_BLOB_AREA = 40

# Rotation-invariant uniform LBP. P=8 neighbors at R=1 gives 10 bins, short
# enough to stay interpretable next to the geometric features.
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"

# The band is normalized so its median is 1.0, and LBP needs integer gray
# levels. Mapping [0, 2] onto [0, 255] puts the median at mid-gray, so the
# bright half of the surface keeps its texture instead of saturating at 255.
LBP_INPUT_CEILING = 2.0

# The vane cavity is the anchor for everything geometric. It is the darkest
# large object in every image and sits at the part's center by construction.
ANCHOR_THRESHOLD = 70
ANCHOR_OPEN_RADIUS = 3
ANCHOR_MIN_AREA = 2000

# Radii in units of the cavity radius. The machined rim that carries the
# defects runs from just outside the cavity to the part's edge. Background, for
# the confound probes only, starts past the edge with a margin.
RIM_INNER_FACTOR = 1.15
RIM_OUTER_FACTOR = 2.05
BACKGROUND_INNER_FACTOR = 2.3

# Polar resampling of the rim band. Angular bins set the rotational resolution;
# radial bins set how finely the band is sampled across its width.
POLAR_ANGULAR_BINS = 360
POLAR_RADIAL_BINS = 64

# A defect occupies a few degrees of arc, so it barely moves the mean over the
# whole ring but dominates its own wedge. Twelve wedges is 30 degrees each,
# comfortably wider than a chip and narrow enough that one does not vanish.
WEDGE_COUNT = 12

# Corner patches for the confound probe: 40 px is well clear of the part in
# every image, so these pixels contain workbench and nothing else.
CORNER_PATCH = 40

# Near-duplicate detection. The same casting was photographed more than once at
# different rotations, so duplicates are found on a rotation-free signature:
# the rim band unwrapped at fine angular resolution, stripped of everything
# every casting shares (the circular machining grooves, the slow lighting
# gradient), leaving the marks unique to one piece of metal.
DETAIL_ANGULAR_BINS = 720
DETAIL_RADIAL_BINS = 96
DETAIL_HIGHPASS_SIGMA = 4
DETAIL_LIGHTING_SIGMA = 12

# Similarity above which two photographs are treated as the same casting. It
# sits above the 99.9th percentile of all 844k pairs, and the notebook checks
# it against a free precision estimate: pairs of different labels cannot be the
# same casting.
DUPLICATE_SIMILARITY = 0.6

# Radial profile for the EDA: 24 rings out to 2.8 cavity radii, which reaches
# past the part's edge so the background is visible on the same axis.
RADIAL_PROFILE_BINS = 24
RADIAL_PROFILE_EXTENT = 2.8

# Defect localization. The top-hat response is compared, radius by radius,
# against what sound castings from the exploration split produce there, so the
# circular structure every casting has is subtracted before anything is boxed.
# The reference is a high quantile of sound metal per ring; the quantile and
# the minimum region area are chosen together on the exploration split as the
# pair that boxes the most defective castings while boxing at most the target
# share of sound ones. When a flagged casting has no region clearing that bar,
# its single most anomalous spot is boxed instead and marked as weak evidence.
LOCALIZER_RADIAL_BINS = 36
LOCALIZER_QUANTILES = (0.999, 0.9995, 0.9999)
LOCALIZER_AREA_GRID = (10, 20, 40, 80, 160)
LOCALIZER_FALSE_BOX_TARGET = 0.10
LOCALIZER_SMOOTHING_SIGMA = 1.5
LOCALIZER_SPOT_PERCENTILE = 99.9
MAX_BOXES = 5

# The sound-metal reference is a quantile of millions of response values per
# radius, accumulated as histograms so it never needs all of them in memory.
# The top-hat response of a band normalized to median 1.0 stays well under 2.
RESPONSE_HISTOGRAM_BINS = 4000
RESPONSE_HISTOGRAM_MAX = 2.0

# The rule-based baseline. Each feature has a physical reading: how dark the
# worst pit is, how much of the band responded, how rough the surface gradient
# is, and how far the worst angular wedge sits from the rest of the ring.
RULE_FEATURES = (
    "tophat_max",
    "tophat_p95",
    "response_area_10",
    "sobel_mean",
    "wedge_worst_gap",
)
RULE_GRID_POINTS = 60

# The alternative the rule replaced: one alarm per feature, each set at this
# percentile of the sound castings, firing if any alarm fires.
OR_GATE_PERCENTILE = 97.5

RANDOM_FOREST_GRID = {
    "n_estimators": (200, 400),
    "max_depth": (None, 8, 16),
    "min_samples_leaf": (1, 2, 4),
    "max_features": ("sqrt", 0.5),
}

# Both axes extend at least one step past where the winner landed, so the
# choice is not pinned to the edge of the grid.
SVM_GRID = {
    "C": (0.1, 1.0, 10.0, 100.0, 1000.0),
    "gamma": ("scale", 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
}

CV_FOLDS = 5
CV_REPEATS = 5

# Exposure matching: a sound and a defective casting are paired when their
# corner brightness is within this many gray levels.
MATCH_TOLERANCE = 1.0
MATCHED_CONTROL_DRAWS = 20

CONFIDENCE_LEVEL = 0.95

# Sample sizes for measurements that do not need every image.
SAMPLE_GRID_PER_CLASS = 4
PROFILE_SAMPLE_PER_CLASS = 40
OTSU_CHECK_SAMPLE = 60
DECOMPOSITION_CHECK_SAMPLE = 8
TIMING_SAMPLE = 100
TIMING_REPEATS = 3
ROBUSTNESS_SAMPLE = 120
DETECTION_GRID_FLAGGED = 8
DETECTION_GRID_PASSED = 4
DUPLICATE_EXAMPLES = 3

# Perturbations for the robustness check: rotation in degrees, gain and gamma
# as multipliers and exponents, and JPEG re-encoding quality.
ROBUSTNESS_PERTURBATIONS = (
    ("rotate", 10.0),
    ("rotate", 37.0),
    ("rotate", 90.0),
    ("gain", 0.9),
    ("gain", 1.1),
    ("gamma", 0.9),
    ("gamma", 1.1),
    ("jpeg", 75.0),
    ("jpeg", 50.0),
)

PALETTE = {
    "ok": "#4C72B0",
    "defective": "#C44E52",
    "neutral": "#8C8C8C",
    "highlight": "#DD8452",
}
FIGURE_DPI = 120
