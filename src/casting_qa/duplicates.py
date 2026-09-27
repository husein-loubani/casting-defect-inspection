"""
Finding the same casting photographed more than once.

Exact duplicates are caught by hashing decoded pixels. This module catches the
harder case: two different photographs of one physical part, taken at
different rotations, with different JPEG noise. Such a pair is not a
duplicate file, but if one lands in the training data and the other in the
test data, the test score partly measures recognition of a part already seen.

The signature is built to keep only what is unique to one piece of metal. The
rim band is unwrapped to polar coordinates in units of the cavity radius, which
makes it independent of scale, and a rotation becomes a cyclic shift along the
angle axis. Then three things every casting shares are removed: slow shading
(a spatial high-pass), the circular machining grooves (the mean over angle at
each radius), and the fixed lighting gradient of the rig (a second high-pass
along the angle). What remains is scratches, chips and casting marks.

Two signatures are compared by their normalized cross-correlation at the best
cyclic shift, computed for every shift at once with an FFT along the angle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from casting_qa.config import (
    DETAIL_ANGULAR_BINS,
    DETAIL_HIGHPASS_SIGMA,
    DETAIL_LIGHTING_SIGMA,
    DETAIL_RADIAL_BINS,
    DUPLICATE_SIMILARITY,
    N_JOBS,
)
from casting_qa.features import denoise, normalize_to_part
from casting_qa.geometry import find_anchor, rim_mask, unwrap


def detail_signature(image: np.ndarray) -> np.ndarray:
    """The rim band's unique surface detail, angle by radius, zero mean and unit variance."""
    clean = denoise(image)
    anchor = find_anchor(clean)
    band = rim_mask(clean, anchor)
    polar = unwrap(normalize_to_part(clean, band), anchor,
                   angular_bins=DETAIL_ANGULAR_BINS, radial_bins=DETAIL_RADIAL_BINS)
    detail = polar - ndimage.gaussian_filter(polar, DETAIL_HIGHPASS_SIGMA, mode=("wrap", "nearest"))
    detail = detail - detail.mean(axis=0, keepdims=True)
    detail = detail - ndimage.gaussian_filter1d(detail, DETAIL_LIGHTING_SIGMA, axis=0, mode="wrap")
    return ((detail - detail.mean()) / (detail.std() + 1e-9)).astype(np.float32)


def signatures(paths, loader: Callable[[str], np.ndarray], n_jobs: int = N_JOBS) -> np.ndarray:
    """Signatures for a list of files, stacked into (n, angles, radii)."""
    return np.stack(Parallel(n_jobs=n_jobs)(delayed(detail_signature)(loader(p)) for p in paths))


def pairwise_similarity(stack: np.ndarray) -> np.ndarray:
    """
    Best-shift normalized cross-correlation for every pair, upper triangle filled.

    Correlation at every cyclic shift is the inverse FFT of one spectrum times
    the conjugate of the other. The sum over radius is taken in the frequency
    domain, where it is linear, so each image is compared with all later ones
    in a single inverse transform.
    """
    count, angles, radii = stack.shape
    spectra = np.fft.rfft(stack, axis=1).astype(np.complex64)
    similarity = np.zeros((count, count), dtype=np.float32)
    for i in range(count - 1):
        cross = np.einsum("fr,nfr->nf", spectra[i], np.conj(spectra[i + 1:]), optimize=True)
        correlation = np.fft.irfft(cross, n=angles, axis=1) / (angles * radii)
        similarity[i, i + 1:] = correlation.max(axis=1)
    return similarity


def similar_pairs(similarity: np.ndarray, filenames, labels,
                  threshold: float = DUPLICATE_SIMILARITY) -> pd.DataFrame:
    """Every pair at or above the threshold, with both labels attached."""
    first, second = np.nonzero(np.triu(similarity, k=1) >= threshold)
    filenames, labels = np.asarray(filenames), np.asarray(labels)
    return pd.DataFrame({
        "first": filenames[first], "second": filenames[second],
        "first_label": labels[first], "second_label": labels[second],
        "similarity": similarity[first, second],
    }).sort_values("similarity", ascending=False).reset_index(drop=True)


def duplicate_groups(similarity: np.ndarray, threshold: float = DUPLICATE_SIMILARITY) -> np.ndarray:
    """
    A group id per image: connected components of the "same casting" graph.

    If A matches B and B matches C, all three are held together even when A and
    C do not match directly, because splitting C away from A would still put
    the same part on both sides through B.
    """
    first, second = np.nonzero(np.triu(similarity, k=1) >= threshold)
    count = similarity.shape[0]
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(count, count))
    _, groups = connected_components(graph, directed=False)
    return groups


def threshold_sweep(similarity: np.ndarray, labels, thresholds) -> pd.DataFrame:
    """
    How many pairs each threshold accepts, and how many of them cross labels.

    A sound and a defective photograph cannot show the same casting, so the
    cross-label share is a free estimate of the matcher's error rate. Random
    pairs disagree on the label about half the time (0.6^2 + 0.4^2 = 0.52 agree),
    so the false-match share is roughly the cross-label share divided by 0.48.
    """
    labels = np.asarray(labels)
    upper = np.triu_indices(similarity.shape[0], k=1)
    scores = similarity[upper]
    agree_by_chance = sum(share ** 2 for share in np.bincount(labels) / len(labels))
    rows = []
    for threshold in thresholds:
        accepted = scores >= threshold
        cross = labels[upper[0][accepted]] != labels[upper[1][accepted]]
        groups = duplicate_groups(similarity, threshold)
        sizes = np.bincount(groups)
        rows.append({
            "threshold": threshold,
            "pairs": int(accepted.sum()),
            "cross_label_pairs": int(cross.sum()),
            "est_false_match_%": round(100 * cross.mean() / (1 - agree_by_chance), 1) if cross.size else 0.0,
            "images_with_a_twin": int((sizes[groups] > 1).sum()),
            "largest_group": int(sizes.max()),
        })
    return pd.DataFrame(rows)


def cross_partition_pairs(pairs: pd.DataFrame, partition: pd.Series) -> pd.DataFrame:
    """The matched pairs whose two photographs sit in different partitions."""
    first = partition.reindex(pairs["first"]).to_numpy()
    second = partition.reindex(pairs["second"]).to_numpy()
    leaked = pairs.assign(first_partition=first, second_partition=second)
    return leaked[leaked["first_partition"] != leaked["second_partition"]].reset_index(drop=True)
