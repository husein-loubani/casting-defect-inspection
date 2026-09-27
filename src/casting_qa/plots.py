"""
Every figure in the project.

Each function returns a Figure and never calls `plt.show()`, so the notebook
decides when to draw and the test suite can inspect axes without a display.
Colors come from the palette in config, and every chart axis carries a label.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import ConfusionMatrixDisplay

from casting_qa.config import CLASS_NAMES, DEFECT, FIGURE_DIR, FIGURE_DPI, OK, PALETTE


def apply_style() -> None:
    """One visual theme for the whole report."""
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def save_figure(figure: Figure, name: str, section: str = "") -> None:
    """Write a figure under reports/figures/<section>/."""
    directory = FIGURE_DIR / section if section else FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", bbox_inches="tight", dpi=FIGURE_DPI)


def _class_color(label: int) -> str:
    return PALETTE["defective"] if label == DEFECT else PALETTE["ok"]


def _hide_ticks(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)


def plot_class_balance(balance: pd.DataFrame) -> Figure:
    """Counts per class, with the share written on each bar."""
    figure, axis = plt.subplots(figsize=(5, 3.4))
    colors = [PALETTE["ok"] if name == "ok" else PALETTE["defective"] for name in balance["label"]]
    bars = axis.bar(balance["label"], balance["images"], color=colors)
    for bar, count, share in zip(bars, balance["images"], balance["share_%"], strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
                  f"{count} ({share:.1f}%)", ha="center", fontsize=9)
    axis.set_title("Class balance")
    axis.set_xlabel("Class")
    axis.set_ylabel("Images")
    axis.set_ylim(0, balance["images"].max() * 1.15)
    figure.tight_layout()
    return figure


def plot_sample_grid(paths_by_class: dict[int, list[str]], loader: Callable,
                     per_class: int) -> Figure:
    """A row of defective castings above a row of sound ones."""
    figure, axes = plt.subplots(2, per_class, figsize=(3.1 * per_class, 6.6))
    for row, label in enumerate((DEFECT, OK)):
        for column in range(per_class):
            axis = axes[row, column]
            axis.imshow(loader(paths_by_class[label][column]), cmap="gray", vmin=0, vmax=255)
            _hide_ticks(axis)
            if column == 0:
                axis.set_ylabel(CLASS_NAMES[label], fontsize=11, color=_class_color(label))
    figure.suptitle("Castings as photographed (exploration split)", y=0.99)
    figure.tight_layout()
    return figure


def plot_quality_by_class(quality: pd.DataFrame, metrics: dict[str, str]) -> Figure:
    """One histogram per image-quality metric, the two classes overlaid."""
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    for axis, (column, label) in zip(np.atleast_1d(axes), metrics.items(), strict=True):
        for cls in (OK, DEFECT):
            axis.hist(quality.loc[quality["label"] == cls, column], bins=30, alpha=0.6,
                      color=_class_color(cls), label=CLASS_NAMES[cls])
        axis.set_title(label, fontsize=10)
        axis.set_xlabel(label)
        axis.set_ylabel("Images")
    np.atleast_1d(axes)[0].legend()
    figure.suptitle("Image quality inside the part, by class (exploration split)", y=1.02)
    figure.tight_layout()
    return figure


def plot_similarity_distribution(scores: np.ndarray, threshold: float) -> Figure:
    """All pairwise detail similarities on a log count axis, with the duplicate threshold."""
    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.hist(scores, bins=120, color=PALETTE["neutral"], log=True)
    axis.axvline(threshold, color=PALETTE["defective"], linestyle="--",
                 label=f"same-casting threshold ({threshold})")
    axis.set_title("Surface-detail similarity of every pair of photographs")
    axis.set_xlabel("Best-rotation normalized cross-correlation of rim detail")
    axis.set_ylabel("Pairs (log scale)")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_duplicate_pairs(pairs: pd.DataFrame, path_of: pd.Series, loader: Callable,
                         count: int) -> Figure:
    """Matched photographs side by side, one pair per row."""
    rows = pairs.head(count)
    figure, axes = plt.subplots(len(rows), 2, figsize=(6.4, 3.3 * len(rows)), squeeze=False)
    for axis_row, (_, pair) in zip(axes, rows.iterrows(), strict=True):
        for axis, name in zip(axis_row, (pair["first"], pair["second"]), strict=True):
            axis.imshow(loader(path_of[name]), cmap="gray", vmin=0, vmax=255)
            _hide_ticks(axis)
            axis.set_title(f"{name}\nsimilarity {pair['similarity']:.2f}", fontsize=8)
    figure.suptitle("The same casting, photographed twice", y=1.0)
    figure.tight_layout()
    return figure


def plot_confound(features: pd.DataFrame, column: str = "nuisance_corner_mean") -> Figure:
    """
    Background brightness by class, the project's headline caveat.

    The corners contain no casting at all, so any separation visible here is
    the lighting rig rather than the metal.
    """
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for label in (OK, DEFECT):
        values = features.loc[features["label"] == label, column]
        axis.hist(values, bins=32, alpha=0.65, label=CLASS_NAMES[label], color=_class_color(label))
    axis.set_title("Background brightness separates the classes, and contains no casting")
    axis.set_xlabel("Mean intensity of the four image corners")
    axis.set_ylabel("Images")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_radial_profile(profiles: dict[str, np.ndarray], edges: np.ndarray,
                        band: tuple[float, float]) -> Figure:
    """Mean intensity against radius, with the inspected band shaded."""
    figure, axis = plt.subplots(figsize=(7, 4))
    centers = (edges[:-1] + edges[1:]) / 2
    axis.axvspan(*band, color=PALETTE["highlight"], alpha=0.15, label="inspected band")
    for name, values in profiles.items():
        color = PALETTE["defective"] if name.startswith("def") else PALETTE["ok"]
        axis.plot(centers, values, label=name, color=color, linewidth=2)
    axis.set_title("Radial intensity profile by class (exploration split)")
    axis.set_xlabel("Radius, in units of the cavity radius")
    axis.set_ylabel("Mean intensity (0-255)")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_pipeline_stages(image: np.ndarray, band: np.ndarray,
                         response: np.ndarray, polar: np.ndarray) -> Figure:
    """The pipeline made visible: original, inspected band, response, unwrapping."""
    figure, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. As photographed")
    axes[1].imshow(np.where(band, image, 0), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("2. Inspected band only")
    axes[2].imshow(response, cmap="inferno")
    axes[2].set_title("3. Black top-hat response")
    axes[3].imshow(polar, cmap="gray", aspect="auto", extent=(0, 1, 360, 0))
    axes[3].set_title("4. Polar unwrapping")
    axes[3].set_xlabel("Position across the band (inner to outer)")
    axes[3].set_ylabel("Angle (degrees)")
    for axis in axes[:3]:
        _hide_ticks(axis)
    figure.tight_layout()
    return figure


def plot_feature_separation(table: pd.DataFrame, top: int) -> Figure:
    """
    Each feature's separation before and after exposure matching, as a dot plot.

    Dots rather than bars because 0.5, not 0, is the no-information level, and
    a bar would suggest a zero baseline that does not exist.
    """
    frame = table.head(top).iloc[::-1]
    figure, axis = plt.subplots(figsize=(7.5, 0.4 * len(frame) + 1.6))
    positions = np.arange(len(frame))
    axis.hlines(positions, frame["separation_matched"], frame["separation_all"],
                color=PALETTE["neutral"], linewidth=1)
    axis.scatter(frame["separation_all"], positions, color=PALETTE["neutral"], label="all images", zorder=3)
    axis.scatter(frame["separation_matched"], positions, color=PALETTE["highlight"],
                 label="exposure-matched", zorder=3)
    axis.axvline(0.5, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(frame["feature"])
    axis.set_title("Class separation per feature, before and after removing the lighting shortcut")
    axis.set_xlabel("max(AUC, 1 - AUC); 0.5 is no information")
    axis.set_ylabel("Feature")
    axis.set_xlim(0.45, 1.0)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure


def plot_model_comparison(board: pd.DataFrame, metric: str = "f1_defect") -> Figure:
    """Validation score per approach, with the background floor drawn as a reference."""
    figure, axis = plt.subplots(figsize=(7, 4))
    best = board.loc[board["model"] != "background floor", metric].idxmax()
    colors = [PALETTE["highlight"] if index == best
              else PALETTE["defective"] if model == "background floor" else PALETTE["ok"]
              for index, model in zip(board.index, board["model"], strict=True)]
    bars = axis.bar(board["model"], board[metric], color=colors)
    for bar, value in zip(bars, board[metric], strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    axis.set_title("Validation defect-class F1 by approach")
    axis.set_xlabel("Approach")
    axis.set_ylabel("Defect-class F1")
    axis.set_ylim(0, 1.08)
    figure.tight_layout()
    return figure


def plot_confusion(labels: np.ndarray, predicted: np.ndarray) -> Figure:
    """The test confusion matrix, drawn by scikit-learn."""
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        labels, predicted, labels=[OK, DEFECT], display_labels=["ok", "defective"],
        cmap="Blues", colorbar=False, ax=axis)
    axis.set_title("Test set confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.grid(False)
    figure.tight_layout()
    return figure


def plot_image_grid(items: list[tuple[np.ndarray, str]], columns: int) -> Figure:
    """A grid of castings with a caption each."""
    rows = int(np.ceil(len(items) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows), squeeze=False)
    for axis, (image, title) in zip(axes.ravel(), items, strict=False):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontsize=8)
        _hide_ticks(axis)
    for axis in axes.ravel()[len(items):]:
        axis.axis("off")
    figure.tight_layout()
    return figure


def _draw_boxes(axis, boxes, style: str) -> None:
    for (min_row, min_col, max_row, max_col), _ in boxes:
        axis.add_patch(Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor=PALETTE["defective"], linewidth=1.8,
                                 linestyle=style))


def plot_detection_grid(items: list[tuple[np.ndarray, dict, str]], columns: int) -> Figure:
    """Castings with their boxes: solid for strong evidence, dashed for weak."""
    rows = int(np.ceil(len(items) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows), squeeze=False)
    for axis, (image, boxes, title) in zip(axes.ravel(), items, strict=False):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        _draw_boxes(axis, boxes["strong"], "-")
        _draw_boxes(axis, boxes["weak"], "--")
        axis.set_title(title, fontsize=8)
        _hide_ticks(axis)
    for axis in axes.ravel()[len(items):]:
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_localizer_reference(edges: np.ndarray, curve: np.ndarray, quantile: float,
                             sound_median: np.ndarray) -> Figure:
    """The sound-metal response level, ring by ring, that a defect has to exceed."""
    figure, axis = plt.subplots(figsize=(7, 3.6))
    centers = (edges[:-1] + edges[1:]) / 2
    axis.plot(centers, sound_median, color=PALETTE["ok"], linewidth=2, label="sound median")
    axis.plot(centers, curve, color=PALETTE["defective"], linewidth=2,
              label=f"reference ({100 * quantile:g}th percentile)")
    axis.set_title("What sound metal produces, by radius (exploration split)")
    axis.set_xlabel("Radius, in units of the cavity radius")
    axis.set_ylabel("Smoothed top-hat response")
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_robustness(table: pd.DataFrame, baseline_accuracy: float) -> Figure:
    """Accuracy under each perturbation against the unmodified images."""
    figure, axis = plt.subplots(figsize=(7, 0.4 * len(table) + 1.6))
    positions = np.arange(len(table))[::-1]
    axis.barh(positions, table["accuracy"], color=PALETTE["ok"])
    axis.axvline(baseline_accuracy, color=PALETTE["defective"], linestyle="--",
                 label=f"unmodified ({baseline_accuracy:.3f})")
    for position, (accuracy, flipped) in zip(positions, zip(table["accuracy"], table["flipped"], strict=True),
                                             strict=True):
        axis.text(accuracy + 0.01, position, f"{accuracy:.3f}, {flipped} flipped", va="center", fontsize=8)
    axis.set_yticks(positions)
    axis.set_yticklabels(table["perturbation"])
    axis.set_title("Validation accuracy under controlled changes to the photograph")
    axis.set_xlabel("Accuracy")
    axis.set_ylabel("Perturbation")
    axis.set_xlim(0, 1.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure
