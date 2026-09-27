"""Every figure function returns a Figure whose chart axes carry labels and a title."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from casting_qa import plots
from tests.conftest import make_casting


def _labeled(figure: Figure) -> bool:
    return all(axis.get_title() and axis.get_xlabel() and axis.get_ylabel()
               for axis in figure.axes if axis.get_visible())


def test_charts_are_figures_with_labeled_axes():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"label": [0, 1] * 20, "nuisance_corner_mean": rng.normal(200, 5, 40),
                          "sharpness": rng.random(40)})
    edges = np.linspace(0, 2, 11)
    table = pd.DataFrame({"feature": ["a", "b"], "separation_all": [0.8, 0.7],
                          "separation_matched": [0.75, 0.6]})
    board = pd.DataFrame({"model": ["background floor", "svm_rbf"], "f1_defect": [0.8, 0.95]})
    robustness = pd.DataFrame({"perturbation": ["gain 1.1"], "accuracy": [0.9], "flipped": [3]})
    charts = [
        plots.plot_class_balance(pd.DataFrame({"label": ["ok", "defective"], "images": [4, 6],
                                               "share_%": [40.0, 60.0]})),
        plots.plot_quality_by_class(frame, {"sharpness": "Sharpness"}),
        plots.plot_similarity_distribution(rng.random(100), 0.6),
        plots.plot_confound(frame),
        plots.plot_radial_profile({"defective": rng.random(10), "ok": rng.random(10)}, edges, (1.15, 2.05)),
        plots.plot_feature_separation(table, top=2),
        plots.plot_model_comparison(board),
        plots.plot_confusion(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])),
        plots.plot_localizer_reference(edges, rng.random(10), 0.9999, rng.random(10)),
        plots.plot_robustness(robustness, 0.95),
    ]
    for figure in charts:
        assert isinstance(figure, Figure)
        assert _labeled(figure), f"unlabeled axis in {figure.axes[0].get_title()!r}"


def test_image_panels_draw_every_box():
    image = make_casting(size=64)
    boxes = {"strong": [((5, 5, 20, 20), 0.3), ((30, 30, 40, 40), 0.2)], "weak": []}
    figure = plots.plot_detection_grid([(image, boxes, "t"), (image, {"strong": [], "weak": [((1, 1, 5, 5), 0.1)]}, "u")],
                                       columns=2)
    assert len(figure.axes[0].patches) == 2 and len(figure.axes[1].patches) == 1
    grid = plots.plot_image_grid([(image, "a"), (image, "b"), (image, "c")], columns=2)
    assert isinstance(grid, Figure)
