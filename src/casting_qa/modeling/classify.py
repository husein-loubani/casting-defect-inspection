"""
Turning features into a defect decision, and measuring how far to trust it.

The brief asks for an algorithm; it does not say the algorithm has to be
learned. So there are two families, compared on equal terms:

- a rule-based score, the mean of a few standardized named quantities with one
  threshold, fitted on the exploration split. A line worker can be told exactly
  what made it reject a part.
- classical classifiers over the whole feature vector, tuned by cross-validation
  on the exploration split and compared on validation.

Everything fitted here (scalers, thresholds, hyperparameters) is fitted on the
exploration split and applied unchanged to validation and test.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from casting_qa.config import (
    CONFIDENCE_LEVEL,
    CV_FOLDS,
    CV_REPEATS,
    DEFECT,
    MATCH_TOLERANCE,
    MATCHED_CONTROL_DRAWS,
    N_JOBS,
    OK,
    OR_GATE_PERCENTILE,
    RANDOM_FOREST_GRID,
    RANDOM_SEED,
    RULE_FEATURES,
    RULE_GRID_POINTS,
    SVM_GRID,
)


def fit_rule(train: pd.DataFrame, labels: np.ndarray,
             columns: tuple[str, ...] = RULE_FEATURES,
             points: int = RULE_GRID_POINTS) -> dict:
    """
    Standardize the rule's features on the training partition and choose the
    score threshold that maximizes defect-class F1 there.

    The scaler is fitted once, here, so a casting scores the same whichever
    batch it arrives in. The threshold is swept over the observed score range
    rather than guessed.
    """
    scaler = StandardScaler().fit(train[list(columns)])
    scores = scaler.transform(train[list(columns)]).mean(axis=1)
    best_threshold, best_f1 = float(scores.min()), -1.0
    for threshold in np.linspace(scores.min(), scores.max(), points):
        score = f1_score(labels, (scores >= threshold).astype(int), pos_label=DEFECT, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return {"columns": list(columns), "scaler": scaler, "threshold": best_threshold, "f1": best_f1}


def rule_score(features: pd.DataFrame, rule: dict) -> np.ndarray:
    """The severity score: the mean of the standardized rule features."""
    return rule["scaler"].transform(features[rule["columns"]]).mean(axis=1)


def apply_rule(features: pd.DataFrame, rule: dict) -> np.ndarray:
    """Predictions from a fitted rule, on any partition."""
    return (rule_score(features, rule) >= rule["threshold"]).astype(int)


def fit_or_gate(train: pd.DataFrame, labels: np.ndarray,
                columns: tuple[str, ...] = RULE_FEATURES,
                percentile: float = OR_GATE_PERCENTILE) -> dict[str, float]:
    """
    The design the rule replaced: one alarm per feature at a high percentile of
    the sound castings, and the part is rejected if any alarm fires.
    """
    sound = train[np.asarray(labels) == OK]
    return {column: float(np.percentile(sound[column], percentile)) for column in columns}


def apply_or_gate(features: pd.DataFrame, gates: dict[str, float]) -> np.ndarray:
    """Predictions from the OR of per-feature alarms."""
    fired = np.zeros(len(features), dtype=bool)
    for column, limit in gates.items():
        fired |= features[column].to_numpy() > limit
    return fired.astype(int)


def build_models(seed: int = RANDOM_SEED) -> dict[str, tuple[Pipeline, dict]]:
    """
    The candidate classifiers and their grids.

    The SVM's scaler sits inside its pipeline, so every cross-validation fold
    refits it on that fold's training part only. The forest gets no scaler:
    a tree splits on thresholds of one feature at a time, and rescaling a
    feature moves the threshold without changing any split.
    """
    forest = Pipeline([("model", RandomForestClassifier(random_state=seed, n_jobs=1))])
    svm = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", random_state=seed))])
    return {
        "random_forest": (forest, {f"model__{k}": list(v) for k, v in RANDOM_FOREST_GRID.items()}),
        "svm_rbf": (svm, {f"model__{k}": list(v) for k, v in SVM_GRID.items()}),
    }


def tune(model: Pipeline, grid: dict, features, labels,
         folds: int = CV_FOLDS, seed: int = RANDOM_SEED) -> GridSearchCV:
    """Grid search with stratified folds, scored on defect-class F1."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(model, grid, scoring="f1", cv=splitter, n_jobs=N_JOBS)
    return search.fit(features, labels)


def decision_scores(model, features) -> np.ndarray:
    """A continuous score for ROC AUC: the SVM's signed margin or the forest's vote share."""
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return model.predict_proba(features)[:, 1]


def evaluate(labels: np.ndarray, predicted: np.ndarray,
             scores: np.ndarray | None = None) -> dict[str, float]:
    """Headline numbers for one model on one partition."""
    matrix = confusion_matrix(labels, predicted, labels=[OK, DEFECT])
    result = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1_defect": float(f1_score(labels, predicted, pos_label=DEFECT, zero_division=0)),
        "recall_defect": float(recall_score(labels, predicted, pos_label=DEFECT, zero_division=0)),
        "precision_defect": float(precision_score(labels, predicted, pos_label=DEFECT, zero_division=0)),
        "missed_defects": int(matrix[1, 0]),
        "false_alarms": int(matrix[0, 1]),
    }
    if scores is not None and len(np.unique(labels)) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    return result


def compare_models(explore: pd.DataFrame, validation: pd.DataFrame, columns: list[str],
                   nuisance: list[str], rule: dict) -> tuple[pd.DataFrame, dict[str, GridSearchCV]]:
    """
    Every approach, fitted on explore and scored on validation, in one table.

    The first row is the background floor: the random forest fitted on the
    workbench probes alone, the score available without looking at any metal.
    """
    y_explore, y_validation = explore["label"].to_numpy(), validation["label"].to_numpy()
    rows, fitted = [], {}

    forest, forest_grid = build_models()["random_forest"]
    floor = tune(forest, forest_grid, explore[nuisance], y_explore)
    rows.append({"model": "background floor", **evaluate(
        y_validation, floor.predict(validation[nuisance]), decision_scores(floor, validation[nuisance]))})

    rows.append({"model": "rule-based", **evaluate(
        y_validation, apply_rule(validation, rule), rule_score(validation, rule))})

    for name, (model, grid) in build_models().items():
        search = tune(model, grid, explore[columns], y_explore)
        fitted[name] = search
        rows.append({"model": name, **evaluate(
            y_validation, search.predict(validation[columns]),
            decision_scores(search, validation[columns]))})
    return pd.DataFrame(rows), fitted


def repeated_cv(model, features, labels, folds: int = CV_FOLDS, repeats: int = CV_REPEATS,
                seed: int = RANDOM_SEED) -> np.ndarray:
    """Defect-class F1 over repeated stratified folds, for the run-to-run spread."""
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return cross_val_score(clone(model), features, labels, scoring="f1", cv=splitter, n_jobs=N_JOBS)


def mcnemar_exact(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """
    Whether two classifiers differ on the same images, beyond chance.

    Only the images where exactly one of them is right carry information. Under
    the null hypothesis each such image is equally likely to favor either, so
    the count favoring one is Binomial(n, 0.5), tested exactly.
    """
    first_right, second_right = first == labels, second == labels
    only_first = int(np.sum(first_right & ~second_right))
    only_second = int(np.sum(~first_right & second_right))
    discordant = only_first + only_second
    p_value = binomtest(only_first, discordant, 0.5).pvalue if discordant else 1.0
    return {"only_first_right": only_first, "only_second_right": only_second, "p_value": float(p_value)}


def proportion_interval(successes: int, total: int,
                        level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    """Wilson score interval for a proportion, which stays sensible at 0 and at 1."""
    interval = binomtest(successes, total).proportion_ci(confidence_level=level, method="wilson")
    return float(interval.low), float(interval.high)


def headline_intervals(labels: np.ndarray, predicted: np.ndarray,
                       level: float = CONFIDENCE_LEVEL) -> pd.DataFrame:
    """Each headline proportion with its confidence interval."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    defective, sound = labels == DEFECT, labels == OK
    flagged = predicted == DEFECT
    quantities = {
        "accuracy": (int(np.sum(predicted == labels)), len(labels)),
        "recall (defects caught)": (int(np.sum(flagged & defective)), int(defective.sum())),
        "precision (flags that were defects)": (int(np.sum(flagged & defective)), int(flagged.sum())),
        "false-alarm rate (sound parts rejected)": (int(np.sum(flagged & sound)), int(sound.sum())),
    }
    rows = []
    for name, (successes, total) in quantities.items():
        low, high = proportion_interval(successes, total, level)
        rows.append({"quantity": name, "count": f"{successes}/{total}",
                     "estimate": round(successes / total, 4),
                     f"{round(level * 100)}%_low": round(low, 4),
                     f"{round(level * 100)}%_high": round(high, 4)})
    return pd.DataFrame(rows)


def report(labels: np.ndarray, predicted: np.ndarray) -> str:
    """The per-class classification report the brief asks for."""
    return classification_report(labels, predicted, labels=[OK, DEFECT],
                                 target_names=["ok", "defective"], digits=3, zero_division=0)


def time_inspection(paths, loader: Callable[[str], np.ndarray],
                    inspect: Callable[[np.ndarray], dict],
                    repeats: int = 1) -> dict[str, float]:
    """
    Wall time per image for the whole product: file on disk to decision and boxes.

    Each image is timed on its own, on one core, so the summary can report the
    median and the 95th percentile rather than one total. The 95th percentile
    matters on a production line, where the slow image sets the pace.
    """
    times = []
    for _ in range(repeats):
        for path in paths:
            started = time.perf_counter()
            inspect(loader(path))
            times.append(time.perf_counter() - started)
    milliseconds = 1000 * np.asarray(times)
    return {
        "images_timed": len(times),
        "median_ms": round(float(np.median(milliseconds)), 1),
        "p95_ms": round(float(np.percentile(milliseconds, 95)), 1),
        "mean_ms": round(float(milliseconds.mean()), 1),
        "images_per_second": round(1000 / float(np.median(milliseconds)), 1),
    }


def exposure_matched_pairs(features: pd.DataFrame, column: str = "nuisance_corner_mean",
                           tolerance: float = MATCH_TOLERANCE) -> pd.DataFrame:
    """
    Pair sound castings with defective ones photographed at the same exposure.

    Greedy nearest-neighbor matching without replacement, keeping only pairs
    within `tolerance` gray levels of corner brightness. Sound castings with no
    defective partner that close are left out, so the subset is balanced 50/50
    and the lighting shortcut carries almost no information inside it.
    """
    if column not in features.columns:
        raise ValueError(f"{column} missing; describe with include_nuisance=True")

    good = features[features["label"] == OK].sort_values(column)
    bad = features[features["label"] == DEFECT].sort_values(column)
    used: set[int] = set()
    rows = []
    bad_values, bad_index = bad[column].to_numpy(), bad.index.to_numpy()
    for position, value in zip(good.index, good[column].to_numpy(), strict=True):
        distances = np.abs(bad_values - value)
        for candidate in np.argsort(distances):
            if distances[candidate] > tolerance:
                break
            if bad_index[candidate] in used:
                continue
            used.add(bad_index[candidate])
            rows.extend([position, bad_index[candidate]])
            break
    return features.loc[rows].copy()


def balanced_random_subset(features: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """A random subset with `per_class` images of each class and no exposure matching."""
    parts = [features[features["label"] == label].sample(per_class, random_state=seed)
             for label in (OK, DEFECT)]
    return pd.concat(parts)


def matched_control(features: pd.DataFrame, matched: pd.DataFrame, model,
                    columns: list[str], nuisance: list[str],
                    draws: int = MATCHED_CONTROL_DRAWS, folds: int = CV_FOLDS,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Cross-validated accuracy on the exposure-matched subset against random
    subsets of the same size and the same 50/50 balance.

    Exactly one thing differs between the two arms, the exposure matching, so
    a drop on the matched arm is what the lighting shortcut was worth. The same
    tuned model is used on both arms and for both feature sets.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def score(frame: pd.DataFrame, subset: list[str]) -> float:
        return float(cross_val_score(clone(model), frame[subset], frame["label"],
                                     cv=splitter, n_jobs=N_JOBS).mean())

    per_class = len(matched) // 2
    random_part = [score(balanced_random_subset(features, per_class, seed + d), columns)
                   for d in range(draws)]
    random_nuisance = [score(balanced_random_subset(features, per_class, seed + d), nuisance)
                       for d in range(draws)]
    return pd.DataFrame([
        {"subset": f"random balanced, {draws} draws", "images": 2 * per_class,
         "part_features": round(float(np.mean(random_part)), 3),
         "part_features_sd": round(float(np.std(random_part)), 3),
         "background_features": round(float(np.mean(random_nuisance)), 3),
         "background_features_sd": round(float(np.std(random_nuisance)), 3)},
        {"subset": "exposure-matched", "images": len(matched),
         "part_features": round(score(matched, columns), 3), "part_features_sd": np.nan,
         "background_features": round(score(matched, nuisance), 3), "background_features_sd": np.nan},
    ])


def separation_table(features: pd.DataFrame, columns: list[str],
                     matched: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    How well each feature separates the classes, whichever way it points.

    AUC below 0.5 is as informative as AUC above it, only inverted, so each
    feature is scored by max(AUC, 1 - AUC) and its direction is recorded. The
    confound share is the separation lost when the lighting shortcut is removed.
    """
    rows = []
    for column in columns:
        auc = float(roc_auc_score(features["label"], features[column]))
        entry = {"feature": column,
                 "higher_in": "defective" if auc >= 0.5 else "ok",
                 "separation_all": round(max(auc, 1 - auc), 3)}
        if matched is not None and len(matched):
            matched_auc = float(roc_auc_score(matched["label"], matched[column]))
            entry["separation_matched"] = round(max(matched_auc, 1 - matched_auc), 3)
            entry["confound_share"] = round(entry["separation_all"] - entry["separation_matched"], 3)
        rows.append(entry)
    frame = pd.DataFrame(rows)
    key = "separation_matched" if "separation_matched" in frame.columns else "separation_all"
    return frame.sort_values(key, ascending=False).reset_index(drop=True)
