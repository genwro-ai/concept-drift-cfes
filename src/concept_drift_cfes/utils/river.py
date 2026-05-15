from collections.abc import Iterator
from typing import Any

import numpy as np
from river import metrics

from concept_drift_cfes.utils.arrays import ensure_2d, resolve_targets


def array_to_river_dict(
    x: np.ndarray,
    feature_names: tuple[Any, ...],
) -> dict[Any, float]:
    x_1d = np.asarray(x, dtype=float).reshape(-1)
    if len(x_1d) != len(feature_names):
        raise ValueError("x and feature_names must have the same length.")
    return {
        feature_name: float(value)
        for feature_name, value in zip(feature_names, x_1d, strict=True)
    }


def iter_river_dicts(
    x: np.ndarray,
    feature_names: tuple[Any, ...],
) -> Iterator[dict[Any, float]]:
    for row in ensure_2d(x):
        yield array_to_river_dict(row, feature_names)


def learn_on_batch(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[Any, ...],
) -> object:
    x_2d = ensure_2d(x)
    y_1d = np.asarray(y, dtype=int).reshape(-1)
    if len(x_2d) != len(y_1d):
        raise ValueError("x and y must have the same number of rows.")

    for x_dict, target in zip(iter_river_dicts(x_2d, feature_names), y_1d, strict=True):
        model.learn_one(x_dict, int(target))
    return model


def predict_batch(
    model: object,
    x: np.ndarray,
    feature_names: tuple[Any, ...],
) -> np.ndarray:
    predictions = [
        model.predict_one(x_dict) for x_dict in iter_river_dicts(x, feature_names)
    ]
    return np.asarray(predictions, dtype=int)


def predict_proba_batch(
    model: object,
    x: np.ndarray,
    feature_names: tuple[Any, ...],
    classes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    rows = []
    for x_dict in iter_river_dicts(x, feature_names):
        if hasattr(model, "predict_proba_one"):
            proba_dict = model.predict_proba_one(x_dict)
        else:
            pred = int(model.predict_one(x_dict))
            proba_dict = {pred: 1.0}
        rows.append(
            [float(proba_dict.get(class_label, 0.0)) for class_label in classes]
        )
    return np.asarray(rows, dtype=float)


def accuracy_on_batch(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[Any, ...],
) -> float:
    metric = metrics.Accuracy()
    x_2d = ensure_2d(x)
    y_1d = np.asarray(y, dtype=int).reshape(-1)
    for x_dict, target in zip(iter_river_dicts(x_2d, feature_names), y_1d, strict=True):
        prediction = model.predict_one(x_dict)
        metric.update(int(target), prediction)
    return float(metric.get())


def target_flip_classes(predictions: np.ndarray) -> np.ndarray:
    predictions_1d = np.asarray(predictions, dtype=int).reshape(-1)
    return resolve_targets(1 - predictions_1d, n_samples=len(predictions_1d))
