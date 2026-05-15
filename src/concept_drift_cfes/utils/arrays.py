from collections.abc import Iterable
from typing import Any

import numpy as np


def ensure_2d(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Expected a 1D or 2D array.")
    return array


def resolve_targets(
    target_class: int | np.ndarray,
    n_samples: int,
) -> np.ndarray:
    targets = np.asarray(target_class)
    if targets.ndim == 0:
        return np.full(n_samples, int(targets), dtype=int)
    if len(targets) != n_samples:
        raise ValueError("target_class must be scalar or have one value per sample.")
    return targets.astype(int)


def sorted_feature_names(row: dict[Any, float]) -> tuple[Any, ...]:
    return tuple(sorted(row.keys()))


def rows_to_batch(
    rows: Iterable[tuple[dict[Any, float], Any]],
    feature_names: tuple[Any, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    row_list = list(rows)
    if not row_list:
        raise ValueError("Cannot convert an empty set of rows to arrays.")

    resolved_feature_names = feature_names or sorted_feature_names(row_list[0][0])
    x = np.asarray(
        [
            [row_features[feature_name] for feature_name in resolved_feature_names]
            for row_features, _ in row_list
        ],
        dtype=float,
    )
    y = np.asarray([int(target) for _, target in row_list], dtype=int)
    return x, y, resolved_feature_names


def validate_split_sizes(
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    sizes = np.asarray([train_size, val_size, test_size], dtype=float)
    if np.any(sizes <= 0.0):
        raise ValueError("train_size, val_size, and test_size must all be positive.")
    if not np.isclose(sizes.sum(), 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0.")
