from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

PredictFn = Callable[[np.ndarray], np.ndarray]
PredictProbaFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CurrentBufferIndex:
    reference_x: np.ndarray
    reference_y: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray
    nn: NearestNeighbors


@dataclass(frozen=True)
class CachedBufferNeighborhood:
    points: np.ndarray
    labels: np.ndarray
    distances: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray


def build_current_buffer_index(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    predict_fn: PredictFn,
    predict_proba_fn: PredictProbaFn,
    neighborhood_size: int,
    algorithm: str = "auto",
) -> CurrentBufferIndex:
    x = np.asarray(reference_x, dtype=float)
    y = np.asarray(reference_y, dtype=int).reshape(-1)
    if len(x) != len(y):
        raise ValueError("reference_x and reference_y must have the same length.")
    if len(x) == 0:
        raise ValueError("reference buffer must not be empty.")

    nn = NearestNeighbors(
        n_neighbors=min(max(1, neighborhood_size), len(x)),
        algorithm=algorithm,
    )
    nn.fit(x)
    return CurrentBufferIndex(
        reference_x=x,
        reference_y=y,
        predicted_labels=np.asarray(predict_fn(x), dtype=int).reshape(-1),
        probabilities=np.asarray(predict_proba_fn(x), dtype=float),
        nn=nn,
    )


def query_cached_buffer_neighborhood(
    counterfactual: np.ndarray,
    index: CurrentBufferIndex,
    neighborhood_size: int,
) -> CachedBufferNeighborhood:
    z = np.asarray(counterfactual, dtype=float).reshape(1, -1)
    k = min(max(1, neighborhood_size), len(index.reference_x))
    distances, indices = index.nn.kneighbors(z, n_neighbors=k)
    indices_1d = indices.reshape(-1)
    return CachedBufferNeighborhood(
        points=index.reference_x[indices_1d],
        labels=index.reference_y[indices_1d],
        distances=distances.reshape(-1),
        predicted_labels=index.predicted_labels[indices_1d],
        probabilities=index.probabilities[indices_1d],
    )


def epanechnikov_kernel_values(
    displacements: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    scaled_squared = np.sum((displacements / bandwidth) ** 2, axis=1)
    weights = 0.75 * (1.0 - scaled_squared)
    weights[scaled_squared > 1.0] = 0.0
    return weights


def fallback_distance_weights(distances: np.ndarray) -> np.ndarray:
    distances_1d = np.asarray(distances, dtype=float).reshape(-1)
    scale = max(float(np.max(distances_1d)), 1e-12)
    return np.exp(-((distances_1d / scale) ** 2))


def target_region_mask(
    neighborhood: CachedBufferNeighborhood,
    target_class: int,
) -> np.ndarray:
    target = int(target_class)
    mask = (np.asarray(neighborhood.labels, dtype=int) == target) & (
        np.asarray(neighborhood.predicted_labels, dtype=int) == target
    )
    if not np.any(mask):
        mask = np.asarray(neighborhood.predicted_labels, dtype=int) == target
    if not np.any(mask):
        mask = np.asarray(neighborhood.labels, dtype=int) == target
    return mask


def buffer_target_region_vector(
    counterfactual: np.ndarray,
    neighborhood: CachedBufferNeighborhood,
    target_class: int,
    bandwidth: float,
) -> np.ndarray:
    z = np.asarray(counterfactual, dtype=float).reshape(-1)
    target_points = np.asarray(neighborhood.points, dtype=float)[
        target_region_mask(neighborhood, target_class)
    ]
    if len(target_points) == 0:
        return np.zeros_like(z)

    displacements = target_points - z
    weights = epanechnikov_kernel_values(displacements, bandwidth=bandwidth)
    if np.allclose(weights.sum(), 0.0):
        weights = fallback_distance_weights(np.linalg.norm(displacements, axis=1))
    return (weights[:, None] * displacements).sum(axis=0) / weights.sum()

