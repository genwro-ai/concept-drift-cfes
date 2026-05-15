from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from concept_drift_cfes.utils.arrays import ensure_2d, resolve_targets

PredictFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CFEMetrics:
    n_samples: int
    validity: float
    l1: float
    l2: float
    sparsity: float
    target_neighbor_distance: float
    target_neighbor_ratio: float
    target_kernel_log_density: float
    isolation_forest_score: float
    local_outlier_factor_score: float


def batch_validity(
    counterfactuals: np.ndarray,
    predict_fn: PredictFn,
    target_class: int | np.ndarray,
) -> np.ndarray:
    x = ensure_2d(counterfactuals)
    targets = resolve_targets(target_class=target_class, n_samples=len(x))
    predictions = np.asarray(predict_fn(x)).reshape(-1).astype(int)
    return predictions == targets


def batch_l1(
    x: np.ndarray,
    counterfactuals: np.ndarray,
) -> np.ndarray:
    x_2d = ensure_2d(x)
    cf_2d = ensure_2d(counterfactuals)
    if x_2d.shape != cf_2d.shape:
        raise ValueError("x and counterfactuals must have the same shape.")
    return np.abs(cf_2d - x_2d).sum(axis=1)


def batch_l2(
    x: np.ndarray,
    counterfactuals: np.ndarray,
) -> np.ndarray:
    x_2d = ensure_2d(x)
    cf_2d = ensure_2d(counterfactuals)
    if x_2d.shape != cf_2d.shape:
        raise ValueError("x and counterfactuals must have the same shape.")
    return np.linalg.norm(cf_2d - x_2d, axis=1)


def batch_sparsity(
    x: np.ndarray,
    counterfactuals: np.ndarray,
    tolerance: float = 1e-5,
) -> np.ndarray:
    x_2d = ensure_2d(x)
    cf_2d = ensure_2d(counterfactuals)
    if x_2d.shape != cf_2d.shape:
        raise ValueError("x and counterfactuals must have the same shape.")
    return np.sum(np.abs(cf_2d - x_2d) > tolerance, axis=1)


def batch_target_neighbor_distance(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactuals: np.ndarray,
    target_class: int | np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    reference_x_2d = ensure_2d(reference_x)
    cf_2d = ensure_2d(counterfactuals)
    reference_y_1d = np.asarray(reference_y, dtype=int)
    targets = resolve_targets(target_class=target_class, n_samples=len(cf_2d))
    scores = np.empty(len(cf_2d), dtype=float)

    for class_label in np.unique(targets):
        mask = targets == class_label
        class_x = reference_x_2d[reference_y_1d == class_label]
        if len(class_x) == 0:
            raise ValueError(f"No reference samples available for class {class_label}.")
        neighbors = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(class_x)),
        ).fit(class_x)
        distances, _ = neighbors.kneighbors(cf_2d[mask])
        scores[mask] = distances.mean(axis=1)

    return scores


def batch_target_neighbor_ratio(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactuals: np.ndarray,
    target_class: int | np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    reference_x_2d = ensure_2d(reference_x)
    cf_2d = ensure_2d(counterfactuals)
    reference_y_1d = np.asarray(reference_y, dtype=int)
    targets = resolve_targets(target_class=target_class, n_samples=len(cf_2d))
    neighbors = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(reference_x_2d)),
    ).fit(reference_x_2d)
    _, indices = neighbors.kneighbors(cf_2d)
    neighbor_labels = reference_y_1d[indices]
    return (neighbor_labels == targets[:, None]).mean(axis=1)


def batch_target_kernel_log_density(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactuals: np.ndarray,
    target_class: int | np.ndarray,
    bandwidth: float = 0.1,
) -> np.ndarray:
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive.")

    reference_x_2d = ensure_2d(reference_x)
    cf_2d = ensure_2d(counterfactuals)
    reference_y_1d = np.asarray(reference_y, dtype=int)
    targets = resolve_targets(target_class=target_class, n_samples=len(cf_2d))
    scores = np.empty(len(cf_2d), dtype=float)
    n_features = cf_2d.shape[1]

    for class_label in np.unique(targets):
        mask = targets == class_label
        class_x = reference_x_2d[reference_y_1d == class_label]
        if len(class_x) == 0:
            raise ValueError(f"No reference samples available for class {class_label}.")

        diff = cf_2d[mask, None, :] - class_x[None, :, :]
        squared_distances = np.sum(diff**2, axis=2)
        log_weights = -squared_distances / (2.0 * bandwidth**2)
        max_log_weights = np.max(log_weights, axis=1)
        log_mean_weights = max_log_weights + np.log(
            np.mean(np.exp(log_weights - max_log_weights[:, None]), axis=1)
        )
        scores[mask] = log_mean_weights - n_features * np.log(bandwidth)

    return scores


def batch_isolation_forest_score(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactuals: np.ndarray,
    target_class: int | np.ndarray,
    random_state: int = 42,
    n_estimators: int = 100,
) -> np.ndarray:
    reference_x_2d = ensure_2d(reference_x)
    cf_2d = ensure_2d(counterfactuals)
    reference_y_1d = np.asarray(reference_y, dtype=int)
    targets = resolve_targets(target_class=target_class, n_samples=len(cf_2d))
    scores = np.empty(len(cf_2d), dtype=float)

    for class_label in np.unique(targets):
        mask = targets == class_label
        class_x = reference_x_2d[reference_y_1d == class_label]
        if len(class_x) == 0:
            raise ValueError(f"No reference samples available for class {class_label}.")
        iforest = IsolationForest(
            random_state=random_state,
            n_estimators=n_estimators,
        ).fit(class_x)
        scores[mask] = iforest.score_samples(cf_2d[mask])

    return scores


def batch_local_outlier_factor_score(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    counterfactuals: np.ndarray,
    target_class: int | np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    reference_x_2d = ensure_2d(reference_x)
    cf_2d = ensure_2d(counterfactuals)
    reference_y_1d = np.asarray(reference_y, dtype=int)
    targets = resolve_targets(target_class=target_class, n_samples=len(cf_2d))
    scores = np.empty(len(cf_2d), dtype=float)

    for class_label in np.unique(targets):
        mask = targets == class_label
        class_x = reference_x_2d[reference_y_1d == class_label]
        if len(class_x) == 0:
            raise ValueError(f"No reference samples available for class {class_label}.")
        if len(class_x) < 3:
            scores[mask] = np.nan
            continue

        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(class_x) - 1),
            novelty=True,
        ).fit(class_x)
        scores[mask] = lof.score_samples(cf_2d[mask])

    return scores


def evaluate_cfe_metrics(
    x: np.ndarray,
    counterfactuals: np.ndarray,
    target_class: int | np.ndarray,
    predict_fn: PredictFn,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    n_neighbors: int = 15,
    lof_n_neighbors: int = 35,
    kernel_density_bandwidth: float = 0.1,
    random_state: int = 42,
    tolerance: float = 1e-5,
    isolation_forest_n_estimators: int = 100,
) -> CFEMetrics:
    validity_scores = batch_validity(counterfactuals, predict_fn, target_class)
    l1_scores = batch_l1(x, counterfactuals)
    l2_scores = batch_l2(x, counterfactuals)
    sparsity_scores = batch_sparsity(x, counterfactuals, tolerance=tolerance)
    neighbor_distance_scores = batch_target_neighbor_distance(
        reference_x=reference_x,
        reference_y=reference_y,
        counterfactuals=counterfactuals,
        target_class=target_class,
        n_neighbors=n_neighbors,
    )
    neighbor_ratio_scores = batch_target_neighbor_ratio(
        reference_x=reference_x,
        reference_y=reference_y,
        counterfactuals=counterfactuals,
        target_class=target_class,
        n_neighbors=n_neighbors,
    )
    target_kernel_log_density_scores = batch_target_kernel_log_density(
        reference_x=reference_x,
        reference_y=reference_y,
        counterfactuals=counterfactuals,
        target_class=target_class,
        bandwidth=kernel_density_bandwidth,
    )
    iforest_scores = batch_isolation_forest_score(
        reference_x=reference_x,
        reference_y=reference_y,
        counterfactuals=counterfactuals,
        target_class=target_class,
        random_state=random_state,
        n_estimators=isolation_forest_n_estimators,
    )
    lof_scores = batch_local_outlier_factor_score(
        reference_x=reference_x,
        reference_y=reference_y,
        counterfactuals=counterfactuals,
        target_class=target_class,
        n_neighbors=lof_n_neighbors,
    )

    return CFEMetrics(
        n_samples=len(ensure_2d(counterfactuals)),
        validity=float(validity_scores.mean()),
        l1=float(l1_scores.mean()),
        l2=float(l2_scores.mean()),
        sparsity=float(sparsity_scores.mean()),
        target_neighbor_distance=float(neighbor_distance_scores.mean()),
        target_neighbor_ratio=float(neighbor_ratio_scores.mean()),
        target_kernel_log_density=float(target_kernel_log_density_scores.mean()),
        isolation_forest_score=float(iforest_scores.mean()),
        local_outlier_factor_score=float(np.nanmean(lof_scores)),
    )
