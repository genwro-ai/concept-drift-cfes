from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from concept_drift_cfes.utils.neighborhoods import (
    CurrentBufferIndex,
    buffer_target_region_vector,
    build_current_buffer_index,
    query_cached_buffer_neighborhood,
)


PredictFn = Callable[[np.ndarray], np.ndarray]
PredictProbaFn = Callable[[np.ndarray], np.ndarray]

VALIDITY_PLAUSIBILITY = "validity_plausibility"
PLAUSIBILITY_LOW_MARGIN = "plausibility_low_margin"


@dataclass(frozen=True)
class CFEUpdateConfig:
    mode: str = VALIDITY_PLAUSIBILITY
    step_size: float = 0.05
    validity_weight: float = 2.0
    plausibility_weight: float = 2.0
    proximity_weight: float = 1.0
    neighborhood_size: int = 64
    low_margin_threshold: float = 0.6
    plausibility_every_steps: int = 60
    max_update_steps: int = 1
    normalize_vectors: bool = True
    target_region_bandwidth: float = 0.3
    gaussian_samples: int = 128
    gaussian_sigma: float = 0.1
    clip_min: float | None = 0.0
    clip_max: float | None = 1.0


@dataclass(frozen=True)
class CounterfactualState:
    x_ref: np.ndarray
    counterfactual: np.ndarray
    target_class: int
    is_active: bool = True
    retirement_reason: str | None = None
    last_update: str = "none"


@dataclass(frozen=True)
class GaussianNeighborhood:
    sampled_points: np.ndarray
    offsets: np.ndarray
    target_probabilities: np.ndarray
    predicted_labels: np.ndarray


def normalize_direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def combine_direction_components(
    components: list[tuple[float, np.ndarray]],
    normalize_vectors: bool,
) -> np.ndarray:
    usable = [(weight, vector) for weight, vector in components if weight > 0.0]
    if not usable:
        raise ValueError("At least one positive-weight component is required.")

    reference_shape = np.asarray(usable[0][1], dtype=float).shape
    combined = np.zeros(reference_shape, dtype=float)
    for weight, vector in usable:
        component = np.asarray(vector, dtype=float)
        if normalize_vectors:
            component = normalize_direction(component)
        combined += weight * component
    if normalize_vectors:
        return normalize_direction(combined)
    return combined


def target_class_probabilities(
    x: np.ndarray,
    predict_proba_fn: PredictProbaFn,
    target_class: int,
) -> np.ndarray:
    probabilities = np.asarray(predict_proba_fn(np.asarray(x, dtype=float)))
    if probabilities.ndim == 1:
        return probabilities.reshape(-1)
    return probabilities[:, int(target_class)]


def target_class_probability(
    counterfactual: np.ndarray,
    predict_proba_fn: PredictProbaFn,
    target_class: int,
) -> float:
    return float(
        target_class_probabilities(
            np.asarray(counterfactual, dtype=float).reshape(1, -1),
            predict_proba_fn=predict_proba_fn,
            target_class=target_class,
        )[0]
    )


def sample_gaussian_neighborhood(
    counterfactual: np.ndarray,
    predict_fn: PredictFn,
    predict_proba_fn: PredictProbaFn,
    target_class: int,
    n_samples: int,
    sigma: float,
    rng: np.random.Generator,
) -> GaussianNeighborhood:
    x = np.asarray(counterfactual, dtype=float).reshape(-1)
    if n_samples <= 0 or sigma <= 0:
        raise ValueError("n_samples and sigma must be positive.")

    offsets = rng.normal(loc=0.0, scale=sigma, size=(n_samples, len(x)))
    sampled_points = x + offsets
    return GaussianNeighborhood(
        sampled_points=sampled_points,
        offsets=offsets,
        target_probabilities=target_class_probabilities(
            sampled_points,
            predict_proba_fn=predict_proba_fn,
            target_class=target_class,
        ),
        predicted_labels=np.asarray(predict_fn(sampled_points), dtype=int).reshape(-1),
    )


def gaussian_sampling_validity_vector(
    neighborhood: GaussianNeighborhood,
    sigma: float,
) -> np.ndarray:
    offsets = np.asarray(neighborhood.offsets, dtype=float)
    probabilities = np.asarray(neighborhood.target_probabilities, dtype=float)
    weights = np.exp(-np.sum(offsets**2, axis=1) / (2.0 * sigma**2))
    weighted_x = offsets * weights[:, None]
    weighted_y = probabilities * weights
    ridge = 1e-6 * np.eye(offsets.shape[1])
    lhs = offsets.T @ weighted_x + ridge
    rhs = offsets.T @ weighted_y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(offsets.shape[1], dtype=float)


def proximity_direction(
    x_ref: np.ndarray,
    counterfactual: np.ndarray,
) -> np.ndarray:
    return np.asarray(x_ref, dtype=float).reshape(-1) - np.asarray(
        counterfactual,
        dtype=float,
    ).reshape(-1)


def scheduled_step_due(update_index: int, every_steps: int) -> bool:
    if every_steps <= 0:
        raise ValueError("every_steps must be positive.")
    return int(update_index) % int(every_steps) == 0


def clip_candidate(candidate: np.ndarray, config: CFEUpdateConfig) -> np.ndarray:
    if config.clip_min is None and config.clip_max is None:
        return candidate
    return np.clip(
        candidate,
        -np.inf if config.clip_min is None else config.clip_min,
        np.inf if config.clip_max is None else config.clip_max,
    )


def plausibility_vector(
    counterfactual: np.ndarray,
    buffer_index: CurrentBufferIndex,
    target_class: int,
    config: CFEUpdateConfig,
) -> np.ndarray:
    neighborhood = query_cached_buffer_neighborhood(
        counterfactual=counterfactual,
        index=buffer_index,
        neighborhood_size=config.neighborhood_size,
    )
    return buffer_target_region_vector(
        counterfactual=counterfactual,
        neighborhood=neighborhood,
        target_class=target_class,
        bandwidth=config.target_region_bandwidth,
    )


def correction_components(
    x_ref: np.ndarray,
    counterfactual: np.ndarray,
    correction_name: str,
    correction_vector: np.ndarray,
    config: CFEUpdateConfig,
) -> list[tuple[float, np.ndarray]]:
    weight = (
        config.validity_weight
        if correction_name == "validity"
        else config.plausibility_weight
    )
    return [
        (weight, correction_vector),
        (config.proximity_weight, proximity_direction(x_ref, counterfactual)),
    ]


def update_counterfactual_state(
    state: CounterfactualState,
    predict_fn: PredictFn,
    predict_proba_fn: PredictProbaFn,
    buffer_index: CurrentBufferIndex,
    config: CFEUpdateConfig,
    update_index: int,
    rng: np.random.Generator,
) -> CounterfactualState:
    if not state.is_active:
        return state

    x_ref = np.asarray(state.x_ref, dtype=float).reshape(-1)
    cf = np.asarray(state.counterfactual, dtype=float).reshape(-1)
    current_state = replace(state, counterfactual=cf)
    target = int(state.target_class)

    if int(predict_fn(x_ref.reshape(1, -1))[0]) == target:
        return replace(
            current_state,
            is_active=False,
            retirement_reason="naturally_resolved",
            last_update="retired",
        )

    last_update = "no_change"
    for _ in range(config.max_update_steps):
        prediction = int(predict_fn(cf.reshape(1, -1))[0])
        probabilities = np.asarray(predict_proba_fn(cf.reshape(1, -1)), dtype=float)[0]
        target_probability = (
            float(probabilities[target]) if 0 <= target < len(probabilities) else 0.0
        )
        low_probability = (
            prediction != target or target_probability < config.low_margin_threshold
        )

        if config.mode == VALIDITY_PLAUSIBILITY:
            if low_probability:
                neighborhood = sample_gaussian_neighborhood(
                    counterfactual=cf,
                    predict_fn=predict_fn,
                    predict_proba_fn=predict_proba_fn,
                    target_class=target,
                    n_samples=config.gaussian_samples,
                    sigma=config.gaussian_sigma,
                    rng=rng,
                )
                correction_name = "validity"
                correction_vector = gaussian_sampling_validity_vector(
                    neighborhood=neighborhood,
                    sigma=config.gaussian_sigma,
                )
            elif scheduled_step_due(update_index, config.plausibility_every_steps):
                correction_name = "plausibility"
                correction_vector = plausibility_vector(
                    counterfactual=cf,
                    buffer_index=buffer_index,
                    target_class=target,
                    config=config,
                )
            else:
                return replace(current_state, counterfactual=cf, last_update=last_update)
        elif config.mode == PLAUSIBILITY_LOW_MARGIN:
            if not low_probability:
                return replace(current_state, counterfactual=cf, last_update=last_update)
            correction_name = "plausibility"
            correction_vector = plausibility_vector(
                counterfactual=cf,
                buffer_index=buffer_index,
                target_class=target,
                config=config,
            )
        else:
            raise ValueError(f"Unknown CFE update mode: {config.mode}")

        if np.allclose(correction_vector, 0.0):
            return replace(current_state, counterfactual=cf, last_update="stalled")

        direction = combine_direction_components(
            components=correction_components(
                x_ref=x_ref,
                counterfactual=cf,
                correction_name=correction_name,
                correction_vector=correction_vector,
                config=config,
            ),
            normalize_vectors=config.normalize_vectors,
        )
        if np.allclose(direction, 0.0):
            return replace(current_state, counterfactual=cf, last_update="stalled")

        cf = clip_candidate(cf + config.step_size * direction, config)
        last_update = f"{correction_name}+proximity"
        current_state = replace(
            current_state,
            counterfactual=cf,
            last_update=last_update,
        )

    return replace(current_state, counterfactual=cf, last_update=last_update)


def update_counterfactual_states(
    states: list[CounterfactualState],
    predict_fn: PredictFn,
    predict_proba_fn: PredictProbaFn,
    buffer_index: CurrentBufferIndex,
    config: CFEUpdateConfig,
    update_index: int,
    rng: np.random.Generator,
) -> list[CounterfactualState]:
    return [
        update_counterfactual_state(
            state=state,
            predict_fn=predict_fn,
            predict_proba_fn=predict_proba_fn,
            buffer_index=buffer_index,
            config=config,
            update_index=update_index,
            rng=rng,
        )
        for state in states
    ]


def refresh_counterfactual_states(
    states: list[CounterfactualState],
    predict_fn: PredictFn,
) -> list[CounterfactualState]:
    refreshed_states = []
    for state in states:
        if not state.is_active:
            refreshed_states.append(state)
            continue

        x_ref = np.asarray(state.x_ref, dtype=float).reshape(1, -1)
        if int(predict_fn(x_ref)[0]) == int(state.target_class):
            refreshed_states.append(
                replace(
                    state,
                    is_active=False,
                    retirement_reason="naturally_resolved",
                    last_update="retired",
                )
            )
            continue

        refreshed_states.append(state)

    return refreshed_states


__all__ = [
    "CFEUpdateConfig",
    "CounterfactualState",
    "CurrentBufferIndex",
    "GaussianNeighborhood",
    "PLAUSIBILITY_LOW_MARGIN",
    "PredictFn",
    "PredictProbaFn",
    "VALIDITY_PLAUSIBILITY",
    "build_current_buffer_index",
    "combine_direction_components",
    "gaussian_sampling_validity_vector",
    "plausibility_vector",
    "proximity_direction",
    "refresh_counterfactual_states",
    "sample_gaussian_neighborhood",
    "scheduled_step_due",
    "target_class_probabilities",
    "target_class_probability",
    "update_counterfactual_state",
    "update_counterfactual_states",
]
