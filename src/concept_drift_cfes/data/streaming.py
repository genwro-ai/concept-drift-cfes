from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np
from river import datasets
from river.datasets import base as datasets_base

StreamFactory = Callable[[], datasets_base.SyntheticDataset]


@dataclass(frozen=True)
class RiverStreamSpec:
    name: str
    description: str
    factory: StreamFactory
    parameters: dict[str, Any] = field(default_factory=dict)


class SmoothHyperplaneDriftStream(datasets_base.SyntheticDataset):
    """Two-dimensional hyperplane stream with sigmoid-interpolated boundary."""

    def __init__(
        self,
        seed: int = 42,
        position: int = 1_500,
        width: int = 1_000,
        noise_percentage: float = 0.0,
    ) -> None:
        super().__init__(
            n_features=2,
            n_classes=2,
            n_outputs=1,
            task=datasets_base.BINARY_CLF,
        )
        if width <= 0:
            raise ValueError("width must be positive.")
        if not 0.0 <= noise_percentage <= 1.0:
            raise ValueError("noise_percentage must be in [0, 1].")

        self.seed = seed
        self.position = position
        self.width = width
        self.noise_percentage = noise_percentage

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        start_normal = np.asarray([1.0, 0.25], dtype=float)
        end_normal = np.asarray([0.25, 1.0], dtype=float)

        sample_idx = 0
        while True:
            sample_idx += 1
            alpha = transition_weight(sample_idx, self.position, self.width)
            normal = normalize_vector((1.0 - alpha) * start_normal + alpha * end_normal)
            x_array = rng.random(2)
            threshold = 0.5 * normal.sum()
            label = int(float(np.dot(normal, x_array)) >= threshold)
            if rng.random() < self.noise_percentage:
                label = 1 - label
            yield {idx: float(value) for idx, value in enumerate(x_array)}, label


class SmoothSineDriftStream(datasets_base.SyntheticDataset):
    """Two-dimensional stream with a continuously interpolated sine boundary."""

    def __init__(
        self,
        seed: int = 42,
        position: int = 1_500,
        width: int = 1_000,
        noise_percentage: float = 0.0,
    ) -> None:
        super().__init__(
            n_features=2,
            n_classes=2,
            n_outputs=1,
            task=datasets_base.BINARY_CLF,
        )
        if width <= 0:
            raise ValueError("width must be positive.")
        if not 0.0 <= noise_percentage <= 1.0:
            raise ValueError("noise_percentage must be in [0, 1].")

        self.seed = seed
        self.position = position
        self.width = width
        self.noise_percentage = noise_percentage

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        sample_idx = 0
        while True:
            sample_idx += 1
            alpha = transition_weight(sample_idx, self.position, self.width)
            x_array = rng.random(2)
            start_boundary = np.sin(x_array[1])
            end_boundary = 0.5 + 0.3 * np.sin(3.0 * np.pi * x_array[1])
            boundary = (1.0 - alpha) * start_boundary + alpha * end_boundary
            label = int(x_array[0] < boundary)
            if rng.random() < self.noise_percentage:
                label = 1 - label
            yield {idx: float(value) for idx, value in enumerate(x_array)}, label


class SmoothSEADriftStream(datasets_base.SyntheticDataset):
    """SEA-like stream with continuously interpolated threshold."""

    def __init__(
        self,
        seed: int = 42,
        position: int = 1_500,
        width: int = 1_000,
        start_threshold: float = 8.0,
        end_threshold: float = 7.0,
        noise_percentage: float = 0.0,
    ) -> None:
        super().__init__(
            n_features=3,
            n_classes=2,
            n_outputs=1,
            task=datasets_base.BINARY_CLF,
        )
        if width <= 0:
            raise ValueError("width must be positive.")
        if not 0.0 <= noise_percentage <= 1.0:
            raise ValueError("noise_percentage must be in [0, 1].")

        self.seed = seed
        self.position = position
        self.width = width
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.noise_percentage = noise_percentage

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        sample_idx = 0
        while True:
            sample_idx += 1
            alpha = transition_weight(sample_idx, self.position, self.width)
            threshold = (
                1.0 - alpha
            ) * self.start_threshold + alpha * self.end_threshold
            x_array = rng.uniform(0.0, 10.0, size=3)
            label = int(x_array[0] + x_array[1] > threshold)
            if rng.random() < self.noise_percentage:
                label = 1 - label
            yield {idx: float(value) for idx, value in enumerate(x_array)}, label


class GaussianDriftStream(datasets_base.SyntheticDataset):
    """Binary Gaussian stream with smooth parameter drift.

    The class-conditional means interpolate continuously from an initial
    configuration to a final configuration. This gives a higher-dimensional
    synthetic stream whose drift is incremental rather than a sample-level
    mixture of two fixed River generators.
    """

    def __init__(
        self,
        seed: int = 42,
        n_features: int = 10,
        position: int = 1_500,
        width: int = 1_000,
        class_sep: float = 1.8,
        drift_strength: float = 1.1,
        noise: float = 0.45,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_classes=2,
            n_outputs=1,
            task=datasets_base.BINARY_CLF,
        )
        if n_features < 2:
            raise ValueError("n_features must be at least 2.")
        if width <= 0:
            raise ValueError("width must be positive.")
        if noise <= 0:
            raise ValueError("noise must be positive.")

        self.seed = seed
        self.position = position
        self.width = width
        self.class_sep = class_sep
        self.drift_strength = drift_strength
        self.noise = noise

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        start_direction = self._unit_vector([1.0, 0.35])
        end_direction = self._unit_vector([0.25, 1.0])
        shared_shift = np.zeros(self.n_features, dtype=float)
        shared_shift[: min(4, self.n_features)] = np.linspace(
            0.15,
            self.drift_strength,
            min(4, self.n_features),
        )

        sample_idx = 0
        while True:
            sample_idx += 1
            alpha = self._transition_weight(sample_idx)
            direction = self._unit_vector(
                (1.0 - alpha) * start_direction + alpha * end_direction
            )
            center = alpha * shared_shift
            label = int(rng.random() >= 0.5)
            sign = 1.0 if label == 1 else -1.0
            mean = center + sign * self.class_sep * direction
            x = rng.normal(loc=mean, scale=self.noise, size=self.n_features)
            yield {idx: float(value) for idx, value in enumerate(x)}, label

    def _transition_weight(self, sample_idx: int) -> float:
        return transition_weight(sample_idx, self.position, self.width)

    def _unit_vector(self, values: list[float] | np.ndarray) -> np.ndarray:
        vector = np.zeros(self.n_features, dtype=float)
        values_array = np.asarray(values, dtype=float)
        vector[: len(values_array)] = values_array
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm


def transition_weight(sample_idx: int, position: int, width: int) -> float:
    exponent = -4.0 * float(sample_idx - position) / float(width)
    return float(1.0 / (1.0 + np.exp(exponent)))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def make_hyperplane(
    seed: int = 42,
    n_features: int = 2,
    n_drift_features: int = 2,
    mag_change: float = 0.01,
    sigma: float = 0.1,
    noise_percentage: float = 0.0,
) -> datasets_base.SyntheticDataset:
    return datasets.synth.Hyperplane(
        seed=seed,
        n_features=n_features,
        n_drift_features=n_drift_features,
        mag_change=mag_change,
        sigma=sigma,
        noise_percentage=noise_percentage,
    )


def make_sine(
    classification_function: int = 0,
    seed: int = 42,
    balance_classes: bool = True,
    has_noise: bool = False,
) -> datasets_base.SyntheticDataset:
    return datasets.synth.Sine(
        classification_function=classification_function,
        seed=seed,
        balance_classes=balance_classes,
        has_noise=has_noise,
    )


def make_sea(
    variant: int = 0,
    noise: float = 0.0,
    seed: int = 42,
) -> datasets_base.SyntheticDataset:
    return datasets.synth.SEA(variant=variant, noise=noise, seed=seed)


def make_concept_drift_stream(
    stream: datasets_base.SyntheticDataset,
    drift_stream: datasets_base.SyntheticDataset,
    position: int = 1_500,
    width: int = 800,
    seed: int = 42,
) -> datasets_base.SyntheticDataset:
    return datasets.synth.ConceptDriftStream(
        stream=stream,
        drift_stream=drift_stream,
        position=position,
        width=width,
        seed=seed,
    )


def make_gaussian_drift(
    seed: int = 42,
    n_features: int = 10,
    position: int = 1_500,
    width: int = 1_000,
    class_sep: float = 1.8,
    drift_strength: float = 1.1,
    noise: float = 0.45,
) -> GaussianDriftStream:
    return GaussianDriftStream(
        seed=seed,
        n_features=n_features,
        position=position,
        width=width,
        class_sep=class_sep,
        drift_strength=drift_strength,
        noise=noise,
    )


def make_smooth_hyperplane_drift(
    seed: int = 42,
    position: int = 1_500,
    width: int = 1_000,
    noise_percentage: float = 0.0,
) -> SmoothHyperplaneDriftStream:
    return SmoothHyperplaneDriftStream(
        seed=seed,
        position=position,
        width=width,
        noise_percentage=noise_percentage,
    )


def make_smooth_sine_drift(
    seed: int = 42,
    position: int = 1_500,
    width: int = 1_000,
    noise_percentage: float = 0.0,
) -> SmoothSineDriftStream:
    return SmoothSineDriftStream(
        seed=seed,
        position=position,
        width=width,
        noise_percentage=noise_percentage,
    )


def make_smooth_sea_drift(
    seed: int = 42,
    position: int = 1_500,
    width: int = 1_000,
    start_threshold: float = 8.0,
    end_threshold: float = 7.0,
    noise_percentage: float = 0.0,
) -> SmoothSEADriftStream:
    return SmoothSEADriftStream(
        seed=seed,
        position=position,
        width=width,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        noise_percentage=noise_percentage,
    )


def make_gradual_sea_drift(
    seed: int = 42,
    start_variant: int = 0,
    end_variant: int = 2,
    noise: float = 0.0,
    position: int = 1_500,
    width: int = 800,
) -> datasets_base.SyntheticDataset:
    return make_concept_drift_stream(
        stream=make_sea(variant=start_variant, noise=noise, seed=seed),
        drift_stream=make_sea(variant=end_variant, noise=noise, seed=seed + 1),
        position=position,
        width=width,
        seed=seed + 2,
    )


def make_gradual_sine_drift(
    seed: int = 42,
    start_function: int = 0,
    end_function: int = 2,
    balance_classes: bool = True,
    has_noise: bool = False,
    position: int = 1_500,
    width: int = 800,
) -> datasets_base.SyntheticDataset:
    return make_concept_drift_stream(
        stream=make_sine(
            classification_function=start_function,
            seed=seed,
            balance_classes=balance_classes,
            has_noise=has_noise,
        ),
        drift_stream=make_sine(
            classification_function=end_function,
            seed=seed + 1,
            balance_classes=balance_classes,
            has_noise=has_noise,
        ),
        position=position,
        width=width,
        seed=seed + 2,
    )


def make_gradual_hyperplane_drift(
    seed: int = 42,
    n_features: int = 2,
    n_drift_features: int = 2,
    initial_mag_change: float = 0.0,
    initial_sigma: float = 0.0,
    drift_mag_change: float = 0.0,
    drift_sigma: float = 0.0,
    noise_percentage: float = 0.0,
    position: int = 1_500,
    width: int = 600,
) -> datasets_base.SyntheticDataset:
    return make_concept_drift_stream(
        stream=make_hyperplane(
            seed=seed,
            n_features=n_features,
            n_drift_features=n_drift_features,
            mag_change=initial_mag_change,
            sigma=initial_sigma,
            noise_percentage=noise_percentage,
        ),
        drift_stream=make_hyperplane(
            seed=seed + 1,
            n_features=n_features,
            n_drift_features=n_drift_features,
            mag_change=drift_mag_change,
            sigma=drift_sigma,
            noise_percentage=noise_percentage,
        ),
        position=position,
        width=width,
        seed=seed + 2,
    )


def get_default_stream_specs(seed: int = 42) -> dict[str, RiverStreamSpec]:
    return {
        "hyperplane": RiverStreamSpec(
            name="hyperplane",
            description=(
                "Two-dimensional hyperplane with continuously interpolated "
                "decision-boundary orientation."
            ),
            factory=lambda: make_smooth_hyperplane_drift(seed=seed),
            parameters={
                "seed": seed,
                "position": 1_500,
                "width": 1_000,
                "noise_percentage": 0.0,
            },
        ),
        "sine_gradual": RiverStreamSpec(
            name="sine_gradual",
            description=(
                "Two-dimensional nonlinear boundary with continuous interpolation "
                "from SINE1 to SINE2."
            ),
            factory=lambda: make_smooth_sine_drift(seed=seed),
            parameters={
                "seed": seed,
                "position": 1_500,
                "width": 1_000,
                "noise_percentage": 0.0,
            },
        ),
        "sea_gradual": RiverStreamSpec(
            name="sea_gradual",
            description=(
                "SEA-like stream with continuously interpolated threshold from "
                "variant 0 to variant 2."
            ),
            factory=lambda: make_smooth_sea_drift(seed=seed),
            parameters={
                "seed": seed,
                "position": 1_500,
                "width": 1_000,
                "start_threshold": 8.0,
                "end_threshold": 7.0,
                "noise_percentage": 0.0,
            },
        ),
        "gaussian_drift": RiverStreamSpec(
            name="gaussian_drift",
            description=(
                "Ten-dimensional class-conditional Gaussian stream with smoothly "
                "interpolated means and separating direction."
            ),
            factory=lambda: make_gaussian_drift(seed=seed),
            parameters={
                "seed": seed,
                "n_features": 10,
                "position": 1_500,
                "width": 1_000,
                "class_sep": 1.8,
                "drift_strength": 1.1,
                "noise": 0.45,
            },
        ),
    }


def stream_to_array(
    stream: datasets_base.SyntheticDataset,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    rows = list(islice(iter(stream), n_samples))
    if not rows:
        raise ValueError("The stream did not yield any samples.")

    feature_keys = tuple(sorted(rows[0][0].keys()))
    x = np.asarray(
        [[features[key] for key in feature_keys] for features, _ in rows],
        dtype=float,
    )
    y = np.asarray([int(target) for _, target in rows], dtype=int)
    return x, y, feature_keys


def make_window_slices(n_samples: int, window_size: int) -> dict[str, slice]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    bounded_window = min(window_size, n_samples)
    middle_start = max((n_samples - bounded_window) // 2, 0)
    late_start = max(n_samples - bounded_window, 0)

    return {
        "Early": slice(0, bounded_window),
        "Middle": slice(middle_start, middle_start + bounded_window),
        "Late": slice(late_start, late_start + bounded_window),
    }
