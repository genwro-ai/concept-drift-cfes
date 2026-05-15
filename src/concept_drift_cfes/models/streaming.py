from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice

import numpy as np
from river import linear_model, optim, preprocessing, tree

from concept_drift_cfes.data.streaming import StreamFactory

ModelFactory = Callable[[], object]


@dataclass(frozen=True)
class OnlineClassifierSpec:
    name: str
    description: str
    factory: ModelFactory


@dataclass(frozen=True)
class OnlineEvaluationResult:
    steps: np.ndarray
    cumulative_accuracy: np.ndarray
    rolling_accuracy: np.ndarray
    final_cumulative_accuracy: float
    final_rolling_accuracy: float


def make_adaptive_hoeffding_tree(
    seed: int = 42,
) -> tree.HoeffdingAdaptiveTreeClassifier:
    return tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=200,
        delta=1e-5,
        seed=seed,
    )


def make_online_logistic_regression(
    learning_rate: float = 0.05,
) -> object:
    return preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(lr=learning_rate),
    )


def get_default_classifier_specs(seed: int = 42) -> dict[str, OnlineClassifierSpec]:
    return {
        "adaptive_hoeffding_tree": OnlineClassifierSpec(
            name="adaptive_hoeffding_tree",
            description="Hoeffding tree with drift adaptation.",
            factory=lambda: make_adaptive_hoeffding_tree(seed=seed),
        ),
        "online_logistic_regression": OnlineClassifierSpec(
            name="online_logistic_regression",
            description="Scaled logistic regression trained one sample at a time.",
            factory=make_online_logistic_regression,
        ),
    }


def evaluate_progressive_accuracy(
    stream_factory: StreamFactory,
    model_factory: ModelFactory,
    n_samples: int,
    window_size: int = 200,
) -> OnlineEvaluationResult:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    stream = stream_factory()
    model = model_factory()

    cumulative_accuracy = np.empty(n_samples, dtype=float)
    rolling_accuracy = np.empty(n_samples, dtype=float)
    recent_correct: deque[int] = deque(maxlen=window_size)
    correct_predictions = 0

    for index, (x, y) in enumerate(islice(iter(stream), n_samples)):
        y_pred = model.predict_one(x)
        is_correct = int(y_pred == y)
        correct_predictions += is_correct
        recent_correct.append(is_correct)
        cumulative_accuracy[index] = correct_predictions / (index + 1)
        rolling_accuracy[index] = sum(recent_correct) / len(recent_correct)
        model.learn_one(x, y)

    steps = np.arange(1, n_samples + 1, dtype=int)
    return OnlineEvaluationResult(
        steps=steps,
        cumulative_accuracy=cumulative_accuracy,
        rolling_accuracy=rolling_accuracy,
        final_cumulative_accuracy=float(cumulative_accuracy[-1]),
        final_rolling_accuracy=float(rolling_accuracy[-1]),
    )


def evaluate_classifier_suite(
    stream_factory: StreamFactory,
    classifier_specs: dict[str, OnlineClassifierSpec],
    n_samples: int,
    window_size: int = 200,
) -> dict[str, OnlineEvaluationResult]:
    return {
        name: evaluate_progressive_accuracy(
            stream_factory=stream_factory,
            model_factory=spec.factory,
            n_samples=n_samples,
            window_size=window_size,
        )
        for name, spec in classifier_specs.items()
    }
