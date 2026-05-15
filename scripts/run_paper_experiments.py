import argparse
import copy
import csv
import logging
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from concept_drift_cfes.data import (
    DatasetBatch,
    InitialDataSplit,
    fit_minmax_feature_scaler,
    make_initial_data_split,
)
from concept_drift_cfes.data.streaming import get_default_stream_specs
from concept_drift_cfes.models.streaming import get_default_classifier_specs
from concept_drift_cfes.reference.growing_spheres import growing_spheres_search
from concept_drift_cfes.reference.robx import RobX
from concept_drift_cfes.update import (
    PLAUSIBILITY_LOW_MARGIN,
    VALIDITY_PLAUSIBILITY,
    CFEUpdateConfig,
    CounterfactualState,
    build_current_buffer_index,
    refresh_counterfactual_states,
    update_counterfactual_states,
)
from concept_drift_cfes.utils import (
    checkpoint_streamed_samples,
    evaluate_cfe_metrics,
    iter_river_dicts,
    learn_on_batch,
    make_checkpoint_batch_indices,
    predict_batch,
    predict_proba_batch,
    target_flip_classes,
)

DEFAULT_STREAMS = ["hyperplane", "sine_gradual", "sea_gradual"]
DEFAULT_CLASSIFIERS = ["adaptive_hoeffding_tree", "online_logistic_regression"]
DEFAULT_INITIAL_GENERATORS = ["robx", "growing_spheres"]
UPDATE_STRATEGIES = ["validity_plausibility_update", "plausibility_low_margin_update"]
BASELINE_STRATEGIES = [
    "frozen",
    "dynamic_nearest_neighbor",
    "dynamic_growing_spheres",
    "final_robx",
]
LOGGER = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper CFE maintenance comparison under concept drift.",
    )
    parser.add_argument("--n-init", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-stream-samples", type=int, default=2_000)
    parser.add_argument("--n-checkpoints", type=int, default=8)
    parser.add_argument("--checkpoint-every-batches", type=int, default=None)
    parser.add_argument("--buffer-size-batches", type=int, default=5)
    parser.add_argument("--update-every-points", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-query", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_runs"))
    parser.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS)
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--initial-generators", nargs="+", default=DEFAULT_INITIAL_GENERATORS)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-feature-normalization", action="store_true")

    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--validity-weight", type=float, default=2.0)
    parser.add_argument("--plausibility-weight", type=float, default=2.0)
    parser.add_argument("--proximity-weight", type=float, default=1.0)
    parser.add_argument("--neighborhood-size", type=int, default=64)
    parser.add_argument("--low-margin-threshold", type=float, default=0.6)
    parser.add_argument("--plausibility-every-steps", type=int, default=60)
    parser.add_argument("--max-update-steps", type=int, default=1)
    parser.add_argument("--target-region-bandwidth", type=float, default=0.3)
    parser.add_argument("--gaussian-samples", type=int, default=128)
    parser.add_argument("--gaussian-sigma", type=float, default=0.1)

    parser.add_argument("--gs-search-samples", type=int, default=1_000)
    parser.add_argument("--gs-step", type=float, default=0.2)
    parser.add_argument("--gs-max-iter", type=int, default=1_000)
    parser.add_argument("--robx-tau", type=float, default=0.45)
    parser.add_argument("--robx-stability-samples", type=int, default=100)
    parser.add_argument("--robx-max-iter", type=int, default=150)
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def make_update_config(args: argparse.Namespace, mode: str) -> CFEUpdateConfig:
    return CFEUpdateConfig(
        mode=mode,
        step_size=args.step_size,
        validity_weight=args.validity_weight,
        plausibility_weight=args.plausibility_weight,
        proximity_weight=args.proximity_weight,
        neighborhood_size=args.neighborhood_size,
        low_margin_threshold=args.low_margin_threshold,
        plausibility_every_steps=args.plausibility_every_steps,
        max_update_steps=args.max_update_steps,
        target_region_bandwidth=args.target_region_bandwidth,
        gaussian_samples=args.gaussian_samples,
        gaussian_sigma=args.gaussian_sigma,
    )


def normalize_split_and_batches(
    split: InitialDataSplit,
    stream_batches: list[DatasetBatch],
) -> tuple[InitialDataSplit, list[DatasetBatch]]:
    scaler = fit_minmax_feature_scaler(split.train)
    normalized_batches = [scaler.transform_batch(batch) for batch in stream_batches]
    normalized_split = InitialDataSplit(
        train=scaler.transform_batch(split.train),
        val=scaler.transform_batch(split.val),
        test=scaler.transform_batch(split.test),
        stream_batches=iter(normalized_batches),
        feature_names=split.feature_names,
    )
    return normalized_split, normalized_batches


def trim_recent_buffer(
    x: np.ndarray,
    y: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_size:
        return x, y
    return x[-max_size:], y[-max_size:]


def extend_recent_buffer(
    recent_x: np.ndarray,
    recent_y: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    combined_x = np.vstack([recent_x, x_new])
    combined_y = np.concatenate([recent_y, y_new])
    return trim_recent_buffer(combined_x, combined_y, max_size=max_size)


def split_batch_for_updates(batch: DatasetBatch, chunk_size: int):
    for start in range(0, len(batch.x), chunk_size):
        end = min(start + chunk_size, len(batch.x))
        yield DatasetBatch(
            x=batch.x[start:end],
            y=batch.y[start:end],
            feature_names=batch.feature_names,
        )


def update_recent_prequential_accuracy(
    recent_correct: list[int],
    model: object,
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    feature_names: tuple[object, ...],
    max_size: int,
) -> float:
    for x_dict, target in zip(
        iter_river_dicts(batch_x, feature_names),
        np.asarray(batch_y, dtype=int).reshape(-1),
        strict=True,
    ):
        prediction = int(model.predict_one(x_dict))
        recent_correct.append(int(prediction == int(target)))
    if len(recent_correct) > max_size:
        del recent_correct[:-max_size]
    return float(np.mean(recent_correct)) if recent_correct else np.nan


def select_queries(
    x: np.ndarray,
    predictions: np.ndarray,
    n_query: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(x))
    if n_query is not None:
        rng.shuffle(indices)
        indices = indices[: min(n_query, len(indices))]
    query_x = np.asarray(x[indices], dtype=float)
    targets = target_flip_classes(np.asarray(predictions[indices], dtype=int))
    return query_x, targets


def generate_initial_counterfactuals(
    generator_name: str,
    split: InitialDataSplit,
    model: object,
    query_x: np.ndarray,
    target_classes: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def predict_fn(x: np.ndarray) -> np.ndarray:
        return predict_batch(model, x, split.feature_names)

    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        return predict_proba_batch(model, x, split.feature_names)

    robx = None
    if generator_name == "robx":
        robx = RobX(
            X_train=split.train.x,
            predict_fn_crisp=predict_fn,
            predict_proba_fn=predict_proba_fn,
        )

    x_refs, cfs, targets = [], [], []
    iterator = zip(query_x, target_classes, strict=True)
    if not args.no_progress:
        iterator = tqdm(
            iterator,
            total=len(query_x),
            desc=f"generate initial {generator_name}",
            leave=False,
        )

    for x_ref, target_class in iterator:
        if generator_name == "robx":
            result = robx.generate(
                start_instance=x_ref,
                target_class=int(target_class),
                tau=args.robx_tau,
                N=args.robx_stability_samples,
                robx_max_iter=args.robx_max_iter,
                gs_n_search_samples=args.gs_search_samples,
                gs_step=args.gs_step,
                gs_max_iter=args.gs_max_iter,
                rng=rng,
            )
            cf = result.counterfactual
        elif generator_name == "growing_spheres":
            cf = growing_spheres_search(
                instance=x_ref,
                pred_fn_crisp=predict_fn,
                target_class=int(target_class),
                n_search_samples=args.gs_search_samples,
                step=args.gs_step,
                max_iter=args.gs_max_iter,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown initial generator: {generator_name}")

        if cf is None:
            continue
        x_refs.append(np.asarray(x_ref, dtype=float))
        cfs.append(np.asarray(cf, dtype=float))
        targets.append(int(target_class))

    if not cfs:
        empty = np.empty((0, query_x.shape[1]), dtype=float)
        return empty, empty, np.empty(0, dtype=int)
    return (
        np.asarray(x_refs, dtype=float),
        np.asarray(cfs, dtype=float),
        np.asarray(targets, dtype=int),
    )


def build_initial_states(
    x_ref: np.ndarray,
    cfs: np.ndarray,
    target_classes: np.ndarray,
) -> list[CounterfactualState]:
    return [
        CounterfactualState(x_ref=x_item, counterfactual=cf_item, target_class=target)
        for x_item, cf_item, target in zip(x_ref, cfs, target_classes, strict=True)
    ]


def state_arrays(
    states: list[CounterfactualState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_states = [state for state in states if state.is_active]
    if not active_states:
        return (
            np.empty((0, 0), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty(0, dtype=int),
        )
    return (
        np.asarray([state.x_ref for state in active_states], dtype=float),
        np.asarray([state.counterfactual for state in active_states], dtype=float),
        np.asarray([state.target_class for state in active_states], dtype=int),
    )


def nearest_model_target_counterfactual(
    x_ref: np.ndarray,
    target_class: int,
    reference_x: np.ndarray,
    predict_fn,
) -> np.ndarray | None:
    predictions = np.asarray(predict_fn(reference_x), dtype=int).reshape(-1)
    candidates = reference_x[predictions == int(target_class)]
    if len(candidates) == 0:
        return None
    distances = np.linalg.norm(candidates - np.asarray(x_ref, dtype=float), axis=1)
    return np.asarray(candidates[int(np.argmin(distances))], dtype=float)


def refresh_dynamic_nearest_neighbor_states(
    states: list[CounterfactualState],
    predict_fn,
    reference_x: np.ndarray,
) -> list[CounterfactualState]:
    refreshed = []
    for state in states:
        if not state.is_active:
            refreshed.append(state)
            continue
        cf = nearest_model_target_counterfactual(
            x_ref=state.x_ref,
            target_class=state.target_class,
            reference_x=reference_x,
            predict_fn=predict_fn,
        )
        refreshed.append(
            state
            if cf is None
            else replace(
                state,
                counterfactual=cf,
                last_update="dynamic_nearest_neighbor",
            )
        )
    return refreshed


def refresh_dynamic_growing_spheres_states(
    states: list[CounterfactualState],
    predict_fn,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[CounterfactualState]:
    refreshed = []
    for state in states:
        if not state.is_active:
            refreshed.append(state)
            continue
        cf = growing_spheres_search(
            instance=state.x_ref,
            pred_fn_crisp=predict_fn,
            target_class=int(state.target_class),
            n_search_samples=args.gs_search_samples,
            step=args.gs_step,
            max_iter=args.gs_max_iter,
            rng=rng,
        )
        refreshed.append(
            state
            if cf is None
            else replace(
                state,
                counterfactual=np.asarray(cf, dtype=float),
                last_update="dynamic_growing_spheres",
            )
        )
    return refreshed


def generate_final_robx_states(
    states: list[CounterfactualState],
    model: object,
    reference_x: np.ndarray,
    split: InitialDataSplit,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[CounterfactualState]:
    def predict_fn(x: np.ndarray) -> np.ndarray:
        return predict_batch(model, x, split.feature_names)

    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        return predict_proba_batch(model, x, split.feature_names)

    robx = RobX(
        X_train=reference_x,
        predict_fn_crisp=predict_fn,
        predict_proba_fn=predict_proba_fn,
    )
    refreshed = []
    for state in states:
        if not state.is_active:
            refreshed.append(state)
            continue
        result = robx.generate(
            start_instance=state.x_ref,
            target_class=int(state.target_class),
            tau=args.robx_tau,
            N=args.robx_stability_samples,
            robx_max_iter=args.robx_max_iter,
            gs_n_search_samples=args.gs_search_samples,
            gs_step=args.gs_step,
            gs_max_iter=args.gs_max_iter,
            rng=rng,
        )
        if result.counterfactual is None:
            refreshed.append(
                replace(
                    state,
                    is_active=False,
                    retirement_reason="regeneration_failed",
                    last_update=f"final_robx_failed:{result.metadata['status']}",
                )
            )
            continue
        refreshed.append(
            replace(
                state,
                counterfactual=np.asarray(result.counterfactual, dtype=float),
                last_update="final_robx",
            )
        )
    return refreshed


def append_row(
    rows: list[dict[str, object]],
    repeat: int,
    strategy: str,
    initial_generator: str,
    stream_name: str,
    classifier_name: str,
    split: InitialDataSplit,
    checkpoint: int,
    streamed_samples: int,
    model: object,
    classifier_accuracy: float,
    states: list[CounterfactualState],
    reference_x: np.ndarray,
    reference_y: np.ndarray,
) -> None:
    active_states = [state for state in states if state.is_active]
    retired_states = [state for state in states if not state.is_active]
    row = {
        "repeat": repeat,
        "stream": stream_name,
        "classifier": classifier_name,
        "initial_generator": initial_generator,
        "strategy": strategy,
        "checkpoint": checkpoint,
        "streamed_samples": streamed_samples,
        "classifier_accuracy": classifier_accuracy,
        "n_total": len(states),
        "n_active": len(active_states),
        "n_retired_natural": sum(
            state.retirement_reason == "naturally_resolved"
            for state in retired_states
        ),
    }

    if not active_states:
        metric_names = [
            "cfe_validity",
            "cfe_l1",
            "cfe_l2",
            "cfe_sparsity",
            "cfe_target_neighbor_distance",
            "cfe_target_neighbor_ratio",
            "cfe_target_kernel_log_density",
            "cfe_isolation_forest_score",
            "cfe_local_outlier_factor_score",
        ]
        row.update({name: np.nan for name in metric_names})
        rows.append(row)
        return

    x_ref, cfs, target_classes = state_arrays(states)

    def predict_fn(x: np.ndarray, current_model=model) -> np.ndarray:
        return predict_batch(current_model, x, split.feature_names)

    try:
        metrics = evaluate_cfe_metrics(
            x=x_ref,
            counterfactuals=cfs,
            target_class=target_classes,
            predict_fn=predict_fn,
            reference_x=reference_x,
            reference_y=reference_y,
        )
        row.update(
            {
                "cfe_validity": metrics.validity,
                "cfe_l1": metrics.l1,
                "cfe_l2": metrics.l2,
                "cfe_sparsity": metrics.sparsity,
                "cfe_target_neighbor_distance": metrics.target_neighbor_distance,
                "cfe_target_neighbor_ratio": metrics.target_neighbor_ratio,
                "cfe_target_kernel_log_density": metrics.target_kernel_log_density,
                "cfe_isolation_forest_score": metrics.isolation_forest_score,
                "cfe_local_outlier_factor_score": metrics.local_outlier_factor_score,
            }
        )
    except ValueError as exc:
        LOGGER.warning("metric evaluation failed: %s", exc)
        row.update(
            {
                "cfe_validity": np.nan,
                "cfe_l1": np.nan,
                "cfe_l2": np.nan,
                "cfe_sparsity": np.nan,
                "cfe_target_neighbor_distance": np.nan,
                "cfe_target_neighbor_ratio": np.nan,
                "cfe_target_kernel_log_density": np.nan,
                "cfe_isolation_forest_score": np.nan,
                "cfe_local_outlier_factor_score": np.nan,
            }
        )
    rows.append(row)


def initialize_states_by_strategy(
    split: InitialDataSplit,
    initial_model: object,
    query_x: np.ndarray,
    target_classes: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[tuple[str, str], list[CounterfactualState]]:
    states_by_key = {}
    for generator_name in args.initial_generators:
        x_ref, cfs, targets = generate_initial_counterfactuals(
            generator_name=generator_name,
            split=split,
            model=initial_model,
            query_x=query_x,
            target_classes=target_classes,
            rng=rng,
            args=args,
        )
        LOGGER.info("generated %d initial CFEs with %s", len(cfs), generator_name)
        if len(cfs) == 0:
            continue
        base_states = build_initial_states(x_ref, cfs, targets)
        for strategy in [*BASELINE_STRATEGIES[:-1], *UPDATE_STRATEGIES]:
            states_by_key[(generator_name, strategy)] = copy.deepcopy(base_states)
    return states_by_key


def append_all_strategy_rows(
    rows: list[dict[str, object]],
    states_by_key: dict[tuple[str, str], list[CounterfactualState]],
    repeat: int,
    stream_name: str,
    classifier_name: str,
    split: InitialDataSplit,
    checkpoint: int,
    streamed_samples: int,
    model: object,
    classifier_accuracy: float,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
) -> None:
    for (initial_generator, strategy), states in states_by_key.items():
        append_row(
            rows=rows,
            repeat=repeat,
            strategy=strategy,
            initial_generator=initial_generator,
            stream_name=stream_name,
            classifier_name=classifier_name,
            split=split,
            checkpoint=checkpoint,
            streamed_samples=streamed_samples,
            model=model,
            classifier_accuracy=classifier_accuracy,
            states=states,
            reference_x=reference_x,
            reference_y=reference_y,
        )


def refresh_dynamic_baselines(
    states_by_key: dict[tuple[str, str], list[CounterfactualState]],
    predict_fn,
    reference_x: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    for key, states in list(states_by_key.items()):
        _, strategy = key
        if strategy == "dynamic_nearest_neighbor":
            states_by_key[key] = refresh_dynamic_nearest_neighbor_states(
                states=states,
                predict_fn=predict_fn,
                reference_x=reference_x,
            )
        elif strategy == "dynamic_growing_spheres":
            states_by_key[key] = refresh_dynamic_growing_spheres_states(
                states=states,
                predict_fn=predict_fn,
                args=args,
                rng=rng,
            )


def run_single_experiment(
    repeat: int,
    stream_name: str,
    classifier_name: str,
    split: InitialDataSplit,
    classifier_factory,
    stream_batches: list[DatasetBatch],
    checkpoint_indices: list[int],
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    max_reference_size = args.buffer_size_batches * max(1, len(stream_batches[0].x))
    validity_config = make_update_config(args, VALIDITY_PLAUSIBILITY)
    low_margin_config = make_update_config(args, PLAUSIBILITY_LOW_MARGIN)

    initial_model = classifier_factory()
    learn_on_batch(initial_model, split.train.x, split.train.y, split.feature_names)
    initial_predictions = predict_batch(initial_model, split.test.x, split.feature_names)
    query_x, target_classes = select_queries(
        x=split.test.x,
        predictions=initial_predictions,
        n_query=args.n_query,
        rng=rng,
    )
    states_by_key = initialize_states_by_strategy(
        split=split,
        initial_model=initial_model,
        query_x=query_x,
        target_classes=target_classes,
        rng=rng,
        args=args,
    )
    if not states_by_key:
        return []

    model = classifier_factory()
    learn_on_batch(model, split.train.x, split.train.y, split.feature_names)
    reference_x, reference_y = trim_recent_buffer(
        split.train.x,
        split.train.y,
        max_size=max_reference_size,
    )
    rows: list[dict[str, object]] = []
    recent_correct: list[int] = []
    recent_accuracy = np.nan

    def current_predict_fn(x: np.ndarray, current_model=model) -> np.ndarray:
        return predict_batch(current_model, x, split.feature_names)

    refresh_dynamic_baselines(
        states_by_key=states_by_key,
        predict_fn=current_predict_fn,
        reference_x=reference_x,
        args=args,
        rng=rng,
    )
    append_all_strategy_rows(
        rows=rows,
        states_by_key=states_by_key,
        repeat=repeat,
        stream_name=stream_name,
        classifier_name=classifier_name,
        split=split,
        checkpoint=0,
        streamed_samples=0,
        model=model,
        classifier_accuracy=recent_accuracy,
        reference_x=reference_x,
        reference_y=reference_y,
    )

    checkpoint_set = set(checkpoint_indices)
    checkpoint_counter = 0
    batch_iterator = enumerate(stream_batches)
    if not args.no_progress:
        batch_iterator = tqdm(
            batch_iterator,
            total=len(stream_batches),
            desc=f"{stream_name}/{classifier_name}",
            leave=False,
        )

    for batch_index, batch in batch_iterator:
        update_index = batch_index * max(
            1,
            int(np.ceil(len(batch.x) / args.update_every_points)),
        )
        for update_batch in split_batch_for_updates(batch, args.update_every_points):
            update_index += 1
            recent_accuracy = update_recent_prequential_accuracy(
                recent_correct=recent_correct,
                model=model,
                batch_x=update_batch.x,
                batch_y=update_batch.y,
                feature_names=split.feature_names,
                max_size=max_reference_size,
            )
            learn_on_batch(model, update_batch.x, update_batch.y, split.feature_names)
            reference_x, reference_y = extend_recent_buffer(
                reference_x,
                reference_y,
                update_batch.x,
                update_batch.y,
                max_size=max_reference_size,
            )

            def predict_fn(x: np.ndarray, current_model=model) -> np.ndarray:
                return predict_batch(current_model, x, split.feature_names)

            def predict_proba_fn(x: np.ndarray, current_model=model) -> np.ndarray:
                return predict_proba_batch(current_model, x, split.feature_names)

            buffer_index = build_current_buffer_index(
                reference_x=reference_x,
                reference_y=reference_y,
                predict_fn=predict_fn,
                predict_proba_fn=predict_proba_fn,
                neighborhood_size=args.neighborhood_size,
            )
            for key, states in list(states_by_key.items()):
                _, strategy = key
                if strategy in {"frozen", "dynamic_nearest_neighbor", "dynamic_growing_spheres"}:
                    states_by_key[key] = refresh_counterfactual_states(
                        states=states,
                        predict_fn=predict_fn,
                    )
                elif strategy == "validity_plausibility_update":
                    states_by_key[key] = update_counterfactual_states(
                        states=states,
                        predict_fn=predict_fn,
                        predict_proba_fn=predict_proba_fn,
                        buffer_index=buffer_index,
                        config=validity_config,
                        update_index=update_index,
                        rng=rng,
                    )
                elif strategy == "plausibility_low_margin_update":
                    states_by_key[key] = update_counterfactual_states(
                        states=states,
                        predict_fn=predict_fn,
                        predict_proba_fn=predict_proba_fn,
                        buffer_index=buffer_index,
                        config=low_margin_config,
                        update_index=update_index,
                        rng=rng,
                    )

        if batch_index not in checkpoint_set:
            continue

        checkpoint_counter += 1
        streamed_samples = sum(len(b.x) for b in stream_batches[: batch_index + 1])

        def checkpoint_predict_fn(x: np.ndarray, current_model=model) -> np.ndarray:
            return predict_batch(current_model, x, split.feature_names)

        refresh_dynamic_baselines(
            states_by_key=states_by_key,
            predict_fn=checkpoint_predict_fn,
            reference_x=reference_x,
            args=args,
            rng=rng,
        )
        append_all_strategy_rows(
            rows=rows,
            states_by_key=states_by_key,
            repeat=repeat,
            stream_name=stream_name,
            classifier_name=classifier_name,
            split=split,
            checkpoint=checkpoint_counter,
            streamed_samples=streamed_samples,
            model=model,
            classifier_accuracy=recent_accuracy,
            reference_x=reference_x,
            reference_y=reference_y,
        )

    final_checkpoint = checkpoint_counter
    final_streamed_samples = sum(len(batch.x) for batch in stream_batches)
    for initial_generator in args.initial_generators:
        base_key = (initial_generator, "frozen")
        if base_key not in states_by_key:
            continue
        final_robx_states = generate_final_robx_states(
            states=states_by_key[base_key],
            model=model,
            reference_x=reference_x,
            split=split,
            args=args,
            rng=rng,
        )
        append_row(
            rows=rows,
            repeat=repeat,
            strategy="final_robx",
            initial_generator=initial_generator,
            stream_name=stream_name,
            classifier_name=classifier_name,
            split=split,
            checkpoint=final_checkpoint,
            streamed_samples=final_streamed_samples,
            model=model,
            classifier_accuracy=recent_accuracy,
            states=final_robx_states,
            reference_x=reference_x,
            reference_y=reference_y,
        )

    return rows


def save_results(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_suite(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    total_cells = args.repeats * len(args.streams) * len(args.classifiers)
    cell_progress = nullcontext()
    if not args.no_progress:
        cell_progress = tqdm(total=total_cells, desc="experiment cells")

    with cell_progress as progress:
        for repeat in range(args.repeats):
            run_seed = args.seed + repeat
            stream_specs = get_default_stream_specs(seed=run_seed)
            classifier_specs = get_default_classifier_specs(seed=run_seed)
            for stream_name in args.streams:
                split = make_initial_data_split(
                    stream_factory=stream_specs[stream_name].factory,
                    n_init=args.n_init,
                    batch_size=args.batch_size,
                    max_stream_samples=args.max_stream_samples,
                    seed=run_seed,
                )
                stream_batches = list(split.stream_batches)
                if not args.no_feature_normalization:
                    split, stream_batches = normalize_split_and_batches(
                        split,
                        stream_batches,
                    )
                checkpoint_indices = make_checkpoint_batch_indices(
                    n_batches=len(stream_batches),
                    n_checkpoints=args.n_checkpoints,
                    every_n_batches=args.checkpoint_every_batches,
                )
                checkpoint_samples = checkpoint_streamed_samples(
                    stream_batches=stream_batches,
                    checkpoint_indices=checkpoint_indices,
                )
                LOGGER.info(
                    "repeat=%d stream=%s checkpoints=%s",
                    repeat,
                    stream_name,
                    checkpoint_samples,
                )
                for classifier_name in args.classifiers:
                    rows = run_single_experiment(
                        repeat=repeat,
                        stream_name=stream_name,
                        classifier_name=classifier_name,
                        split=split,
                        classifier_factory=classifier_specs[classifier_name].factory,
                        stream_batches=stream_batches,
                        checkpoint_indices=checkpoint_indices,
                        args=args,
                        seed=run_seed,
                    )
                    all_rows.extend(rows)
                    if not args.no_progress:
                        progress.update(1)

    output_path = args.output_dir / "paper_reference_drift_results.csv"
    save_results(all_rows, output_path)
    LOGGER.info("saved %d rows to %s", len(all_rows), output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.update_every_points <= 0:
        raise ValueError("--update-every-points must be positive.")

    log_context = nullcontext() if args.no_progress else logging_redirect_tqdm()
    with log_context:
        run_experiment_suite(args)


if __name__ == "__main__":
    main()
