import argparse
import csv
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

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
    update_counterfactual_states,
)
from concept_drift_cfes.utils import (
    learn_on_batch,
    make_checkpoint_batch_indices,
    predict_batch,
    predict_proba_batch,
    target_flip_classes,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_STREAMS = ["hyperplane", "sea_gradual"]
DEFAULT_CLASSIFIERS = ["adaptive_hoeffding_tree", "online_logistic_regression"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time update methods against RoBx and Growing Spheres regeneration.",
    )
    parser.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS)
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--n-init", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-stream-samples", type=int, default=2_000)
    parser.add_argument("--n-checkpoints", type=int, default=8)
    parser.add_argument("--checkpoint-every-batches", type=int, default=None)
    parser.add_argument("--buffer-size-batches", type=int, default=5)
    parser.add_argument("--update-every-points", type=int, default=10)
    parser.add_argument("--n-query", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_timing"))
    parser.add_argument("--no-feature-normalization", action="store_true")

    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--validity-weight", type=float, default=2.0)
    parser.add_argument("--plausibility-weight", type=float, default=2.0)
    parser.add_argument("--proximity-weight", type=float, default=1.0)
    parser.add_argument("--neighborhood-size", type=int, default=64)
    parser.add_argument("--low-margin-threshold", type=float, default=0.6)
    parser.add_argument("--plausibility-every-steps", type=int, default=60)
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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
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


def make_config(args: argparse.Namespace, mode: str) -> CFEUpdateConfig:
    return CFEUpdateConfig(
        mode=mode,
        step_size=args.step_size,
        validity_weight=args.validity_weight,
        plausibility_weight=args.plausibility_weight,
        proximity_weight=args.proximity_weight,
        neighborhood_size=args.neighborhood_size,
        low_margin_threshold=args.low_margin_threshold,
        plausibility_every_steps=args.plausibility_every_steps,
        target_region_bandwidth=args.target_region_bandwidth,
        gaussian_samples=args.gaussian_samples,
        gaussian_sigma=args.gaussian_sigma,
    )


def select_queries(
    x: np.ndarray,
    predictions: np.ndarray,
    n_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(x))
    rng.shuffle(indices)
    indices = indices[: min(n_query, len(indices))]
    query_x = np.asarray(x[indices], dtype=float)
    targets = target_flip_classes(np.asarray(predictions[indices], dtype=int))
    return query_x, targets


def initial_growing_spheres_states(
    query_x: np.ndarray,
    target_classes: np.ndarray,
    predict_fn,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[CounterfactualState]:
    states = []
    for x_ref, target in zip(query_x, target_classes, strict=True):
        cf = growing_spheres_search(
            instance=x_ref,
            pred_fn_crisp=predict_fn,
            target_class=int(target),
            n_search_samples=args.gs_search_samples,
            step=args.gs_step,
            max_iter=args.gs_max_iter,
            rng=rng,
        )
        if cf is not None:
            states.append(
                CounterfactualState(
                    x_ref=np.asarray(x_ref, dtype=float),
                    counterfactual=np.asarray(cf, dtype=float),
                    target_class=int(target),
                )
            )
    return states


def retire_naturally_resolved(
    states: list[CounterfactualState],
    predict_fn,
) -> list[CounterfactualState]:
    refreshed = []
    for state in states:
        if not state.is_active:
            refreshed.append(state)
            continue
        if int(predict_fn(np.asarray(state.x_ref).reshape(1, -1))[0]) == int(
            state.target_class
        ):
            refreshed.append(
                replace(
                    state,
                    is_active=False,
                    retirement_reason="naturally_resolved",
                    last_update="retired",
                )
            )
        else:
            refreshed.append(state)
    return refreshed


def dynamic_nearest_neighbor_states(
    states: list[CounterfactualState],
    reference_x: np.ndarray,
    predict_fn,
) -> tuple[list[CounterfactualState], int]:
    refreshed = []
    success_count = 0
    reference_predictions = np.asarray(predict_fn(reference_x), dtype=int).reshape(-1)
    for state in retire_naturally_resolved(states, predict_fn):
        if not state.is_active:
            refreshed.append(state)
            continue
        candidates = reference_x[reference_predictions == int(state.target_class)]
        if len(candidates) == 0:
            refreshed.append(state)
            continue
        distances = np.linalg.norm(
            candidates - np.asarray(state.x_ref, dtype=float).reshape(1, -1),
            axis=1,
        )
        refreshed.append(
            replace(
                state,
                counterfactual=np.asarray(candidates[int(np.argmin(distances))]),
                last_update="dynamic_nearest_neighbor",
            )
        )
        success_count += 1
    return refreshed, success_count


def dynamic_growing_spheres_states(
    states: list[CounterfactualState],
    predict_fn,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[CounterfactualState], int]:
    refreshed = []
    success_count = 0
    for state in retire_naturally_resolved(states, predict_fn):
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
        if cf is None:
            refreshed.append(state)
            continue
        refreshed.append(
            replace(
                state,
                counterfactual=np.asarray(cf, dtype=float),
                last_update="dynamic_growing_spheres",
            )
        )
        success_count += 1
    return refreshed, success_count


def update_method_states(
    states: list[CounterfactualState],
    predict_fn,
    predict_proba_fn,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    config: CFEUpdateConfig,
    update_index: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[CounterfactualState], int]:
    buffer_index = build_current_buffer_index(
        reference_x=reference_x,
        reference_y=reference_y,
        predict_fn=predict_fn,
        predict_proba_fn=predict_proba_fn,
        neighborhood_size=args.neighborhood_size,
    )
    updated = update_counterfactual_states(
        states=states,
        predict_fn=predict_fn,
        predict_proba_fn=predict_proba_fn,
        buffer_index=buffer_index,
        config=config,
        update_index=update_index,
        rng=rng,
    )
    return updated, sum(state.is_active for state in updated)


def time_robx(
    states: list[CounterfactualState],
    train_x: np.ndarray,
    predict_fn,
    predict_proba_fn,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    robx = RobX(
        X_train=train_x,
        predict_fn_crisp=predict_fn,
        predict_proba_fn=predict_proba_fn,
    )
    count = 0
    for state in states:
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
        count += int(result.counterfactual is not None)
    return count


def elapsed_seconds(callable_obj) -> tuple[float, int]:
    start = time.perf_counter()
    count = callable_obj()
    return time.perf_counter() - start, int(count)


def elapsed_state_update(callable_obj) -> tuple[float, list[CounterfactualState], int]:
    start = time.perf_counter()
    states, count = callable_obj()
    return time.perf_counter() - start, states, int(count)


def time_checkpoint_regeneration(
    states_by_method: dict[str, list[CounterfactualState]],
    reference_x: np.ndarray,
    predict_fn,
    args: argparse.Namespace,
    rng: np.random.Generator,
    cumulative_seconds: dict[str, float],
    cumulative_success: dict[str, int],
    cumulative_steps: dict[str, int],
) -> None:
    checkpoint_methods = {
        "nearest_neighbor_regeneration": lambda: dynamic_nearest_neighbor_states(
            states=states_by_method["nearest_neighbor_regeneration"],
            reference_x=reference_x,
            predict_fn=predict_fn,
        ),
        "growing_spheres_regeneration": lambda: dynamic_growing_spheres_states(
            states=states_by_method["growing_spheres_regeneration"],
            predict_fn=predict_fn,
            args=args,
            rng=rng,
        ),
    }
    for method_name, callable_obj in checkpoint_methods.items():
        elapsed, updated_states, success_count = elapsed_state_update(callable_obj)
        states_by_method[method_name] = updated_states
        cumulative_seconds[method_name] += elapsed
        cumulative_success[method_name] += success_count
        cumulative_steps[method_name] += 1


def run_timing(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        rng = np.random.default_rng(seed)
        stream_specs = get_default_stream_specs(seed=seed)
        classifier_specs = get_default_classifier_specs(seed=seed)

        for stream_name in args.streams:
            for classifier_name in args.classifiers:
                rows.extend(
                    run_timing_cell(
                        repeat=repeat,
                        seed=seed,
                        rng=rng,
                        stream_name=stream_name,
                        classifier_name=classifier_name,
                        stream_spec=stream_specs[stream_name],
                        classifier_spec=classifier_specs[classifier_name],
                        args=args,
                    )
                )
    return rows


def run_timing_cell(
    repeat: int,
    seed: int,
    rng: np.random.Generator,
    stream_name: str,
    classifier_name: str,
    stream_spec,
    classifier_spec,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    LOGGER.info(
        "timing repeat=%d stream=%s classifier=%s",
        repeat,
        stream_name,
        classifier_name,
    )
    split = make_initial_data_split(
        stream_factory=stream_spec.factory,
        n_init=args.n_init,
        batch_size=args.batch_size,
        max_stream_samples=args.max_stream_samples,
        seed=seed,
    )
    stream_batches = list(split.stream_batches)
    if not args.no_feature_normalization:
        split, stream_batches = normalize_split_and_batches(split, stream_batches)

    model = classifier_spec.factory()
    learn_on_batch(model, split.train.x, split.train.y, split.feature_names)

    def predict_fn(x: np.ndarray, current_model=model) -> np.ndarray:
        return predict_batch(current_model, x, split.feature_names)

    def predict_proba_fn(x: np.ndarray, current_model=model) -> np.ndarray:
        return predict_proba_batch(current_model, x, split.feature_names)

    predictions = predict_fn(split.test.x)
    query_x, target_classes = select_queries(
        split.test.x,
        predictions,
        args.n_query,
        rng,
    )
    states = initial_growing_spheres_states(
        query_x=query_x,
        target_classes=target_classes,
        predict_fn=predict_fn,
        args=args,
        rng=rng,
    )
    if not states:
        LOGGER.warning(
            "no initial CFEs generated for repeat=%d stream=%s classifier=%s",
            repeat,
            stream_name,
            classifier_name,
        )
        return rows

    max_reference_size = args.buffer_size_batches * max(1, len(stream_batches[0].x))
    reference_x, reference_y = trim_recent_buffer(
        split.train.x,
        split.train.y,
        max_size=max_reference_size,
    )
    states_by_method = {
        "validity_plausibility_update": [replace(state) for state in states],
        "plausibility_low_margin_update": [replace(state) for state in states],
        "nearest_neighbor_regeneration": [replace(state) for state in states],
        "growing_spheres_regeneration": [replace(state) for state in states],
    }
    cumulative_seconds = dict.fromkeys(states_by_method, 0.0)
    cumulative_success = dict.fromkeys(states_by_method, 0)
    cumulative_steps = dict.fromkeys(states_by_method, 0)
    validity_config = make_config(args, VALIDITY_PLAUSIBILITY)
    low_margin_config = make_config(args, PLAUSIBILITY_LOW_MARGIN)
    checkpoint_indices = make_checkpoint_batch_indices(
        n_batches=len(stream_batches),
        n_checkpoints=args.n_checkpoints,
        every_n_batches=args.checkpoint_every_batches,
    )
    checkpoint_set = {index for index in checkpoint_indices if index >= 0}

    time_checkpoint_regeneration(
        states_by_method=states_by_method,
        reference_x=reference_x,
        predict_fn=predict_fn,
        args=args,
        rng=rng,
        cumulative_seconds=cumulative_seconds,
        cumulative_success=cumulative_success,
        cumulative_steps=cumulative_steps,
    )

    update_index = 0
    for batch_index, batch in enumerate(stream_batches):
        for update_batch in split_batch_for_updates(
            batch,
            chunk_size=args.update_every_points,
        ):
            update_index += 1
            learn_on_batch(
                model,
                update_batch.x,
                update_batch.y,
                split.feature_names,
            )
            reference_x, reference_y = extend_recent_buffer(
                reference_x,
                reference_y,
                update_batch.x,
                update_batch.y,
                max_size=max_reference_size,
            )

            def current_predict_fn(
                x: np.ndarray,
                current_model=model,
            ) -> np.ndarray:
                return predict_batch(current_model, x, split.feature_names)

            def current_predict_proba_fn(
                x: np.ndarray,
                current_model=model,
            ) -> np.ndarray:
                return predict_proba_batch(current_model, x, split.feature_names)

            step_methods = {
                "validity_plausibility_update": lambda: update_method_states(
                    states=states_by_method["validity_plausibility_update"],
                    predict_fn=current_predict_fn,
                    predict_proba_fn=current_predict_proba_fn,
                    reference_x=reference_x,
                    reference_y=reference_y,
                    config=validity_config,
                    update_index=update_index,
                    args=args,
                    rng=rng,
                ),
                "plausibility_low_margin_update": lambda: update_method_states(
                    states=states_by_method["plausibility_low_margin_update"],
                    predict_fn=current_predict_fn,
                    predict_proba_fn=current_predict_proba_fn,
                    reference_x=reference_x,
                    reference_y=reference_y,
                    config=low_margin_config,
                    update_index=update_index,
                    args=args,
                    rng=rng,
                ),
            }
            for method_name, callable_obj in step_methods.items():
                elapsed, updated_states, success_count = elapsed_state_update(
                    callable_obj
                )
                states_by_method[method_name] = updated_states
                cumulative_seconds[method_name] += elapsed
                cumulative_success[method_name] += success_count
                cumulative_steps[method_name] += 1

        if batch_index in checkpoint_set:
            time_checkpoint_regeneration(
                states_by_method=states_by_method,
                reference_x=reference_x,
                predict_fn=current_predict_fn,
                args=args,
                rng=rng,
                cumulative_seconds=cumulative_seconds,
                cumulative_success=cumulative_success,
                cumulative_steps=cumulative_steps,
            )

    for method_name, seconds_total in cumulative_seconds.items():
        rows.append(
            {
                "repeat": repeat,
                "stream": stream_name,
                "classifier": classifier_name,
                "timing_scope": "cumulative_stream"
                if method_name.endswith("_update")
                else "checkpoint_cumulative",
                "method": method_name,
                "n_cfes_initial": len(states),
                "n_steps": cumulative_steps[method_name],
                "n_success": cumulative_success[method_name],
                "seconds_total": seconds_total,
                "seconds_per_cfe_step": seconds_total
                / max(len(states) * cumulative_steps[method_name], 1),
            }
        )
        LOGGER.info(
            "repeat=%d stream=%s classifier=%s method=%s steps=%d seconds=%.4f",
            repeat,
            stream_name,
            classifier_name,
            method_name,
            cumulative_steps[method_name],
            seconds_total,
        )

    elapsed, count = elapsed_seconds(
        lambda: time_robx(
            states,
            reference_x,
            predict_fn,
            predict_proba_fn,
            args,
            rng,
        )
    )
    rows.append(
        {
            "repeat": repeat,
            "stream": stream_name,
            "classifier": classifier_name,
            "timing_scope": "final_only",
            "method": "robx_regeneration",
            "n_cfes_initial": len(states),
            "n_steps": 1,
            "n_success": count,
            "seconds_total": elapsed,
            "seconds_per_cfe_step": elapsed / max(len(states), 1),
        }
    )
    LOGGER.info(
        "repeat=%d stream=%s classifier=%s method=robx_regeneration seconds=%.4f",
        repeat,
        stream_name,
        classifier_name,
        elapsed,
    )
    return rows


def save_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_logging()
    args = parse_args()
    rows = run_timing(args)
    output_path = args.output_dir / "paper_timing_results.csv"
    save_rows(rows, output_path)
    LOGGER.info("saved %d timing rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
