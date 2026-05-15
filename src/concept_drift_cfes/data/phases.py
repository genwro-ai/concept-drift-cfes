from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from concept_drift_cfes.data.streaming import StreamFactory
from concept_drift_cfes.utils import rows_to_batch, validate_split_sizes


@dataclass(frozen=True)
class DatasetBatch:
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[Any, ...]


@dataclass
class InitialDataSplit:
    train: DatasetBatch
    val: DatasetBatch
    test: DatasetBatch
    stream_batches: Iterator[DatasetBatch]
    feature_names: tuple[Any, ...]


@dataclass(frozen=True)
class MinMaxFeatureScaler:
    minimum: np.ndarray
    scale: np.ndarray

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.minimum) / self.scale

    def transform_batch(self, batch: DatasetBatch) -> DatasetBatch:
        return DatasetBatch(
            x=self.transform_x(batch.x),
            y=batch.y,
            feature_names=batch.feature_names,
        )


def fit_minmax_feature_scaler(
    batch: DatasetBatch,
    min_scale: float = 1e-12,
) -> MinMaxFeatureScaler:
    x = np.asarray(batch.x, dtype=float)
    minimum = x.min(axis=0)
    maximum = x.max(axis=0)
    scale = np.maximum(maximum - minimum, min_scale)
    return MinMaxFeatureScaler(minimum=minimum, scale=scale)


def materialize_stream(
    stream_factory: StreamFactory,
    n_samples: int,
) -> DatasetBatch:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rows = list(islice(iter(stream_factory()), n_samples))
    if len(rows) < n_samples:
        raise ValueError(
            f"Requested {n_samples} samples, but the stream only yielded {len(rows)}."
        )

    x, y, feature_names = rows_to_batch(rows)
    return DatasetBatch(x=x, y=y, feature_names=feature_names)


def split_dataset(
    batch: DatasetBatch,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> tuple[DatasetBatch, DatasetBatch, DatasetBatch]:
    validate_split_sizes(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    stratify_labels = batch.y if stratify else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        batch.x,
        batch.y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    val_fraction = val_size / (train_size + val_size)
    stratify_train_val = y_train_val if stratify else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_train_val,
    )

    feature_names = batch.feature_names
    train = DatasetBatch(x=x_train, y=y_train, feature_names=feature_names)
    val = DatasetBatch(x=x_val, y=y_val, feature_names=feature_names)
    test = DatasetBatch(x=x_test, y=y_test, feature_names=feature_names)
    return train, val, test


def make_initial_data_split(
    stream_factory: StreamFactory,
    n_init: int,
    batch_size: int = 1,
    max_stream_samples: int | None = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> InitialDataSplit:
    if n_init <= 0:
        raise ValueError("n_init must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    stream_iterator = iter(stream_factory())
    init_rows = list(islice(stream_iterator, n_init))
    if len(init_rows) < n_init:
        raise ValueError(
            f"Requested {n_init} initial samples, but the stream only yielded "
            f"{len(init_rows)}."
        )

    initial_x, initial_y, feature_names = rows_to_batch(init_rows)
    initial_batch = DatasetBatch(
        x=initial_x,
        y=initial_y,
        feature_names=feature_names,
    )
    train, val, test = split_dataset(
        batch=initial_batch,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        stratify=stratify,
        seed=seed,
    )

    def iter_stream_batches() -> Iterator[DatasetBatch]:
        yielded = 0
        while True:
            if max_stream_samples is not None:
                remaining = max_stream_samples - yielded
                if remaining <= 0:
                    break
                chunk_size = min(batch_size, remaining)
            else:
                chunk_size = batch_size

            rows = list(islice(stream_iterator, chunk_size))
            if not rows:
                break

            x_batch, y_batch, _ = rows_to_batch(rows, feature_names=feature_names)
            yield DatasetBatch(
                x=x_batch,
                y=y_batch,
                feature_names=feature_names,
            )
            yielded += len(rows)

    return InitialDataSplit(
        train=train,
        val=val,
        test=test,
        stream_batches=iter_stream_batches(),
        feature_names=feature_names,
    )
