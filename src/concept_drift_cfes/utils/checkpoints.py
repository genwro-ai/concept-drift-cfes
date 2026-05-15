from collections.abc import Sequence

import numpy as np


def make_checkpoint_batch_indices(
    n_batches: int,
    n_checkpoints: int,
    every_n_batches: int | None = None,
) -> list[int]:
    if n_batches == 0:
        return [-1]

    if every_n_batches is not None:
        if every_n_batches <= 0:
            raise ValueError("every_n_batches must be positive")
        indices = list(range(0, n_batches, every_n_batches))
        if indices[-1] != n_batches - 1:
            indices.append(n_batches - 1)
        return [-1] + indices

    if n_checkpoints <= 1:
        return [-1]

    target_count = min(n_checkpoints - 1, n_batches)
    indices = np.linspace(0, n_batches - 1, num=target_count, dtype=int)
    return [-1] + sorted(set(int(index) for index in indices))


def checkpoint_streamed_samples(
    stream_batches: Sequence[object],
    checkpoint_indices: Sequence[int],
) -> list[int]:
    batch_sizes = [len(batch.x) for batch in stream_batches]
    cumulative_sizes = np.cumsum(batch_sizes).tolist()

    streamed_samples = []
    for batch_index in checkpoint_indices:
        if batch_index < 0:
            streamed_samples.append(0)
            continue
        streamed_samples.append(int(cumulative_sizes[batch_index]))
    return streamed_samples
