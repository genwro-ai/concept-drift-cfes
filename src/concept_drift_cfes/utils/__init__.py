from concept_drift_cfes.utils.arrays import (
    ensure_2d,
    resolve_targets,
    rows_to_batch,
    sorted_feature_names,
    validate_split_sizes,
)
from concept_drift_cfes.utils.cfe_metrics import CFEMetrics, evaluate_cfe_metrics
from concept_drift_cfes.utils.checkpoints import (
    checkpoint_streamed_samples,
    make_checkpoint_batch_indices,
)
from concept_drift_cfes.utils.neighborhoods import (
    CachedBufferNeighborhood,
    CurrentBufferIndex,
    buffer_target_region_vector,
    build_current_buffer_index,
    epanechnikov_kernel_values,
    query_cached_buffer_neighborhood,
)
from concept_drift_cfes.utils.river import (
    accuracy_on_batch,
    array_to_river_dict,
    iter_river_dicts,
    learn_on_batch,
    predict_batch,
    predict_proba_batch,
    target_flip_classes,
)

__all__ = [
    "CFEMetrics",
    "CachedBufferNeighborhood",
    "CurrentBufferIndex",
    "accuracy_on_batch",
    "array_to_river_dict",
    "buffer_target_region_vector",
    "build_current_buffer_index",
    "checkpoint_streamed_samples",
    "ensure_2d",
    "epanechnikov_kernel_values",
    "evaluate_cfe_metrics",
    "iter_river_dicts",
    "learn_on_batch",
    "make_checkpoint_batch_indices",
    "predict_batch",
    "predict_proba_batch",
    "query_cached_buffer_neighborhood",
    "resolve_targets",
    "rows_to_batch",
    "sorted_feature_names",
    "target_flip_classes",
    "validate_split_sizes",
]
