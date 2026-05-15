import numpy as np


def hyper_sphere_coordinates(
    n_search_samples: int,
    instance: np.ndarray,
    high: float,
    low: float,
    p_norm: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    center = np.asarray(instance).reshape(1, -1)
    deltas = rng.standard_normal((n_search_samples, center.shape[1]))
    dist = rng.random(n_search_samples) * (high - low) + low
    norm_p = np.linalg.norm(deltas, ord=p_norm, axis=1)
    scale = dist / (norm_p + 1e-12)
    deltas = deltas * scale.reshape(-1, 1)
    candidates = center + deltas
    return candidates, dist


def growing_spheres_search(
    instance: np.ndarray,
    pred_fn_crisp: callable,
    target_class: int | None = None,
    n_search_samples: int = 1000,
    p_norm: int = 2,
    step: float = 0.2,
    max_iter: int = 1000,
    binary_indices: list[int] | None = None,
    feature_min: np.ndarray | None = None,
    feature_max: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    if rng is None:
        rng = np.random.default_rng()

    instance = np.asarray(instance).reshape(1, -1)
    if not np.all(np.isfinite(instance)):
        return None
    instance_label = int(pred_fn_crisp(instance)[0])
    if target_class is None:
        if instance_label in (0, 1):
            target_class = 1 - instance_label
        else:
            raise ValueError("target_class must be provided for multiclass settings.")

    n_features = instance.shape[1]
    if binary_indices:
        binary_indices = sorted(set(binary_indices))
    else:
        binary_indices = []

    binary_mask = np.zeros(n_features, dtype=bool)
    if binary_indices:
        binary_mask[binary_indices] = True
    continuous_indices = np.where(~binary_mask)[0]

    low = 0.0
    high = low + step
    for _ in range(max_iter):
        candidates = np.repeat(instance, n_search_samples, axis=0)

        if continuous_indices.size > 0:
            cont_center = instance[:, continuous_indices]
            cont_samples, _ = hyper_sphere_coordinates(
                n_search_samples,
                cont_center,
                high=high,
                low=low,
                p_norm=p_norm,
                rng=rng,
            )
            candidates[:, continuous_indices] = cont_samples

        if binary_indices:
            bin_samples = rng.binomial(
                n=1, p=0.5, size=(n_search_samples, len(binary_indices))
            )
            candidates[:, binary_indices] = bin_samples

        if feature_min is not None or feature_max is not None:
            if feature_min is None or feature_max is None:
                raise ValueError("feature_min and feature_max must both be provided.")
            candidates = np.clip(candidates, feature_min, feature_max)

        preds = pred_fn_crisp(candidates)
        mask = preds == target_class
        if np.any(mask):
            if p_norm == 1:
                distances = np.abs(candidates - instance).sum(axis=1)
            elif p_norm == 2:
                distances = np.square(candidates - instance).sum(axis=1)
            else:
                raise ValueError("Unsupported p_norm. Use 1 or 2.")

            candidate_indices = np.where(mask)[0]
            best_idx = candidate_indices[np.argmin(distances[mask])]
            return candidates[best_idx]

        low = high
        high = low + step

    return None


class GrowingSpheres:

    def __init__(
        self,
        pred_fn_crisp: callable,
        target_class: int | None = None,
        max_iter: int = 1000,
        n_search_samples: int = 1000,
        p_norm: int = 2,
        step: float = 0.1,
        binary_indices: list[int] | None = None,
        feature_min: np.ndarray | None = None,
        feature_max: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pred_fn = pred_fn_crisp
        self.target_class = target_class
        self.max_iter = max_iter
        self.p_norm = p_norm
        self.step = step
        self.n_search_samples = n_search_samples
        self.binary_indices = binary_indices
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.rng = rng

    def generate(self, query_instance: np.ndarray) -> np.ndarray | None:
        return growing_spheres_search(
            instance=query_instance,
            pred_fn_crisp=self.pred_fn,
            target_class=self.target_class,
            n_search_samples=self.n_search_samples,
            p_norm=self.p_norm,
            step=self.step,
            max_iter=self.max_iter,
            binary_indices=self.binary_indices,
            feature_min=self.feature_min,
            feature_max=self.feature_max,
            rng=self.rng,
        )
