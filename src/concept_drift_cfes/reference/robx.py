from dataclasses import dataclass
from typing import Any

import numpy as np

from concept_drift_cfes.reference.growing_spheres import growing_spheres_search


def _counterfactual_stability_batch(
    xs: np.ndarray,
    pred_func: callable,
    variance: np.ndarray | float,
    N: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    B, d = xs.shape
    results = np.full(B, np.nan)

    valid_mask = np.all(np.isfinite(xs), axis=1)
    if not np.any(valid_mask):
        return results

    valid_xs = xs[valid_mask]
    B_valid = valid_xs.shape[0]

    if isinstance(variance, np.ndarray):
        variance = np.asarray(variance)
        if variance.ndim == 1:
            if len(variance) != d:
                raise ValueError("variance must have the same length as x")
            if not np.all(np.isfinite(variance)) or np.any(variance < 0):
                return results
            std = np.sqrt(np.clip(variance, 1e-12, None))
            noise = rng.standard_normal((B_valid, N, d)) * std[None, None, :]
        elif variance.ndim == 2:
            cov = variance
            if cov.shape != (d, d):
                raise ValueError("variance matrix has invalid shape")
            if not np.all(np.isfinite(cov)):
                return results
            try:
                noise = rng.multivariate_normal(np.zeros(d), cov, size=(B_valid * N,))
                noise = noise.reshape(B_valid, N, d)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return results
        else:
            raise ValueError("variance must be 1D or 2D")
    else:
        var_scalar = float(variance)
        if not np.isfinite(var_scalar) or var_scalar <= 0:
            return results
        std = np.sqrt(var_scalar)
        noise = rng.standard_normal((B_valid, N, d)) * std

    X_p = valid_xs[:, None, :] + noise  # (B_valid, N, d)

    finite_per_point = np.all(np.isfinite(X_p.reshape(B_valid, -1)), axis=1)

    orig_preds = pred_func(valid_xs)  # (B_valid,)
    cf_classes = (orig_preds > gamma).astype(int)  # (B_valid,)

    X_p_flat = X_p.reshape(B_valid * N, d)
    all_preds = pred_func(X_p_flat).reshape(B_valid, N)

    X_pred = np.where(cf_classes[:, None] == 1, all_preds, 1.0 - all_preds)
    c_mean = np.mean(X_pred, axis=1)
    c_std = np.std(X_pred, axis=1)
    stabilities = c_mean - c_std

    stabilities[~finite_per_point] = np.nan
    results[valid_mask] = stabilities
    return results


def counterfactual_stability(
    x: np.ndarray,
    pred_func: callable,
    variance: np.ndarray | float = 0.1,
    N: int = 100,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x).reshape(1, -1)
    return float(
        _counterfactual_stability_batch(x, pred_func, variance, N, gamma, rng)[0]
    )


def counterfactual_stability_test(counterfactual_stability: float, tau: float) -> bool:
    return counterfactual_stability > tau


def get_conservative_counterfactuals(
    counterfactual: np.ndarray,
    data_X: np.ndarray,
    predict_class_proba_fn: callable,
    variance: np.ndarray | float = 0.1,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
    _batch_size: int = 128,
) -> np.ndarray | None:
    cf_prob = predict_class_proba_fn(counterfactual.reshape(1, -1))[0]
    cf_class = 1 if cf_prob > gamma else 0

    data_probs = predict_class_proba_fn(data_X)
    data_classes = (data_probs > gamma).astype(int)
    correct_class_mask = data_classes == cf_class
    data = data_X[correct_class_mask]

    if data.size == 0:
        return None

    dist = np.sum(np.abs(data - counterfactual), axis=1)
    indices = np.argsort(dist)
    data = data[indices]

    conservative_counterfactuals = []
    for start in range(0, len(data), _batch_size):
        batch = data[start : start + _batch_size]
        stabilities = _counterfactual_stability_batch(
            batch, predict_class_proba_fn, variance, N, gamma, rng
        )
        for x, s in zip(batch, stabilities):
            if s > tau:
                conservative_counterfactuals.append(x)
                if len(conservative_counterfactuals) == k:
                    return np.array(conservative_counterfactuals)

    if not conservative_counterfactuals:
        return None

    return np.array(conservative_counterfactuals)


def robx_algorithm(
    X_train: np.ndarray,
    predict_class_proba_fn: callable,
    start_counterfactual: np.ndarray,
    variance: np.ndarray | float = 0.1,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    robx_max_iter: int = 100,
    robx_lambda: float = 0.1,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if rng is None:
        rng = np.random.default_rng()

    init_stability = _counterfactual_stability_batch(
        start_counterfactual.reshape(1, -1),
        predict_class_proba_fn,
        variance,
        N,
        gamma,
        rng,
    )[0]
    if counterfactual_stability_test(init_stability, tau):
        return start_counterfactual, None

    conservative_counterfactuals = get_conservative_counterfactuals(
        counterfactual=start_counterfactual,
        data_X=X_train,
        predict_class_proba_fn=predict_class_proba_fn,
        variance=variance,
        tau=tau,
        N=N,
        k=k,
        gamma=gamma,
        rng=rng,
    )

    if conservative_counterfactuals is None:
        return None, None

    n_cfs = len(conservative_counterfactuals)
    counterfactuals = np.tile(start_counterfactual, (n_cfs, 1))

    for _ in range(robx_max_iter):
        counterfactuals = (
            robx_lambda * conservative_counterfactuals
            + (1.0 - robx_lambda) * counterfactuals
        )

        stabilities = _counterfactual_stability_batch(
            counterfactuals,
            predict_class_proba_fn,
            variance,
            N,
            gamma,
            rng,
        )

        stable_mask = stabilities > tau
        if np.any(stable_mask):
            first_stable = np.argmax(stable_mask)
            return counterfactuals[first_stable], conservative_counterfactuals

    return None, None


@dataclass
class RobXResult:
    counterfactual: np.ndarray | None
    start_counterfactual: np.ndarray | None
    conservative_counterfactuals: np.ndarray | None
    metadata: dict[str, Any]


class RobX:
    
    def __init__(
        self,
        X_train: np.ndarray,
        predict_fn_crisp: callable,
        predict_proba_fn: callable,
        binary_indices: list[int] | None = None,
        feature_min: np.ndarray | None = None,
        feature_max: np.ndarray | None = None,
    ) -> None:
        self.X_train = np.asarray(X_train)
        self.predict_fn_crisp = predict_fn_crisp
        self.predict_proba_fn = predict_proba_fn
        self.binary_indices = binary_indices
        self.feature_min = feature_min
        self.feature_max = feature_max

    def _predict_target_proba(self, X: np.ndarray, target_class: int) -> np.ndarray:
        proba = self.predict_proba_fn(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 1:
            return proba.reshape(-1)
        return proba[:, target_class]

    def generate(
        self,
        start_instance: np.ndarray,
        target_class: int,
        variance: np.ndarray | float = 0.1,
        tau: float = 0.4,
        N: int = 100,
        k: int = 3,
        robx_max_iter: int = 100,
        robx_lambda: float = 0.1,
        gamma: float = 0.5,
        gs_n_search_samples: int = 1000,
        gs_p_norm: int = 2,
        gs_step: float = 0.2,
        gs_max_iter: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> RobXResult:
        if rng is None:
            rng = np.random.default_rng()

        start_instance = np.asarray(start_instance).reshape(-1)

        start_cf = growing_spheres_search(
            instance=start_instance,
            pred_fn_crisp=self.predict_fn_crisp,
            target_class=target_class,
            n_search_samples=gs_n_search_samples,
            p_norm=gs_p_norm,
            step=gs_step,
            max_iter=gs_max_iter,
            binary_indices=self.binary_indices,
            feature_min=self.feature_min,
            feature_max=self.feature_max,
            rng=rng,
        )

        if start_cf is None:
            return RobXResult(
                counterfactual=None,
                start_counterfactual=None,
                conservative_counterfactuals=None,
                metadata={"status": "no_start_cf"},
            )

        def predict_target_proba(X: np.ndarray) -> np.ndarray:
            return self._predict_target_proba(X, target_class)

        robust_cf, conservative_cfs = robx_algorithm(
            X_train=self.X_train,
            predict_class_proba_fn=predict_target_proba,
            start_counterfactual=start_cf,
            variance=variance,
            tau=tau,
            N=N,
            k=k,
            robx_max_iter=robx_max_iter,
            robx_lambda=robx_lambda,
            gamma=gamma,
            rng=rng,
        )

        status = "ok" if robust_cf is not None else "no_robust_cf"
        return RobXResult(
            counterfactual=robust_cf,
            start_counterfactual=start_cf,
            conservative_counterfactuals=conservative_cfs,
            metadata={"status": status},
        )
