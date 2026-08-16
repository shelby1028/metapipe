"""Mixed-effects meta-regression for continuous and categorical moderators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from metapipe.effects import EffectSize
from metapipe.models import TauMethod, random_effects

EstimationMethod = Literal["wls", "ml"]


@dataclass(frozen=True)
class MetaRegressionResult:
    """Summary of a mixed-effects meta-regression fit.

    Attributes:
        coefficients: One row per intercept or moderator coefficient, containing
            estimate, standard error, 95% interval, and two-sided p value.
        tau_squared: Estimated residual between-study variance.
        r_squared: Proportion of null-model heterogeneity explained by moderators.
        residual_q: Weighted residual heterogeneity statistic.
        residual_q_p_value: Upper-tail chi-squared p value for residual Q.
        log_likelihood: Gaussian log likelihood at the fitted parameters.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        fitted_values: Model-fitted effect estimates in input study order.
        residuals: Study-level observed-minus-fitted residuals.
        estimation_method: Requested fitting method.
        moderator_columns: Design-matrix columns after categorical encoding.
    """

    coefficients: pd.DataFrame
    tau_squared: float
    r_squared: float
    residual_q: float
    residual_q_p_value: float
    log_likelihood: float
    aic: float
    bic: float
    fitted_values: tuple[float, ...]
    residuals: tuple[float, ...]
    estimation_method: EstimationMethod
    moderator_columns: tuple[str, ...]


def _coerce_effects(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert numeric or EffectSize inputs to validated arrays."""
    if len(effects) < 3:
        raise ValueError("At least three studies are required for meta-regression.")
    first = effects[0]
    if isinstance(first, EffectSize):
        if variances is not None:
            raise ValueError("Do not provide variances with EffectSize objects.")
        if not all(isinstance(effect, EffectSize) for effect in effects):
            raise ValueError(
                "Effects must all be EffectSize objects or all be numbers."
            )
        effect_array = np.asarray([effect.effect for effect in effects], dtype=float)
        variance_array = np.asarray(
            [effect.variance for effect in effects], dtype=float
        )
    else:
        if variances is None:
            raise ValueError("Variances are required with numeric effects.")
        effect_array = np.asarray(effects, dtype=float)
        variance_array = np.asarray(variances, dtype=float)
    if effect_array.ndim != 1 or variance_array.ndim != 1:
        raise ValueError("Effects and variances must be one-dimensional.")
    if effect_array.size != variance_array.size:
        raise ValueError("Effects and variances must have equal length.")
    if not np.all(np.isfinite(effect_array)):
        raise ValueError("Effects must be finite.")
    if not np.all(np.isfinite(variance_array)) or np.any(variance_array <= 0):
        raise ValueError("Variances must be finite and greater than zero.")
    return effect_array, variance_array


def _moderator_frame(
    moderators: pd.DataFrame | Mapping[str, Sequence[object]],
    study_count: int,
) -> pd.DataFrame:
    """Create a numeric design frame and one-hot encode categorical moderators."""
    dataframe = pd.DataFrame(moderators).copy()
    if dataframe.empty:
        raise ValueError("At least one moderator is required for meta-regression.")
    if len(dataframe) != study_count:
        raise ValueError("Moderators must have the same number of rows as effects.")
    if dataframe.columns.duplicated().any():
        raise ValueError("Moderator column names must be unique.")
    if dataframe.isna().any().any():
        raise ValueError("Moderators must not contain missing values.")

    categorical_columns = [
        column
        for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    encoded = pd.get_dummies(
        dataframe,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    if encoded.shape[1] == 0:
        raise ValueError("Moderators must produce at least one design column.")
    encoded = encoded.astype(float)
    if not np.all(np.isfinite(encoded.to_numpy(dtype=float))):
        raise ValueError("Moderators must contain finite values.")
    return encoded


def _fit_weighted_regression(
    effects: np.ndarray,
    variances: np.ndarray,
    design_matrix: np.ndarray,
    tau_squared: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit weighted least squares and return beta, covariance, residuals, Q, logL."""
    weights = 1.0 / (variances + tau_squared)
    weighted_design = design_matrix * np.sqrt(weights)[:, None]
    weighted_effects = effects * np.sqrt(weights)
    beta, _, _, _ = np.linalg.lstsq(weighted_design, weighted_effects, rcond=None)
    fitted_values = design_matrix @ beta
    residuals = effects - fitted_values
    residual_q = float(np.sum(weights * residuals**2))
    information = design_matrix.T @ (weights[:, None] * design_matrix)
    covariance = np.linalg.pinv(information)
    log_likelihood = float(
        -0.5
        * np.sum(
            np.log(2 * np.pi * (variances + tau_squared))
            + residuals**2 / (variances + tau_squared)
        )
    )
    return beta, covariance, residuals, residual_q, log_likelihood


def _ml_tau_squared(
    effects: np.ndarray,
    variances: np.ndarray,
    design_matrix: np.ndarray,
) -> float:
    """Estimate residual tau-squared by maximizing the Gaussian profile likelihood."""

    def objective(tau_squared: float) -> float:
        _, _, _, _, log_likelihood = _fit_weighted_regression(
            effects,
            variances,
            design_matrix,
            tau_squared,
        )
        return -log_likelihood

    upper = max(float(np.var(effects)), float(np.max(variances)), 1.0)
    result = minimize_scalar(objective, bounds=(0.0, upper * 20), method="bounded")
    if not result.success:
        raise RuntimeError("Maximum-likelihood tau-squared estimation failed.")
    return float(max(0.0, result.x))


def _wls_residual_tau_squared(
    residual_q: float,
    variances: np.ndarray,
    predictor_count: int,
) -> float:
    """Estimate residual tau-squared with a moment estimator after WLS fitting."""
    weights = 1.0 / variances
    denominator = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    degrees_of_freedom = variances.size - predictor_count
    if denominator <= 0:
        return 0.0
    return max(0.0, (residual_q - degrees_of_freedom) / denominator)


def meta_regression(
    effects: Sequence[float] | Sequence[EffectSize],
    moderators: pd.DataFrame | Mapping[str, Sequence[object]],
    variances: Sequence[float] | None = None,
    *,
    method: EstimationMethod = "wls",
    tau_method: TauMethod = "reml",
) -> MetaRegressionResult:
    """Fit a mixed-effects meta-regression with continuous or categorical moderators.

    Categorical moderators are treatment-coded with the first observed category as
    the reference group. The ``"wls"`` method estimates tau-squared from the
    null random-effects model and refits a residual moment estimate. The ``"ml"``
    method maximizes the Gaussian profile likelihood for residual tau-squared.

    Args:
        effects: Study-level estimates or EffectSize objects.
        moderators: Data frame or mapping containing moderator columns.
        variances: Sampling variances for numeric effects.
        method: ``"wls"`` for inverse-variance weighted least squares or ``"ml"``
            for profile maximum-likelihood residual variance estimation.
        tau_method: Null-model tau-squared estimator used by ``"wls"``.

    Returns:
        Coefficients, uncertainty, heterogeneity explanation, and fit statistics.

    Raises:
        ValueError: If data are invalid, the design has insufficient degrees of
            freedom, or the requested method is unknown.
    """
    effect_array, variance_array = _coerce_effects(effects, variances)
    encoded_moderators = _moderator_frame(moderators, effect_array.size)
    design_columns = ["Intercept", *encoded_moderators.columns.tolist()]
    design_matrix = np.column_stack(
        [np.ones(effect_array.size), encoded_moderators.to_numpy(dtype=float)]
    )
    predictor_count = design_matrix.shape[1]
    if effect_array.size <= predictor_count:
        raise ValueError("Meta-regression requires more studies than predictors.")
    if np.linalg.matrix_rank(design_matrix) < predictor_count:
        raise ValueError("Moderator design matrix is rank deficient.")
    if method not in {"wls", "ml"}:
        raise ValueError("method must be either 'wls' or 'ml'.")

    null_model = random_effects(effect_array, variance_array, tau_method=tau_method)
    if method == "ml":
        tau_squared = _ml_tau_squared(effect_array, variance_array, design_matrix)
    else:
        _, _, initial_residuals, initial_q, _ = _fit_weighted_regression(
            effect_array,
            variance_array,
            design_matrix,
            null_model.tau_squared,
        )
        del initial_residuals
        tau_squared = _wls_residual_tau_squared(
            initial_q,
            variance_array,
            predictor_count,
        )

    beta, covariance, residuals, residual_q, log_likelihood = _fit_weighted_regression(
        effect_array,
        variance_array,
        design_matrix,
        tau_squared,
    )
    standard_errors = np.sqrt(np.clip(np.diag(covariance), a_min=0, a_max=None))
    z_scores = np.divide(
        beta,
        standard_errors,
        out=np.zeros_like(beta),
        where=standard_errors > 0,
    )
    coefficients = pd.DataFrame(
        {
            "term": design_columns,
            "coefficient": beta,
            "standard_error": standard_errors,
            "ci_lower": beta - 1.95996398454 * standard_errors,
            "ci_upper": beta + 1.95996398454 * standard_errors,
            "p_value": 2 * norm.sf(np.abs(z_scores)),
        }
    )
    residual_degrees_of_freedom = effect_array.size - predictor_count
    residual_q_p_value = float(chi2.sf(residual_q, residual_degrees_of_freedom))
    if null_model.tau_squared <= 0:
        r_squared = 0.0
    else:
        r_squared = max(0.0, 1 - tau_squared / null_model.tau_squared)
    parameter_count = predictor_count + 1
    aic = float(-2 * log_likelihood + 2 * parameter_count)
    bic = float(-2 * log_likelihood + np.log(effect_array.size) * parameter_count)
    return MetaRegressionResult(
        coefficients=coefficients,
        tau_squared=tau_squared,
        r_squared=float(r_squared),
        residual_q=residual_q,
        residual_q_p_value=residual_q_p_value,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        fitted_values=tuple(float(value) for value in design_matrix @ beta),
        residuals=tuple(float(value) for value in residuals),
        estimation_method=method,
        moderator_columns=tuple(encoded_moderators.columns.tolist()),
    )
