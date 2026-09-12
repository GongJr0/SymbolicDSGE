"""Serializable estimation specification + result metadata (text only).

Stdlib dataclasses — the core ``estimation`` module must stay pydantic-free
(pydantic is only present transitively under the ``[ui]`` extra). The UI keeps
its pydantic request models in :mod:`SymbolicDSGE.ui.schemas` and converts via
:meth:`EstimationRunRequest.to_core`. This is the text representation a
``.sdsge`` bundle stores for the estimation tab.

Bulk arrays (observed data ``y``, MCMC ``samples``, ``logpost_trace``, and ``logjac_trace``) are
not carried here — they ride sibling Parquet members and pair with this
metadata at load time, mirroring the
:mod:`SymbolicDSGE.monte_carlo.serialize` split.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, TypedDict
from numpy.typing import NDArray


def _prior_from_spec(prior: PriorSpec) -> Any:
    from . import make_prior

    return make_prior(
        distribution=prior["distribution"],
        parameters=prior["parameters"],
        transform=prior["transform"],
        transform_kwargs=prior["transform_kwargs"],
    )


class PriorSpec(TypedDict):
    """Specification of a prior for serialization.

    Attributes
    ----------
    distribution : str
        Distribution family name.
    parameters : dict[str, float]
        Distribution parameters keyed by name.
    transform : str
        Transform method name.
    transform_kwargs : dict[str, float]
        Transform method keyword arguments keyed by name.

    """

    distribution: str
    parameters: dict[str, float]
    transform: str
    transform_kwargs: dict[str, float]


def _coerce_ss_seed(
    ss_seed: Mapping[str, float] | Sequence[float] | NDArray | None,
) -> dict[str, float] | list[float] | None:
    """A steady-state seed with JSON-safe values, in the shape it was given.

    A mapping names the variables it seeds and a sequence covers the declared
    set in order, so the two say different things and neither can be normalized
    into the other without knowing the model.
    """
    if ss_seed is None:
        return None
    if isinstance(ss_seed, Mapping):
        return {str(name): float(value) for name, value in ss_seed.items()}
    return [float(x) for x in ss_seed]


class EstimatorParams(TypedDict):
    """Estimator parameters for serialization.

    Attributes
    ----------
    observables : Sequence[str] | None
        Selected observables for the estimation. If ``None``, all observables are used.
    filter_mode : str
        Kalman filter mode. One of "linear", "extended", or "unscented".
    P0 : Sequence[Sequence[float]] | None
        Initial state covariance matrix. If ``None``, the stationary covariance is used.
    R : Sequence[Sequence[float]] | None
        Observation noise covariance matrix. If ``None``, the kalman config is consulted. Raises when both are ``None``.
    estimated_params : Sequence[str] | None
        List of parameter names to estimate. If ``None``, all parameters are estimated.
    priors : Mapping[str, PriorSpec] | None
        Mapping of parameter names to their prior specifications. If ``None``, no priors are used.
    ss_seed : Sequence[float] | Mapping[str, float] | None
        Steady-state seed values. Can be a sequence in declaration order or a mapping (with variable names as keys).
    x0 : Sequence[float] | None
        Kalman filter initial state vector. If ``None``, the steady-state is used.
    jitter : float
        Jitter to add to a matrix when cholesky decomposition fails. If zero, no jitter is added.
    symmetrize : bool
        Whether to symmetrize the covariance matrices in the Kalman kernels.
    joseph_cov : bool
        Whether to use Joseph form for the covariance update in the Kalman filter.

    """

    observables: Sequence[str] | None
    filter_mode: str
    P0: Sequence[Sequence[float]] | None
    R: Sequence[Sequence[float]] | None
    estimated_params: Sequence[str] | None
    priors: Mapping[str, PriorSpec] | None
    ss_seed: Sequence[float] | Mapping[str, float] | None
    x0: Sequence[float] | None
    jitter: float
    symmetrize: bool
    joseph_cov: bool


@dataclass
class EstimatorSpec:
    """Estimator specification (meta + data) for serialization.

    Attributes
    ----------
    y : Sequence[Sequence[float]]
        Observed data to estimate against.
    params : EstimatorParams
        JSON-format parameters for the :class:`Estimator`.

    """

    y: Sequence[Sequence[float]]
    params: EstimatorParams


class OptimizationResultSpec(TypedDict):
    """Point estimate optimization result metadata for serialization.

    Attributes
    ----------
    x : Sequence[float]
        Optimized parameter values in the order of ``param_names``.
    theta : Sequence[float]
        Optimized parameter values as a mapping from parameter names to values.
    success : bool
        Whether the optimization converged successfully.
    message : str
        Error message or success message from the optimizer.
    fun : float
        Objective function value at the optimum.
    nfev : int
        Number of function evaluations performed by the optimizer.
    nit : int | None
        Number of iternations if applicable for the method used.
    vcov : Sequence[Sequence[float]] | None
        Covariance (Hessian) of the objective funtion at the optimum. Computed when opted-in.
    cov_status : int
        Covariance solve status code. Non-zero is a failure.
    se : Mapping[str, float] | None
        Standard errors of the estimated parameters, computed from the covariance matrix.
    optimizer_config : dict[str, Any]
        Routine configuration to reproduce this run.
    loglik : float
        Log-likelihood at the optimum. Only present for MLE results.
    logpost : float
        Log-posterior at the optimum. Only present for MAP results.
    logprior : float
        Log-prior at the optimum. Only present for MAP results.

    """

    x: Sequence[float]
    theta: Mapping[str, float]
    success: bool
    message: str
    fun: float
    nfev: int
    nit: int | None
    vcov: Sequence[Sequence[float]] | None
    cov_status: int
    se: Mapping[str, float] | None

    # Run reprodiction needs call arguments
    optimizer_config: dict[str, Any]


class MLEResultSpec(OptimizationResultSpec):
    """Serializable form of a maximum likelihood estimation result.

    Extends the shared optimization spec with the achieved likelihood.

    Attributes
    ----------
    loglik : float
        Log likelihood at the reported optimum.
    """

    loglik: float


class MAPResultSpec(OptimizationResultSpec):
    """Serializable form of a maximum a posteriori estimation result.

    Extends the shared optimization spec with the posterior decomposition.

    Attributes
    ----------
    logpost : float
        Log posterior at the reported optimum.
    logprior : float
        Log prior at the reported optimum.
    """

    logpost: float
    logprior: float


class MCMCResultMeta(TypedDict):
    """Text-only metadata for an :class:`MCMCResult`.

    Bulk ``samples`` (``n_draws × len(param_names)``), ``logpost_trace``, and `logjac_trace``` ride
    a sibling Parquet member via :func:`SymbolicDSGE.bundle.columns_to_parquet`;
    pairing this metadata with that trace dict reconstructs the full result.
    """

    param_names: Sequence[str]
    accept_rate: float
    n_draws: int
    burn_in: int
    thin: int
    sampler_config: Mapping[str, Any]


@dataclass
class MCMCResultSpec:
    """Posterior sampling result spec for serialization.

    Attributes
    ----------
    samples : Sequence[Sequence[float]]
        Posterior samples in bundle-ready ND array form.
    logpost_trace : Sequence[float]
        Log-posterior trace for the samples.
    logjac_trace : Sequence[float]
        Log-Jacobian trace for the samples.
    meta : MCMCResultMeta
        Inline JSON metadata for the MCMC run, including parameter names, sampler config, and trace lengths.

    """

    samples: Sequence[Sequence[float]]
    logpost_trace: Sequence[float]
    logjac_trace: Sequence[float]
    meta: MCMCResultMeta
