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
    y: Sequence[Sequence[float]]
    params: EstimatorParams


class OptimizationResultSpec(TypedDict):
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
    loglik: float


class MAPResultSpec(OptimizationResultSpec):
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
    samples: Sequence[Sequence[float]]
    logpost_trace: Sequence[float]
    logjac_trace: Sequence[float]
    meta: MCMCResultMeta
