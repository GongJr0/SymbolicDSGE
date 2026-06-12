"""Serializable estimation specification + result metadata (text only).

Stdlib dataclasses — the core ``estimation`` module must stay pydantic-free
(pydantic is only present transitively under the ``[ui]`` extra). The UI keeps
its pydantic request models in :mod:`SymbolicDSGE.ui.schemas` and converts via
:meth:`EstimationRunRequest.to_core`. This is the text representation a
``.sdsge`` bundle stores for the estimation tab.

Bulk arrays (observed data ``y``, MCMC ``samples`` and ``logpost_trace``) are
not carried here — they ride sibling Parquet members and pair with this
metadata at load time, mirroring the
:mod:`SymbolicDSGE.monte_carlo.serialize` split.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

EstimationMethod = Literal["mle", "map", "mcmc"]
PosteriorPoint = Literal["mean", "map", "last"]

ESTIMATION_METHODS: frozenset[str] = frozenset(get_args(EstimationMethod))
POSTERIOR_POINTS: frozenset[str] = frozenset(get_args(PosteriorPoint))


@dataclass
class PriorSpec:
    distribution: str = "normal"
    parameters: dict[str, float] = field(default_factory=dict)
    transform: str = "identity"
    transform_kwargs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "transform": self.transform,
            "transform_kwargs": {
                str(k): float(v) for k, v in self.transform_kwargs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriorSpec:
        return cls(
            distribution=str(data.get("distribution", "normal")),
            parameters={
                str(k): float(v) for k, v in dict(data.get("parameters", {})).items()
            },
            transform=str(data.get("transform", "identity")),
            transform_kwargs={
                str(k): float(v)
                for k, v in dict(data.get("transform_kwargs", {})).items()
            },
        )


@dataclass
class EstimationParameterSpec:
    name: str
    initial: float
    estimate: bool = False
    lower: float | None = None
    upper: float | None = None
    prior: PriorSpec | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EstimationParameterSpec.name must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "initial": float(self.initial),
            "estimate": bool(self.estimate),
        }
        if self.lower is not None:
            out["lower"] = float(self.lower)
        if self.upper is not None:
            out["upper"] = float(self.upper)
        if self.prior is not None:
            out["prior"] = self.prior.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EstimationParameterSpec:
        prior_data = data.get("prior")
        return cls(
            name=str(data["name"]),
            initial=float(data["initial"]),
            estimate=bool(data.get("estimate", False)),
            lower=None if data.get("lower") is None else float(data["lower"]),
            upper=None if data.get("upper") is None else float(data["upper"]),
            prior=PriorSpec.from_dict(prior_data) if prior_data is not None else None,
        )


@dataclass
class EstimationSpec:
    """Text representation of an estimation run (no observed data)."""

    method: str = "mle"
    parameters: list[EstimationParameterSpec] = field(default_factory=list)
    observables: list[str] | None = None
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    compile_kwargs: dict[str, Any] = field(default_factory=dict)
    steady_state: list[float] | None = None
    posterior_point: str = "mean"

    def __post_init__(self) -> None:
        if self.method not in ESTIMATION_METHODS:
            raise ValueError(
                f"Unknown estimation method {self.method!r}; "
                f"expected one of {sorted(ESTIMATION_METHODS)}."
            )
        if not self.parameters:
            raise ValueError("EstimationSpec.parameters must be non-empty.")
        if self.posterior_point not in POSTERIOR_POINTS:
            raise ValueError(
                f"Unknown posterior_point {self.posterior_point!r}; "
                f"expected one of {sorted(POSTERIOR_POINTS)}."
            )
        active = [p.name for p in self.parameters if p.estimate]
        if len(set(active)) != len(active):
            raise ValueError("Estimated parameter names must be unique.")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "parameters": [p.to_dict() for p in self.parameters],
            "method_kwargs": dict(self.method_kwargs),
            "compile_kwargs": dict(self.compile_kwargs),
            "posterior_point": self.posterior_point,
        }
        if self.observables is not None:
            out["observables"] = list(self.observables)
        if self.steady_state is not None:
            out["steady_state"] = [float(x) for x in self.steady_state]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EstimationSpec:
        return cls(
            method=str(data.get("method", "mle")),
            parameters=[
                EstimationParameterSpec.from_dict(p) for p in data.get("parameters", [])
            ],
            observables=(
                list(data["observables"])
                if data.get("observables") is not None
                else None
            ),
            method_kwargs=dict(data.get("method_kwargs", {})),
            compile_kwargs=dict(data.get("compile_kwargs", {})),
            steady_state=(
                [float(x) for x in data["steady_state"]]
                if data.get("steady_state") is not None
                else None
            ),
            posterior_point=str(data.get("posterior_point", "mean")),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> EstimationSpec:
        return cls.from_dict(json.loads(text))


@dataclass
class OptimizationResultMeta:
    """Text-only metadata for an :class:`OptimizationResult`.

    The raw ``scipy.optimize.OptimizeResult`` and the flat ``x`` vector aren't
    carried — ``theta`` covers the same information by name, and ``raw`` is
    opaque scipy state. Sufficient to repaint the MLE/MAP summary on load.
    """

    kind: str
    theta: dict[str, float]
    success: bool
    message: str
    fun: float
    loglik: float
    logprior: float
    logpost: float
    nfev: int
    nit: int | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("OptimizationResultMeta.kind must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "theta": {str(k): float(v) for k, v in self.theta.items()},
            "success": bool(self.success),
            "message": self.message,
            "fun": float(self.fun),
            "loglik": float(self.loglik),
            "logprior": float(self.logprior),
            "logpost": float(self.logpost),
            "nfev": int(self.nfev),
            "nit": None if self.nit is None else int(self.nit),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationResultMeta:
        return cls(
            kind=str(data["kind"]),
            theta={str(k): float(v) for k, v in dict(data["theta"]).items()},
            success=bool(data["success"]),
            message=str(data["message"]),
            fun=float(data["fun"]),
            loglik=float(data["loglik"]),
            logprior=float(data["logprior"]),
            logpost=float(data["logpost"]),
            nfev=int(data["nfev"]),
            nit=None if data.get("nit") is None else int(data["nit"]),
        )


@dataclass
class MCMCResultMeta:
    """Text-only metadata for an :class:`MCMCResult`.

    Bulk ``samples`` (``n_draws × len(param_names)``) and ``logpost_trace`` ride
    a sibling Parquet member via :func:`SymbolicDSGE.bundle.trace_to_json`;
    pairing this metadata with that trace dict reconstructs the full result.
    """

    param_names: list[str]
    accept_rate: float
    n_draws: int
    burn_in: int
    thin: int

    def __post_init__(self) -> None:
        if not self.param_names:
            raise ValueError("MCMCResultMeta.param_names must be non-empty.")
        if self.n_draws <= 0:
            raise ValueError("MCMCResultMeta.n_draws must be positive.")
        if self.burn_in < 0:
            raise ValueError("MCMCResultMeta.burn_in must be non-negative.")
        if self.thin <= 0:
            raise ValueError("MCMCResultMeta.thin must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_names": list(self.param_names),
            "accept_rate": float(self.accept_rate),
            "n_draws": int(self.n_draws),
            "burn_in": int(self.burn_in),
            "thin": int(self.thin),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCMCResultMeta:
        return cls(
            param_names=[str(name) for name in data["param_names"]],
            accept_rate=float(data["accept_rate"]),
            n_draws=int(data["n_draws"]),
            burn_in=int(data["burn_in"]),
            thin=int(data["thin"]),
        )
