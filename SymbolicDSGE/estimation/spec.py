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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

EstimationMethod = Literal["mle", "map", "mcmc"]
PosteriorPoint = Literal["mean", "map", "last"]

ESTIMATION_METHODS: frozenset[str] = frozenset(get_args(EstimationMethod))
POSTERIOR_POINTS: frozenset[str] = frozenset(get_args(PosteriorPoint))


@dataclass
class EstimatorInputs:
    """Concrete arguments lowered from an :class:`EstimationSpec`.

    The pydantic-free, core-resident result of compiling a spec for execution
    (the inverse of authoring it). ``priors`` holds built
    :class:`~SymbolicDSGE.bayesian.priors.Prior` objects (``None`` for MLE);
    ``bounds`` is ``None`` unless at least one parameter sets a bound, matching
    what :meth:`SymbolicDSGE.core.solver.DSGESolver.estimate` expects. ``theta0``
    is ``None`` when block (matrix) priors are present, so the estimator derives
    all initials from calibration.
    """

    estimated_params: list[str]
    theta0: dict[str, float] | None
    priors: dict[str, Any] | None = None
    bounds: list[tuple[float | None, float | None]] | None = None


def _prior_from_spec(prior: PriorSpec) -> Any:
    # Lazy import: keeps this module's import light (bayesian pulls numba/scipy);
    # only paid when a spec is actually lowered for a MAP/MCMC run.
    from ..bayesian.priors import make_prior

    return make_prior(
        distribution=prior.distribution,
        parameters=dict(prior.parameters),
        transform=prior.transform,
        transform_kwargs=dict(prior.transform_kwargs),
    )


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
    #: Block (LKJ) priors keyed by reserved matrix target ("R_corr"/"Q_corr").
    #: These cover whole correlation matrices, not scalar parameters, so they
    #: live apart from ``parameters`` (no scalar ``initial``/``bounds``).
    matrix_priors: dict[str, PriorSpec] = field(default_factory=dict)
    observables: list[str] | None = None
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    steady_state: list[float] | None = None
    posterior_point: str = "mean"

    def __post_init__(self) -> None:
        if self.method not in ESTIMATION_METHODS:
            raise ValueError(
                f"Unknown estimation method {self.method!r}; "
                f"expected one of {sorted(ESTIMATION_METHODS)}."
            )
        if not self.parameters and not self.matrix_priors:
            raise ValueError(
                "EstimationSpec requires at least one parameter or matrix prior."
            )
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
            "posterior_point": self.posterior_point,
        }
        if self.matrix_priors:
            out["matrix_priors"] = {
                target: prior.to_dict() for target, prior in self.matrix_priors.items()
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
            matrix_priors={
                str(target): PriorSpec.from_dict(prior)
                for target, prior in dict(data.get("matrix_priors", {})).items()
            },
            observables=(
                list(data["observables"])
                if data.get("observables") is not None
                else None
            ),
            method_kwargs=dict(data.get("method_kwargs", {})),
            steady_state=(
                [float(x) for x in data["steady_state"]]
                if data.get("steady_state") is not None
                else None
            ),
            posterior_point=str(data.get("posterior_point", "mean")),
        )

    @classmethod
    def from_targets(
        cls,
        estimated_params: Sequence[str],
        *,
        method: str = "mle",
        initial: Mapping[str, float] | None = None,
        priors: Mapping[str, PriorSpec] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        matrix_priors: Mapping[str, PriorSpec] | None = None,
        observables: Sequence[str] | None = None,
        method_kwargs: Mapping[str, Any] | None = None,
        steady_state: Sequence[float] | None = None,
        posterior_point: str = "mean",
    ) -> EstimationSpec:
        """Build a spec from estimation *targets* alone, mirroring
        :meth:`SymbolicDSGE.core.solver.DSGESolver.estimate`.

        Only the parameters you intend to estimate are listed — each is flagged
        ``estimate=True`` for you, so the GUI-shaped ``estimate`` toggle never
        has to be set by hand. ``initial`` supplies starting values (default
        ``0.0`` when omitted; calibration values are a better source, which is
        why :meth:`SymbolicDSGE.estimation.estimator.Estimator.to_spec` fills
        them in). ``priors``/``bounds`` are keyed by parameter name;
        ``matrix_priors`` carries block (LKJ) priors keyed by reserved target
        (``"R_corr"``/``"Q_corr"``).
        """
        if not estimated_params and not matrix_priors:
            raise ValueError(
                "from_targets requires at least one estimated parameter or "
                "matrix prior."
            )
        initial = dict(initial or {})
        prior_map = dict(priors or {})
        bound_map = dict(bounds or {})
        parameters: list[EstimationParameterSpec] = []
        for name in estimated_params:
            lower, upper = bound_map.get(name, (None, None))
            parameters.append(
                EstimationParameterSpec(
                    name=name,
                    initial=float(initial.get(name, 0.0)),
                    estimate=True,
                    lower=None if lower is None else float(lower),
                    upper=None if upper is None else float(upper),
                    prior=prior_map.get(name),
                )
            )
        return cls(
            method=method,
            parameters=parameters,
            matrix_priors=dict(matrix_priors or {}),
            observables=list(observables) if observables is not None else None,
            method_kwargs=dict(method_kwargs or {}),
            steady_state=(
                [float(x) for x in steady_state] if steady_state is not None else None
            ),
            posterior_point=posterior_point,
        )

    def to_estimator_inputs(self) -> EstimatorInputs:
        """Lower this spec to concrete :class:`EstimatorInputs` for a run.

        The inverse of authoring: selects the ``estimate=True`` parameters,
        collects their initials and bounds, and (for MAP/MCMC) builds the
        :class:`~SymbolicDSGE.bayesian.priors.Prior` objects from each
        :class:`PriorSpec`. Block (LKJ) ``matrix_priors`` are appended to
        ``estimated_params`` under their reserved target name and built as
        priors; the estimator expands them to correlation members and derives
        those initials from calibration, so ``theta0`` is left ``None`` whenever
        matrix priors are present. Lets a loaded ``.sdsge`` bundle drive
        ``DSGESolver.estimate`` without the ``[ui]`` extra.
        """
        active = [p for p in self.parameters if p.estimate]
        matrix_targets = list(self.matrix_priors)
        if not active and not matrix_targets:
            raise ValueError(
                "EstimationSpec has no estimated parameters or matrix priors."
            )
        if matrix_targets and self.method not in {"map", "mcmc"}:
            raise ValueError(
                "Matrix (LKJ) priors require method 'map' or 'mcmc'; "
                f"got {self.method!r}."
            )
        names = [p.name for p in active]
        if len(set(names)) != len(names):
            raise ValueError("Estimated parameter names must be unique.")

        bounds = [(p.lower, p.upper) for p in active]
        bound_arg = (
            bounds
            if any(low is not None or high is not None for low, high in bounds)
            else None
        )
        # Matrix targets expand to correlation members whose initials come from
        # calibration, so defer theta0 derivation to the estimator entirely.
        theta0: dict[str, float] | None = (
            None if matrix_targets else {p.name: float(p.initial) for p in active}
        )

        priors: dict[str, Any] | None = None
        if self.method in {"map", "mcmc"}:
            priors = {}
            for p in active:
                if p.prior is None:
                    raise ValueError(
                        f"Parameter '{p.name}' requires a prior for "
                        f"{self.method.upper()}."
                    )
                priors[p.name] = _prior_from_spec(p.prior)
            for target, prior in self.matrix_priors.items():
                priors[target] = _prior_from_spec(prior)

        return EstimatorInputs(
            estimated_params=names + matrix_targets,
            theta0=theta0,
            priors=priors,
            bounds=bound_arg,
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> EstimationSpec:
        return cls.from_dict(json.loads(text))


@dataclass
class OptimizationResultMeta:
    """Text-only metadata for an :class:`OptimizationResult`.

    The flat ``x`` vector isn't carried — ``theta`` covers the same point
    estimate by name. Sufficient to repaint the MLE/MAP summary on load and to
    rebuild a first-class :class:`OptimizationResult` (``x`` from ``theta``).
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
    optimizer_config: dict[str, Any] = field(default_factory=dict)

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
            "optimizer_config": dict(self.optimizer_config),
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
            optimizer_config=dict(data.get("optimizer_config", {})),
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
    sampler_config: dict[str, Any] = field(default_factory=dict)

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
            "sampler_config": dict(self.sampler_config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MCMCResultMeta:
        return cls(
            param_names=[str(name) for name in data["param_names"]],
            accept_rate=float(data["accept_rate"]),
            n_draws=int(data["n_draws"]),
            burn_in=int(data["burn_in"]),
            thin=int(data["thin"]),
            sampler_config=dict(data.get("sampler_config", {})),
        )
