"""Optimization result containers for DSGE model estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ...core.config import ModelConfig
from ..spec import (
    OptimizationResultSpec,
    MLEResultSpec,
    MAPResultSpec,
)

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class OptimizationResult:
    """Point estimate optimization result container.

    Attributes
    ----------
    x : NDArray[np.float64]
        Optimal parameter vector in the unconstrained space.
    theta : dict[str, float64]
        Optimal parameter vector in the constrained space, keyed by parameter names.
    success : bool
        Whether the optimization was successful.
    message : str
        Error/success message from the optimizer.
    fun : float64
        Objective function value at the optimum.
    nfev : int
        Number of function evaluations performed by the optimizer.
    nit : int | None
        Number of iterations performed by the optimizer. None if not applicable.
    optimizer_config : dict[str, Any]
        Configuration to reproduce the optimization run (optimizer method, bounds, options).
    vcov : NDArray[float64] | None
        Covariance matrix of the unconstrained parameters at the optimum, derived from the finite-difference Hessian.
        In unconstrained space; directly passable to MCMC as the proposal covariance.
    se : dict[str, float64] | None
        Standard errors of the constrained parameters, computed if covariance was requested.
    cov_status : int
        Status of the covariance computation: 0 if successful, otherwise indicates why it was not computed.

    """

    x: NDF
    theta: dict[str, float64]
    success: bool
    message: str
    fun: float64
    nfev: int
    nit: int | None
    #: Call configuration for the ``mle``/``map`` run (optimizer ``method``,
    #: ``bounds``, ``options``) — recorded so the run is reconstructable.
    optimizer_config: dict[str, Any]
    #: Asymptotic covariance at the optimum, from the finite-difference Hessian
    #: there. It describes ``x``, the unconstrained vector, so its rows and
    #: columns follow ``list(theta)`` and it is the covariance an MCMC proposal
    #: wants. ``None`` when the run was asked not to compute it; NaN throughout
    #: when it was asked and the Hessian at the optimum is not positive definite.
    vcov: NDF | None = field(default=None, kw_only=True)
    #: Standard errors keyed like ``theta`` and in the same constrained space,
    #: so ``theta[name]`` and ``se[name]`` are read together. Derived from
    #: ``vcov`` through the prior transforms' Jacobian, which is why this is not
    #: simply ``sqrt(diag(vcov))`` unless every transform is the identity.
    se: dict[str, float64] | None = field(default=None, kw_only=True)
    #: 0 when the covariance was produced, otherwise why it was not.
    cov_status: int = field(default=0, kw_only=True)

    def common_spec(self) -> OptimizationResultSpec:
        """Base spec for JSON-serialization of MLE and MAP result containers.

        Returns
        -------
        OptimizationResultSpec
            Shared components of point estimate results.

        """
        return OptimizationResultSpec(
            x=self.x.tolist(),
            theta={k: float(v) for k, v in self.theta.items()},
            success=bool(self.success),
            message=str(self.message),
            fun=float(self.fun),
            nfev=int(self.nfev),
            nit=int(self.nit) if self.nit is not None else None,
            optimizer_config=dict(self.optimizer_config),
            vcov=self.vcov.tolist() if self.vcov is not None else None,
            cov_status=int(self.cov_status),
            se=(
                {k: float(v) for k, v in self.se.items()}
                if self.se is not None
                else None
            ),
        )

    def apply_to(self, config: ModelConfig) -> ModelConfig:
        """Return a new config with the parameter values set to the point estimate.

        Parameters
        ----------
        config : ModelConfig
            Config to take as a base.
            Must contain all parameter names in the result's ``theta``.

        Returns
        -------
        ModelConfig
            New config object with updated parameters.

        """
        return config.with_parameters(self.theta)


@dataclass(frozen=True)
class MLEResult(OptimizationResult):
    """Maximum-likelihood point estimate result."""

    loglik: float64

    def to_spec(self) -> MLEResultSpec:
        """Return a JSON-serializable dict representation of the result."""
        return MLEResultSpec(**self.common_spec(), loglik=float(self.loglik))

    @classmethod
    def from_spec(cls, data: MLEResultSpec) -> MLEResult:
        """Construct a MLEResult from a JSON-serializable dict representation."""
        se = data.get("se") or {}
        return cls(
            x=np.asarray(data["x"], dtype=float64),
            theta={k: float64(v) for k, v in data["theta"].items()},
            success=bool(data["success"]),
            message=str(data["message"]),
            fun=float64(data["fun"]),
            nfev=int(data["nfev"]),
            nit=int(data["nit"]) if data["nit"] is not None else None,
            optimizer_config=dict(data.get("optimizer_config", {})),
            vcov=(
                np.asarray(data["vcov"], dtype=float64)
                if data.get("vcov") is not None
                else None
            ),
            se={k: float64(v) for k, v in se.items()} or None,
            cov_status=int(data.get("cov_status", 0)),
            loglik=float64(data["loglik"]),
        )


@dataclass(frozen=True)
class MAPResult(OptimizationResult):
    """Maximum-a-posteriori point estimate result."""

    logpost: float64
    logprior: float64

    def to_spec(self) -> MAPResultSpec:
        """Return a JSON-serializable dict representation of the result."""
        return MAPResultSpec(
            **self.common_spec(),
            logpost=float(self.logpost),
            logprior=float(self.logprior),
        )

    @classmethod
    def from_spec(cls, data: MAPResultSpec) -> MAPResult:
        """Construct a MAPResult from a JSON-serializable dict representation."""
        se = data.get("se") or {}
        return cls(
            x=np.asarray(data["x"], dtype=float64),
            theta={k: float64(v) for k, v in data["theta"].items()},
            success=bool(data["success"]),
            message=str(data["message"]),
            fun=float64(data["fun"]),
            nfev=int(data["nfev"]),
            nit=int(data["nit"]) if data["nit"] is not None else None,
            optimizer_config=dict(data.get("optimizer_config", {})),
            vcov=(
                np.asarray(data["vcov"], dtype=float64)
                if data.get("vcov") is not None
                else None
            ),
            se={k: float64(v) for k, v in se.items()} or None,
            cov_status=int(data.get("cov_status", 0)),
            logpost=float64(data["logpost"]),
            logprior=float64(data["logprior"]),
        )
