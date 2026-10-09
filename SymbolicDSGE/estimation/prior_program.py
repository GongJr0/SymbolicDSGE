"""Mirror of the native prior program, packing the prior distribution and transform arguments into flat arrays for the kernel to read."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..bayesian.distributions import (
    Beta,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InvGamma,
    LKJChol,
    LogNormal,
    Normal,
    TruncNormal,
    Uniform,
)
from ..bayesian.transforms import (
    AffineLogitTransform,
    AffineProbitTransform,
    CholeskyCorrTransform,
    Identity,
    LogTransform,
    LogitTransform,
    LowerBoundedTransform,
    ProbitTransform,
    SoftplusTransform,
    TanhTransform,
    UpperBoundedTransform,
)
from ..bayesian.priors import Prior
from ..bayesian.support import Support
from ..core.compiled_model import CompiledModel
from ..kalman.config import KalmanConfig
from .resolvers import active_Q, active_R

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]

_STD_SCALAR_SUPPORT = Support(
    low=float64(0.0), high=float64(np.inf), low_inclusive=True, high_inclusive=False
)
_CORR_SCALAR_SUPPORT = Support(
    low=float64(-1.0), high=float64(1.0), low_inclusive=True, high_inclusive=True
)


class DistCode(IntEnum):
    """Integer dispatch codes for scalar prior families in the packed kernel.

    Mirrors ``SdsgeDistCode`` in ``prior_program.h`` -- the two MUST stay in
    lockstep (same names, same values). The Python side packs these into the
    int64 code arrays the native kernel switches on; numba evaluates the members
    directly inside the cached njit hot loop.
    """

    NO_DENSITY = 0
    NORMAL = 1
    LOG_NORMAL = 2
    HALF_NORMAL = 3
    TRUNC_NORMAL = 4
    HALF_CAUCHY = 5
    BETA = 6
    GAMMA = 7
    INV_GAMMA = 8
    UNIFORM = 9
    LKJ_CHOL = 10


class TransformCode(IntEnum):
    """Integer dispatch codes for prior transforms in the packed kernel.

    Mirrors ``SdsgeTransformCode`` in ``prior_program.h``; see :class:`DistCode`.
    """

    IDENTITY = 0
    LOG = 1
    SOFTPLUS = 2
    LOGIT = 3
    PROBIT = 4
    AFFINE_LOGIT = 5
    AFFINE_PROBIT = 6
    LOWER_BOUNDED = 7
    UPPER_BOUNDED = 8
    TANH = 9
    CHOLESKY_CORR = 10


#: Packed-row strides (mirror ``SDSGE_N_DIST_PARAMS`` / ``SDSGE_N_TRANSFORM_PARAMS``
#: in ``prior_program.h``).
N_DIST_PARAMS = 5
N_TRANSFORM_PARAMS = 3


@dataclass(frozen=True, slots=True)
class PyPriorTables:
    """Mirror of ``sdsge_prior_tables``: the packed per-theta prior program.

    Every column runs to ``n_theta``, so column ``i`` describes theta slot
    ``i`` and the kernel reads it by the loop counter with no gather.
    ``dist_params`` is n_theta*5 and ``transform_params`` n_theta*3, both read
    row-major flat by C. ``has_prior`` gates the density half only; the
    transform half is populated on every leg, since the z -> x map runs whether
    or not a density does. A CPC block's row repeats across its whole run, so
    a reader that enters the run must advance past it by ``K(K-1)/2``.
    """

    has_prior: bool
    dist_codes: NDI  # n_theta
    transform_codes: NDI  # n_theta
    dist_params: NDF  # n_theta*5
    transform_params: NDF  # n_theta*3


def build_prior_tables(
    *,
    param_index: Mapping[str, int],
    priors: Mapping[str, Any] | None,
    matrix_blocks: Mapping[str, Any],
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    observables: Sequence[str],
) -> PyPriorTables:
    """Pack the transform and density columns for one theta layout.

    A block's run carries ``CHOLESKY_CORR`` and its ``K`` on every slot,
    whatever the leg, since the z -> x map runs without a density; its LKJ row
    lands only when priored. Elsewhere ``priors`` covers every slot or none of
    them: with priors each slot takes its own density and transform, without
    them the zero fill stands and the region is left to the bounds at the call.
    """
    n_theta = len(param_index)
    dist_codes = np.zeros(n_theta, dtype=np.int64)
    transform_codes = np.zeros(n_theta, dtype=np.int64)
    dist_params = np.zeros((n_theta, N_DIST_PARAMS), dtype=float64)
    transform_params = np.zeros((n_theta, N_TRANSFORM_PARAMS), dtype=float64)

    blocked = np.zeros(n_theta, dtype=bool)
    has_prior = priors is not None

    for key, block in matrix_blocks.items():
        run = block.theta_slice
        blocked[run] = True
        transform_codes[run] = TransformCode.CHOLESKY_CORR
        transform_params[run, 0] = float(block.K)
        if priors is None:
            continue
        dist_code, dist_row = _pack_distribution(priors[key].dist)
        if dist_code is None:
            raise TypeError(
                f"Matrix block '{key}' uses distribution "
                f"{type(priors[key].dist).__name__!r}, which the native prior program "
                f"has no code for."
            )
        dist_codes[run] = dist_code
        dist_params[run] = dist_row

    if priors is None:
        return PyPriorTables(
            has_prior=has_prior,
            dist_codes=dist_codes,
            transform_codes=transform_codes,
            dist_params=dist_params,
            transform_params=transform_params,
        )
    else:
        names = list(param_index)
        stdQ, corrQ = active_Q(names, compiled)
        stdR, corrR = active_R(names, kalman, observables)
        std = stdQ | stdR
        corr = corrQ | corrR

        warn_std = []
        warn_corr = []

        for name, i in param_index.items():
            if blocked[i]:
                continue
            prior = priors[name]
            if not isinstance(prior, Prior):
                raise TypeError(
                    f"Prior on '{name}' must be a Prior; got {type(prior).__name__}."
                )
            transform = prior.transform
            if name in std and not (_STD_SCALAR_SUPPORT << transform.support):
                warn_std.append((name, type(transform).__name__))
                transform = LogTransform()
            elif name in corr and not (_CORR_SCALAR_SUPPORT << transform.support):
                warn_corr.append((name, type(transform).__name__))
                transform = TanhTransform()

            transform_code, transform_row = _pack_transform(transform)
            if transform_code is None:
                raise TypeError(
                    f"Prior on '{name}' uses transform "
                    f"{type(prior.transform).__name__!r}, which the native prior "
                    f"program has no code for."
                )

            dist_code, dist_row = _pack_distribution(prior.dist)
            if dist_code is None:
                raise TypeError(
                    f"Prior on '{name}' uses distribution "
                    f"{type(prior.dist).__name__!r}, which the native prior program "
                    f"has no code for."
                )
            if (
                dist_code == DistCode.LKJ_CHOL
                or transform_code == TransformCode.CHOLESKY_CORR
            ):
                raise ValueError(
                    f"Prior on '{name}' packs a block correlation code "
                    f"(dist={DistCode(dist_code).name}, "
                    f"transform={TransformCode(transform_code).name}), which only a "
                    f"reserved matrix key ('R_corr' or 'Q_corr') can carry."
                )
            dist_codes[i] = dist_code
            dist_params[i] = dist_row
            transform_codes[i] = transform_code
            transform_params[i] = transform_row

        _warn_std_corr_support(warn_std, warn_corr)
        return PyPriorTables(
            has_prior=has_prior,
            dist_codes=dist_codes,
            transform_codes=transform_codes,
            dist_params=dist_params,
            transform_params=transform_params,
        )


def _blank_dist_params() -> list[float]:
    return [0.0] * N_DIST_PARAMS


def _blank_transform_params() -> list[float]:
    return [0.0] * N_TRANSFORM_PARAMS


def _pack_distribution(dist: Any) -> tuple[int | None, list[float]]:
    params = _blank_dist_params()
    if isinstance(dist, Normal):
        params[0] = float(getattr(dist, "_mean"))
        params[1] = float(getattr(dist, "_var"))
        return DistCode.NORMAL, params
    if isinstance(dist, LogNormal):
        params[0] = float(getattr(dist, "_meanlog"))
        params[1] = float(getattr(dist, "_stdlog"))
        return DistCode.LOG_NORMAL, params
    if isinstance(dist, HalfNormal):
        params[0] = float(getattr(dist, "_std"))
        return DistCode.HALF_NORMAL, params
    if isinstance(dist, TruncNormal):
        params[0] = float(getattr(dist, "_mean"))
        params[1] = float(getattr(dist, "_std"))
        params[2] = float(getattr(dist, "_low_trunc"))
        params[3] = float(getattr(dist, "_high_trunc"))
        params[4] = float(getattr(dist, "_log_norm"))
        return DistCode.TRUNC_NORMAL, params
    if isinstance(dist, HalfCauchy):
        params[0] = float(getattr(dist, "_gamma"))
        return DistCode.HALF_CAUCHY, params
    if isinstance(dist, Beta):
        params[0] = float(getattr(dist, "_a"))
        params[1] = float(getattr(dist, "_b"))
        params[2] = float(getattr(dist, "_log_norm"))
        return DistCode.BETA, params
    if isinstance(dist, Gamma):
        params[0] = float(getattr(dist, "_a"))
        params[1] = float(getattr(dist, "_theta"))
        params[2] = float(getattr(dist, "_log_norm"))
        return DistCode.GAMMA, params
    if isinstance(dist, InvGamma):
        params[0] = float(getattr(dist, "_a"))
        params[1] = float(getattr(dist, "_beta"))
        params[2] = float(getattr(dist, "_log_prefactor"))
        return DistCode.INV_GAMMA, params
    if isinstance(dist, Uniform):
        params[0] = float(getattr(dist, "_low"))
        params[1] = float(getattr(dist, "_high"))
        params[2] = float(getattr(dist, "_width"))
        return DistCode.UNIFORM, params
    if isinstance(dist, LKJChol):
        params[0] = float(getattr(dist, "_eta"))
        params[1] = float(getattr(dist, "_K"))
        params[2] = float(getattr(dist, "_log_norm"))
        return DistCode.LKJ_CHOL, params
    return None, params


def _pack_transform(transform: Any) -> tuple[int | None, list[float]]:
    params = _blank_transform_params()
    if isinstance(transform, Identity):
        return TransformCode.IDENTITY, params
    if isinstance(transform, LogTransform):
        return TransformCode.LOG, params
    if isinstance(transform, SoftplusTransform):
        return TransformCode.SOFTPLUS, params
    if isinstance(transform, LogitTransform):
        return TransformCode.LOGIT, params
    if isinstance(transform, ProbitTransform):
        return TransformCode.PROBIT, params
    if isinstance(transform, TanhTransform):
        return TransformCode.TANH, params
    if isinstance(transform, AffineLogitTransform):
        params[0] = float(transform.low)
        params[1] = float(transform.high)
        params[2] = float(transform.high - transform.low)
        return TransformCode.AFFINE_LOGIT, params
    if isinstance(transform, AffineProbitTransform):
        params[0] = float(transform.low)
        params[1] = float(transform.high)
        params[2] = float(transform.high - transform.low)
        return TransformCode.AFFINE_PROBIT, params
    if isinstance(transform, LowerBoundedTransform):
        params[0] = float(transform.low)
        return TransformCode.LOWER_BOUNDED, params
    if isinstance(transform, UpperBoundedTransform):
        params[0] = float(transform.high)
        return TransformCode.UPPER_BOUNDED, params
    if isinstance(transform, CholeskyCorrTransform):
        params[0] = float(transform.K)
        return TransformCode.CHOLESKY_CORR, params
    return None, params


def _warn_std_corr_support(
    warn_std: list[tuple[str, str]], warn_corr: list[tuple[str, str]]
) -> None:
    if warn_std:
        warnings.warn(
            f"Standard deviation parameters [{', '.join(f'{name} ({transform})' for name, transform in warn_std)}] "
            "have transforms that can map draws outside of the valid support for standard deviations [0, inf). "
            "Transforms of each of these parameters have been replaced with `LogTransform` to ensure valid draws.",
            UserWarning,
        )
    if warn_corr:
        warnings.warn(
            f"Correlation parameters [{', '.join(f'{name} ({transform})' for name, transform in warn_corr)}] "
            "have transforms that can map draws outside of the valid support for correlations [-1, 1]. "
            "Transforms of each of these parameters have been replaced with `TanhTransform` to ensure valid draws.",
            UserWarning,
        )
