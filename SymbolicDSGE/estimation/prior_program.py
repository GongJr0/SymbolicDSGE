from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..bayesian.distributions.beta import Beta
from ..bayesian.distributions.gamma import Gamma
from ..bayesian.distributions.half_cauchy import HalfCauchy
from ..bayesian.distributions.half_norm import HalfNormal
from ..bayesian.distributions.inv_gamma import InvGamma
from ..bayesian.distributions.lkj_chol import LKJChol, _log_lkj_normalizer_C
from ..bayesian.distributions.log_norm import LogNormal
from ..bayesian.distributions.norm import Normal
from ..bayesian.distributions.trunc_norm import TruncNormal
from ..bayesian.distributions.uniform import Uniform
from ..bayesian.priors import Prior
from ..bayesian.transforms.affine_logit import AffineLogitTransform
from ..bayesian.transforms.affine_probit import AffineProbitTransform
from ..bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from ..bayesian.transforms.identity import Identity
from ..bayesian.transforms.log import LogTransform
from ..bayesian.transforms.logit import LogitTransform
from ..bayesian.transforms.lower_bounded import LowerBoundedTransform
from ..bayesian.transforms.probit import ProbitTransform
from ..bayesian.transforms.softplus import SoftplusTransform
from ..bayesian.transforms.tanh import TanhTransform
from ..bayesian.transforms.upper_bounded import UpperBoundedTransform

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


class DistCode(IntEnum):
    """Integer dispatch codes for scalar prior families in the packed kernel.

    Mirrors ``SdsgeDistCode`` in ``prior_program.h`` -- the two MUST stay in
    lockstep (same names, same values). The Python side packs these into the
    int64 code arrays the native kernel switches on; numba evaluates the members
    directly inside the cached njit hot loop.
    """

    NORMAL = 1
    LOG_NORMAL = 2
    HALF_NORMAL = 3
    TRUNC_NORMAL = 4
    HALF_CAUCHY = 5
    BETA = 6
    GAMMA = 7
    INV_GAMMA = 8
    UNIFORM = 9


class TransformCode(IntEnum):
    """Integer dispatch codes for prior transforms in the packed kernel.

    Mirrors ``SdsgeTransformCode`` in ``prior_program.h``; see :class:`DistCode`.
    """

    IDENTITY = 1
    LOG = 2
    SOFTPLUS = 3
    LOGIT = 4
    PROBIT = 5
    AFFINE_LOGIT = 6
    AFFINE_PROBIT = 7
    LOWER_BOUNDED = 8
    UPPER_BOUNDED = 9
    TANH = 10


#: Packed-row strides (mirror ``SDSGE_N_DIST_PARAMS`` / ``SDSGE_N_TRANSFORM_PARAMS``
#: in ``prior_program.h``).
N_DIST_PARAMS = 5
N_TRANSFORM_PARAMS = 3


@dataclass(frozen=True, slots=True)
class PyPriorTables:
    """Mirror of ``sdsge_prior_tables``: packed log-prior program arguments.

    ``has_prior`` gates the whole block. Scalar columns run to ``n_scalar`` =
    ``len(scalar_indices)``; ``scalar_dist_params`` is n_scalar*5 and
    ``scalar_transform_params`` n_scalar*3, both read row-major flat by C.
    Matrix (CPC/LKJ) block columns run to ``n_blocks`` =
    ``len(matrix_offsets)``."""

    has_prior: bool
    scalar_indices: NDI  # n_scalar
    scalar_dist_codes: NDI  # n_scalar
    scalar_transform_codes: NDI  # n_scalar
    scalar_dist_params: NDF  # n_scalar*5
    scalar_transform_params: NDF  # n_scalar*3
    matrix_offsets: NDI  # n_blocks
    matrix_dims: NDI  # n_blocks
    matrix_lengths: NDI  # n_blocks
    matrix_etas: NDF  # n_blocks
    matrix_log_constants: NDF  # n_blocks

    @classmethod
    def empty(cls) -> "PyPriorTables":
        """The disabled table: no priors, or a prior the packer could not
        represent. Every column is length zero, so the kernel sums nothing."""
        empty_i = np.empty(0, dtype=np.int64)
        return cls(
            has_prior=False,
            scalar_indices=empty_i,
            scalar_dist_codes=empty_i,
            scalar_transform_codes=empty_i,
            scalar_dist_params=np.empty((0, N_DIST_PARAMS), dtype=float64),
            scalar_transform_params=np.empty((0, N_TRANSFORM_PARAMS), dtype=float64),
            matrix_offsets=empty_i,
            matrix_dims=empty_i,
            matrix_lengths=empty_i,
            matrix_etas=np.empty(0, dtype=float64),
            matrix_log_constants=np.empty(0, dtype=float64),
        )


def build_packed_logprior(
    *,
    priors: Mapping[str, Any] | None,
    param_index: Mapping[str, int],
    matrix_blocks: Mapping[str, Any],
    matrix_member_names: set[str],
) -> PyPriorTables | None:
    if priors is None:
        return None

    scalar_indices: list[int] = []
    scalar_dist_codes: list[int] = []
    scalar_transform_codes: list[int] = []
    scalar_dist_params: list[list[float]] = []
    scalar_transform_params: list[list[float]] = []

    matrix_offsets: list[int] = []
    matrix_dims: list[int] = []
    matrix_lengths: list[int] = []
    matrix_etas: list[float] = []
    matrix_log_constants: list[float] = []

    for name, prior in priors.items():
        if name in matrix_blocks:
            block = matrix_blocks[name]
            # A matrix key may be given as a bare LKJChol; the block carries the
            # coerced Prior, so read that and fall back to the raw entry.
            block_prior = getattr(block, "prior", None)
            if block_prior is not None:
                prior = block_prior
            if not isinstance(prior, Prior):
                raise TypeError(
                    f"Prior on matrix key '{name}' must be an LKJChol "
                    f"distribution or a Prior wrapping one; got "
                    f"{type(prior).__name__}."
                )
            if not isinstance(prior.dist, LKJChol) or not isinstance(
                prior.transform, CholeskyCorrTransform
            ):
                raise TypeError(
                    f"Prior on matrix key '{name}' must pair LKJChol with "
                    f"CholeskyCorrTransform; got {type(prior.dist).__name__} "
                    f"and {type(prior.transform).__name__}."
                )
            dim = int(block.dim)
            sl = block.theta_slice
            matrix_offsets.append(int(sl.start))
            matrix_dims.append(dim)
            matrix_lengths.append(int(sl.stop - sl.start))
            eta = float(getattr(prior.dist, "_eta"))
            matrix_etas.append(eta)
            matrix_log_constants.append(float(_log_lkj_normalizer_C(dim, eta)))
            continue

        if name in matrix_member_names:
            continue
        if name not in param_index:
            raise ValueError(f"Prior on '{name}', which is not an estimated parameter.")
        if not isinstance(prior, Prior):
            raise TypeError(
                f"Prior on '{name}' must be a Prior; got {type(prior).__name__}."
            )

        dist_code, dist_params = _pack_distribution(prior.dist)
        transform_code, transform_params = _pack_transform(prior.transform)
        if dist_code is None:
            raise TypeError(
                f"Prior on '{name}' uses distribution "
                f"{type(prior.dist).__name__}, which the native prior program "
                f"has no code for."
            )
        if transform_code is None:
            raise TypeError(
                f"Prior on '{name}' uses transform "
                f"{type(prior.transform).__name__}, which the native prior "
                f"program has no code for."
            )

        scalar_indices.append(int(param_index[name]))
        scalar_dist_codes.append(dist_code)
        scalar_transform_codes.append(transform_code)
        scalar_dist_params.append(dist_params)
        scalar_transform_params.append(transform_params)

    n_scalar = len(scalar_indices)

    return PyPriorTables(
        has_prior=True,
        scalar_indices=np.asarray(scalar_indices, dtype=np.int64),
        scalar_dist_codes=np.asarray(scalar_dist_codes, dtype=np.int64),
        scalar_transform_codes=np.asarray(scalar_transform_codes, dtype=np.int64),
        scalar_dist_params=np.asarray(
            scalar_dist_params if n_scalar else np.empty((0, N_DIST_PARAMS)),
            dtype=float64,
        ).reshape(n_scalar, N_DIST_PARAMS),
        scalar_transform_params=np.asarray(
            scalar_transform_params if n_scalar else np.empty((0, N_TRANSFORM_PARAMS)),
            dtype=float64,
        ).reshape(n_scalar, N_TRANSFORM_PARAMS),
        matrix_offsets=np.asarray(matrix_offsets, dtype=np.int64),
        matrix_dims=np.asarray(matrix_dims, dtype=np.int64),
        matrix_lengths=np.asarray(matrix_lengths, dtype=np.int64),
        matrix_etas=np.asarray(matrix_etas, dtype=float64),
        matrix_log_constants=np.asarray(matrix_log_constants, dtype=float64),
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
    return None, params
