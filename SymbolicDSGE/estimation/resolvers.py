from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence, Mapping

import numpy as np
from numpy.typing import NDArray

from . import backend as b

from ..core.compiled_model import CompiledModel
from ..bayesian.distributions.lkj_chol import LKJChol
from ..bayesian.priors import Prior
from ..bayesian.support import OutOfSupportError
from ..bayesian.transforms import CholeskyCorrTransform, Transform
from ..kalman.config import KalmanConfig
from ..core.config import PairGetterDict

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]

_RESERVED_MATRIX_KEYS = ("R_corr", "Q_corr")
_RESERVED_TO_NAME = {
    "R_corr": "R",
    "Q_corr": "Q",
}

# --- Name and Membership Validation ---


def assert_only_prior_or_parameters(
    estimated_parameters: Sequence[str] | None, priors: Mapping[str, Prior] | None
) -> list[str]:
    if estimated_parameters is not None:
        if priors is not None:
            mismatch = sorted(set(estimated_parameters) ^ set(priors))
            if mismatch:
                raise ValueError(
                    f"`estimated_parameters` and `priors` must name the same "
                    f"parameters; {mismatch} appear in one and not the other. If you "
                    f"specified a block prior, pass its name to "
                    f"`estimated_parameters` instead of the members."
                )
        return list(estimated_parameters)
    if priors is not None:
        return list(priors.keys())
    raise ValueError("Either `estimated_parameters` or `priors` must be specified.")


def check_parameters(
    compiled: CompiledModel,
    estimated_parameters: Sequence[str],
) -> None:
    base = b.extract_base_params(compiled)
    allowed = set(base.keys()).union(_RESERVED_MATRIX_KEYS)
    seen = set()
    for param in estimated_parameters:
        if param not in allowed:
            raise ValueError(
                f"Parameter {param!r} is not found in the compiled model specification."
                f"Available parameters are: {list(base.keys())}"
            )
        if param in seen:
            raise ValueError(f"Parameter {param!r} is specified multiple times.")
        seen.add(param)


def resolve_theta_layout(
    requested: Sequence[str],
    priored: Mapping[str, Prior] | None,
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    observables: list[str],
) -> tuple[dict[str, int], dict[str, b.MatrixPriorBlock]]:
    """Resolve the requested names into the theta layout and its CPC blocks.

    The returned mapping is ``name -> theta index`` in theta order, so
    ``list(...)`` recovers the parameter names. Each reserved matrix key that is
    an estimation target contributes one dense CPC block over a contiguous run,
    recorded on the block's ``theta_slice``.
    """
    present = set(requested)
    blocks: dict[str, b.MatrixPriorBlock] = {}
    folded: dict[str, str] = {}

    for key in _RESERVED_MATRIX_KEYS:
        by_key = key in present
        try:
            block = (
                _resolve_R(kalman=kalman, observables=observables)
                if key == "R_corr"
                else _resolve_Q(compiled=compiled)
            )
        except Exception:
            # Only a key asked for by name owes an explanation; an unresolvable
            # matrix is simply not a fold target.
            if by_key:
                raise
            continue

        members = block.member_names
        expected = (block.K * (block.K - 1)) // 2
        if by_key:
            if block.K < 2:
                raise ValueError(f"{key} requires a matrix of dimension at least 2.")
            _assert_dense_block(key, block)
            if priored is not None:
                _assert_lkj_prior(key, priored[key], block.K)
            doubled = sorted(set(members) & present)
            if doubled:
                raise ValueError(
                    f"Correlations {doubled} are members of '{key}', which is itself an "
                    f"estimation target. Estimate the block or its members, not both."
                )
        elif members and len(members) == expected and set(members) <= present:
            member_priors = sorted(
                name for name in members if priored is not None and name in priored
            )
            if member_priors:
                matrix_name = _RESERVED_TO_NAME[key]
                raise ValueError(
                    f"Correlations {member_priors} carry scalar priors but are the complete "
                    f"{matrix_name} correlation set, so independent per-parameter densities "
                    f"cannot guarantee a joint positive-definite matrix. Estimate the block "
                    f"via '{key}' with an LKJChol prior instead."
                )
            folded.update(dict.fromkeys(members, key))
        else:
            continue

        blocks[key] = block

    if len(blocks) == len(_RESERVED_MATRIX_KEYS):
        shared = sorted(
            set(blocks["R_corr"].member_names) & set(blocks["Q_corr"].member_names)
        )
        if shared:
            raise ValueError(
                f"Correlation blocks on R and Q cannot share member parameters. "
                f"Overlap: {shared}."
            )

    if blocks:
        std_names = _matrix_std_names(compiled, kalman, observables)
        for key, block in blocks.items():
            clash = sorted(std_names & set(block.member_names))
            if clash:
                raise ValueError(
                    f"Correlations {clash} of '{key}' are also named as standard "
                    f"deviations in Q or R. A block member reaches the covariance "
                    f"through the Cholesky reparameterization and is never scattered "
                    f"into the parameter vector; its standard deviation would silently "
                    f"hold its calibrated value. Use distinct parameter names."
                )

    names: list[str] = []
    pending = set(blocks)
    for name in requested:
        k = name if name in blocks else folded.get(name)
        if k is None:
            names.append(name)
            continue
        if k not in pending:
            continue
        pending.discard(k)
        block = blocks[k]
        start = len(names)
        names.extend(block.member_names)
        blocks[k] = block._replace(
            theta_slice=slice(start, start + len(block.member_names))
        )

    return {name: i for i, name in enumerate(names)}, blocks


def _matrix_std_names(
    compiled: CompiledModel,
    kalman: KalmanConfig | None,
    observables: list[str],
) -> set[str]:
    """Std parameter names that reach a cov spec's ``std_slots``, across Q and R.

    R contributes only the stds of the active observables, matching the
    ``obs_list`` the R spec indexes; Q contributes every named shock std.
    """
    active = {str(o) for o in observables}
    out: set[str] = set()
    std_map = None if kalman is None else kalman.R_std_param_map
    for obs, name in (std_map or {}).items():
        if name is not None and str(obs) in active:
            out.add(name)
    shock_std = getattr(compiled.config.calibration, "shock_std", None) or {}
    for name in shock_std.values():
        if name is not None:
            out.add(name)
    return out


def _assert_dense_block(key: str, block: b.MatrixPriorBlock) -> None:
    named = {(int(row), int(col)) for row, col in block.positions}
    triangle = [(row, col) for row in range(1, block.K) for col in range(row)]
    missing = [pair for pair in triangle if pair not in named]
    if missing:
        raise ValueError(_dense_matrix_error(key, block.labels, missing))
    if len(block.member_names) != len(triangle):
        raise ValueError(_dense_matrix_error(key, block.labels, triangle))


def _dense_matrix_error(
    key: str,
    labels: Sequence[str],
    pairs: Sequence[tuple[int, int]],
) -> str:
    matrix_name = _RESERVED_TO_NAME[key]
    pair_text = ", ".join(f"({labels[row]}, {labels[col]})" for row, col in pairs)
    return (
        f"LKJChol prior on {key} requires a dense correlation block for estimation, "
        f"but the configured {matrix_name} matrix is sparse. Missing named correlation parameters for pairs: "
        f"{pair_text}. Outside estimation, unnamed correlations fall back to their defaults "
        "(typically zero). For estimation with LKJChol, declare a named parameter for each missing "
        "pair in the config DSL and give it a placeholder default value (for example 0.0) so the "
        f"estimator can reparameterize the full {matrix_name} correlation matrix."
    )


def _assert_lkj_prior(key: str, prior: Prior, K: int) -> None:
    if not isinstance(prior, Prior):
        raise TypeError(
            f"Prior on matrix key '{key}' must be a Prior wrapping an LKJChol "
            f"distribution; got {type(prior).__name__}."
        )
    dist = prior.dist
    transform = prior.transform
    if not isinstance(dist, LKJChol) or not isinstance(
        transform, CholeskyCorrTransform
    ):
        raise ValueError(
            f"Block correlation estimation {key} requires a LKJChol distribution and a "
            "CholeskyCorrTransform. Got "
            f"distribution={type(dist).__name__}, "
            f"transform={type(transform).__name__}."
        )
    # make_prior reconciles the two Ks; a Prior built directly can disagree, and
    # the transform's K is what sizes the block's correlation factor.
    dist_k = int(getattr(dist, "_K", -1))
    if dist_k != transform.K:
        raise ValueError(
            f"Block correlation estimation {key} requires matching K between the "
            f"LKJChol distribution and its CholeskyCorrTransform. Got "
            f"distribution K={dist_k}, transform K={transform.K}."
        )
    if dist_k != K:
        raise ValueError(
            f"LKJChol prior on {key} has K={dist_k}, but the resolved {key} "
            f"correlation dimension is {K}."
        )


def check_estimated_versus_constant_R(
    requested: Sequence[str], kalman: KalmanConfig | None, R: NDF | None
) -> None:
    r_block_target = "R_corr" in requested
    r_component_target = False
    if kalman is not None:
        r_component_target = any(
            name in requested for name in (kalman.R_param_names or [])
        )
    is_target = r_block_target or r_component_target

    if is_target and R is not None:
        raise ValueError(
            "R cannot be supplied as a constant when 'R_corr' or any of its members are an estimation target."
        )


# --- Matrix Prior Resolution ---


def _build_matrix_resolution(
    key: str,
    labels: list[str],
    corr_param_map: PairGetterDict[str | None],
) -> b.MatrixPriorBlock:
    """Resolve the named std/correlation parameters for one matrix into a partial :class:`_MatrixPriorBlock` (``theta_slice`` empty, ``prior`` ``None``).

    Validates a unique named variance per diagonal and that no parameter name is
    reused. Missing off-diagonal pairs are simply absent from
    ``positions``/``member_names``; the caller derives and reports them against
    the expected dense set.
    """
    K = len(labels)
    used_names: set[str] = set()
    member_names: list[str] = []
    positions: list[tuple[int, int]] = []

    for row in range(1, K):
        for col in range(row):
            pair = (labels[row], labels[col])
            corr_name = corr_param_map[pair]
            if corr_name is None:
                continue
            if corr_name in used_names:
                raise ValueError(
                    f"LKJChol prior on {key} requires a unique named parameter per correlation pair. "
                    f"Parameter '{corr_name}' is reused."
                )
            used_names.add(corr_name)
            member_names.append(corr_name)
            positions.append((row, col))

    return b.MatrixPriorBlock(
        K=K,
        labels=list(labels),
        member_names=member_names,
        positions=np.asarray(positions, dtype=np.int64).reshape(-1, 2),
        theta_slice=slice(0, 0),
    )


def _resolve_R(
    kalman: KalmanConfig | None, observables: list[str]
) -> b.MatrixPriorBlock:
    if kalman is None:
        raise ValueError(
            "Block estimation of R requires a KalmanConfig to specify symbolic R std/correlation metadata."
        )
    std_param_map = kalman.R_std_param_map
    corr_param_map = kalman.R_corr_param_map
    if std_param_map is None or corr_param_map is None:
        raise ValueError(
            "LKJChol prior on R_corr requires parser-generated R std/correlation metadata."
        )
    return _build_matrix_resolution(
        key="R_corr",
        labels=observables,
        corr_param_map=corr_param_map,
    )


def _resolve_Q(compiled: CompiledModel) -> b.MatrixPriorBlock:
    shock_corr = compiled.config.calibration.shock_corr
    labels = list(compiled.shock_names)
    return _build_matrix_resolution(
        key="Q_corr",
        labels=labels,
        corr_param_map=shock_corr,
    )


# --- Prior Collection ---


def active_Q(
    requested: Sequence[str],
    compiled: CompiledModel,
) -> tuple[set[str], set[str]]:
    """Estimated Q parameters split by role, as ``(stds, correlations)``.

    Labels are ``shock_names``, the order the Q covariance spec is built over.
    """
    calib = compiled.config.calibration
    present = set(requested)
    shocks = compiled.shock_names

    corr_map: Mapping[Any, str | None] = calib.shock_corr

    std = {calib.shock_std[s] for s in shocks} & present
    corr = {
        name
        for pair in combinations(shocks, 2)
        if (name := corr_map.get(pair)) is not None and name in present
    }
    return std, corr


def active_R(
    requested: Sequence[str],
    kalman: KalmanConfig | None,
    observables: Sequence[str],
) -> tuple[set[str], set[str]]:
    """Estimated R parameters split by role, as ``(stds, correlations)``.

    Labels are the active observables, the order the R covariance spec is built
    over; a name reaching only an inactive observable is not estimated.
    """
    if kalman is None:
        return set(), set()

    present = set(requested)
    std_map = kalman.R_std_param_map
    corr_map: Mapping[Any, str | None] | None = kalman.R_corr_param_map

    std: set[str] = set()
    if std_map is not None:
        std = {std_map[obs] for obs in observables} & present

    corr: set[str] = set()
    if corr_map is not None:
        corr = {
            name
            for pair in combinations(observables, 2)
            if (name := corr_map.get(pair)) is not None and name in present
        }
    return std, corr


# --- Theta Construction and Bounds ---

#: Interior of the role box the transform-free leg is optimized over. Both sit
#: strictly inside the role's own region, which an optimizer projecting onto a
#: closed box would otherwise be free to sit on.
_STD_FLOOR = 1e-8
_CORR_LIMIT = 1.0 - 1e-6


def _blocked(
    n_theta: int, matrix_blocks: Mapping[str, b.MatrixPriorBlock]
) -> NDArray[np.bool_]:
    mask = np.zeros(n_theta, dtype=bool)
    for block in matrix_blocks.values():
        mask[block.theta_slice] = True
    return mask


def corr_from_members(block: b.MatrixPriorBlock, values: NDF) -> NDF:
    """A block's member values as the full correlation matrix."""
    corr = np.eye(block.K, dtype=np.float64)
    rows = block.positions[:, 0]
    cols = block.positions[:, 1]
    vals = np.asarray(values, dtype=np.float64)
    corr[rows, cols] = vals
    corr[cols, rows] = vals
    return corr


def theta_from_params(
    values: NDF,
    param_names: Sequence[str],
    matrix_blocks: Mapping[str, b.MatrixPriorBlock],
    transforms: Sequence[Transform],
) -> NDF:
    """Forward-map a parameter-space vector in theta order to theta.

    Both passes run over one buffer: a block's run is overwritten with the CPC
    coordinates of the correlation its members describe, and the scalar pass
    then reads what the block pass wrote, so the ``Identity`` standing on a
    block slot carries it through instead of restoring the member value.
    """
    z = np.array(values, dtype=np.float64, copy=True)
    if z.ndim != 1 or z.shape[0] != len(param_names):
        raise ValueError(
            f"Parameter vector of length {z.size} does not match the "
            f"{len(param_names)} estimated parameters."
        )

    for key, block in matrix_blocks.items():
        corr = corr_from_members(block, z[block.theta_slice])
        try:
            z[block.theta_slice] = b._unconstrained_from_corr(corr)
        except ValueError as exc:
            raise ValueError(
                f"Correlations of '{key}' do not form a positive-definite "
                f"correlation matrix over {block.labels}: {exc}"
            ) from exc

    for i, (name, transform) in enumerate(zip(param_names, transforms)):
        try:
            z[i] = transform.safe_forward(np.float64(z[i]))
        except OutOfSupportError as exc:
            raise ValueError(
                f"Value for '{name}' lies outside the region its transform maps "
                f"from: {exc}"
            ) from exc
    return z


def resolve_bounds(
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    param_index: Mapping[str, int],
    matrix_blocks: Mapping[str, b.MatrixPriorBlock],
    transforms: Sequence[Transform],
    roles: tuple[set[str], set[str]] | None,
) -> tuple[NDF, NDF]:
    """One call's theta box, as two dense ``n_theta`` buffers.

    Seeded from the region each transform maps from, which is how parameter
    space spells an unbounded side: the endpoint's image is the infinity theta
    space spells it with. So nothing restates a transform, and a bound binds
    only by sitting strictly inside it.

    ``roles`` arrives on the transform-free leg alone and is written last.
    There it is the only thing holding a standard deviation positive or a
    correlation inside its interval, which makes it the only restriction
    allowed to override what the caller asked for.
    """
    blocked = _blocked(len(param_index), matrix_blocks)
    lo = np.array([t.support.low for t in transforms], dtype=np.float64)
    hi = np.array([t.support.high for t in transforms], dtype=np.float64)

    _write_bounds(lo, hi, bounds, param_index, transforms, blocked)
    if roles is not None:
        _write_roles(lo, hi, param_index, roles, blocked)
        empty = np.flatnonzero(lo > hi)
        if empty.size:
            slot = int(empty[0])
            raise ValueError(
                f"The role box on '{list(param_index)[slot]}' leaves nothing of the "
                f"bound it was given: [{lo[slot]}, {hi[slot]}] is empty."
            )

    for i, transform in enumerate(transforms):
        # UpperBounded decreases, so the pair is placed by its image rather
        # than by the side it came from.
        low = transform.forward(np.float64(lo[i]))
        high = transform.forward(np.float64(hi[i]))
        lo[i], hi[i] = min(low, high), max(low, high)
    return lo, hi


def _write_bounds(
    lo: NDF,
    hi: NDF,
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    param_index: Mapping[str, int],
    transforms: Sequence[Transform],
    blocked: NDArray[np.bool_],
) -> None:
    """Write the caller's bounds over the seed, in place.

    Assigned rather than intersected: a value reaching the write has already
    been shown to lie in the region its transform maps from, so a ``max``
    against the seed could only ever return the value itself.
    """
    for name, (bound_lo, bound_hi) in (bounds or {}).items():
        idx = param_index.get(name)
        if idx is None:
            raise ValueError(f"Bound on {name!r}, which is not an estimated parameter.")
        if blocked[idx]:
            raise ValueError(
                f"'{name}' is a member of an estimated correlation block and reaches "
                f"the covariance through its Cholesky factor, so it owns no parameter "
                f"of its own to bound."
            )
        low = _side(name, bound_lo, transforms[idx], "lower", lo[idx])
        high = _side(name, bound_hi, transforms[idx], "upper", hi[idx])
        if low > high:
            raise ValueError(f"Bound ({bound_lo}, {bound_hi}) on '{name}' is reversed.")
        lo[idx], hi[idx] = low, high


def _write_roles(
    lo: NDF,
    hi: NDF,
    param_index: Mapping[str, int],
    roles: tuple[set[str], set[str]] | None,
    blocked: NDArray[np.bool_],
) -> None:
    """Close the slots a role names over whatever is already there, in place.

    A blocked slot is skipped: the role names reach it as a correlation, but
    the slot holds a Cholesky coordinate whose block keeps the matrix valid
    without restricting the coordinate.
    """
    std, corr = roles if roles is not None else (set(), set())
    for name, i in param_index.items():
        if blocked[i]:
            continue
        if name in std:
            lo[i] = max(lo[i], _STD_FLOOR)
        elif name in corr:
            lo[i] = max(lo[i], -_CORR_LIMIT)
            hi[i] = min(hi[i], _CORR_LIMIT)


def _side(
    name: str,
    value: float | None,
    transform: Transform,
    side: str,
    seed: np.float64,
) -> np.float64:
    """The number one side of a caller's pair ends up at.

    ``None`` and the side's own infinity are the same statement, that the
    caller bounded nothing here, and both leave the seed standing; the opposite
    infinity is not an open side but an empty one. A finite value must lie in
    the closure of the region the transform maps from, endpoints included,
    which is how a bound that only restates the transform costs nothing instead
    of raising.
    """
    if value is None:
        return seed

    out = np.float64(value)
    if np.isnan(out):
        raise ValueError(f"The {side} bound on '{name}' is not a number.")
    if np.isinf(out):
        if out != np.float64(-np.inf if side == "lower" else np.inf):
            raise ValueError(
                f"The {side} bound on '{name}' is {out}, which admits nothing."
            )
        return seed

    support = transform.support
    if not (support.low <= out <= support.high):
        raise ValueError(
            f"The {side} bound on '{name}' is {out}, outside "
            f"[{support.low}, {support.high}], the region "
            f"{type(transform).__name__} maps from."
        )
    return out


def serialize_bounds(
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
) -> dict[str, list[float | None]] | None:
    if bounds is None:
        return None
    return {
        name: [
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        ]
        for name, (lo, hi) in bounds.items()
    }
