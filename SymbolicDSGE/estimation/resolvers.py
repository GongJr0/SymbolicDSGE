from typing import Sequence, Mapping

import numpy as np
from numpy.typing import NDArray

from . import backend as b

from ..core.compiled_model import CompiledModel
from ..bayesian.distributions.lkj_chol import LKJChol
from ..bayesian.priors import Prior
from ..bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from ..kalman.config import KalmanConfig
from ..core.config import PairGetterDict

NDF = NDArray[np.float64]

_RESERVED_MATRIX_KEYS = ("R_corr", "Q_corr")
_RESERVED_TO_NAME = {
    "R_corr": "R",
    "Q_corr": "Q",
}


# --- Name and Membership Validation ---


def assert_only_prior_or_parameters(
    estimated_parameters: Sequence[str] | None, priors: Mapping[str, Prior] | None
) -> list[str]:
    if estimated_parameters is not None and priors is not None:
        raise ValueError(
            "Specify `estimated_parameters` only when you intend to use MLE alone. "
            "`priors` is keyed by estimated parameter names and is mandatory for MAP and MCMC. "
            "Calling MLE with `priors` will estimate the parameters named as keys but ignore the prior distributions/transforms."
        )
    if estimated_parameters is not None:
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
        prior=None,
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
