from __future__ import annotations

from copy import deepcopy
import warnings
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, NamedTuple, Sequence, cast

import numpy as np
import pandas as pd
from numpy import asarray, float64
from numpy.typing import NDArray
from scipy import optimize

from ..bayesian.distributions.lkj_chol import LKJChol
from ..bayesian.priors import Prior
from ..bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from ..bayesian.transforms.identity import Identity
from ..bayesian.transforms.log import LogTransform
from ..bayesian.transforms.tanh import TanhTransform
from ..bayesian.transforms.transform import Transform
from ..core.compiled_model import CompiledModel
from ..core.solver import DSGESolver
from . import backend
from .prior_program import PackedLogPrior, build_packed_logprior
from .results import MCMCResult, OptimizationResult
from .spec import EstimationSpec, PriorSpec

NDF = NDArray[np.float64]
_MatrixName = Literal["R", "Q"]
_MatrixPriorKey = Literal["R_corr", "Q_corr"]


class _MatrixPriorBlock(NamedTuple):
    """Minimal per-matrix LKJ metadata.

    ``positions`` is an ``(n_members, 2)`` int array of ``(row, col)`` targets in
    the correlation matrix, parallel to ``member_names``. The block's members
    occupy a contiguous theta run, so ``theta_slice`` (a plain ``slice``) covers
    them all: ``theta[theta_slice]`` is the block's unconstrained z with no gather.
    Resolution yields a partial block (``theta_slice`` empty, ``prior`` ``None``);
    it is completed via ``_replace`` once the theta layout is known. The reserved
    matrix key ("R_corr"/"Q_corr") is the dict key the block is stored under in
    ``_matrix_blocks``, so it is not repeated on the block itself.
    """

    dim: int
    labels: list[str]
    member_names: list[str]
    positions: NDArray[np.int64]
    theta_slice: slice
    prior: Prior | None


class MissingConfigError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Estimator:
    """
    Estimation interface exposing three public methods:
    - maximum likelihood estimation (`mle`)
    - maximum a posteriori estimation (`map`)
    - adaptive random-walk Metropolis MCMC (`mcmc`)
    """

    @staticmethod
    def make_prior(
        *,
        distribution: str,
        parameters: dict[str, Any],
        transform: str,
        transform_kwargs: dict[str, Any] | None = None,
    ) -> Prior:
        from ..bayesian.priors import make_prior as _make_prior

        return _make_prior(
            distribution=distribution,
            parameters=parameters,
            transform=transform,
            transform_kwargs=transform_kwargs,
        )

    @property
    def _reserved_matrix_keys(self) -> tuple[_MatrixPriorKey, _MatrixPriorKey]:
        return ("R_corr", "Q_corr")

    @staticmethod
    def _matrix_name_for_reserved_key(name: str) -> _MatrixName:
        if name == "R_corr":
            return "R"
        if name == "Q_corr":
            return "Q"
        raise KeyError(f"Unknown reserved matrix key '{name}'.")

    def __init__(
        self,
        *,
        solver: DSGESolver,
        compiled: CompiledModel,
        y: NDF | pd.DataFrame,
        observables: list[str] | None = None,
        filter_mode: str = "linear",
        estimated_params: Sequence[str] | None = None,
        priors: Mapping[str, Any] | None = None,
        ss_seed: NDF | dict[str, float] | None = None,
        x0: NDF | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool | None = None,
        R: NDF | None = None,
        P0: NDF | None = None,
    ) -> None:
        self.solver = solver
        self.compiled = compiled

        kalman = compiled.kalman
        if kalman is None:
            raise MissingConfigError(
                "Estimation requires a Kalman configuration; the compiled model has none. "
                "The likelihood is a Kalman filter loglik and cannot be formed without it."
            )
        self.kalman = kalman

        self.observables = observables
        self.filter_mode = filter_mode
        self._input_priors = dict(priors) if priors is not None else None

        self.ss_seed = ss_seed
        self.x0 = x0
        self.jitter = jitter
        self.symmetrize = symmetrize
        self.R = R
        self.P0 = P0

        self._prepared_filter = backend.prepare_filter_run(
            compiled=compiled,
            kalman=self.kalman,
            y=y,
            observables=observables,
            filter_mode=self.filter_mode,
            jitter=jitter,
            symmetrize=symmetrize,
            P0=P0,
        )
        self.y = self._prepared_filter.y_reordered  # Don't use user order directly.

        self._base_params = backend.extract_base_params(compiled)

        default_params = list(self._base_params.keys())
        requested_names_raw = self._requested_param_keys(estimated_params)
        allowed_names = set(default_params).union(self._reserved_matrix_keys)
        unknown = [p for p in requested_names_raw if p not in allowed_names]
        if unknown:
            raise ValueError(
                f"Unknown estimated parameters {unknown}. "
                f"Known calibration parameters: {default_params}"
            )

        # A fully-estimated dense correlation set is the reserved key by another
        # name; fold it so those correlations take the CPC block, not scalar tanh.
        requested_names_raw = self._promote_full_dense_corr_sets(requested_names_raw)

        r_is_target = ("R_corr" in requested_names_raw) or any(
            name
            for name in requested_names_raw
            if name in (self.kalman.R_param_names or [])
        )

        if r_is_target and self.R is not None:
            raise ValueError(
                "R cannot be supplied as a constant when 'R_corr' or any of its members are an estimation target."
            )

        # A reserved matrix key requested for estimation builds a CPC (Cholesky)
        # correlation block whether or not an LKJ prior is attached; the prior is
        # optional density on top of the reparameterization.
        self._requested_reserved_keys: tuple[_MatrixPriorKey, ...] = tuple(
            k for k in self._reserved_matrix_keys if k in requested_names_raw
        )
        self.priors = self._select_active_priors(requested_names_raw)
        self.param_names = self._expand_requested_params(requested_names_raw)
        self._param_index = {name: i for i, name in enumerate(self.param_names)}
        self._matrix_blocks = self._build_matrix_prior_blocks()
        self._matrix_member_names = {
            name
            for block in self._matrix_blocks.values()
            for name in block.member_names
        }
        self._spd_std_members, self._spd_corr_members = self._spd_member_names()
        self._corr_pairs = self._corr_pairs_by_name()
        identity = Identity()
        # Support the constraining transform must map onto, so loglik and
        # logprior share one theta<->param map.
        std_support = (float64(0.0), float64(np.inf))
        corr_support = (float64(-1.0), float64(1.0))
        self._param_transforms: dict[str, Transform] = {}
        for name in self.param_names:
            if name in self._matrix_member_names:
                # Correlation member of a CPC block: the block owns its
                # reparameterization (CholeskyCorr), so this scalar transform is
                # never consulted.
                self._param_transforms[name] = identity
                continue
            if name in self._spd_std_members:
                # A variance is positivity-constrained by its role in Q/R,
                # authoritatively. A conflicting prior transform is rejected.
                self._param_transforms[name] = self._role_transform_for(
                    name, LogTransform(), std_support
                )
                continue
            if name in self._spd_corr_members:
                # A correlation estimated as a standalone scalar (not via a block):
                # tanh into (-1, 1). The joint-SPD gate governs only the prior-free
                # role default. An explicit prior is the user's deliberate choice
                # (its transform still bounds it, and non-SPD draws fall to -inf),
                has_prior = self.priors is not None and name in self.priors
                if not has_prior:
                    self._assert_scalar_corr_spd_safe(name)
                self._param_transforms[name] = self._role_transform_for(
                    name, TanhTransform(), corr_support
                )
                continue
            # Plain calibration parameter: honor an explicit prior transform.
            self._param_transforms[name] = self._prior_transform_or(name, identity)
        self._packed_logprior: PackedLogPrior | None = build_packed_logprior(
            priors=self.priors,
            param_index=self._param_index,
            matrix_blocks=self._matrix_blocks,
            matrix_member_names=self._matrix_member_names,
        )
        self._warning_signal_count = 0

    def _spd_member_names(self) -> tuple[set[str], set[str]]:
        """Names of the SPD-relevant std (diagonal) and correlation (off-diagonal)
        parameters across the R and Q matrices, read straight from the parser's
        name maps.

        The two roles are kept separate because they need different constraining
        transforms: a variance wants a positivity map, a correlation a (-1, 1)
        map. Membership is deliberately independent of whether a prior exists, so
        this drives the transform defaults on the prior-free (MLE) path, not just
        the prior-gated CPC block.
        """
        std_members: set[str] = set()
        corr_members: set[str] = set()
        observed = self._active_observable_names()
        active_shocks = self._active_shock_names()

        r_std_map = getattr(self.kalman, "R_std_param_map", None) or {}
        for obs, v in r_std_map.items():
            if v is not None and (observed is None or str(obs) in observed):
                std_members.add(v)
        r_corr_map = getattr(self.kalman, "R_corr_param_map", None) or {}
        for pair, v in r_corr_map.items():
            if v is not None and (
                observed is None or {str(x) for x in pair} <= observed
            ):
                corr_members.add(v)

        calibration = self.compiled.config.calibration
        shock_std = getattr(calibration, "shock_std", None) or {}
        for shock, sym in shock_std.items():
            if sym is not None and (
                active_shocks is None or str(shock) in active_shocks
            ):
                std_members.add(sym.name)
        shock_corr = getattr(calibration, "shock_corr", None) or {}
        for pair, sym in shock_corr.items():
            if sym is not None and (
                active_shocks is None or {str(s) for s in pair} <= active_shocks
            ):
                corr_members.add(sym.name)

        return std_members, corr_members

    def _active_observable_names(self) -> set[str] | None:
        """Observable labels actually in the R matrix, or ``None`` if unavailable
        (then no filtering is applied). Correlations/variances of unobserved
        variables never enter R, so they are not SPD-relevant."""
        obs = getattr(self._prepared_filter, "observables", None)
        if obs is None:
            return None
        return {str(o) for o in obs}

    def _active_shock_names(self) -> set[str] | None:
        compiled = self.compiled
        try:
            shock_map = compiled.config.shock_map
            exogs = [str(v) for v in compiled.var_names[: compiled.n_exog]]
            rev = {str(exo): str(shock) for shock, exo in shock_map.items()}
            return {rev[e] for e in exogs if e in rev}
        except Exception:
            return None

    def _promote_full_dense_corr_sets(self, requested: Sequence[str]) -> list[str]:
        """Fold a fully-estimated *dense* correlation set into its reserved key.

        When every off-diagonal correlation of R or Q is a dense named set and all
        of its members are requested individually (e.g. the estimate-all default),
        that is the same estimation target as the reserved key. Promoting it here
        routes those correlations to the SPD-by-construction CPC block instead of
        per-scalar tanh, and groups them into one contiguous theta run.
        """
        result = list(requested)
        for key in self._reserved_matrix_keys:
            if key in result:
                continue
            matrix_name = self._matrix_name_for_reserved_key(key)
            try:
                block = self._resolve_R() if matrix_name == "R" else self._resolve_Q()
            except Exception:
                continue
            if block.dim < 2:
                continue
            expected = (block.dim * (block.dim - 1)) // 2
            members = set(block.member_names)
            dense = len(block.member_names) == expected
            if not (dense and members and members.issubset(result)):
                continue
            folded: list[str] = []
            inserted = False
            for name in result:
                if name in members:
                    if not inserted:
                        folded.append(key)
                        inserted = True
                    continue
                folded.append(name)
            result = folded
        return result

    def _corr_pairs_by_name(self) -> dict[str, tuple[str, frozenset[str]]]:
        """Map each named correlation parameter to ``(matrix_key, {var_a, var_b})``,
        for the joint-SPD safety gate on standalone scalar correlations."""
        out: dict[str, tuple[str, frozenset[str]]] = {}
        observed = self._active_observable_names()
        active_shocks = self._active_shock_names()
        r_corr_map = getattr(self.kalman, "R_corr_param_map", None) or {}
        for pair, nm in r_corr_map.items():
            vars_ = frozenset(str(v) for v in pair)
            if nm is not None and (observed is None or vars_ <= observed):
                out[nm] = ("R_corr", vars_)
        shock_corr = getattr(self.compiled.config.calibration, "shock_corr", None) or {}
        for pair, sym in shock_corr.items():
            vars_ = frozenset(str(s) for s in pair)
            if sym is not None and (active_shocks is None or vars_ <= active_shocks):
                out[sym.name] = ("Q_corr", vars_)
        return out

    def _prior_transform_or(self, name: str, default: Transform) -> Transform:
        """The transform an explicit prior on ``name`` carries, else ``default``."""
        if self.priors is not None and name in self.priors:
            prior_obj = self.priors[name]
            if hasattr(prior_obj, "transform"):
                return cast(Transform, getattr(prior_obj, "transform"))
        return default

    def _role_transform_for(
        self,
        name: str,
        default: Transform,
        role_support: tuple[float64, float64],
    ) -> Transform:
        """Role-authoritative constraining transform for an SPD member.

        With no prior on the member, returns the role default (Log for a
        variance, Tanh for a correlation). With a prior, the prior's transform is
        honored only if it constrains to the same domain.
        """
        low, high = role_support
        if self.priors is not None and name in self.priors:
            prior_obj = self.priors[name]
            if hasattr(prior_obj, "transform"):
                tr = cast(Transform, getattr(prior_obj, "transform"))
                sup = tr.support
                if not (sup.low == low and sup.high == high):
                    raise ValueError(
                        f"Prior on SPD parameter '{name}' uses a transform constraining "
                        f"to ({sup.low}, {sup.high}), but the parameter's role in Q/R "
                        f"requires a constraint to ({low}, {high}). Supply a prior whose "
                        f"transform matches that domain, or drop the prior to take the "
                        f"role default."
                    )
                return tr
        return default

    def _assert_scalar_corr_spd_safe(self, name: str) -> None:
        """Fail fast when estimating ``name`` as a standalone scalar correlation
        can't guarantee a joint-SPD matrix.
        """
        info = self._corr_pairs.get(name)
        if info is None:
            return
        matrix_key, pair = info
        for other_name, (other_key, other_pair) in self._corr_pairs.items():
            if other_name == name or other_key != matrix_key:
                continue
            if not (pair & other_pair):
                continue
            estimated = other_name in self._param_index
            fixed_nonzero = float(self._base_params.get(other_name, 0.0)) != 0.0
            if estimated or fixed_nonzero:
                shared = ", ".join(sorted(pair & other_pair))
                raise ValueError(
                    f"Correlation '{name}' is estimated as a standalone scalar, but "
                    f"variable(s) [{shared}] also carry another estimated or nonzero "
                    f"correlation ('{other_name}') in the same matrix, so a per-parameter "
                    f"tanh cannot guarantee joint positive-definiteness. Estimate the whole "
                    f"correlation block via '{matrix_key}' (Cholesky reparameterization) instead."
                )

    def _requested_param_keys(
        self,
        estimated_params: Sequence[str] | None,
    ) -> list[str]:
        if estimated_params is None:
            if self._input_priors is not None:
                return list(self._input_priors.keys())
            return [str(p) for p in self.compiled.calib_params]
        return list(estimated_params)

    def _select_active_priors(
        self,
        requested_names_raw: Sequence[str],
    ) -> dict[str, Any] | None:
        if self._input_priors is None:
            return None
        requested = set(requested_names_raw)
        active = {
            name: prior
            for name, prior in self._input_priors.items()
            if name in requested
        }
        return active or None

    def _expand_requested_params(
        self,
        requested_names_raw: Sequence[str],
    ) -> list[str]:
        expanded: list[str] = []
        owner: dict[str, str] = {}
        for name in requested_names_raw:
            if name in self._reserved_matrix_keys:
                matrix_name = self._matrix_name_for_reserved_key(name)
                block = self._resolve_R() if matrix_name == "R" else self._resolve_Q()
                members = block.member_names
            else:
                members = [name]

            for member in members:
                if member in owner:
                    raise ValueError(
                        f"Estimated parameter '{member}' is specified more than once via "
                        f"'{owner[member]}' and '{name}'."
                    )
                owner[member] = name
                expanded.append(member)
        return expanded

    @staticmethod
    def _coerce_lkj_prior(name: str, prior_obj: Any) -> Prior:
        if isinstance(prior_obj, LKJChol):
            return Prior(
                dist=prior_obj,
                transform=CholeskyCorrTransform(K=int(getattr(prior_obj, "_K", -1))),
            )
        if isinstance(prior_obj, Prior) and isinstance(prior_obj.dist, LKJChol):
            if not isinstance(prior_obj.transform, CholeskyCorrTransform):
                raise TypeError(
                    f"Prior '{name}' must use LKJChol directly or a Prior wrapping LKJChol with CholeskyCorrTransform."
                )
            if int(getattr(prior_obj.dist, "_K", -1)) != prior_obj.transform.K:
                raise TypeError(
                    f"Prior '{name}' must use matching K values between LKJChol and CholeskyCorrTransform."
                )
            return prior_obj
        raise TypeError(
            f"Prior '{name}' must be an LKJChol distribution or a Prior wrapping LKJChol with CholeskyCorrTransform."
        )

    @staticmethod
    def _format_pairs(pairs: Sequence[tuple[str, str]]) -> str:
        return ", ".join(f"({a}, {b})" for a, b in pairs)

    def _dense_matrix_error(
        self,
        key: _MatrixPriorKey,
        matrix_name: _MatrixName,
        missing_pairs: Sequence[tuple[str, str]],
    ) -> str:
        pair_text = self._format_pairs(missing_pairs)
        return (
            f"LKJChol prior on {key} requires a dense correlation block for estimation, "
            f"but the configured {matrix_name} matrix is sparse. Missing named correlation parameters for pairs: "
            f"{pair_text}. Outside estimation, unnamed correlations fall back to their defaults "
            "(typically zero). For estimation with LKJChol, declare a named parameter for each missing "
            "pair in the config DSL and give it a placeholder default value (for example 0.0) so the "
            f"estimator can reparameterize the full {matrix_name} correlation matrix."
        )

    @staticmethod
    def _cov_to_corr(cov: NDF, key: str) -> tuple[NDF, NDF]:
        cov = np.asarray(cov, dtype=float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"{key} must resolve to a square covariance matrix.")
        if not np.allclose(cov, cov.T, atol=1e-10, rtol=0.0):
            raise ValueError(f"{key} must resolve to a symmetric covariance matrix.")
        variances = np.diag(cov).astype(float64, copy=False)
        if np.any(variances <= 0.0):
            raise ValueError(f"{key} must have strictly positive diagonal variances.")

        std = np.sqrt(variances).astype(float64, copy=False)
        corr = cov / np.outer(std, std)
        corr = np.asarray(corr, dtype=float64)
        np.fill_diagonal(corr, 1.0)
        return std, corr

    def _build_matrix_resolution(
        self,
        *,
        key: _MatrixPriorKey,
        labels: list[str],
        std_param_map: Mapping[str, str | None],
        corr_param_map: Mapping[frozenset[str], str | None],
    ) -> _MatrixPriorBlock:
        """Resolve the named std/correlation parameters for one matrix into a
        partial :class:`_MatrixPriorBlock` (``theta_slice`` empty, ``prior``
        ``None``). Validates a unique named variance per diagonal and that no
        parameter name is reused. Missing off-diagonal pairs are simply absent
        from ``positions``/``member_names``; the caller derives and reports them
        against the expected dense set."""
        dim = len(labels)
        used_names: set[str] = set()
        member_names: list[str] = []
        positions: list[tuple[int, int]] = []

        for label in labels:
            std_name = std_param_map.get(label)
            if std_name is None:
                raise ValueError(
                    f"LKJChol prior on {key} requires a named variance parameter for "
                    f"{key}[{label}, {label}]."
                )
            if std_name in used_names:
                raise ValueError(
                    f"LKJChol prior on {key} requires a unique named variance parameter per "
                    f"diagonal entry. Parameter '{std_name}' is reused."
                )
            used_names.add(std_name)

        for row in range(1, dim):
            for col in range(row):
                pair = (labels[row], labels[col])
                corr_name = corr_param_map.get(frozenset(pair))
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

        return _MatrixPriorBlock(
            dim=dim,
            labels=list(labels),
            member_names=member_names,
            positions=np.asarray(positions, dtype=np.int64).reshape(-1, 2),
            theta_slice=slice(0, 0),
            prior=None,
        )

    def _resolve_R(self) -> _MatrixPriorBlock:
        labels = self._prepared_filter.observables
        std_param_map = self.kalman.R_std_param_map
        corr_param_map = self.kalman.R_corr_param_map
        if std_param_map is None or corr_param_map is None:
            raise ValueError(
                "LKJChol prior on R_corr requires parser-generated R std/correlation metadata."
            )
        return self._build_matrix_resolution(
            key="R_corr",
            labels=labels,
            std_param_map=std_param_map,
            corr_param_map=corr_param_map,
        )

    def _resolve_Q(self) -> _MatrixPriorBlock:
        Q_cov = backend.build_Q(self.compiled, self._base_params)
        self._cov_to_corr(Q_cov, "Q")

        shock_map = self.compiled.config.shock_map
        shock_std = self.compiled.config.calibration.shock_std
        shock_corr = self.compiled.config.calibration.shock_corr
        exogs = [str(v) for v in self.compiled.var_names[: self.compiled.n_exog]]
        rev = {str(exo): str(shock) for shock, exo in shock_map.items()}
        labels = [rev[exo] for exo in exogs]
        std_param_map: dict[str, str | None] = {}
        corr_param_map: dict[frozenset[str], str | None] = {}

        for label in labels:
            sym = shock_std[label]
            std_param_map[label] = None if sym is None else sym.name
        for row in range(1, len(labels)):
            for col in range(row):
                pair = (labels[row], labels[col])
                try:
                    sym = shock_corr[pair]
                except KeyError:
                    sym = None
                corr_param_map[frozenset(pair)] = None if sym is None else sym.name

        return self._build_matrix_resolution(
            key="Q_corr",
            labels=labels,
            std_param_map=std_param_map,
            corr_param_map=corr_param_map,
        )

    def _build_matrix_prior_blocks(self) -> dict[str, _MatrixPriorBlock]:
        # A reserved key requested for estimation builds a dense CPC correlation
        # block regardless of priors -- this is the SPD-by-construction Cholesky
        # reparameterization. An LKJChol prior, when present, is validated and
        # attached as optional density; without one the block carries prior=None
        # (pure reparameterization, e.g. the MLE path).
        blocks: dict[str, _MatrixPriorBlock] = {}
        claimed_names: set[str] = set()
        for key in self._requested_reserved_keys:
            matrix_name = self._matrix_name_for_reserved_key(key)
            block = self._resolve_R() if matrix_name == "R" else self._resolve_Q()
            if block.dim < 2:
                raise ValueError(f"{key} requires a matrix of dimension at least 2.")
            present = {(int(r), int(c)) for r, c in block.positions}
            missing_pairs = [
                (block.labels[row], block.labels[col])
                for row in range(1, block.dim)
                for col in range(row)
                if (row, col) not in present
            ]
            if missing_pairs:
                raise ValueError(
                    self._dense_matrix_error(key, matrix_name, missing_pairs)
                )

            expected = (block.dim * (block.dim - 1)) // 2
            if len(block.member_names) != expected:
                expected_pairs = [
                    (block.labels[row], block.labels[col])
                    for row in range(1, block.dim)
                    for col in range(row)
                ]
                raise ValueError(
                    self._dense_matrix_error(key, matrix_name, expected_pairs)
                )

            missing_estimated = [
                name for name in block.member_names if name not in self._param_index
            ]
            if missing_estimated:
                raise ValueError(
                    f"{key} requires all correlation members to be estimated. "
                    f"Missing from estimated_params: {missing_estimated}."
                )

            overlap = sorted(claimed_names.intersection(block.member_names))
            if overlap:
                raise ValueError(
                    f"Correlation blocks on R and Q cannot share member parameters. Overlap: {overlap}."
                )

            indices = [self._param_index[name] for name in block.member_names]
            start = indices[0]
            stop = start + len(indices)
            if indices != list(range(start, stop)):
                raise ValueError(
                    f"{key} expects its correlation members to occupy a contiguous "
                    f"theta range; got scattered indices {indices} for "
                    f"{block.member_names}."
                )

            lkj_prior = None
            if self.priors is not None and key in self.priors:
                lkj_prior = self._coerce_lkj_prior(key, self.priors[key])
                scalar_conflicts = [
                    name
                    for name in block.member_names
                    if self._input_priors is not None
                    and name in self._input_priors
                    and name not in self._reserved_matrix_keys
                ]
                if scalar_conflicts:
                    raise ValueError(
                        f"LKJChol prior on {key} cannot be combined with scalar priors on the same "
                        f"correlation members: {scalar_conflicts}."
                    )
                prior_dim = int(getattr(lkj_prior.dist, "_K", -1))
                if prior_dim != block.dim:
                    raise ValueError(
                        f"LKJChol prior on {key} has K={prior_dim}, but the resolved {key} "
                        f"correlation dimension is {block.dim}."
                    )

            blocks[key] = block._replace(
                theta_slice=slice(start, stop), prior=lkj_prior
            )
            claimed_names.update(block.member_names)

        return blocks

    @staticmethod
    def _corr_from_member_values(block: _MatrixPriorBlock, values: NDF) -> NDF:
        corr = np.eye(block.dim, dtype=float64)
        rows = block.positions[:, 0]
        cols = block.positions[:, 1]
        vals = np.asarray(values, dtype=float64)
        corr[rows, cols] = vals
        corr[cols, rows] = vals
        return corr

    @staticmethod
    def _block_cpc_from_corr(block: _MatrixPriorBlock, corr: NDF) -> NDF:
        try:
            return backend._unconstrained_from_corr(corr)
        except ValueError as exc:
            raise ValueError(
                f"Correlation values do not form a valid positive-definite "
                f"correlation matrix over {block.labels}: {exc}"
            ) from exc

    @staticmethod
    def _block_corr_from_theta(
        block: _MatrixPriorBlock, theta_block: NDF
    ) -> tuple[NDF, NDF]:
        Lcorr = backend._corr_chol_from_unconstrained(theta_block, block.dim)
        corr = np.asarray(Lcorr @ Lcorr.T, dtype=float64)
        return corr, np.asarray(Lcorr, dtype=float64)

    def to_spec(
        self,
        *,
        method: str | None = None,
        result: OptimizationResult | MCMCResult | None = None,
        priors: Mapping[str, PriorSpec] | None = None,
        observables: Sequence[str] | None = None,
        method_kwargs: Mapping[str, Any] | None = None,
        posterior_point: str = "mean",
    ) -> EstimationSpec:
        """Project this estimator (and optionally a run ``result``) to a
        serializable :class:`~SymbolicDSGE.estimation.spec.EstimationSpec`.

        Captures the estimated scalar parameters (calibration values as
        ``initial``), the observables, scalar priors (reversed losslessly from
        the live :class:`Prior` objects), and any block (LKJ) priors on
        ``R_corr``/``Q_corr``.

        When ``result`` is supplied the run is folded in: ``method`` is inferred
        from it, and its recorded ``method_kwargs`` (optimizer/sampler config)
        and parameter ``bounds`` are merged into the spec — so the spec fully
        reproduces the run. Explicit ``method``/``method_kwargs``/``priors``
        override the inferred values. Provide at least one of ``method`` or
        ``result``.
        """
        resolved_method = method or _method_from_result(result)
        if resolved_method is None:
            raise ValueError(
                "Provide method= or result= to determine the estimation method."
            )

        scalar_names = [
            name for name in self.param_names if name not in self._matrix_member_names
        ]

        scalar_priors: dict[str, PriorSpec]
        if priors is not None:
            scalar_priors = dict(priors)
        else:
            scalar_priors = {}
            for name in scalar_names:
                prior = (self.priors or {}).get(name)
                if prior is not None and hasattr(prior, "to_spec"):
                    scalar_priors[name] = prior.to_spec()

        matrix_priors = {
            target: block.prior.to_spec()
            for target, block in self._matrix_blocks.items()
            if block.prior is not None
        }

        if method_kwargs is not None:
            resolved_kwargs = dict(method_kwargs)
        elif result is not None:
            resolved_kwargs = _method_kwargs_from_result(result)
        else:
            resolved_kwargs = {}

        bounds_map = _bounds_from_result(result, self.param_names)
        scalar_bounds = {
            name: bounds_map[name] for name in scalar_names if name in bounds_map
        } or None

        return EstimationSpec.from_targets(
            scalar_names,
            method=resolved_method,
            initial={name: float(self._base_params[name]) for name in scalar_names},
            priors=scalar_priors or None,
            matrix_priors=matrix_priors or None,
            bounds=scalar_bounds,
            observables=(
                list(observables) if observables is not None else self.observables
            ),
            method_kwargs=resolved_kwargs or None,
            posterior_point=posterior_point,
        )

    def theta0(self) -> NDF:
        constrained = asarray(
            [self._base_params[name] for name in self.param_names],
            dtype=float64,
        )
        return self.params_to_theta(constrained)

    def resolve_theta0(self, theta0: NDF | Mapping[str, float] | None) -> NDF:
        """Coerce a user ``theta0`` to the unconstrained theta vector.

        ``None`` seeds from the model calibration (:meth:`theta0`); a mapping is
        validated against the estimated parameter names and converted through
        :meth:`params_to_theta`; an array is taken as-is.
        """
        if theta0 is None:
            return self.theta0()
        if isinstance(theta0, Mapping):
            missing = [name for name in self.param_names if name not in theta0]
            if missing:
                raise ValueError(
                    f"theta0 dictionary is missing estimated parameters: {missing}"
                )
            unknown = [key for key in theta0 if key not in self.param_names]
            if unknown:
                raise ValueError(f"theta0 dictionary has unknown parameters: {unknown}")
            return self.params_to_theta(
                {name: float64(theta0[name]) for name in self.param_names}
            )
        return asarray(theta0, dtype=float64)

    def _validate_prior_initial_guess(self, theta: NDF) -> None:
        """Fail fast when the initial guess sits outside the priors' support or
        breaks a prior transform. No-op when no priors are set."""
        if self.priors is None:
            return
        params = self.theta_to_params(theta)
        invalid: list[str] = []
        for name, prior in self.priors.items():
            if name not in params:
                continue
            val = float64(params[name])
            try:
                if hasattr(prior, "transform"):
                    z = float64(
                        cast(Transform, getattr(prior, "transform")).safe_forward(val)
                    )
                else:
                    z = val
                prior.logpdf(z)
            except Exception as exc:  # pragma: no cover - exact type is prior-dependent
                invalid.append(f"{name}={val} ({type(exc).__name__}: {exc})")
        if invalid:
            raise ValueError(
                "Initial calibration values are incompatible with the provided "
                "priors or their transforms: " + ", ".join(invalid)
            )

    def params_to_theta(self, params: Mapping[str, float] | NDF) -> NDF:
        if isinstance(params, Mapping):
            missing = [name for name in self.param_names if name not in params]
            if missing:
                raise ValueError(
                    f"Parameter mapping is missing estimated parameters: {missing}"
                )
            vals = asarray(
                [float64(params[name]) for name in self.param_names], dtype=float64
            )
        else:
            vals = asarray(params, dtype=float64)
            if vals.ndim != 1:
                raise ValueError("params array must be 1D.")
            if vals.shape[0] != len(self.param_names):
                raise ValueError(
                    f"params length {vals.shape[0]} does not match estimated parameter count {len(self.param_names)}."
                )
        out = np.empty_like(vals, dtype=float64)
        handled = np.zeros((len(self.param_names),), dtype=bool)
        for block in self._matrix_blocks.values():
            corr_vals = np.asarray(vals[block.theta_slice], dtype=float64)
            corr = self._corr_from_member_values(block, corr_vals)
            out[block.theta_slice] = self._block_cpc_from_corr(block, corr)
            handled[block.theta_slice] = True

        for i, name in enumerate(self.param_names):
            if handled[i]:
                continue
            out[i] = float64(
                self._param_transforms[name].safe_forward(float64(vals[i]))
            )
        return out

    def _resolve_theta(self, theta: NDF) -> tuple[dict[str, float64], dict[str, NDF]]:
        """Single materialization site for a theta draw.

        Returns the named parameter dict (the boundary view every caller expects)
        alongside the matrix blocks it built on the way — keyed by reserved matrix
        key ("Q_corr"/"R_corr"). The hot path takes the matrices straight to the
        Q/R assembly instead of re-gathering them from the named scalars.
        """
        theta = asarray(theta, dtype=float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match estimated parameter count {len(self.param_names)}."
            )
        full = dict(self._base_params)
        matrices: dict[str, NDF] = {}
        handled = np.zeros((len(self.param_names),), dtype=bool)
        for key, block in self._matrix_blocks.items():
            theta_block = np.asarray(theta[block.theta_slice], dtype=float64)
            corr, _ = self._block_corr_from_theta(block, theta_block)
            matrices[key] = corr
            member_vals = corr[block.positions[:, 0], block.positions[:, 1]]
            for name, val in zip(block.member_names, member_vals):
                full[name] = float64(val)
            handled[block.theta_slice] = True

        for i, name in enumerate(self.param_names):
            if handled[i]:
                continue
            full[name] = float64(
                self._param_transforms[name].safe_inverse(float64(theta[i]))
            )
        return full, matrices

    def theta_to_params(self, theta: NDF) -> dict[str, float64]:
        return self._resolve_theta(theta)[0]

    def _loglik_from_params(
        self,
        params: Mapping[str, float64],
        *,
        q_corr: NDF | None = None,
    ) -> float64:
        return backend.evaluate_loglik(
            solver=self.solver,
            compiled=self.compiled,
            kalman=self.kalman,
            y=self.y,
            params=params,
            filter_mode=self.filter_mode,
            observables=self.observables,
            ss_seed=self.ss_seed,
            x0=self.x0,
            jitter=self.jitter,
            symmetrize=self.symmetrize,
            R=self.R,
            P0=self.P0,
            prepared=self._prepared_filter,
            q_corr=q_corr,
        )

    def loglik(self, theta: NDF) -> float64:
        params, matrices = self._resolve_theta(theta)
        return self._loglik_from_params(params, q_corr=matrices.get("Q_corr"))

    def _logprior_python(self, theta: NDF) -> float64:
        if self.priors is None:
            return float64(0.0)
        lp = float64(0.0)
        for block in self._matrix_blocks.values():
            if block.prior is None:
                # Prior-free block (pure CPC reparameterization) contributes no
                # density -- a flat prior over the correlation manifold.
                continue
            theta_block = np.asarray(theta[block.theta_slice], dtype=float64)
            lp += float64(block.prior.logpdf(theta_block))

        for name, prior in self.priors.items():
            if name in self._matrix_blocks or name in self._matrix_member_names:
                continue
            if name in self._param_index:
                z = float64(theta[self._param_index[name]])
            elif name in self._base_params:
                x0 = float64(self._base_params[name])
                if hasattr(prior, "transform"):
                    z = float64(
                        cast(Transform, getattr(prior, "transform")).safe_forward(x0)
                    )
                else:
                    z = x0
            else:
                raise KeyError(f"Prior specified for unknown parameter '{name}'.")
            lp += float64(prior.logpdf(z))
        return lp

    def logprior(self, theta: NDF) -> float64:
        if self.priors is None:
            return float64(0.0)
        theta = asarray(theta, dtype=float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match estimated parameter count {len(self.param_names)}."
            )
        packed = self._packed_logprior
        if packed is not None and packed.matches(self.priors):
            lp_fast = float64(packed.logpdf(theta))
            if not np.isnan(lp_fast):
                return lp_fast
        return self._logprior_python(theta)

    def logpost(self, theta: NDF) -> float64:
        return float64(self.loglik(theta) + self.logprior(theta))

    def _logpost(self, theta: NDF) -> float64:
        # Single ``_resolve_theta`` pass shared by the loglik and the Q
        # correlation block (cheaper than ``logpost``, which resolves twice).
        params, matrices = self._resolve_theta(theta)
        return float64(
            self._loglik_from_params(params, q_corr=matrices.get("Q_corr"))
            + self.logprior(theta)
        )

    def _eval_with_warning_capture(
        self, fn: Callable[[NDF], float64], theta: NDF
    ) -> tuple[float64, int]:
        # A misconfigured model can emit a warning on every evaluation; left to
        # print, thousands of them flood the console and can take an IPython
        # kernel down on its output-buffer limit. Intercept at the source: a
        # counting ``showwarning`` tallies each warning and discards it, so
        # nothing is printed and nothing is retained (O(1) memory, no per-eval
        # buffer scan). stderr is deliberately left alone -- genuine errors there
        # halt the evaluation and are caught by the callers.
        signals = 0

        def _count(*_args: Any, **_kwargs: Any) -> None:
            nonlocal signals
            signals += 1

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = _count
            val = float64(fn(theta))
        return val, int(signals)

    def _reset_search_warning_count(self) -> None:
        self._warning_signal_count = 0

    def _report_search_warning_count(self, kind: str) -> None:
        print(
            f"[Estimator:{kind}] BK stability warnings encountered during search: {self._warning_signal_count}"
        )

    @staticmethod
    def _clone_generator(rng: np.random.Generator) -> np.random.Generator:
        # Snapshot the caller-provided generator so repeated runs can reuse the
        # same fixed seed/state without inheriting previous sampler consumption.
        bitgen = type(rng.bit_generator)()
        bitgen.state = deepcopy(rng.bit_generator.state)
        return np.random.Generator(bitgen)

    def _safe_loglik(self, theta: NDF) -> float64:
        try:
            ll, n_signals = self._eval_with_warning_capture(self.loglik, theta)
            self._warning_signal_count += n_signals
            if n_signals > 0 or not np.isfinite(ll):
                return float64(-np.inf)
            return float64(ll)
        except BaseException:
            return float64(-np.inf)

    def _safe_logpost(self, theta: NDF) -> float64:
        try:
            lp, n_signals = self._eval_with_warning_capture(self._logpost, theta)
            self._warning_signal_count += n_signals
            if n_signals > 0 or not np.isfinite(lp):
                return float64(-np.inf)
            return float64(lp)
        except BaseException:
            return float64(-np.inf)

    def _safe_logprior(self, theta: NDF) -> float64:
        try:
            lp = float64(self.logprior(theta))
            if not np.isfinite(lp):
                return float64(-np.inf)
            return lp
        except BaseException:
            return float64(-np.inf)

    @staticmethod
    def _serialize_bounds(
        bounds: Sequence[tuple[float, float]] | None,
    ) -> list[list[float | None]] | None:
        if bounds is None:
            return None
        return [
            [None if lo is None else float(lo), None if hi is None else float(hi)]
            for lo, hi in bounds
        ]

    def _pack_opt_result(
        self,
        kind: str,
        res: optimize.OptimizeResult,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        x = asarray(res.x, dtype=float64)
        theta = self.theta_to_params(x)

        ll = self._safe_loglik(x)
        lpr = self._safe_logprior(x)
        lpo = (
            float64(ll + lpr)
            if np.isfinite(ll) and np.isfinite(lpr)
            else float64(-np.inf)
        )

        return OptimizationResult(
            kind=kind,
            x=x,
            theta=theta,
            success=bool(res.success),
            message=str(res.message),
            fun=float64(res.fun),
            loglik=float64(ll),
            logprior=float64(lpr),
            logpost=float64(lpo),
            nfev=int(res.nfev),
            nit=(int(res.nit) if hasattr(res, "nit") and res.nit is not None else None),
            optimizer_config=dict(config or {}),
        )

    def mle(
        self,
        *,
        theta0: NDF | Mapping[str, float] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        method: str = "L-BFGS-B",
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        self._reset_search_warning_count()
        init = self.resolve_theta0(theta0)

        def obj(theta: NDF) -> float64:
            ll = self._safe_loglik(theta)
            if not np.isfinite(ll):
                return float64(np.inf)
            return float64(-ll)

        minimize = cast(Any, optimize.minimize)
        res = cast(
            optimize.OptimizeResult,
            minimize(
                obj,
                x0=init,
                method=method,
                bounds=bounds,
                options=(dict(options) if options is not None else None),
            ),
        )
        out = self._pack_opt_result(
            "mle",
            res,
            config={
                "method": method,
                "bounds": self._serialize_bounds(bounds),
                "options": dict(options) if options is not None else {},
            },
        )
        self._report_search_warning_count("mle")
        return out

    def map(
        self,
        *,
        theta0: NDF | Mapping[str, float] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        method: str = "L-BFGS-B",
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        if self.priors is None:
            raise ValueError("MAP requires priors. No priors were provided.")

        self._reset_search_warning_count()
        init = self.resolve_theta0(theta0)
        self._validate_prior_initial_guess(init)

        def obj(theta: NDF) -> float64:
            lp = self._safe_logpost(theta)
            if not np.isfinite(lp):
                return float64(np.inf)
            return float64(-lp)

        minimize = cast(Any, optimize.minimize)
        res = cast(
            optimize.OptimizeResult,
            minimize(
                obj,
                x0=init,
                method=method,
                bounds=bounds,
                options=(dict(options) if options is not None else None),
            ),
        )
        out = self._pack_opt_result(
            "map",
            res,
            config={
                "method": method,
                "bounds": self._serialize_bounds(bounds),
                "options": dict(options) if options is not None else {},
            },
        )
        self._report_search_warning_count("map")
        return out

    def mcmc(
        self,
        *,
        n_draws: int,
        burn_in: int = 1000,
        thin: int = 1,
        theta0: NDF | Mapping[str, float] | None = None,
        random_state: int | np.random.Generator | None = None,
        adapt: bool = True,
        adapt_start: int = 100,
        adapt_interval: int = 25,
        proposal_scale: float = 0.1,
        adapt_epsilon: float = 1e-8,
    ) -> MCMCResult:
        if n_draws <= 0:
            raise ValueError("n_draws must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if thin <= 0:
            raise ValueError("thin must be positive.")
        if self.priors is None:
            raise ValueError("MCMC requires priors to define a posterior.")
        self._reset_search_warning_count()

        rng = (
            self._clone_generator(random_state)
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        current = self.resolve_theta0(theta0)
        self._validate_prior_initial_guess(current)
        d = current.shape[0]
        if d == 0:
            raise ValueError("No estimated parameters were provided.")

        total_steps = burn_in + n_draws * thin
        cov = (float64(proposal_scale) ** 2) * np.eye(d, dtype=float64)
        scale = float64((2.38**2) / d)

        cur_lp = self._safe_logpost(current)
        accepted = 0

        history = np.empty((total_steps, d), dtype=float64)
        lp_trace = np.empty((total_steps,), dtype=float64)
        kept = np.empty((n_draws, d), dtype=float64)
        kept_lp = np.empty((n_draws,), dtype=float64)

        keep_i = 0
        eye_d = np.eye(d, dtype=float64)
        t0 = perf_counter()

        for t in range(total_steps):
            prop = rng.multivariate_normal(current, cov)
            prop_lp = self._safe_logpost(prop)

            if np.isfinite(prop_lp):
                log_alpha = prop_lp - cur_lp
                if np.log(rng.random()) < log_alpha:
                    current = prop
                    cur_lp = prop_lp
                    accepted += 1

            history[t] = current
            lp_trace[t] = cur_lp

            if (
                adapt
                and t < burn_in
                and t >= adapt_start
                and (t + 1) % adapt_interval == 0
            ):
                hist = history[: t + 1]
                if d == 1:
                    var = (
                        np.var(hist[:, 0], ddof=1)
                        if hist.shape[0] > 1
                        else float64(1.0)
                    )
                    cov = np.array([[scale * var + adapt_epsilon]], dtype=float64)
                else:
                    emp = np.cov(hist.T, ddof=1) if hist.shape[0] > 1 else eye_d
                    cov = scale * (asarray(emp, dtype=float64) + adapt_epsilon * eye_d)

            if t >= burn_in and (t - burn_in) % thin == 0:
                kept[keep_i] = current
                kept_lp[keep_i] = cur_lp
                keep_i += 1
        elapsed = max(perf_counter() - t0, np.finfo(float).eps)

        kept_params = np.empty_like(kept, dtype=float64)
        for i in range(n_draws):
            p = self.theta_to_params(kept[i])
            for j, name in enumerate(self.param_names):
                kept_params[i, j] = float64(p[name])

        print(
            f"MCMC sampling concluded in {elapsed:.2f} seconds with {float(total_steps / elapsed):.2f} iterations per second."
        )

        out = MCMCResult(
            param_names=list(self.param_names),
            samples=kept_params,
            logpost_trace=kept_lp,
            accept_rate=float64(accepted / total_steps),
            n_draws=n_draws,
            burn_in=burn_in,
            thin=thin,
            sampler_config={
                "adapt": bool(adapt),
                "adapt_start": int(adapt_start),
                "adapt_interval": int(adapt_interval),
                "proposal_scale": float(proposal_scale),
                "adapt_epsilon": float(adapt_epsilon),
                "random_state": (
                    int(random_state)
                    if isinstance(random_state, (int, np.integer))
                    else None
                ),
            },
        )
        self._report_search_warning_count("mcmc")
        return out


def _method_from_result(
    result: OptimizationResult | MCMCResult | None,
) -> str | None:
    """The estimation method (``mle``/``map``/``mcmc``) implied by a result."""
    if result is None:
        return None
    if isinstance(result, MCMCResult):
        return "mcmc"
    if isinstance(result, OptimizationResult):
        return result.kind
    raise TypeError(f"Unsupported estimation result type: {type(result).__name__}")


def _method_kwargs_from_result(
    result: OptimizationResult | MCMCResult,
) -> dict[str, Any]:
    """The method kwargs recorded on a result (sans bounds, folded separately)."""
    if isinstance(result, MCMCResult):
        return {
            "n_draws": int(result.n_draws),
            "burn_in": int(result.burn_in),
            "thin": int(result.thin),
            **dict(result.sampler_config),
        }
    cfg = dict(result.optimizer_config)
    out: dict[str, Any] = {}
    if cfg.get("method") is not None:
        out["method"] = cfg["method"]
    if cfg.get("options"):
        out["options"] = cfg["options"]
    return out


def _bounds_from_result(
    result: OptimizationResult | MCMCResult | None,
    param_names: Sequence[str],
) -> dict[str, tuple[float | None, float | None]]:
    """Per-parameter bounds recorded on an optimization result (empty otherwise)."""
    if isinstance(result, OptimizationResult):
        raw = result.optimizer_config.get("bounds")
        if raw:
            return {name: (pair[0], pair[1]) for name, pair in zip(param_names, raw)}
    return {}
