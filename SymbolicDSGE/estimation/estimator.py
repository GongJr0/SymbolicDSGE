from __future__ import annotations

import warnings
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from numpy import asarray, float64
from numpy.typing import NDArray

from ..bayesian.distributions.lkj_chol import LKJChol
from ..bayesian.priors import Prior
from ..bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from ..bayesian.transforms.identity import Identity
from ..bayesian.transforms.log import LogTransform
from ..bayesian.transforms.tanh import TanhTransform
from ..bayesian.transforms.transform import Transform

from ..core.compiled_model import CompiledModel

from .._ckernels.estimation import (
    run_estimation,
    run_mcmc,
    loglik,
    logprior,
    logpost,
)

from .prior_program import PyPriorTables, build_packed_logprior
from .results import MCMCResult, MLEResult, MAPResult, OptimizationResult
from .spec import EstimatorSpec, EstimatorParams, PriorSpec, _coerce_ss_seed

from . import backend
from .backend import (
    MatrixName,
    MatrixPriorKey,
    MatrixPriorBlock,
    PyExtendedContext,
    PyLinearContext,
    PyUnscentedContext,
    build_linear_context,
    build_extended_context,
    build_unscented_context,
    build_obj_common,
)

NDF = NDArray[np.float64]


class Estimator:
    """
    Estimation interface exposing three public methods:
    - maximum likelihood estimation (`mle`)
    - maximum a posteriori estimation (`map`)
    - adaptive random-walk Metropolis MCMC (`mcmc`)
    """

    @property
    def _reserved_matrix_keys(self) -> tuple[MatrixPriorKey, MatrixPriorKey]:
        return ("R_corr", "Q_corr")

    @staticmethod
    def _matrix_name_for_reserved_key(name: str) -> MatrixName:
        if name == "R_corr":
            return "R"
        if name == "Q_corr":
            return "Q"
        raise KeyError(f"Unknown reserved matrix key '{name}'.")

    def __init__(
        self,
        *,
        compiled: CompiledModel,
        y: NDF | pd.DataFrame,
        observables: Sequence[str] | None = None,
        filter_mode: str = "linear",
        estimated_params: Sequence[str] | None = None,
        priors: Mapping[str, Prior] | None = None,
        ss_seed: Sequence[float] | NDF | Mapping[str, float] | None = None,
        x0: NDF | None = None,
        jitter: float | float64 | None = None,
        symmetrize: bool = True,
        joseph_cov: bool = True,
        R: NDF | None = None,
        P0: NDF | None = None,
    ) -> None:

        self.estimated_params = estimated_params
        self.compiled = compiled
        if compiled.kalman is None and R is None:
            raise ValueError(
                "R must be provided in symbolic or scalar form, either through the "
                "model's Kalman configuration or as a parameter override."
            )

        self.kalman = compiled.kalman

        self.observables = observables
        self.y = y

        self.ss_seed = ss_seed
        self.x0 = x0
        self.R = R
        self.P0 = P0

        self._prepared_filter = backend.prepare_filter_run(
            compiled=compiled,
            kalman=self.kalman,
            y=y,
            observables=observables,
            filter_mode=filter_mode,
            jitter=jitter,
            symmetrize=symmetrize,
            joseph_cov=bool(joseph_cov),
            P0=P0,
        )

        self._base_params = backend.extract_base_params(compiled)

        self.priors = dict(priors) if priors is not None else None

        # The reserved block keys are estimable targets without being calibration
        # parameters, so they join the allowed set the requested names validate against.
        allowed_names = set(self._base_params).union(self._reserved_matrix_keys)
        requested_names_raw = self._requested_param_keys(
            allowed_names, estimated_params, self.priors
        )

        # A fully-estimated dense correlation set is the reserved key by another
        # name; fold it so those correlations take the CPC block, not scalar tanh.
        requested_names_raw = self._promote_full_dense_corr_sets(
            requested_names_raw, self.priors
        )

        r_block_target = "R_corr" in requested_names_raw
        if self.kalman is not None:
            r_component_target = any(
                name
                for name in requested_names_raw
                if name in (self.kalman.R_param_names or [])
            )
        else:
            r_component_target = False

        r_is_target = r_block_target or r_component_target

        if r_is_target and self.R is not None:
            raise ValueError(
                "R cannot be supplied as a constant when 'R_corr' or any of its members are an estimation target."
            )

        # A reserved matrix key requested for estimation builds a CPC (Cholesky)
        # correlation block whether or not an LKJ prior is attached; the prior is
        # optional density on top of the reparameterization.
        self._requested_reserved_keys: tuple[MatrixPriorKey, ...] = tuple(
            k for k in self._reserved_matrix_keys if k in requested_names_raw
        )
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
            self._param_transforms[name] = self._get_transform(name)

    def _get_transform(self, name: str) -> Transform:
        if self.priors is not None and name in self.priors:
            return self.priors[name].transform
        return Identity()

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
        try:
            return set(self.compiled.shock_names)
        except Exception:
            return None

    def _promote_full_dense_corr_sets(
        self,
        requested: Sequence[str],
        priors: Mapping[str, Prior] | None,
    ) -> list[str]:
        """Fold a fully-estimated *dense* correlation set into its reserved key.

        When every off-diagonal correlation of R or Q is a dense named set and all
        of its members are requested individually (e.g. the estimate-all default),
        that is the same estimation target as the reserved key. Promoting it here
        routes those correlations to the SPD-by-construction CPC block instead of
        per-scalar tanh, and groups them into one contiguous theta run.

        Scalar priors on the members are rejected rather than folded: independent
        per-parameter densities cannot keep the matrix positive-definite, which is
        the guarantee the block's LKJChol prior exists to provide.
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
            priored = sorted(
                name
                for name in block.member_names
                if priors is not None and name in priors
            )
            if priored:
                raise ValueError(
                    f"Correlations {priored} carry scalar priors but are the complete "
                    f"{matrix_name} correlation set, so independent per-parameter densities "
                    f"cannot guarantee a joint positive-definite matrix. Estimate the block "
                    f"via '{key}' with an LKJChol prior instead."
                )
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
        tr = self._get_transform(name)
        sup = tr.support
        if not (sup.low >= low and sup.high <= high):
            warnings.warn(
                f"SPD parameter '{name}' uses {type(tr).__name__} transform constraining "
                f"to ({sup.low}, {sup.high}), but the parameter's role in Q/R "
                f"requires a constraint to ({low}, {high}). The default role "
                f"transform ({type(default).__name__}) is used instead.",
                UserWarning,
            )
            tr = default
        return tr

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

    @staticmethod
    def _requested_param_keys(
        allowed_names: set[str],
        estimated_params: Sequence[str] | None,
        priors: Mapping[str, Prior] | None = None,
    ) -> list[str]:
        if estimated_params is not None:
            if not all(param in allowed_names for param in estimated_params):
                missing = set(estimated_params) - allowed_names
                raise ValueError(
                    f"Parameters {{{missing}}} are not estimable targets of the model: {sorted(allowed_names)}"
                )
            if not all(param in estimated_params for param in priors or {}):
                missing = set(priors or {}) - set(estimated_params)
                raise ValueError(
                    f"Priors specified for parameters {{{missing}}} which are not in the estimated parameters: {list(estimated_params)}"
                )
            return list(estimated_params)
        if priors is not None:
            if not all(param in allowed_names for param in priors):
                missing = set(priors) - allowed_names
                raise ValueError(
                    f"Parameters {{{missing}}} are not estimable targets of the model: {sorted(allowed_names)}"
                )
            return list(priors)
        raise ValueError(
            "Either estimated_params or priors must be provided to determine the requested parameters."
        )

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
    def _is_lkj_prior(name: str, prior: Prior) -> Prior:
        dist = prior.dist
        transform = prior.transform

        if not isinstance(dist, LKJChol) or not isinstance(
            transform, CholeskyCorrTransform
        ):
            raise ValueError(
                f"Block correlation estimation {name} requires a LKJChol distribution and a "
                "CholeskyCorrTransform. Got "
                f"distribution={type(prior.dist).__name__}, "
                f"transform={type(prior.transform).__name__}."
            )
        # make_prior reconciles the two Ks; a Prior built directly can disagree,
        # and the transform's K is what sizes the block's correlation factor.
        dist_k = int(getattr(dist, "_K", -1))
        if dist_k != transform.K:
            raise ValueError(
                f"Block correlation estimation {name} requires matching K between the "
                f"LKJChol distribution and its CholeskyCorrTransform. Got "
                f"distribution K={dist_k}, transform K={transform.K}."
            )
        return prior

    @staticmethod
    def _format_pairs(pairs: Sequence[tuple[str, str]]) -> str:
        return ", ".join(f"({a}, {b})" for a, b in pairs)

    def _dense_matrix_error(
        self,
        key: MatrixPriorKey,
        matrix_name: MatrixName,
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
        key: MatrixPriorKey,
        labels: list[str],
        std_param_map: Mapping[str, str | None],
        corr_param_map: Mapping[frozenset[str], str | None],
    ) -> MatrixPriorBlock:
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

        return MatrixPriorBlock(
            dim=dim,
            labels=list(labels),
            member_names=member_names,
            positions=np.asarray(positions, dtype=np.int64).reshape(-1, 2),
            theta_slice=slice(0, 0),
            prior=None,
        )

    def _resolve_R(self) -> MatrixPriorBlock:
        if self.kalman is None:
            raise ValueError(
                "Block estimation of R requires a KalmanConfig to specify symbolic R std/correlation metadata."
            )
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

    def _resolve_Q(self) -> MatrixPriorBlock:
        Q_cov = backend.build_Q(self.compiled, self._base_params)
        self._cov_to_corr(Q_cov, "Q")

        shock_std = self.compiled.config.calibration.shock_std
        shock_corr = self.compiled.config.calibration.shock_corr
        labels = list(self.compiled.shock_names)
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

    def _build_matrix_prior_blocks(self) -> dict[str, MatrixPriorBlock]:
        # A reserved key requested for estimation builds a dense CPC correlation
        # block regardless of priors; this is the SPD-by-construction Cholesky
        # reparameterization. An LKJChol prior, when present, is validated and
        # attached as optional density; without one the block carries prior=None
        # (pure reparameterization, e.g. the MLE path).
        blocks: dict[str, MatrixPriorBlock] = {}
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
                lkj_prior = self._is_lkj_prior(key, self.priors[key])
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
    def _corr_from_member_values(block: MatrixPriorBlock, values: NDF) -> NDF:
        corr = np.eye(block.dim, dtype=float64)
        rows = block.positions[:, 0]
        cols = block.positions[:, 1]
        vals = np.asarray(values, dtype=float64)
        corr[rows, cols] = vals
        corr[cols, rows] = vals
        return corr

    @staticmethod
    def _block_cpc_from_corr(block: MatrixPriorBlock, corr: NDF) -> NDF:
        try:
            return backend._unconstrained_from_corr(corr)
        except ValueError as exc:
            raise ValueError(
                f"Correlation values do not form a valid positive-definite "
                f"correlation matrix over {block.labels}: {exc}"
            ) from exc

    @staticmethod
    def _block_corr_from_theta(
        block: MatrixPriorBlock, theta_block: NDF
    ) -> tuple[NDF, NDF]:
        Lcorr = backend._corr_chol_from_unconstrained(theta_block, block.dim)
        corr = np.asarray(Lcorr @ Lcorr.T, dtype=float64)
        return corr, np.asarray(Lcorr, dtype=float64)

    def to_spec(self) -> EstimatorSpec:
        priors = {name: prior.to_spec() for name, prior in (self.priors or {}).items()}

        params = EstimatorParams(
            observables=self.observables,
            filter_mode=self._prepared_filter.mode,
            P0=self.P0.tolist() if self.P0 is not None else None,
            R=self.R.tolist() if self.R is not None else None,
            estimated_params=self.estimated_params,
            priors=priors or None,
            ss_seed=_coerce_ss_seed(self.ss_seed),
            x0=list(self.x0) if self.x0 is not None else None,
            jitter=self._prepared_filter.kf_jitter,
            symmetrize=self._prepared_filter.kf_sym,
            joseph_cov=self._prepared_filter.kf_joseph_cov,
        )

        if isinstance(self.y, pd.DataFrame):
            y = self.y.to_numpy().tolist()
        else:
            y = self.y.tolist()

        return EstimatorSpec(
            y=y,
            params=params,
        )

    @classmethod
    def from_spec(cls, spec: EstimatorSpec, compiled: CompiledModel) -> "Estimator":
        params = spec.params
        y = np.asarray(spec.y, dtype=float64)
        R = np.asarray(params["R"], dtype=float64) if params["R"] is not None else None
        P0 = (
            np.asarray(params["P0"], dtype=float64)
            if params["P0"] is not None
            else None
        )

        x0 = (
            np.asarray(params["x0"], dtype=float64)
            if params["x0"] is not None
            else None
        )
        priors = {
            name: Prior.from_spec(prior_spec)
            for name, prior_spec in (params["priors"] or {}).items()
        }
        return cls(
            compiled=compiled,
            y=y,
            observables=params["observables"],
            filter_mode=params["filter_mode"],
            estimated_params=params["estimated_params"],
            priors=priors or None,
            ss_seed=params["ss_seed"],
            x0=x0,
            jitter=params["jitter"],
            symmetrize=params["symmetrize"],
            joseph_cov=params["joseph_cov"],
            R=R,
            P0=P0,
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

    def _validate_theta0(self, theta: NDF) -> None:
        """Fail fast on an initial guess the objective cannot score.

        A transform's inverse lands inside its own support, so a theta only
        arrives unusable by being non-finite itself or by saturating its
        transform: a std that overflows to infinity, or one that underflows to
        the zero its support excludes. The support is the role's, so this holds
        for an MLE start as much as a MAP one.
        """
        invalid: list[str] = []
        for i, name in enumerate(self.param_names):
            z = float64(theta[i])
            if not np.isfinite(z):
                invalid.append(f"{name}={z}")
                continue
            if name in self._matrix_member_names:
                # A block's run decodes to a valid correlation for any finite z.
                continue
            transform = self._param_transforms[name]
            value = float64(transform.safe_inverse(z))
            if not np.isfinite(value) or not transform.support.contains(value):
                invalid.append(f"{name}={value}")
        if invalid:
            raise ValueError(
                "Initial guess maps to parameter values the objective cannot "
                "score: " + ", ".join(invalid)
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

    def theta_to_params(self, theta: NDF) -> dict[str, float64]:
        """A theta draw as the named parameters, over the base calibration.

        A CPC block's members come off the correlation its run decodes to;
        every other estimated entry comes through its own inverse transform.
        """
        theta = asarray(theta, dtype=float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} does not match estimated parameter count {len(self.param_names)}."
            )
        full = dict(self._base_params)
        handled = np.zeros((len(self.param_names),), dtype=bool)
        for block in self._matrix_blocks.values():
            theta_block = np.asarray(theta[block.theta_slice], dtype=float64)
            corr, _ = self._block_corr_from_theta(block, theta_block)
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
        return full

    def loglik(self, theta: NDF) -> float64:
        ctx, mode = self._build_native_context()
        return loglik(ctx, mode, theta)

    def logprior(self, theta: NDF, include_logjac: bool = False) -> float64:
        ctx, _ = self._build_native_context()
        return logprior(ctx, theta, include_logjac)

    def logpost(self, theta: NDF, include_logjac: bool = False) -> float64:
        ctx, mode = self._build_native_context()
        return logpost(ctx, mode, theta, include_logjac)

    def _report_search_warning_count(self, kind: str, n_err: int) -> None:
        print(
            f"[Estimator:{kind}] BK stability warnings encountered during search: {n_err}"
        )

    @staticmethod
    def _serialize_bounds(
        bounds: Sequence[tuple[float | None, float | None]] | None,
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
        res: dict[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        x = asarray(res["x"], dtype=float64)
        # res["params"] is the calib-order parameter vector at x_best. Only the
        # estimated names are this result's own; every other parameter sits at
        # the calibration it entered with and belongs to the model, not here.
        params_at_x = dict(
            zip((str(name) for name in self.compiled.calib_params), res["params"])
        )
        theta = {name: float64(params_at_x[name]) for name in self.param_names}

        vcov = res.get("vcov")
        se = res.get("se")
        common: dict[str, Any] = dict(
            x=x,
            theta=theta,
            vcov=vcov,
            se=dict(zip(self.param_names, se)) if se is not None else None,
            cov_status=int(res.get("cov_status", 0)),
            success=bool(res["success"]),
            message=str(res["message"]),
            fun=float64(res["fun"]),
            nfev=int(res["nfev"]),
            nit=int(res["nit"]),
            optimizer_config=dict(config or {}),
        )
        if kind == "mle":
            return MLEResult(**common, loglik=-res["fun"])
        if kind == "map":
            return MAPResult(**common, logpost=-res["fun"], logprior=res["logprior"])
        raise ValueError(f"unknown result kind {kind!r}")

    def _build_native_context(
        self,
    ) -> tuple[PyLinearContext | PyExtendedContext | PyUnscentedContext, str]:
        """Build the native objective context DTO for the current filter mode.

        Method-agnostic: it depends only on ``self`` (model, data, priors, Q/R
        specs, transforms), so the same ctx serves the MLE/MAP optimizer driver
        and the MCMC mainloop. The driver decides how to drive it (minimized
        ``-logpost`` vs ``+logpost``); the ctx is identical.
        """
        common = build_obj_common(
            compiled=self.compiled,
            kalman=self.kalman,
            prepared=self._prepared_filter,
            param_names=self.param_names,
            param_index=self._param_index,
            matrix_member_names=self._matrix_member_names,
            matrix_blocks=self._matrix_blocks,
            param_transforms=self._param_transforms,
            priors=self.priors,
            ss_seed=self.ss_seed,
            x0=self.x0,
            R_override=self.R,
        )

        ctx: PyLinearContext | PyExtendedContext | PyUnscentedContext
        if (mode := self._prepared_filter.mode) == "linear":
            ctx = build_linear_context(common)
        elif mode == "extended":
            ctx = build_extended_context(common)
        elif mode == "unscented":
            ctx = build_unscented_context(common, compiled=self.compiled, x0=self.x0)
        else:
            raise ValueError(f"Unknown filter_mode {mode!r}.")
        return ctx, mode

    def _point_estimate(
        self,
        routine: Literal["mle", "map"],
        has_priors: bool,
        jacobian: bool = False,
        theta0: NDF | Mapping[str, float] | None = None,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        m: int = 10,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        fd_step: float = 0.0,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        cov: bool = True,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> OptimizationResult:

        init = self.resolve_theta0(theta0)
        self._validate_theta0(init)

        ctx, mode = self._build_native_context()

        res = run_estimation(
            ctx,
            mode,
            method,
            include_logjac=jacobian,
            theta0=init,
            bounds=bounds,
            has_priors=has_priors,
            m=m,
            maxiter=maxiter,
            maxfun=maxfun,
            maxls=maxls,
            factr=factr,
            pgtol=pgtol,
            fd_step=fd_step,
            xatol=xatol,
            fatol=fatol,
            compute_cov=cov,
            cov_fd_step_scale=cov_fd_step_scale,
            cov_fd_absolute_floor=cov_fd_absolute_floor,
        )

        out = self._pack_opt_result(
            routine,
            res,
            config={
                "theta0": theta0.tolist() if isinstance(theta0, np.ndarray) else theta0,
                "method": method,
                "bounds": self._serialize_bounds(bounds),
                "options": {
                    "m": m,
                    "maxiter": maxiter,
                    "maxfun": maxfun,
                    "maxls": maxls,
                    "factr": factr,
                    "pgtol": pgtol,
                    "fd_step": fd_step,
                    "xatol": xatol,
                    "fatol": fatol,
                    "jacobian": jacobian,
                    "cov": cov,
                    "cov_fd_step_scale": cov_fd_step_scale,
                    "cov_fd_absolute_floor": cov_fd_absolute_floor,
                },
            },
        )
        self._report_search_warning_count(routine, res["bk_violations"])
        return out

    def mle(
        self,
        *,
        theta0: NDF | Mapping[str, float] | None = None,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        m: int = 10,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        fd_step: float = 0.0,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        cov: bool = True,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> MLEResult:
        if self.priors is not None:
            warnings.warn(
                "MLE will ignore any provided priors. Use MAP or MCMC for prior-informed estimation.",
                UserWarning,
            )

        return cast(
            MLEResult,
            self._point_estimate(
                routine="mle",
                has_priors=False,
                jacobian=False,
                theta0=theta0,
                bounds=bounds,
                method=method,
                m=m,
                maxiter=maxiter,
                maxfun=maxfun,
                maxls=maxls,
                factr=factr,
                pgtol=pgtol,
                fd_step=fd_step,
                xatol=xatol,
                fatol=fatol,
                cov=cov,
                cov_fd_step_scale=cov_fd_step_scale,
                cov_fd_absolute_floor=cov_fd_absolute_floor,
            ),
        )

    def map(
        self,
        *,
        theta0: NDF | Mapping[str, float] | None = None,
        bounds: Sequence[tuple[float | None, float | None]] | None = None,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        jacobian: bool = False,
        m: int = 10,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        fd_step: float = 0.0,
        xatol: float = 1e-4,
        fatol: float = 1e-4,
        cov: bool = True,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> MAPResult:
        if self.priors is None:
            raise ValueError("MAP requires priors. No priors were provided.")

        return cast(
            MAPResult,
            self._point_estimate(
                routine="map",
                has_priors=True,
                jacobian=jacobian,
                theta0=theta0,
                bounds=bounds,
                method=method,
                m=m,
                maxiter=maxiter,
                maxfun=maxfun,
                maxls=maxls,
                factr=factr,
                pgtol=pgtol,
                fd_step=fd_step,
                xatol=xatol,
                fatol=fatol,
                cov=cov,
                cov_fd_step_scale=cov_fd_step_scale,
                cov_fd_absolute_floor=cov_fd_absolute_floor,
            ),
        )

    def mcmc(
        self,
        *,
        n_draws: int,
        burn_in: int = 1000,
        thin: int = 1,
        theta0: NDF | Mapping[str, float] | None = None,
        random_state: int | None = None,
        adapt: bool = True,
        adapt_start: int = 100,
        proposal_scale: float = 0.1,
        adapt_epsilon: float = 1e-8,
        compute_map: bool = True,
        map_options: dict[str, Any] | None = None,
        proposal_cov: NDF | None = None,
        cov_fd_step_scale: float = 1.0,
        cov_fd_absolute_floor: float = 0.1,
    ) -> MCMCResult:
        if self.priors is None:
            raise ValueError("MCMC requires priors to define a posterior.")

        rng = np.random.default_rng(random_state)

        current = self.resolve_theta0(theta0)
        self._validate_theta0(current)
        if current.shape[0] == 0:
            raise ValueError("No estimated parameters were provided.")

        ctx, mode = self._build_native_context()

        # The chain runs entirely in native nogil code; ``rng`` (numpy's own
        # PCG64) is borrowed for the run and must outlive it, which the local
        # reference here guarantees. Timing wraps only the native call.
        t0 = perf_counter()
        out = run_mcmc(
            ctx,
            mode,
            current,
            rng,
            n_draws=n_draws,
            burn_in=burn_in,
            thin=thin,
            adapt=adapt,
            adapt_start=adapt_start,
            proposal_scale=proposal_scale,
            proposal_cov=proposal_cov,
            cov_fd_step_scale=cov_fd_step_scale,
            cov_fd_absolute_floor=cov_fd_absolute_floor,
            adapt_epsilon=adapt_epsilon,
            compute_map=compute_map,
            map_options=map_options,
        )
        elapsed = max(perf_counter() - t0, np.finfo(float).eps)

        total_steps = int(out["total_steps"])
        kept = out["samples"]

        print(
            f"MCMC sampling concluded in {elapsed:.2f} seconds with {float(total_steps / elapsed):.2f} iterations per second."
        )

        # Recorded, not re-used: the run already consumed map_options, so this
        # copy exists only to carry a JSON-safe bounds shape into the config.
        recorded_map_options: dict[str, Any] | None = None
        if map_options is not None:
            recorded_map_options = dict(map_options)
            if recorded_map_options.get("bounds") is not None:
                recorded_map_options["bounds"] = self._serialize_bounds(
                    recorded_map_options["bounds"]
                )

        result = MCMCResult(
            param_names=list(self.param_names),
            samples=kept,
            logpost_trace=out["logpost_trace"],
            logjac_trace=out["logjac_trace"],
            accept_rate=float64(out["n_accepted"] / total_steps),
            n_draws=n_draws,
            burn_in=burn_in,
            thin=thin,
            sampler_config={
                "theta0": theta0.tolist() if isinstance(theta0, np.ndarray) else theta0,
                "adapt": bool(adapt),
                "adapt_start": int(adapt_start),
                "proposal_scale": float(proposal_scale),
                "adapt_epsilon": float(adapt_epsilon),
                "compute_map": bool(compute_map),
                "map_options": recorded_map_options,
                "proposal_cov": (
                    proposal_cov.tolist() if proposal_cov is not None else None
                ),
                "cov_fd_step_scale": float(cov_fd_step_scale),
                "cov_fd_absolute_floor": float(cov_fd_absolute_floor),
                "random_state": (None if random_state is None else int(random_state)),
            },
        )
        self._report_search_warning_count("mcmc", out["bk_violations"])
        return result
