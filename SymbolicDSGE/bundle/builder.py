"""Assemble a ``.sdsge`` bundle from model/estimation/Monte-Carlo artifacts.

:class:`BundleBuilder` accumulates members and emits the archive. Text specs
(model YAML, estimation/MC JSON) ride as deflated text; bulk numeric data
(observed ``y``, MCMC posteriors, MC traces) flows through
:func:`SymbolicDSGE.bundle.parquet.columns_to_parquet`, and raw observable files,
whose cells may be strings, through :func:`SymbolicDSGE.bundle.parquet.to_parquet`.
This is the writer half of the container (#142) and the assembly point the
future ``sdsge-compile`` CLI calls.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, Sequence

import numpy as np
from numpy.typing import NDArray

from ..estimation.results import MCMCResult, OptimizationResult

if TYPE_CHECKING:
    from ..estimation.estimator import Estimator
from ..estimation.spec import (
    EstimatorSpec,
    MLEResultSpec,
    MAPResultSpec,
    MCMCResultSpec,
    MCMCResultMeta,
)
from ..core.shock_generators import Shock
from ..estimation.results import OptimizationResult, MLEResult, MAPResult

from ..monte_carlo.core import MCPipeline
from ..monte_carlo.mc_constructs import MCPipelineResult, MCStep
from ..monte_carlo.serialize import (
    json_safe,
    serialize_run_meta,
    serialize_test_results,
    serialize_regression_results,
    serialize_transform_results,
    serialize_postproc_results,
)
from ..monte_carlo.spec import PipelineSpec
from ..monte_carlo.spec_compile import raw_model_data_arrays
from .container import write_bundle
from .manifest import Manifest, Member, MemberKind, SimSpec
from .parquet import (
    arrays_to_parquet,
    columns_to_parquet,
    csv_to_json,
    to_parquet,
    trace_to_csv,
)

NDF = NDArray[np.float64]

# Ref/DGP top level models
_MODEL_PATH = "model/{role}.yaml"

# MLE, MAP, MCMC estimation tab members
_ESTIMATION_SPEC = "estimation/spec.json"
_ESTIMATION_RESULT = "estimation/result.json"
_ESTIMATION_DATA_PARQUET = "estimation/observed.parquet"
_ESTIMATION_DATA_CSV = "estimation/observed.csv"
_ESTIMATION_POSTERIOR_PARQUET = "estimation/posterior.parquet"
_ESTIMATION_POSTERIOR_CSV = "estimation/posterior.csv"

# Monte Carlo pipeline spec and per-rep custom members
_MC_PIPELINE = "montecarlo/pipeline.json"
_MC_CUSTOM_OP = "montecarlo/custom/{ref}.pkl"
_MC_RAW_MODEL_DATA = "montecarlo/data/{ref}.parquet"

# Monte Carlo result tab members
_MC_RESULT_META = "montecarlo/result/meta.json"

_MC_TEST_STEPS = "montecarlo/result/tests/test_steps.json"
_MC_TEST_PARQUET = "montecarlo/result/tests/test_traces.parquet"
_MC_TEST_CSV = "montecarlo/result/tests/test_traces.csv"

_MC_REGRESSION_STEPS = "montecarlo/result/regressions/regression_steps.json"
_MC_REGRESSION_PARQUET = "montecarlo/result/regressions/regression_traces.parquet"
_MC_REGRESSION_CSV = "montecarlo/result/regressions/regression_traces.csv"

_MC_TRANSFORM_STEPS = "montecarlo/result/transforms/transform_steps.json"
_MC_TRANSFORM_PARQUET = "montecarlo/result/transforms/{ref}_{field}.parquet"
_MC_TRANSFORM_CSV = "montecarlo/result/transforms/{ref}_{field}.csv"

_MC_POSTPROC_STEPS = "montecarlo/result/postproc/postproc_steps.json"
_MC_POSTPROC_PARQUET = "montecarlo/result/postproc/{ref}_{field}.parquet"
_MC_POSTPROC_CSV = "montecarlo/result/postproc/{ref}_{field}.csv"

#: Fill for the rows a shorter column contributes to a shared block. Negative so
#: it can never be read as a rep index; no reader looks past ``n_retained``.
_PAD = -1


def _pad_columns(columns: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    """Bring every column up to the tallest one's height, filling with ``_PAD``.

    A column block is rectangular, but ``n_retain`` is per step, so two steps in
    one kind can retain different numbers of replications.
    """
    if not columns:
        return {}
    height = max(int(arr.shape[0]) for arr in columns.values())
    if height == 0:
        return {}
    out: dict[str, NDArray[Any]] = {}
    for name, arr in columns.items():
        rows = int(arr.shape[0])
        if rows == height:
            out[name] = arr
            continue
        pad = np.full((height - rows, *arr.shape[1:]), _PAD, dtype=arr.dtype)
        out[name] = np.concatenate([arr, pad])
    return out


def _library_version() -> str:
    try:
        return f"SymbolicDSGE {version('symbolicdsge')}"
    except PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "SymbolicDSGE (unknown)"


class BundleBuilder:
    """Collect bundle members, then :meth:`write` (or :meth:`build`) the archive."""

    def __init__(self, *, created_by: str | None = None) -> None:
        self._created_by = created_by or _library_version()
        self._members: list[Member] = []
        self._files: dict[str, bytes] = {}
        self._simulation: dict[str, SimSpec] = {}

    # Models

    def add_model(
        self,
        role: str,
        yaml_text: str,
        *,
        compile_kwargs: Mapping[str, Any] | None = None,
        solve_kwargs: Mapping[str, Any] | None = None,
    ) -> BundleBuilder:
        """Add a model config (its source YAML) under ``role`` (reference/dgp).

        ``compile_kwargs``/``solve_kwargs`` are recorded so the loader rebuilds an
        identical :class:`SolvedModel`.
        """
        path = _MODEL_PATH.format(role=role)
        options: dict[str, Any] = {}
        if compile_kwargs:
            options["compile_kwargs"] = dict(compile_kwargs)
        if solve_kwargs:
            options["solve_kwargs"] = dict(solve_kwargs)
        self._add(
            Member(path=path, kind="model_config", role=role, options=options),
            yaml_text.encode("utf-8"),
        )
        return self

    # Raw data

    def add_raw_data(
        self,
        name: str,
        data: bytes | str,
        *,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add a raw observable file. CSV input is converted to Parquet by
        default (``as_parquet``); pass ``as_parquet=False`` to store the CSV
        verbatim (still a valid, format-agnostic member)."""
        if as_parquet:
            self._add(
                Member(path=f"data/{name}.parquet", kind="raw_data"),
                to_parquet(csv_to_json(data)),
            )
        else:
            text = data.encode("utf-8") if isinstance(data, str) else data
            self._add(Member(path=f"data/{name}.csv", kind="raw_data"), text)
        return self

    # Estimation

    def add_estimation(
        self,
        source: Estimator,
        *,
        result: MLEResult | MAPResult | MCMCResult | None = None,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the estimation tab from a live :class:`Estimator`.

        ``result`` accepts live any result object an :class:`Estimator` can produce.
        ``as_parquet`` controls whether relevant bulk data gets compressed as Parquet
        or stays user-readable as CSV.
        """
        spec = source.to_spec()

        # The observed matrix and the posterior traces are different shapes with
        # different column names, so each gets its own encoder rather than one
        # serializer that would have to hardcode a column name for both.
        observable_names = (
            None
            if spec.params["observables"] is None
            else list(spec.params["observables"])
        )
        if as_parquet:
            dpath = _ESTIMATION_DATA_PARQUET
            ppath = _ESTIMATION_POSTERIOR_PARQUET

            def observed_bytes(y: Any) -> bytes:
                return columns_to_parquet({"y": y})

            def posterior_bytes(columns: Mapping[str, Any]) -> bytes:
                return columns_to_parquet(columns)

        else:
            dpath = _ESTIMATION_DATA_CSV
            ppath = _ESTIMATION_POSTERIOR_CSV

            def observed_bytes(y: Any) -> bytes:
                return _observed_to_csv(y, observable_names)

            def posterior_bytes(columns: Mapping[str, Any]) -> bytes:
                return trace_to_csv(dict(columns))

        self._add(
            Member(path=_ESTIMATION_SPEC, kind="estimation_spec"),
            json.dumps(spec.params, indent=2).encode("utf-8"),
        )
        self._add(
            Member(path=dpath, kind="estimation_data", columns=observable_names),
            observed_bytes(spec.y),
        )
        if result is not None:
            # One name per arm: the three specs are unrelated types, and only the
            # MCMC one is a container whose bulk fields split off to their own member.
            kind: str
            payload: MCMCResultMeta | MLEResultSpec | MAPResultSpec
            match result:
                case MCMCResult():
                    mcmc_spec = result.to_spec()
                    kind = "mcmc"
                    payload = mcmc_spec.meta

                    self._add(
                        Member(path=ppath, kind="estimation_trace"),
                        posterior_bytes(
                            {
                                "samples": mcmc_spec.samples,
                                "logpost": mcmc_spec.logpost_trace,
                                "logjac": mcmc_spec.logjac_trace,
                            }
                        ),
                    )
                case MLEResult():
                    kind = "mle"
                    payload = result.to_spec()
                case MAPResult():
                    kind = "map"
                    payload = result.to_spec()
                case _:
                    raise ValueError(f"Unknown result type {type(result).__name__}.")

            result_data = json.dumps({"type": kind, "data": payload}, indent=2)
            self._add(
                Member(path=_ESTIMATION_RESULT, kind="estimation_result"),
                result_data.encode("utf-8"),
            )

        return self

    # Monte Carlo

    def add_mc(
        self,
        pipeline: MCPipeline,
        *,
        result: MCPipelineResult | None = None,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the MC tab from a live :class:`MCPipeline`.

        ``add_mc(pipeline)``: A live pipeline is compiled to
        its graph spec via :meth:`MCPipeline.to_spec`, and its bulk side-channels
        are shipped as their own members: ``raw_model_data`` datagen arrays as
        Parquet, and ``custom`` ops as cloudpickle blobs (each callable is
        enforced/wrapped as a :class:`NumpyCustomFunc` so its source travels for
        receiver audit).

        Optionally records a run ``result``. Its run-level config and each step
        kind's metas ride their own JSON members; the bulk traces ride separate
        members, as Parquet or, with ``as_parquet=False``, as CSV. The loader
        reads either.
        """
        self._add(
            Member(path=_MC_PIPELINE, kind="mc_pipeline"),
            json.dumps(pipeline.to_spec(), indent=2).encode("utf-8"),
        )
        self._add_mc_resources(pipeline)
        if result is not None:
            self._add(
                Member(path=_MC_RESULT_META, kind="mc_result_meta"),
                json.dumps(json_safe(serialize_run_meta(result)), indent=2).encode(
                    "utf-8"
                ),
            )
            tests = serialize_test_results(result.test_summaries)
            regressions = serialize_regression_results(result.regression_summaries)
            transforms = serialize_transform_results(
                result.transform_outputs, result.n_rep
            )
            postprocs = serialize_postproc_results(result.postproc)

            self._add_step_metas(_MC_TEST_STEPS, "mc_test_steps", tests)
            self._add_step_metas(
                _MC_REGRESSION_STEPS, "mc_regression_steps", regressions
            )
            self._add_step_metas(_MC_TRANSFORM_STEPS, "mc_transform_steps", transforms)
            self._add_step_metas(_MC_POSTPROC_STEPS, "mc_postproc_steps", postprocs)

            self._add_trace_block(
                _MC_TEST_PARQUET, _MC_TEST_CSV, "mc_test_traces", tests, as_parquet
            )
            self._add_trace_block(
                _MC_REGRESSION_PARQUET,
                _MC_REGRESSION_CSV,
                "mc_regression_traces",
                regressions,
                as_parquet,
            )
            self._add_trace_arrays(
                _MC_TRANSFORM_PARQUET,
                _MC_TRANSFORM_CSV,
                "mc_transform_trace",
                transforms,
                as_parquet,
            )
            self._add_trace_arrays(
                _MC_POSTPROC_PARQUET,
                _MC_POSTPROC_CSV,
                "mc_postproc_raw",
                postprocs,
                as_parquet,
            )
        return self

    def _add_step_metas(
        self,
        path: str,
        kind: MemberKind,
        steps: Mapping[str, tuple[Any, Mapping[str, NDArray[Any]]]],
    ) -> None:
        """Collect one step kind's metas into a single member, keyed by step name.

        The traces half is dropped here; it rides its own members. A kind with no
        steps writes nothing, so the loader reads an absent member as empty.
        """
        if not steps:
            return
        payload = {name: json_safe(meta) for name, (meta, _) in steps.items()}
        self._add(
            Member(path=path, kind=kind),
            json.dumps(payload, indent=2).encode("utf-8"),
        )

    def _add_trace_block(
        self,
        parquet_path: str,
        csv_path: str,
        kind: MemberKind,
        steps: Mapping[str, tuple[Any, Mapping[str, NDArray[Any]]]],
        as_parquet: bool,
    ) -> None:
        """Pack one step kind's traces into a single column block.

        Columns are qualified ``{step}.{field}``, which the 2-D expansion extends
        to ``{step}.{field}.{j}``. Steps that retained different numbers of
        replications are padded to the tallest; the rows past a step's own
        ``n_retained`` are never read, since its meta carries the count.
        """
        columns = _pad_columns(
            {
                f"{name}.{field}": arr
                for name, (_, traces) in steps.items()
                for field, arr in traces.items()
            }
        )
        if not columns:
            return
        if as_parquet:
            self._add(Member(path=parquet_path, kind=kind), columns_to_parquet(columns))
        else:
            self._add(Member(path=csv_path, kind=kind), trace_to_csv(columns))

    def _add_trace_arrays(
        self,
        parquet_path: str,
        csv_path: str,
        kind: MemberKind,
        steps: Mapping[str, tuple[Any, Mapping[str, NDArray[Any]]]],
        as_parquet: bool,
    ) -> None:
        """Ship one step kind's traces as a member per array.

        These are arbitrary-shape payloads that share no height with each other,
        so none of them pack. An array above 2-D is flattened to ``(-1, last)``
        and restored from the ``shape`` its step's meta records.
        """
        for name, (_, traces) in steps.items():
            for field, arr in traces.items():
                if arr.size == 0:
                    continue
                flat = arr if arr.ndim <= 1 else arr.reshape(-1, arr.shape[-1])
                columns = {f"{name}.{field}": flat}
                path = (parquet_path if as_parquet else csv_path).format(
                    ref=name, field=field
                )
                data = (
                    columns_to_parquet(columns) if as_parquet else trace_to_csv(columns)
                )
                self._add(
                    Member(
                        path=path, kind=kind, options={"name": name, "field": field}
                    ),
                    data,
                )

    def _add_mc_resources(self, pipeline: MCPipeline) -> None:
        """Ship the bulk side-channels a live pipeline references by key.

        ``raw_model_data`` datagens become array members; ``custom`` ops
        become cloudpickle members (wrapped as :class:`NumpyCustomFunc` first,
        which enforces the author-side contract and carries the source for audit).
        """
        for step in (*pipeline.per_rep_steps, *pipeline.postproc_steps):
            if step.step_type == "raw_model_data":
                arrays = raw_model_data_arrays(step.kwargs)
                if not arrays:
                    continue
                data, _ = arrays_to_parquet(arrays)
                self._add(
                    Member(
                        path=_MC_RAW_MODEL_DATA.format(ref=step.name),
                        kind="mc_raw_model_data",
                        options={"ref": step.name},
                    ),
                    data,
                )
            elif step.step_type in ("transform:custom", "postproc:custom"):
                self._add(
                    Member(
                        path=_MC_CUSTOM_OP.format(ref=step.name),
                        kind="mc_custom_op",
                        options={"ref": step.name},
                    ),
                    _custom_op_blob(step),
                )

    # Simulation prefill

    def set_simulation(
        self,
        role: str,
        *,
        T: int,
        shocks: Mapping[str, Shock | NDF] | None = None,
        shock_scale: float = 1.0,
        x0: Mapping[str, float] | Sequence[float] | NDF | None = None,
        observables: bool = False,
    ) -> BundleBuilder:
        """Attach a simulation prefill under ``role``, taking ``SolvedModel.sim``'s
        keywords and lowering them to the stored form.

        Each shock is a live :class:`Shock` or a raw path array, the two shapes
        ``sim`` draws from; the parameters are read off the object rather than
        hand-written. A callable cannot be stored, since only its result would
        travel and the receiver could not redraw it.
        """
        self._simulation[role] = SimSpec(
            T=int(T),
            x0=_prefill_x0(x0),
            observables=bool(observables),
            shock_scale=float(shock_scale),
            shocks=_prefill_shocks(shocks),
        )
        return self

    # Low-level passthrough

    def add_member(self, member: Member, data: bytes) -> BundleBuilder:
        """Append a pre-encoded member at its declared path.

        Public seam for callers that already hold the final member bytes, for
        example the ``sdsge-compile`` CLI copying a Parquet ``data/`` file through
        or staging a pre-split MC result + traces pair. The higher-level ``add_*`` methods would otherwise re-encode.
        """
        self._add(member, data)
        return self

    # Emit

    def manifest(self) -> Manifest:
        return Manifest(
            created_by=self._created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            members=list(self._members),
            simulation=self._simulation or None,
            checksums={
                path: hashlib.sha256(data).hexdigest()
                for path, data in self._files.items()
            },
        )

    def build(self) -> tuple[Manifest, dict[str, bytes]]:
        return self.manifest(), dict(self._files)

    def write(self, path: str | Path) -> Path:
        write_bundle(path, self.manifest(), self._files)
        return Path(path)

    def _add(self, member: Member, data: bytes) -> None:
        if member.path in self._files:
            raise ValueError(f"Duplicate bundle member path {member.path!r}.")
        self._members.append(member)
        self._files[member.path] = data


def _prefill_x0(
    x0: Mapping[str, float] | Sequence[float] | NDF | None,
) -> Mapping[str, float] | list[float] | None:
    """Lower ``sim``'s ``x0`` to its stored shape, name-keyed or positional."""
    if x0 is None:
        return None
    if isinstance(x0, Mapping):
        return {str(name): float(value) for name, value in x0.items()}
    if isinstance(x0, (Sequence, np.ndarray)) and not isinstance(x0, (str, bytes)):
        return [float(value) for value in x0]
    raise TypeError(
        f"x0 must be a mapping of variable names to values, a sequence in "
        f"declaration order, or an ndarray; got {type(x0).__name__}."
    )


def _prefill_shocks(
    shocks: Mapping[str, Shock | NDF] | None,
) -> dict[str, Any] | None:
    """Lower each shock to its stored shape: parameters, or a raw path.

    A :class:`Shock` travels as its parameters so the receiver redraws it under
    the author's seed. An array travels as itself. A callable is rejected: only
    the path it returned would survive, which is not the same run.
    """
    if shocks is None:
        return None
    lowered: dict[str, Any] = {}
    for key, shock in shocks.items():
        if isinstance(shock, Shock):
            lowered[key] = shock.to_dict()
        elif isinstance(shock, np.ndarray):
            lowered[key] = shock.tolist()
        else:
            raise TypeError(
                f"Shock {key!r} must be a Shock or a raw path array; got "
                f"{type(shock).__name__}. A callable is rejected here too: only "
                f"the path it returned would travel, and the receiver could not "
                f"redraw it."
            )
    return lowered


def _custom_op_blob(step: MCStep) -> bytes:
    """Wrap a custom step's callable in the phase wrapper and cloudpickle it.

    Wrapping enforces the author-side contract (top-level def, safe namespace)
    and snapshots the source + captured globals, so the receiver can audit the
    op at load. Post-loop (POSTPROC) ops get the looser pandas namespace; every
    other phase gets numpy. An already-wrapped callable passes through; a pandas
    wrapper outside the post-loop phase is rejected.
    """
    import cloudpickle

    from ..monte_carlo.custom_op import (
        CustomFunc,
        CustomOpValidationError,
        NumpyCustomFunc,
        PandasCustomFunc,
    )
    from ..monte_carlo.mc_constructs import OpType

    if step.func is None:
        raise ValueError(f"Custom step {step.name!r} has no callable.")
    wrapper = PandasCustomFunc if step.op_type is OpType.POSTPROC else NumpyCustomFunc
    if isinstance(step.func, PandasCustomFunc) and wrapper is NumpyCustomFunc:
        raise CustomOpValidationError(
            f"{step.name!r}: a PandasCustomFunc is only allowed in a post-loop "
            f"(POSTPROC) step, not a {step.op_type.value!r} step."
        )
    wrapped = step.func if isinstance(step.func, CustomFunc) else wrapper(step.func)
    return cast(bytes, cloudpickle.dumps(wrapped))


def _estimator_observed(
    estimator: "Estimator",
) -> tuple[NDArray[Any], list[str] | None]:
    """Extract the observed matrix + observable names from an estimator's ``y``."""
    y = estimator.y
    if hasattr(y, "columns"):  # pandas DataFrame
        frame = cast(Any, y)
        return (
            np.asarray(frame.to_numpy(), dtype=np.float64),
            [str(column) for column in frame.columns],
        )
    matrix = np.asarray(y, dtype=np.float64)
    names = list(estimator.observables) if estimator.observables else None
    return matrix, names


def _observed_to_csv(
    y: Sequence[Sequence[float]], names: Sequence[str] | None
) -> bytes:
    """Render a 2-D observed matrix as CSV with user-friendly headers.

    Uses ``names`` as the header row when provided (paired with
    ``Member.columns`` so the loader can stack semantic-header CSVs back into
    the matrix). Falls back to mechanical ``y.{j}`` headers. Round-trips
    through :func:`SymbolicDSGE.bundle.parquet.collapse_columns` the same way
    Parquet observed data does.
    """
    n = len(y)
    p = len(y[0]) if n > 0 else 0
    if names is not None and len(names) != p:
        raise ValueError(
            f"observable_names length {len(names)} does not match observed "
            f"column count {p}."
        )
    headers = list(names) if names is not None else [f"y.{j}" for j in range(p)]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(headers)
    for row in y:
        writer.writerow([_float_cell(v) for v in row])
    return out.getvalue().encode("utf-8")


def _float_cell(value: float) -> str:
    number = float(value)
    return "" if not math.isfinite(number) else repr(number)
