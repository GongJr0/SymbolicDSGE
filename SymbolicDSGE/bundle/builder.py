"""Assemble a ``.sdsge`` bundle from model/estimation/Monte-Carlo artifacts.

:class:`BundleBuilder` accumulates members and emits the archive. Text specs
(model YAML, estimation/MC JSON) ride as deflated text; bulk numeric data (raw
observable files, observed ``y``, MCMC posteriors, MC traces) flows through
:func:`SymbolicDSGE.bundle.parquet.to_parquet`. This is the writer half of the
container (#142) and the assembly point the future ``sdsge-compile`` CLI calls.
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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ..estimation.results import MCMCResult, OptimizationResult

if TYPE_CHECKING:
    from ..estimation.estimator import Estimator
from ..estimation.spec import (
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
)
from ..monte_carlo.core import MCPipeline
from ..monte_carlo.mc_constructs import MCPipelineResult, MCStep
from ..monte_carlo.serialize import (
    result_document,
    result_postproc_arrays,
    result_postproc_tables,
    result_traces,
)
from ..monte_carlo.spec import PipelineSpec
from ..monte_carlo.spec_compile import raw_data_arrays
from .container import write_bundle
from .manifest import Manifest, Member, SimSpec
from .parquet import (
    arrays_to_parquet,
    csv_to_json,
    frame_to_json,
    to_parquet,
    trace_to_csv,
    trace_to_json,
)

_MODEL_PATH = "model/{role}.yaml"
_ESTIMATION_SPEC = "estimation/spec.json"
_ESTIMATION_RESULT = "estimation/result.json"
_ESTIMATION_DATA_PARQUET = "estimation/observed.parquet"
_ESTIMATION_DATA_CSV = "estimation/observed.csv"
_ESTIMATION_POSTERIOR_PARQUET = "estimation/posterior.parquet"
_ESTIMATION_POSTERIOR_CSV = "estimation/posterior.csv"
_MC_PIPELINE = "montecarlo/pipeline.json"
_MC_RESULT = "montecarlo/result.json"
_MC_TRACE_PARQUET = "montecarlo/traces.parquet"
_MC_TRACE_CSV = "montecarlo/traces.csv"
_MC_RAW_DATA = "montecarlo/data/{ref}.parquet"
_MC_CUSTOM_OP = "montecarlo/custom/{ref}.pkl"
_MC_POSTPROC = "montecarlo/postproc/{ref}.parquet"
_MC_POSTPROC_TABLE = "montecarlo/postproc/table/{ref}.parquet"

#: Single-array column name inside a postproc parquet member (avoids dotted
#: artifact names colliding with the ``"{name}.{j}"`` 2-D column expansion).
_POSTPROC_COL = "a"


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
        self._simulation: SimSpec | None = None

    # -- models ---------------------------------------------------------------

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

    # -- raw data -------------------------------------------------------------

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

    # -- estimation -----------------------------------------------------------

    def add_estimation(
        self,
        source: EstimationSpec | "Estimator",  # pyright: ignore
        *,
        result: (
            OptimizationResult
            | MCMCResult
            | OptimizationResultMeta
            | MCMCResultMeta
            | None
        ) = None,
        observed: NDArray[Any] | None = None,
        observable_names: list[str] | None = None,
        posterior: Mapping[str, NDArray[Any]] | None = None,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the estimation tab from a live :class:`Estimator` or a spec.

        Two entry shapes:

        - ``add_estimation(estimator, result=run)`` — the high-level path. The
          spec is built from the estimator and the live ``result`` (method,
          ``method_kwargs``, bounds, priors, and observed ``y`` all flattened
          for you); MCMC posteriors are pulled from the result automatically.
        - ``add_estimation(spec, result=..., observed=..., ...)`` — the explicit
          path for a hand-authored :class:`EstimationSpec`.

        ``result`` accepts a live ``OptimizationResult``/``MCMCResult`` (projected
        via ``to_meta()``, and a live ``MCMCResult`` auto-supplies ``posterior``)
        or its projected ``*Meta``. ``as_parquet=False`` writes observed/posterior
        as CSV; the format-agnostic loader reads either.
        """
        if isinstance(source, EstimationSpec):
            spec = source
        else:
            from ..estimation.estimator import Estimator

            if not isinstance(source, Estimator):
                raise TypeError(
                    "add_estimation expects an EstimationSpec or Estimator as the "
                    f"first argument, got {type(source).__name__}."
                )
            if not isinstance(result, (OptimizationResult, MCMCResult)):
                raise ValueError(
                    "add_estimation(estimator, ...) requires a live "
                    "OptimizationResult or MCMCResult."
                )
            spec = source.to_spec(result=result)
            if observed is None:
                observed, derived_names = _estimator_observed(source)
                if observable_names is None:
                    observable_names = derived_names

        self._add(
            Member(path=_ESTIMATION_SPEC, kind="estimation_spec"),
            spec.to_json(indent=2).encode("utf-8"),
        )
        if isinstance(result, (OptimizationResult, MCMCResult)):
            if isinstance(result, MCMCResult) and posterior is None:
                posterior = result.posterior_arrays()
            result = result.to_meta()
        if result is not None:
            kind = "mcmc" if isinstance(result, MCMCResultMeta) else "optimization"
            payload = json.dumps({"type": kind, "data": result.to_dict()}, indent=2)
            self._add(
                Member(path=_ESTIMATION_RESULT, kind="estimation_result"),
                payload.encode("utf-8"),
            )
        if observed is not None:
            matrix = np.asarray(observed, dtype=np.float64)
            if matrix.ndim != 2:
                raise ValueError("observed must be a 2-D (n, k) array.")
            if as_parquet:
                path = _ESTIMATION_DATA_PARQUET
                data = to_parquet(trace_to_json({"y": matrix}))
            else:
                path = _ESTIMATION_DATA_CSV
                data = _observed_to_csv(matrix, observable_names)
            self._add(
                Member(path=path, kind="estimation_data", columns=observable_names),
                data,
            )
        if posterior is not None:
            if as_parquet:
                path = _ESTIMATION_POSTERIOR_PARQUET
                data = to_parquet(trace_to_json(dict(posterior)))
            else:
                path = _ESTIMATION_POSTERIOR_CSV
                data = trace_to_csv(dict(posterior))
            self._add(Member(path=path, kind="estimation_trace"), data)
        return self

    # -- monte carlo ----------------------------------------------------------

    def add_mc(
        self,
        pipeline: MCPipeline | PipelineSpec,
        *,
        result: MCPipelineResult | None = None,
        run_id: str = "",
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the MC tab from a live :class:`MCPipeline` or a :class:`PipelineSpec`.

        ``add_mc(pipeline)`` — the high-level path. A live pipeline is compiled to
        its graph spec via :meth:`MCPipeline.to_spec`, and its bulk side-channels
        are shipped as their own members: ``raw_data`` datagen arrays as Parquet,
        and ``custom`` ops as cloudpickle blobs (each callable is enforced/wrapped
        as a :class:`NumpyCustomFunc` so its source travels for receiver audit).

        ``add_mc(spec)`` — the explicit path for a hand-authored spec (any
        ``raw_data``/``custom`` members must be staged separately).

        Optionally records a run ``result`` (split into a trace-free document + a
        trace member); ``as_parquet=False`` writes traces as CSV — the
        format-agnostic loader reads either.
        """
        if isinstance(pipeline, MCPipeline):
            spec = pipeline.to_spec()
            self._add_mc_resources(pipeline)
        else:
            spec = pipeline
        self._add(
            Member(path=_MC_PIPELINE, kind="mc_pipeline"),
            spec.to_json(indent=2).encode("utf-8"),
        )
        if result is not None:
            document = result_document(result, run_id=run_id)
            self._add(
                Member(path=_MC_RESULT, kind="mc_result"),
                json.dumps(document, indent=2).encode("utf-8"),
            )
            traces = result_traces(result)
            if traces:
                if as_parquet:
                    path = _MC_TRACE_PARQUET
                    data = to_parquet(trace_to_json(traces))
                else:
                    path = _MC_TRACE_CSV
                    data = trace_to_csv(traces)
                self._add(Member(path=path, kind="mc_trace"), data)
            self._add_postproc_arrays(result)
            self._add_postproc_tables(result)
        return self

    def _add_postproc_tables(self, result: MCPipelineResult) -> None:
        """Ship tabular POSTPROC artifacts as columnar parquet members.

        Tables are mixed-dtype, so they ride the columnar NDJSON seam
        (:func:`frame_to_json` + :func:`to_parquet`) rather than the float
        shape-manifest one; column/dtype/index metadata lives in the result
        document, the artifact name in the member options. An empty (0-row) table
        carries no parquet rows and is skipped — the loader rebuilds its columns.
        """
        for index, (name, columns) in enumerate(result_postproc_tables(result).items()):
            payload = frame_to_json(columns)
            if not payload:  # 0-row table -> nothing to encode
                continue
            self._add(
                Member(
                    path=_MC_POSTPROC_TABLE.format(ref=index),
                    kind="mc_postproc_table",
                    options={"name": name},
                ),
                to_parquet(payload),
            )

    def _add_postproc_arrays(self, result: MCPipelineResult) -> None:
        """Ship bulk POSTPROC ndarray artifacts as shape-manifest parquet members.

        Each artifact is an arbitrary-shape payload (e.g. a KDE ``N x 2`` curve),
        so — unlike the uniform-length ``mc_trace`` columns — they cannot share
        one column block; each rides its own member with its shape recorded in the
        member options for restore. Scalar artifacts stay inline in the result
        document.
        """
        for index, (name, arr) in enumerate(result_postproc_arrays(result).items()):
            data, _ = arrays_to_parquet({_POSTPROC_COL: arr})
            self._add(
                Member(
                    path=_MC_POSTPROC.format(ref=index),
                    kind="mc_postproc",
                    options={"name": name, "shape": list(arr.shape)},
                ),
                data,
            )

    def _add_mc_resources(self, pipeline: MCPipeline) -> None:
        """Ship the bulk side-channels a live pipeline references by key.

        ``raw_data`` datagens become Parquet array members; ``custom`` ops become
        cloudpickle members (wrapped as :class:`NumpyCustomFunc` first, which
        enforces the author-side contract and carries the source for audit).
        """
        for step in pipeline.steps:
            if step.step_type == "raw_data":
                arrays = raw_data_arrays(step.kwargs)
                if not arrays:
                    continue
                data, _ = arrays_to_parquet(arrays)
                self._add(
                    Member(
                        path=_MC_RAW_DATA.format(ref=step.name),
                        kind="mc_raw_data",
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

    # -- simulation prefill ---------------------------------------------------

    def set_simulation(self, simulation: SimSpec) -> BundleBuilder:
        self._simulation = simulation
        return self

    # -- low-level passthrough ------------------------------------------------

    def add_member(self, member: Member, data: bytes) -> BundleBuilder:
        """Append a pre-encoded member at its declared path.

        Public seam for callers that already hold the final member bytes —
        e.g. the ``sdsge-compile`` CLI copying a Parquet ``data/`` file through
        verbatim, or staging a pre-split MC result + traces pair. The
        higher-level ``add_*`` methods would otherwise re-encode.
        """
        self._add(member, data)
        return self

    # -- emit -----------------------------------------------------------------

    def manifest(self) -> Manifest:
        return Manifest(
            created_by=self._created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            members=list(self._members),
            simulation=self._simulation,
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

    # -- internals ------------------------------------------------------------

    def _add(self, member: Member, data: bytes) -> None:
        if member.path in self._files:
            raise ValueError(f"Duplicate bundle member path {member.path!r}.")
        self._members.append(member)
        self._files[member.path] = data


def _custom_op_blob(step: MCStep) -> bytes:
    """Wrap a custom step's callable as a NumpyCustomFunc and cloudpickle it.

    Wrapping enforces the author-side contract (top-level def, safe namespace)
    and snapshots the source + captured globals, so the receiver can audit the
    op at load. Already-wrapped callables pass through idempotently.
    """
    import cloudpickle

    from ..monte_carlo.custom_op import NumpyCustomFunc

    wrapped = NumpyCustomFunc(step.func)
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


def _observed_to_csv(matrix: NDArray[Any], names: list[str] | None) -> bytes:
    """Render a 2-D observed matrix as CSV with user-friendly headers.

    Uses ``names`` as the header row when provided (paired with
    ``Member.columns`` so the loader can stack semantic-header CSVs back into
    the matrix). Falls back to mechanical ``y.{j}`` headers — round-trips
    through :func:`SymbolicDSGE.bundle.parquet.collapse_columns` the same way
    Parquet observed data does.
    """
    if names is not None and len(names) != matrix.shape[1]:
        raise ValueError(
            f"observable_names length {len(names)} does not match observed "
            f"column count {matrix.shape[1]}."
        )
    headers = (
        list(names) if names is not None else [f"y.{j}" for j in range(matrix.shape[1])]
    )
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(headers)
    for i in range(matrix.shape[0]):
        writer.writerow([_float_cell(matrix[i, j]) for j in range(matrix.shape[1])])
    return out.getvalue().encode("utf-8")


def _float_cell(value: float) -> str:
    number = float(value)
    return "" if not math.isfinite(number) else repr(number)
