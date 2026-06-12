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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..estimation.results import MCMCResult, OptimizationResult
from ..estimation.spec import (
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
)
from ..monte_carlo.mc_constructs import MCPipelineResult
from ..monte_carlo.serialize import result_document, result_traces
from ..monte_carlo.spec import PipelineSpec
from .container import write_bundle
from .manifest import Manifest, Member, SimSpec
from .parquet import csv_to_json, to_parquet, trace_to_csv, trace_to_json

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
        spec: EstimationSpec,
        *,
        result: OptimizationResultMeta | MCMCResultMeta | None = None,
        observed: NDArray[Any] | None = None,
        observable_names: list[str] | None = None,
        posterior: Mapping[str, NDArray[Any]] | None = None,
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the estimation tab: spec (always), and optionally its result
        metadata, observed ``y`` matrix, and MCMC posterior traces.

        ``as_parquet=False`` writes observed/posterior as CSV instead — the
        format-agnostic loader reads either. ``observed.csv`` uses the
        ``observable_names`` as headers when supplied (user-friendly for
        hand-authoring); posterior CSV uses the mechanical ``{name}.{j}``
        expansion that mirrors :func:`trace_to_csv`.
        """
        self._add(
            Member(path=_ESTIMATION_SPEC, kind="estimation_spec"),
            spec.to_json(indent=2).encode("utf-8"),
        )
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
        pipeline: PipelineSpec,
        *,
        result: MCPipelineResult | None = None,
        run_id: str = "",
        as_parquet: bool = True,
    ) -> BundleBuilder:
        """Add the MC tab: pipeline spec (always), and optionally a run result
        (split into a trace-free document + a trace member).

        ``as_parquet=False`` writes traces as CSV instead — the format-agnostic
        loader reads either.
        """
        self._add(
            Member(path=_MC_PIPELINE, kind="mc_pipeline"),
            pipeline.to_json(indent=2).encode("utf-8"),
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
        return self

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


def _meta_from_optimization(result: OptimizationResult) -> OptimizationResultMeta:
    """Helper: project a live :class:`OptimizationResult` to its text metadata."""
    return OptimizationResultMeta(
        kind=result.kind,
        theta={k: float(v) for k, v in result.theta.items()},
        success=bool(result.success),
        message=str(result.message),
        fun=float(result.fun),
        loglik=float(result.loglik),
        logprior=float(result.logprior),
        logpost=float(result.logpost),
        nfev=int(result.nfev),
        nit=None if result.nit is None else int(result.nit),
    )


def _meta_from_mcmc(result: MCMCResult) -> MCMCResultMeta:
    """Helper: project a live :class:`MCMCResult` to its text metadata."""
    return MCMCResultMeta(
        param_names=list(result.param_names),
        accept_rate=float(result.accept_rate),
        n_draws=int(result.n_draws),
        burn_in=int(result.burn_in),
        thin=int(result.thin),
    )
