"""Reconstruct in-code objects from a ``.sdsge`` bundle.

:func:`build_from` opens a bundle and rebuilds what it carries: the
:class:`SolvedModel`(s) (re-parsed and re-solved from the stored YAML using the
recorded compile/solve options), the estimation artifacts, the Monte-Carlo
pipeline/result, and the simulation prefill. The read counterpart to
:class:`SymbolicDSGE.bundle.builder.BundleBuilder`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from numpy.typing import NDArray

from ..core.model_parser import ModelParser
from ..core.solved_model import SolvedModel
from ..core.solver import DSGESolver
from ..estimation.spec import (
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
)
from ..monte_carlo.serialize import pipeline_result_wire
from ..monte_carlo.spec import PipelineSpec
from .container import BundleArchive
from .manifest import Manifest, SimSpec
from .parquet import collapse_columns, from_parquet_columns


@dataclass
class LoadedEstimation:
    """Estimation artifacts recovered from a bundle."""

    spec: EstimationSpec
    result: OptimizationResultMeta | MCMCResultMeta | None = None
    observed: NDArray[Any] | None = None
    posterior: dict[str, NDArray[Any]] | None = None


@dataclass
class LoadedMC:
    """Monte-Carlo pipeline + (optional) run result recovered from a bundle."""

    spec: PipelineSpec
    document: dict[str, Any] | None = None
    traces: dict[str, NDArray[Any]] | None = None

    def wire(self) -> dict[str, Any] | None:
        """Re-merge document + traces into the UI wire shape, when both exist."""
        if self.document is None or self.traces is None:
            return None
        return pipeline_result_wire(self.document, self.traces)


@dataclass
class LoadedBundle:
    """Everything reconstructed from a ``.sdsge`` bundle."""

    manifest: Manifest
    reference: SolvedModel | None = None
    dgp: SolvedModel | None = None
    estimation: LoadedEstimation | None = None
    mc: LoadedMC | None = None
    simulation: SimSpec | None = None


def build_from(path: str | Path) -> LoadedBundle:
    """Open a ``.sdsge`` bundle and rebuild its in-code objects."""
    archive = BundleArchive.open(path)
    manifest = archive.manifest
    return LoadedBundle(
        manifest=manifest,
        reference=_load_model(archive, manifest, "reference"),
        dgp=_load_model(archive, manifest, "dgp"),
        estimation=_load_estimation(archive, manifest),
        mc=_load_mc(archive, manifest),
        simulation=manifest.simulation,
    )


def _load_model(
    archive: BundleArchive, manifest: Manifest, role: str
) -> SolvedModel | None:
    member = manifest.model_member(role)
    if member is None:
        return None
    parser = ModelParser.from_string(archive.read_text(member.path))
    model, kalman = parser.get_all()
    solver = DSGESolver(model, cast(Any, kalman))
    compile_kwargs = dict(member.options.get("compile_kwargs", {}))
    solve_kwargs = dict(member.options.get("solve_kwargs", {}))
    compiled = solver.compile(**compile_kwargs)
    return solver.solve(compiled, **solve_kwargs)


def _load_estimation(
    archive: BundleArchive, manifest: Manifest
) -> LoadedEstimation | None:
    spec_members = manifest.members_by_kind("estimation_spec")
    if not spec_members:
        return None
    spec = EstimationSpec.from_json(archive.read_text(spec_members[0].path))

    result: OptimizationResultMeta | MCMCResultMeta | None = None
    result_members = manifest.members_by_kind("estimation_result")
    if result_members:
        payload = json.loads(archive.read_text(result_members[0].path))
        data = payload["data"]
        if payload.get("type") == "mcmc":
            result = MCMCResultMeta.from_dict(data)
        else:
            result = OptimizationResultMeta.from_dict(data)

    observed: NDArray[Any] | None = None
    data_members = manifest.members_by_kind("estimation_data")
    if data_members:
        columns = collapse_columns(
            from_parquet_columns(archive.read(data_members[0].path))
        )
        observed = columns.get("y")

    posterior: dict[str, NDArray[Any]] | None = None
    trace_members = manifest.members_by_kind("estimation_trace")
    if trace_members:
        posterior = collapse_columns(
            from_parquet_columns(archive.read(trace_members[0].path))
        )

    return LoadedEstimation(
        spec=spec, result=result, observed=observed, posterior=posterior
    )


def _load_mc(archive: BundleArchive, manifest: Manifest) -> LoadedMC | None:
    pipeline_members = manifest.members_by_kind("mc_pipeline")
    if not pipeline_members:
        return None
    spec = PipelineSpec.from_json(archive.read_text(pipeline_members[0].path))

    document: dict[str, Any] | None = None
    result_members = manifest.members_by_kind("mc_result")
    if result_members:
        document = json.loads(archive.read_text(result_members[0].path))

    traces: dict[str, NDArray[Any]] | None = None
    trace_members = manifest.members_by_kind("mc_trace")
    if trace_members:
        traces = collapse_columns(
            from_parquet_columns(archive.read(trace_members[0].path))
        )

    return LoadedMC(spec=spec, document=document, traces=traces)
