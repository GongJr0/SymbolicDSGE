"""Reconstruct in-code objects from a ``.sdsge`` bundle.

:func:`build_from` opens a bundle and rebuilds what it carries: the
:class:`SolvedModel`(s) (re-parsed and re-solved from the stored YAML using the
recorded compile/solve options), the estimation artifacts, the Monte-Carlo
pipeline/result, and the simulation prefill. The read counterpart to
:class:`SymbolicDSGE.bundle.builder.BundleBuilder`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ..core.model_parser import ModelParser
from ..core.solved_model import SolvedModel
from ..core.solver import DSGESolver
from ..estimation.results import MCMCResult, OptimizationResult
from ..estimation.spec import (
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
)
from ..monte_carlo.serialize import pipeline_result_wire
from ..monte_carlo.spec import PipelineSpec
from .container import BundleArchive
from .manifest import Manifest, Member, SimSpec
from .parquet import (
    arrays_from_parquet,
    collapse_columns,
    csv_to_columns,
    from_parquet_columns,
)

if TYPE_CHECKING:
    from ..monte_carlo.core import MCPipeline


@dataclass
class LoadedEstimation:
    """Estimation artifacts recovered from a bundle.

    ``result`` is a first-class :class:`OptimizationResult` / :class:`MCMCResult`
    (rebuilt from the stored metadata + posterior traces), not the on-disk
    ``*Meta`` shape. ``posterior`` still carries the raw ``samples``/``logpost``
    columns for callers that want them directly.
    """

    spec: EstimationSpec
    result: OptimizationResult | MCMCResult | None = None
    observed: NDArray[Any] | None = None
    posterior: dict[str, NDArray[Any]] | None = None


@dataclass
class LoadedMC:
    """Monte-Carlo pipeline + (optional) run result recovered from a bundle.

    ``pipeline`` is the live, runnable :class:`MCPipeline`, rebuilt eagerly at
    load from ``spec`` + ``resources``. No model is needed to build it (simulation
    shocks come from the explicit registry, not a model), so it is ready to run
    against models supplied at ``pipeline.run(reference=..., dgp=...)`` time. The
    raw ``spec`` stays available for the UI to consume.

    ``resources`` reattaches the bulk side-channels the spec references by key:
    each ``raw_data`` ``data_ref`` maps to its restored ``{name: ndarray}`` arrays
    and each ``custom`` ``func_ref`` (transform *or* post-loop) to its callable.

    Recovered run artifacts of a POSTPROC phase: ``postproc_arrays`` (bulk ndarray
    artifacts) and ``postproc_tables`` (tabular/DataFrame artifacts as columnar
    dicts); scalar artifacts ride inline in ``document``. :meth:`wire` re-merges
    all three back into the canonical UI wire shape.
    """

    spec: PipelineSpec
    pipeline: MCPipeline
    document: dict[str, Any] | None = None
    traces: dict[str, NDArray[Any]] | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    postproc_arrays: dict[str, NDArray[Any]] = field(default_factory=dict)
    postproc_tables: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def wire(self) -> dict[str, Any] | None:
        """Re-merge document + traces into the UI wire shape, when both exist."""
        if self.document is None or self.traces is None:
            return None
        return pipeline_result_wire(
            self.document, self.traces, self.postproc_arrays, self.postproc_tables
        )


@dataclass
class LoadedBundle:
    """Everything reconstructed from a ``.sdsge`` bundle."""

    manifest: Manifest
    reference: SolvedModel | None = None
    dgp: SolvedModel | None = None
    estimation: LoadedEstimation | None = None
    mc: LoadedMC | None = None
    simulation: dict[str, SimSpec] | None = None


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


def _load_columns(archive: BundleArchive, member: Member) -> dict[str, list[Any]]:
    """Format-agnostic column read: dispatch on ``member.format`` (#142)."""
    raw = archive.read(member.path)
    if member.format == "parquet":
        return from_parquet_columns(raw)
    if member.format == "csv":
        return csv_to_columns(raw)
    raise ValueError(
        f"Cannot load member {member.path!r} as columns: format "
        f"{member.format!r} is neither 'parquet' nor 'csv'."
    )


def _stack_observed(cols: dict[str, list[Any]], member: Member) -> NDArray[np.float64]:
    """Reconstruct the observed ``(n, k)`` matrix from CSV or Parquet columns.

    Handles both the mechanical ``y.{j}`` layout (Parquet path and CSV without
    ``observable_names``) and the semantic-header CSV layout (columns named by
    ``Member.columns``).
    """
    collapsed = collapse_columns(cols)
    y = collapsed.get("y")
    if isinstance(y, np.ndarray) and y.ndim == 2:
        return y.astype(np.float64, copy=False)
    if member.columns:
        return np.column_stack([_float_column(cols[name]) for name in member.columns])
    raise ValueError(
        f"Cannot reconstruct observed matrix from {member.path!r}: no 'y.*' "
        f"columns and no Member.columns metadata to stack semantic headers."
    )


def _float_column(values: list[Any]) -> NDArray[np.float64]:
    """Coerce a column of numbers/Nones to ``float64`` (None -> NaN)."""
    return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)


def _load_estimation(
    archive: BundleArchive, manifest: Manifest
) -> LoadedEstimation | None:
    spec_members = manifest.members_by_kind("estimation_spec")
    if not spec_members:
        return None
    spec = EstimationSpec.from_json(archive.read_text(spec_members[0].path))

    observed: NDArray[Any] | None = None
    data_members = manifest.members_by_kind("estimation_data")
    if data_members:
        observed = _stack_observed(
            _load_columns(archive, data_members[0]), data_members[0]
        )

    # Load the posterior first: the MCMC result is rebuilt from metadata + these
    # traces (the optimization result needs no traces).
    posterior: dict[str, NDArray[Any]] | None = None
    trace_members = manifest.members_by_kind("estimation_trace")
    if trace_members:
        posterior = collapse_columns(_load_columns(archive, trace_members[0]))

    result: OptimizationResult | MCMCResult | None = None
    result_members = manifest.members_by_kind("estimation_result")
    if result_members:
        payload = json.loads(archive.read_text(result_members[0].path))
        data = payload["data"]
        if payload.get("type") == "mcmc":
            result = _rebuild_mcmc_result(data, posterior)
        else:
            result = _rebuild_optimization_result(data)

    return LoadedEstimation(
        spec=spec, result=result, observed=observed, posterior=posterior
    )


def _rebuild_mcmc_result(
    data: dict[str, Any], posterior: dict[str, NDArray[Any]] | None
) -> MCMCResult:
    """Recombine MCMC metadata with its posterior traces into a live result.

    The Meta/trace split is the designed inverse (see ``MCMCResultMeta``): the
    scalar metadata rides the JSON member, the ``samples``/``logpost`` columns
    ride the parquet trace member.
    """
    meta = MCMCResultMeta.from_dict(data)
    if posterior is None or "samples" not in posterior or "logpost" not in posterior:
        raise ValueError(
            "MCMC bundle result requires an 'estimation_trace' member carrying "
            "'samples' and 'logpost' columns."
        )
    return MCMCResult(
        param_names=meta.param_names,
        samples=np.asarray(posterior["samples"], dtype=np.float64),
        logpost_trace=np.asarray(posterior["logpost"], dtype=np.float64),
        accept_rate=np.float64(meta.accept_rate),
        n_draws=meta.n_draws,
        burn_in=meta.burn_in,
        thin=meta.thin,
        sampler_config=dict(meta.sampler_config),
    )


def _rebuild_optimization_result(data: dict[str, Any]) -> OptimizationResult:
    """Rebuild an MLE/MAP result from its metadata (point estimate, no traces).

    ``x`` is recovered from ``theta`` (same ordering the estimator built it from).
    """
    meta = OptimizationResultMeta.from_dict(data)
    theta = {str(k): np.float64(v) for k, v in meta.theta.items()}
    return OptimizationResult(
        kind=meta.kind,
        x=np.asarray(list(theta.values()), dtype=np.float64),
        theta=theta,
        success=meta.success,
        message=meta.message,
        fun=np.float64(meta.fun),
        loglik=np.float64(meta.loglik),
        logprior=np.float64(meta.logprior),
        logpost=np.float64(meta.logpost),
        nfev=meta.nfev,
        nit=meta.nit,
        optimizer_config=dict(meta.optimizer_config),
    )


def _load_mc(archive: BundleArchive, manifest: Manifest) -> LoadedMC | None:
    from ..monte_carlo.builder import build_pipeline, validate_pipeline_spec

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
        traces = collapse_columns(_load_columns(archive, trace_members[0]))

    postproc_arrays = _load_mc_postproc(archive, manifest)
    postproc_tables = _load_mc_postproc_tables(archive, manifest)
    resources = _load_mc_resources(archive, manifest, spec)

    # Build the runnable pipeline eagerly. This needs no model: every step
    # compiles from its parameters alone. A malformed stored spec raises here, so
    # ``load_bundle`` fails fast on structure. The model-availability gate is a
    # run concern: both ``reference`` and ``dgp`` are supplied later at
    # ``pipeline.run(...)``, so both are declared available here (the pipeline is
    # runnable once the caller passes the models it needs).
    ordered, postprocs = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    pipeline = build_pipeline(ordered, postprocs, resources=resources)

    return LoadedMC(
        spec=spec,
        pipeline=pipeline,
        document=document,
        traces=traces,
        resources=resources,
        postproc_arrays=postproc_arrays,
        postproc_tables=postproc_tables,
    )


def _load_mc_postproc_tables(
    archive: BundleArchive, manifest: Manifest
) -> dict[str, dict[str, list[Any]]]:
    """Restore tabular POSTPROC artifacts as columnar dicts, keyed by artifact name.

    Each member is a columnar parquet table (mixed dtype); the artifact name rides
    the member options. Dropped all-null columns are rebuilt by
    :func:`pipeline_result_wire` from the document's column metadata."""
    out: dict[str, dict[str, list[Any]]] = {}
    for member in manifest.members_by_kind("mc_postproc_table"):
        name = str(member.options.get("name", ""))
        out[name] = from_parquet_columns(archive.read(member.path))
    return out


def _load_mc_postproc(
    archive: BundleArchive, manifest: Manifest
) -> dict[str, NDArray[Any]]:
    """Restore bulk POSTPROC ndarray artifacts, keyed by artifact name.

    Each member holds one shape-manifest array under the fixed ``"a"`` column;
    its name and shape ride the member options. An all-NaN array is dropped to
    nothing by the Parquet encoder, so a missing column is rebuilt as a NaN array
    of the recorded shape (matching the wire's null-trace convention)."""
    out: dict[str, NDArray[Any]] = {}
    for member in manifest.members_by_kind("mc_postproc"):
        name = str(member.options.get("name", ""))
        shape = tuple(int(d) for d in member.options.get("shape", []))
        raw = archive.read(member.path)
        try:
            out[name] = arrays_from_parquet(raw, {"a": shape})["a"]
        except KeyError:
            out[name] = np.full(shape, np.nan)
    return out


def _load_mc_resources(
    archive: BundleArchive, manifest: Manifest, spec: PipelineSpec
) -> dict[str, Any]:
    """Restore the bulk side-channels referenced by the MC spec.

    ``raw_data`` parquet members are reshaped using the ``data_shapes`` recorded
    on their spec node; ``custom`` op members are unpickled. Keyed by the node's
    ``data_ref`` / ``func_ref`` so :func:`build_pipeline` can reattach them.
    """
    resources: dict[str, Any] = {}

    shapes_by_ref = {
        node.params["data_ref"]: node.params.get("data_shapes", {})
        for node in spec.nodes
        if node.step_type == "raw_data" and "data_ref" in node.params
    }
    for member in manifest.members_by_kind("mc_raw_data"):
        ref = str(member.options.get("ref", ""))
        shapes = shapes_by_ref.get(ref, {})
        resources[ref] = arrays_from_parquet(archive.read(member.path), shapes)

    custom_members = manifest.members_by_kind("mc_custom_op")
    if custom_members:
        import cloudpickle

        for member in custom_members:
            ref = str(member.options.get("ref", ""))
            resources[ref] = cloudpickle.loads(archive.read(member.path))

    return resources
