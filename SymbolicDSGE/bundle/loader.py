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

from ..core.compiled_model import CompiledModel
from ..core.model_parser import ModelParser
from ..core.solved_model import SolvedModel
from ..core.solver import DSGESolver
from ..estimation.estimator import Estimator
from ..estimation.results import MCMCResult, MLEResult, MAPResult
from ..estimation.spec import (
    EstimatorParams,
    EstimatorSpec,
    MLEResultSpec,
    MAPResultSpec,
    MCMCResultMeta,
    MCMCResultSpec,
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
    document shape. ``estimator`` is the live object the ``spec`` describes,
    bound to the reference model the bundle was loaded with.
    """

    spec: EstimatorSpec
    _compiled: CompiledModel
    result: MLEResult | MAPResult | MCMCResult | None = None
    _estimator: Estimator | None = field(default=None, init=False, repr=False)

    @property
    def estimator(self) -> Estimator:
        """The live estimator this spec describes, built on first access.

        Deferred rather than built at load: ``Estimator`` construction compiles
        the measurement and observable-jacobian cfuncs, which a caller that only
        reads ``result`` never needs.
        """
        if self._estimator is None:
            self._estimator = Estimator.from_spec(self.spec, self._compiled)
        return self._estimator


@dataclass
class LoadedMC:
    """Monte-Carlo pipeline + (optional) run result recovered from a bundle.

    ``pipeline`` is the live, runnable :class:`MCPipeline`, rebuilt eagerly at
    load from ``spec`` + ``resources``. No model is needed to build it (simulation
    shocks come from the explicit registry, not a model), so it is ready to run
    against models supplied at ``pipeline.run(reference=..., dgp=...)`` time. The
    raw ``spec`` stays available for the UI to consume.

    ``resources`` reattaches the bulk side-channels the spec references by key:
    each ``raw_model_data`` ``data_ref`` maps to its restored ``{name: ndarray}``
    arrays and each ``custom`` ``func_ref`` (transform *or* post-loop) to its callable.

    Recovered run artifacts of a POSTPROC phase: ``postproc_arrays`` holds each
    step's bulk ``Raw`` array; its ``summary`` slot rides inline in ``document``.
    :meth:`wire` re-merges the two back into the canonical UI wire shape.
    """

    spec: PipelineSpec
    pipeline: MCPipeline
    document: dict[str, Any] | None = None
    traces: dict[str, NDArray[Any]] | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    postproc_arrays: dict[str, NDArray[Any]] = field(default_factory=dict)

    def wire(self) -> dict[str, Any] | None:
        """Re-merge document + traces into the UI wire shape, when both exist."""
        if self.document is None or self.traces is None:
            return None
        return pipeline_result_wire(self.document, self.traces, self.postproc_arrays)


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
    reference = _load_model(archive, manifest, "reference")
    return LoadedBundle(
        manifest=manifest,
        reference=reference,
        dgp=_load_model(archive, manifest, "dgp"),
        estimation=_load_estimation(archive, manifest, reference),
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


def _stack_observed(cols: dict[str, list[Any]], member: Member) -> list[list[float]]:
    """Reconstruct the observed ``(n, k)`` matrix from CSV or Parquet columns.

    Handles both the mechanical ``y.{j}`` layout (Parquet path and CSV without
    ``observable_names``) and the semantic-header CSV layout (columns named by
    ``Member.columns``).
    """
    collapsed = collapse_columns(cols)
    y = collapsed.get("y")
    if isinstance(y, np.ndarray) and y.ndim == 2:
        return cast(list[list[float]], y.tolist())
    if member.columns:
        return cast(
            list[list[float]],
            np.column_stack(
                [_float_column(cols[name]) for name in member.columns]
            ).tolist(),
        )
    raise ValueError(
        f"Cannot reconstruct observed matrix from {member.path!r}: no 'y.*' "
        f"columns and no Member.columns metadata to stack semantic headers."
    )


def _float_column(values: list[Any]) -> NDArray[np.float64]:
    """Coerce a column of numbers/Nones to ``float64`` (None -> NaN)."""
    return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)


def _load_estimation(
    archive: BundleArchive, manifest: Manifest, reference: SolvedModel | None
) -> LoadedEstimation | None:
    param_members = manifest.members_by_kind("estimation_spec")
    if not param_members:
        return None
    if reference is None:
        raise ValueError(
            "Bundle carries an estimation section but no reference model, so the "
            "estimator it describes cannot be bound to one."
        )
    data_members = manifest.members_by_kind("estimation_data")
    if not data_members:
        raise ValueError(
            "Bundle carries an estimation section but no 'estimation_data' member; "
            "an estimator is not defined without the data it conditions on."
        )

    params = cast(EstimatorParams, json.loads(archive.read_text(param_members[0].path)))
    y = _stack_observed(_load_columns(archive, data_members[0]), data_members[0])
    spec = EstimatorSpec(y=y, params=params)

    # Load the posterior first: the MCMC result is rebuilt from metadata + these
    # traces (the optimization result needs no traces).
    posterior: dict[str, NDArray[Any]] | None = None
    trace_members = manifest.members_by_kind("estimation_trace")
    if trace_members:
        posterior = collapse_columns(_load_columns(archive, trace_members[0]))

    result: MLEResult | MAPResult | MCMCResult | None = None
    result_members = manifest.members_by_kind("estimation_result")
    if result_members:
        payload = json.loads(archive.read_text(result_members[0].path))
        data = payload["data"]
        if (typ := payload.get("type")) == "mcmc":
            result = _rebuild_mcmc_result(data, posterior)
        elif typ == "mle":
            result = MLEResult.from_spec(cast(MLEResultSpec, data))
        elif typ == "map":
            result = MAPResult.from_spec(cast(MAPResultSpec, data))

    return LoadedEstimation(spec=spec, _compiled=reference.compiled, result=result)


def _rebuild_mcmc_result(
    data: dict[str, Any], posterior: dict[str, NDArray[Any]] | None
) -> MCMCResult:
    """Recombine MCMC metadata with its posterior traces into a live result.

    The Meta/trace split is the designed inverse (see ``MCMCResultMeta``): the
    scalar metadata rides the JSON member, the ``samples``/``logpost`` columns
    ride the parquet trace member.
    """
    meta = cast(MCMCResultMeta, data)
    if (
        posterior is None
        or "samples" not in posterior
        or "logpost" not in posterior
        or "logjac" not in posterior
    ):
        raise ValueError(
            "MCMC bundle result requires an 'estimation_trace' member carrying "
            "'samples', 'logpost', and 'logjac' columns."
        )
    spec = MCMCResultSpec(
        samples=posterior["samples"].tolist(),
        logpost_trace=posterior["logpost"].tolist(),
        logjac_trace=posterior["logjac"].tolist(),
        meta=meta,
    )
    return MCMCResult.from_spec(spec)


def _load_mc(archive: BundleArchive, manifest: Manifest) -> LoadedMC | None:
    from ..monte_carlo.builder import build_pipeline

    pipeline_members = manifest.members_by_kind("mc_pipeline")
    if not pipeline_members:
        return None
    spec = cast(PipelineSpec, json.loads(archive.read_text(pipeline_members[0].path)))

    document: dict[str, Any] | None = None
    result_members = manifest.members_by_kind("mc_result")
    if result_members:
        document = json.loads(archive.read_text(result_members[0].path))

    traces: dict[str, NDArray[Any]] | None = None
    trace_members = manifest.members_by_kind("mc_trace")
    if trace_members:
        traces = collapse_columns(_load_columns(archive, trace_members[0]))

    postproc_arrays = _load_mc_postproc(archive, manifest)
    resources = _load_mc_resources(archive, manifest, spec)

    # Build the runnable pipeline eagerly. This needs no model: every step
    # compiles from its parameters alone, and the models are supplied later at
    # ``pipeline.run(...)``. A malformed stored spec raises here, so
    # ``load_bundle`` fails fast on structure.
    pipeline = build_pipeline(spec, resources=resources)

    return LoadedMC(
        spec=spec,
        pipeline=pipeline,
        document=document,
        traces=traces,
        resources=resources,
        postproc_arrays=postproc_arrays,
    )


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

    ``raw_model_data`` parquet members are reshaped using the ``data_shapes`` recorded
    on their spec node; ``custom`` op members are unpickled. Keyed by the node's
    ``data_ref`` / ``func_ref`` so :func:`build_pipeline` can reattach them.
    """
    resources: dict[str, Any] = {}

    shapes_by_ref = {
        (params := node["params"])["data_ref"]: params.get("data_shapes", {})
        for node in spec["nodes"]
        if node["step_type"] == "raw_model_data" and "data_ref" in node["params"]
    }
    for member in manifest.members_by_kind("mc_raw_model_data"):
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
