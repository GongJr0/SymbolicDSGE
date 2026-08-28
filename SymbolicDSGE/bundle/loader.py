"""Reconstruct in-code objects from a ``.sdsge`` bundle.

:func:`build_from` opens a bundle and rebuilds what it carries: the
:class:`SolvedModel`(s) (re-parsed and re-solved from the stored YAML using the
recorded compile/solve options), the estimation artifacts, the Monte-Carlo
pipeline/result, and the simulation prefill. The read counterpart to
:class:`SymbolicDSGE.bundle.builder.BundleBuilder`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

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
from .._diag_tests.result import MCTestResult
from ..regression.ols.ols_result import MCRegressionResult
from ..monte_carlo.builder import build_pipeline
from ..monte_carlo.postproc import Artifact, Raw, Summary
from ..monte_carlo.spec import (
    MCRegressionResultMeta,
    MCRegressionResultSpec,
    MCTestResultMeta,
    MCTestResultSpec,
    PipelineSpec,
)
from ..monte_carlo.mc_constructs import (
    MCFailure,
    MCMeta,
    MCPipelineResult,
    failed_postproc_names,
    failed_step_counts,
)
from .container import BundleArchive
from .manifest import Manifest, Member
from .parquet import collapse_columns, csv_to_columns, from_parquet_columns

if TYPE_CHECKING:
    from ..monte_carlo.core import MCPipeline

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass
class LoadedEstimation:
    """Estimation artifacts recovered from a bundle.
    ``estimator`` is the bundled :class:`Estimator` instance.
    ``result``, if present, is the run result bundled with the estimator.
    """

    estimator: Estimator
    result: MLEResult | MAPResult | MCMCResult | None = None


@dataclass
class LoadedMC:
    """Monte-Carlo pipeline + (optional) run result recovered from a bundle.

    ``pipeline`` is the live, runnable :class:`MCPipeline`, rebuilt eagerly at
    load with its bulk side-channels reattached: each ``raw_model_data``
    ``data_ref`` to its restored arrays and each ``custom`` ``func_ref``
    (transform *or* post-loop) to its callable. No model is needed to build it
    (simulation shocks come from the explicit registry, not a model), so it is
    ready to run against models supplied at ``pipeline.run(reference=...,
    dgp=...)`` time.

    ``result`` is the run the bundle recorded, rebuilt as the same
    :class:`MCPipelineResult` the run returned, or ``None`` when the bundle
    carries a pipeline alone.
    """

    pipeline: MCPipeline
    result: MCPipelineResult | None = None


@dataclass
class LoadedBundle:
    """Everything reconstructed from a ``.sdsge`` bundle."""

    manifest: Manifest
    reference: SolvedModel | None = None
    dgp: SolvedModel | None = None
    estimation: LoadedEstimation | None = None
    mc: LoadedMC | None = None
    #: ``SolvedModel.sim`` keywords per role, ready to unpack into a run.
    simulation: dict[str, dict[str, Any]] | None = None


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
        simulation=_load_simulation(manifest),
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


def _load_simulation(manifest: Manifest) -> dict[str, dict[str, Any]] | None:
    """Each role's prefill as the ``sim`` keywords it replays through.

    The stored :class:`SimSpec` is the manifest's own carrier; a caller receives
    the materialized keywords, so ``model.sim(**loaded.simulation[role])`` is the
    whole of replaying one.
    """
    if not manifest.simulation:
        return None
    return {role: spec.to_sim_kwargs() for role, spec in manifest.simulation.items()}


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
    estimator = Estimator.from_spec(spec, compiled=reference.compiled)
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

    return LoadedEstimation(estimator=estimator, result=result)


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
    pipeline_members = manifest.members_by_kind("mc_pipeline")
    if not pipeline_members:
        return None
    spec = cast(PipelineSpec, json.loads(archive.read_text(pipeline_members[0].path)))

    resources = _load_mc_resources(archive, manifest, spec)

    pipeline = build_pipeline(spec, resources=resources)
    result = _load_mc_result(archive, manifest)
    return LoadedMC(pipeline=pipeline, result=result)


def _mc_json(archive: BundleArchive, manifest: Manifest, kind: str) -> dict[str, Any]:
    """One JSON member's object, or ``{}`` when the kind is absent.

    A step kind with no steps writes no member, so absence is emptiness.
    """
    members = manifest.members_by_kind(kind)
    if not members:
        return {}
    return cast(dict[str, Any], json.loads(archive.read_text(members[0].path)))


def _mc_block(
    archive: BundleArchive, manifest: Manifest, kind: str
) -> dict[str, list[Any]]:
    """One shared column block's raw columns, keyed ``{step}.{field}[.{idx}]``."""
    members = manifest.members_by_kind(kind)
    if not members:
        return {}
    return _load_columns(archive, members[0])


def _mc_array_columns(
    archive: BundleArchive, manifest: Manifest, kind: str
) -> dict[tuple[str, str], dict[str, list[Any]]]:
    """Each one-array member's raw columns, keyed by its ``(name, field)``."""
    return {
        (str(member.options["name"]), str(member.options["field"])): _load_columns(
            archive, member
        )
        for member in manifest.members_by_kind(kind)
    }


def _float_trace(cols: Mapping[str, list[Any]], key: str, rows: int) -> NDF:
    """One float column, or a NaN column of ``rows`` when the encoder dropped it.

    A float column that is null in every row carries no values, so Parquet drops
    it. The author saw NaNs there and so does the reader.
    """
    values = cols.get(key)
    if values is None:
        return np.full(rows, np.nan)
    return _float_column(values)[:rows]


def _int_trace(cols: Mapping[str, list[Any]], key: str, rows: int) -> NDI:
    """One integer column. Integers are never null, so absence is corruption."""
    return np.asarray(cols[key], dtype=np.int64)[:rows]


def _float_matrix(
    cols: Mapping[str, list[Any]], key: str, rows: int, width: int
) -> NDF:
    """A 2-D float trace from its ``{key}.{j}`` columns, NaN-filling dropped ones."""
    if width == 0:
        return np.empty((rows, 0), dtype=np.float64)
    return np.column_stack(
        [_float_trace(cols, f"{key}.{j}", rows) for j in range(width)]
    )


def _mc_array(cols: Mapping[str, list[Any]], key: str, shape: tuple[int, ...]) -> NDF:
    """Restore one member's array from its columns, using the meta's shape.

    The writer flattened anything above 2-D to ``(-1, last)``, so the row count
    is the product of the leading axes and the width is the last one.
    """
    if len(shape) < 2:
        return _float_trace(cols, key, int(shape[0]) if shape else 1).reshape(shape)
    rows = int(np.prod(shape[:-1]))
    return _float_matrix(cols, key, rows, int(shape[-1])).reshape(shape)


def _load_mc_tests(
    archive: BundleArchive, manifest: Manifest
) -> dict[str, MCTestResult]:
    metas = _mc_json(archive, manifest, "mc_test_steps")
    if not metas:
        return {}
    cols = _mc_block(archive, manifest, "mc_test_traces")
    return {
        name: MCTestResult.from_spec(
            MCTestResultSpec(
                meta=cast(MCTestResultMeta, meta),
                statistic_trace=_float_trace(
                    cols, f"{name}.statistic_trace", int(meta["n_retained"])
                ),
                _raw_status=_int_trace(
                    cols, f"{name}.status_trace", int(meta["n_retained"])
                ),
                retained_reps=_int_trace(
                    cols, f"{name}.retained_reps", int(meta["n_retained"])
                ),
            )
        )
        for name, meta in metas.items()
    }


def _load_mc_regressions(
    archive: BundleArchive, manifest: Manifest
) -> dict[str, MCRegressionResult]:
    metas = _mc_json(archive, manifest, "mc_regression_steps")
    if not metas:
        return {}
    cols = _mc_block(archive, manifest, "mc_regression_traces")
    out: dict[str, MCRegressionResult] = {}
    for name, meta in metas.items():
        rows = int(meta["n_retained"])
        width = int(meta["k"])
        out[name] = MCRegressionResult.from_spec(
            MCRegressionResultSpec(
                meta=cast(MCRegressionResultMeta, meta),
                coef_trace=_float_matrix(cols, f"{name}.coef_trace", rows, width),
                ssr_trace=_float_trace(cols, f"{name}.ssr_trace", rows),
                sst_trace=_float_trace(cols, f"{name}.sst_trace", rows),
                retained_reps=_int_trace(cols, f"{name}.retained_reps", rows),
                _raw_status=_int_trace(cols, f"{name}.status_trace", rows),
                _se_trace=(
                    _float_matrix(cols, f"{name}.se_trace", rows, width)
                    if meta["kind"] == "ols"
                    else None
                ),
            )
        )
    return out


def _load_mc_transforms(archive: BundleArchive, manifest: Manifest) -> dict[str, NDF]:
    metas = _mc_json(archive, manifest, "mc_transform_steps")
    if not metas:
        return {}
    columns = _mc_array_columns(archive, manifest, "mc_transform_trace")
    return {
        name: _mc_array(
            columns.get((name, "value"), {}),
            f"{name}.value",
            tuple(int(d) for d in meta["shape"]),
        )
        for name, meta in metas.items()
    }


def _load_mc_postprocs(
    archive: BundleArchive, manifest: Manifest
) -> dict[str, Artifact]:
    metas = _mc_json(archive, manifest, "mc_postproc_steps")
    if not metas:
        return {}
    columns = _mc_array_columns(archive, manifest, "mc_postproc_raw")
    out: dict[str, Artifact] = {}
    for name, meta in metas.items():
        shape, summary = meta["shape"], meta["summary"]
        out[name] = Artifact(
            raw=(
                None
                if shape is None
                else Raw(
                    value=_mc_array(
                        columns.get((name, "value"), {}),
                        f"{name}.value",
                        tuple(int(d) for d in shape),
                    )
                )
            ),
            summary=None if summary is None else Summary(value=_mc_summary(summary)),
        )
    return out


def _mc_summary(value: Any) -> Any:
    """Undo the table schema a DataFrame summary was written with."""
    if (
        isinstance(value, dict)
        and isinstance(value.get("schema"), dict)
        and "fields" in value["schema"]
        and "data" in value
    ):
        import pandas as pd

        return pd.read_json(StringIO(json.dumps(value)), orient="table")
    return value


def _load_mc_result(
    archive: BundleArchive, manifest: Manifest
) -> MCPipelineResult | None:
    """Rebuild the run result from its meta member and each kind's traces."""
    run = _mc_json(archive, manifest, "mc_result_meta")
    if not run:
        return None
    failures = [MCFailure(**failure) for failure in run["failures"]]
    meta = MCMeta(
        n_rep=int(run["n_rep"]),
        n_retained_by_step={
            name: int(value) for name, value in run["n_retained_by_step"].items()
        },
        elapsed_s=float(run["elapsed_s"]),
        step_elapsed_s={
            name: float(value) for name, value in run["step_elapsed_s"].items()
        },
        step_counts={name: int(value) for name, value in run["step_counts"].items()},
        step_failures={
            name: int(value) for name, value in run["step_failures"].items()
        },
        postproc_elapsed_s={
            name: float(value) for name, value in run["postproc_elapsed_s"].items()
        },
        # Derived from the failures on the write side too, so never stored.
        failed_steps=failed_step_counts(failures),
        failed_postprocs=failed_postproc_names(failures),
    )
    return MCPipelineResult(
        meta=meta,
        n_rep=int(run["n_rep"]),
        n_successful=int(run["n_successful"]),
        test_summaries=_load_mc_tests(archive, manifest),
        transform_outputs=_load_mc_transforms(archive, manifest),
        regression_summaries=_load_mc_regressions(archive, manifest),
        failures=tuple(failures),
        postproc=_load_mc_postprocs(archive, manifest),
        run_config=dict(run["run_config"]),
    )


def _load_mc_resources(
    archive: BundleArchive, manifest: Manifest, spec: PipelineSpec
) -> dict[str, Any]:
    """Restore the bulk side-channels referenced by the MC spec.

    ``raw_model_data`` members are read format-agnostically and reshaped using the
    ``data_shapes`` recorded on their spec node; ``custom`` op members are
    unpickled. Keyed by the node's ``data_ref`` / ``func_ref`` so
    :func:`build_pipeline` can reattach them.
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
        columns = collapse_columns(_load_columns(archive, member))
        resources[ref] = {
            name: np.asarray(columns[name], dtype=np.float64).reshape(
                tuple(int(d) for d in shape)
            )
            for name, shape in shapes.items()
        }

    custom_members = manifest.members_by_kind("mc_custom_op")
    if custom_members:
        import cloudpickle

        for member in custom_members:
            ref = str(member.options.get("ref", ""))
            resources[ref] = cloudpickle.loads(archive.read(member.path))

    return resources
