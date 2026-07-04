from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from SymbolicDSGE.bundle import BundleBuilder, build_from
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline, build_pipeline, validate_pipeline_spec
from SymbolicDSGE.monte_carlo.custom_op import (
    CustomOpValidationError,
    NumpyCustomFunc,
    PandasCustomFunc,
)
from SymbolicDSGE.monte_carlo.operations.core import raw_data_step
from SymbolicDSGE.monte_carlo.operations.postproc import postproc_step
from SymbolicDSGE.monte_carlo.operations.tests import jarque_bera_test_step
from SymbolicDSGE.monte_carlo.operations.transforms import transform_step
from SymbolicDSGE.monte_carlo.serialize import serialize_pipeline_result

# Bundling a result that retained per-rep payloads/test-results/contexts warns
# (those don't travel in a .sdsge). That's expected for every result-bundling
# test here and is important communication to a real bundler, not test noise, so
# suppress it at the boundary while leaving the source warning intact.
pytestmark = pytest.mark.filterwarnings("ignore:MC results were ran with ")


def zscore(*, context, **kwargs):
    """Top-level custom op (NumpyCustomFunc-eligible)."""
    arr = context.require_data().observables
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def pval_table(*, traces):
    """Top-level pandas post-loop op returning a DataFrame (references `pd`)."""
    pvals = traces["test.jb.pval"]
    return pd.DataFrame({"rep": np.arange(pvals.size), "pval": pvals})


def summary_bundle(*, traces, threshold):
    """Post-loop op emitting all three artifact kinds: scalar, array, table."""
    pvals = traces["test.jb.pval"]
    flags = (pvals < threshold).astype(float)
    return {
        "pcs": float(flags.mean()),  # scalar -> Summary (document)
        "flags": flags,  # array -> Raw (mc_postproc)
        "table": pd.DataFrame(  # DataFrame -> Summary table (mc_postproc_table)
            {"rep": np.arange(pvals.size), "pval": pvals}
        ),
    }


def selection_rate(*, traces):
    """Top-level postproc op: bare values auto-wrap (scalar -> Summary, array ->
    Raw), so the op body references no captured types."""
    pval = traces["test.jb.pval"]
    indicator = (pval < 0.5).astype(float)
    return {"rate": float(indicator.mean()), "flags": indicator}


def _raw_data_pipeline() -> MCPipeline:
    rng = np.random.default_rng(0)
    observables = rng.normal(size=(3, 20, 2))  # n_rep, T, k
    return MCPipeline(
        [
            raw_data_step(
                "dat",
                observables=observables,
                n_exog=1,
                observable_names=("y", "x"),
            ),
            jarque_bera_test_step("jb", source="observables", column=0),
        ]
    )


def test_add_mc_ships_raw_data_member_and_loader_rehydrates(tmp_path) -> None:
    pipe = _raw_data_pipeline()
    expected = np.asarray(pipe.per_rep_steps[0].kwargs["observables"], dtype=np.float64)

    target = (
        BundleBuilder(created_by="mc-test").add_mc(pipe).write(tmp_path / "raw.sdsge")
    )

    loaded = build_from(target)
    assert loaded.mc is not None
    # The parquet side-channel member exists and rehydrated under data_ref.
    assert any(m.kind == "mc_raw_data" for m in loaded.manifest.members)
    arrays = loaded.mc.resources["dat"]
    np.testing.assert_allclose(arrays["observables"], expected)

    # The loaded spec + resources rebuild an equivalent runnable pipeline.
    ordered, postprocs = validate_pipeline_spec(
        loaded.mc.spec, has_reference=True, has_dgp=False
    )
    rebuilt = build_pipeline(ordered, postprocs, resources=loaded.mc.resources)
    np.testing.assert_allclose(rebuilt.per_rep_steps[0].kwargs["observables"], expected)
    assert [s.step_type for s in rebuilt.per_rep_steps] == ["raw_data", "jarque_bera"]


def test_add_mc_ships_custom_op_member_and_loader_rebuilds(tmp_path) -> None:
    observables = np.random.default_rng(1).normal(size=(2, 15, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            transform_step("z", zscore, source="observables"),
            jarque_bera_test_step("jb", source="payload", payload_key="z"),
        ]
    )

    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe)
        .write(tmp_path / "custom.sdsge")
    )

    loaded = build_from(target)
    assert loaded.mc is not None
    assert any(m.kind == "mc_custom_op" for m in loaded.manifest.members)

    func = loaded.mc.resources["z"]
    assert isinstance(func, NumpyCustomFunc)  # wrapped + source-carrying
    assert "zscore" in func.source

    ordered, postprocs = validate_pipeline_spec(
        loaded.mc.spec, has_reference=True, has_dgp=False
    )
    rebuilt = build_pipeline(ordered, postprocs, resources=loaded.mc.resources)
    z_step = {s.name: s for s in rebuilt.per_rep_steps}["z"]
    assert z_step.step_type == "transform:custom"
    assert callable(z_step.func)


def test_add_mc_ships_postproc_artifacts_and_wire_round_trips(tmp_path) -> None:
    observables = np.random.default_rng(2).normal(size=(4, 20, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
        ],
        [postproc_step("post", selection_rate)],
    )
    result = pipe.run(reference=cast(SolvedModel, object()), n_rep=4, verbosity=0)

    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe, result=result, run_id="r1")
        .write(tmp_path / "pp.sdsge")
    )

    loaded = build_from(target)
    assert loaded.mc is not None
    # The bulk array artifact rides its own shape-manifest parquet member.
    assert any(m.kind == "mc_postproc" for m in loaded.manifest.members)
    np.testing.assert_array_equal(
        loaded.mc.postproc_arrays["post.flags"], result.postproc["post"]["flags"]
    )

    # document (scalar inline) + parquet array re-merge to the live wire shape.
    wire = loaded.mc.wire()
    assert wire is not None
    assert (
        wire["postproc"] == serialize_pipeline_result(result, run_id="r1")["postproc"]
    )


def test_add_mc_warns_when_bundling_a_result_with_retained_per_rep_data(
    tmp_path,
) -> None:
    # A default run retains per-rep payloads / test results, none of which travel
    # in the bundle. Bundling that result must warn the author so they know to
    # re-run if the dropped fields matter (the rest of this module suppresses the
    # warning as expected noise; here we assert it actually fires).
    observables = np.random.default_rng(0).normal(size=(4, 20, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
        ],
    )
    result = pipe.run(reference=cast(SolvedModel, object()), n_rep=4, verbosity=0)
    assert result.meta.test_results_retained is True  # sanity: the run retained

    with pytest.warns(UserWarning, match="not supported in the bundle"):
        BundleBuilder(created_by="mc-test").add_mc(
            pipe, result=result, run_id="r1"
        ).write(tmp_path / "warn.sdsge")


def test_add_mc_ships_postproc_table_and_wire_round_trips(tmp_path) -> None:
    from SymbolicDSGE.monte_carlo.operations.postproc import kde_step

    observables = np.random.default_rng(3).normal(size=(12, 30, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
        ],
        [kde_step("kde", trace="test.jb.statistic", grid_points=32)],
    )
    result = pipe.run(reference=cast(SolvedModel, object()), n_rep=12, verbosity=0)

    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe, result=result, run_id="r1")
        .write(tmp_path / "kde.sdsge")
    )

    loaded = build_from(target)
    assert loaded.mc is not None
    # KDE emits a Raw curve (array member) and a descriptives table member.
    assert any(m.kind == "mc_postproc" for m in loaded.manifest.members)
    assert any(m.kind == "mc_postproc_table" for m in loaded.manifest.members)
    assert "kde.descriptives" in loaded.mc.postproc_tables

    wire = loaded.mc.wire()
    assert wire is not None
    assert (
        wire["postproc"] == serialize_pipeline_result(result, run_id="r1")["postproc"]
    )


def test_add_mc_ships_pandas_postproc_op_under_pandas_namespace(tmp_path) -> None:
    observables = np.random.default_rng(5).normal(size=(6, 20, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
        ],
        [postproc_step("ptab", pval_table)],  # plain func -> auto-wrapped at ship
    )
    result = pipe.run(reference=cast(SolvedModel, object()), n_rep=6, verbosity=0)

    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe, result=result, run_id="r1")
        .write(tmp_path / "pandas_pp.sdsge")
    )

    loaded = build_from(target)
    assert loaded.mc is not None
    # The post-loop op was wrapped under the pandas namespace and round-trips.
    func = loaded.mc.resources["ptab"]
    assert isinstance(func, PandasCustomFunc)
    out = func(traces={"test.jb.pval": np.array([0.1, 0.2])})
    assert isinstance(out, pd.DataFrame)
    # The DataFrame artifact also lands as a table member.
    assert any(m.kind == "mc_postproc_table" for m in loaded.manifest.members)


def test_postproc_custom_op_full_round_trip(tmp_path) -> None:
    # Acceptance (#183): a pipeline with a custom POSTPROC op emitting scalar +
    # array + table artifacts writes to .sdsge and reloads to an equivalent
    # runnable pipeline whose re-run reproduces the recovered artifacts.
    observables = np.random.default_rng(7).normal(size=(8, 25, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
        ],
        [postproc_step("sum", summary_bundle, threshold=0.5)],
    )
    ref = cast(SolvedModel, object())
    result = pipe.run(reference=ref, n_rep=8, verbosity=0)

    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe, result=result, run_id="r1")
        .write(tmp_path / "pp_full.sdsge")
    )
    loaded = build_from(target)
    assert loaded.mc is not None

    # All three artifact channels plus the custom-op blob shipped.
    kinds = {m.kind for m in loaded.manifest.members}
    assert {"mc_custom_op", "mc_postproc", "mc_postproc_table"} <= kinds

    # The post-loop op rehydrated under the pandas namespace, kwargs preserved.
    func = loaded.mc.resources["sum"]
    assert isinstance(func, PandasCustomFunc)

    # Recovered artifacts (document + parquet members) == the live wire.
    assert (
        loaded.mc.wire()["postproc"]
        == serialize_pipeline_result(result, run_id="r1")["postproc"]
    )
    assert loaded.mc.postproc_tables["sum.table"]["pval"] == list(
        result.postproc["sum"]["table"]["pval"]
    )

    # Rebuild from spec + resources -> equivalent runnable pipeline.
    ordered, postprocs = validate_pipeline_spec(
        loaded.mc.spec, has_reference=True, has_dgp=False
    )
    rebuilt = build_pipeline(ordered, postprocs, resources=loaded.mc.resources)
    assert [s.step_type for s in (*rebuilt.per_rep_steps, *rebuilt.postproc_steps)] == [
        "raw_data",
        "jarque_bera",
        "postproc:custom",
    ]

    # Re-running reproduces every artifact (deterministic replayed data).
    rerun = rebuilt.run(reference=ref, n_rep=8, verbosity=0)
    assert rerun.postproc["sum"]["pcs"] == result.postproc["sum"]["pcs"]
    np.testing.assert_array_equal(
        rerun.postproc["sum"]["flags"], result.postproc["sum"]["flags"]
    )
    assert list(rerun.postproc["sum"]["table"]["pval"]) == list(
        result.postproc["sum"]["table"]["pval"]
    )


def test_pandas_wrapper_rejected_outside_postproc() -> None:
    # A PandasCustomFunc on a per-rep transform is rejected at pipeline build.
    wrapped = PandasCustomFunc(pval_table)
    with pytest.raises((ValueError, CustomOpValidationError), match="POSTPROC"):
        transform_step("bad", wrapped, source="observables")


def test_add_mc_rejects_unshippable_custom_op(tmp_path) -> None:
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=np.zeros((2, 5, 2))),
            transform_step("z", lambda **_: None, source="observables"),
        ]
    )
    with pytest.raises(Exception, match="[Ll]ambda"):
        BundleBuilder().add_mc(pipe)


def test_add_mc_still_accepts_a_plain_pipeline_spec(tmp_path) -> None:
    # The explicit spec path is unchanged: no side-channel members emitted.
    spec = _raw_data_pipeline().to_spec()
    builder = BundleBuilder().add_mc(spec)
    manifest, _files = builder.build()
    kinds = {m.kind for m in manifest.members}
    assert "mc_pipeline" in kinds
    assert "mc_raw_data" not in kinds
