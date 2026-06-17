from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.bundle import BundleBuilder, build_from
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline, build_pipeline, validate_pipeline_spec
from SymbolicDSGE.monte_carlo.custom_op import NumpyCustomFunc
from SymbolicDSGE.monte_carlo.operations.core import raw_data_step
from SymbolicDSGE.monte_carlo.operations.postproc import postproc_step
from SymbolicDSGE.monte_carlo.operations.tests import jarque_bera_test_step
from SymbolicDSGE.monte_carlo.operations.transforms import transform_step
from SymbolicDSGE.monte_carlo.serialize import serialize_pipeline_result


def zscore(*, context, **kwargs):
    """Top-level custom op (NumpyCustomFunc-eligible)."""
    arr = context.require_data().observables
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def selection_rate(*, traces, reference, dgp):
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
    expected = np.asarray(pipe.steps[0].kwargs["observables"], dtype=np.float64)

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
    ordered = validate_pipeline_spec(loaded.mc.spec, has_reference=True, has_dgp=False)
    rebuilt = build_pipeline(ordered, resources=loaded.mc.resources)
    np.testing.assert_allclose(rebuilt.steps[0].kwargs["observables"], expected)
    assert [s.step_type for s in rebuilt.steps] == ["raw_data", "jarque_bera"]


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

    ordered = validate_pipeline_spec(loaded.mc.spec, has_reference=True, has_dgp=False)
    rebuilt = build_pipeline(ordered, resources=loaded.mc.resources)
    z_step = {s.name: s for s in rebuilt.steps}["z"]
    assert z_step.step_type == "transform:custom"
    assert callable(z_step.func)


def test_add_mc_ships_postproc_artifacts_and_wire_round_trips(tmp_path) -> None:
    observables = np.random.default_rng(2).normal(size=(4, 20, 2))
    pipe = MCPipeline(
        [
            raw_data_step("dat", observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
            postproc_step("post", selection_rate),
        ]
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
        loaded.mc.postproc_arrays["post.flags"], result.postproc["post.flags"].value
    )

    # document (scalar inline) + parquet array re-merge to the live wire shape.
    wire = loaded.mc.wire()
    assert wire is not None
    assert (
        wire["postproc"] == serialize_pipeline_result(result, run_id="r1")["postproc"]
    )


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
