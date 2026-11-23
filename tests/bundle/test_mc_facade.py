from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.bundle import BundleBuilder, load_bundle
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.custom_op import NumbaCustomFunc
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
    transform_step,
)

# Bundling a result that retained per-rep payloads/test-results/contexts warns
# (those don't travel in a .sdsge). That's expected for every result-bundling
# test here and is important communication to a real bundler, not test noise, so
# suppress it at the boundary while leaving the source warning intact.
pytestmark = pytest.mark.filterwarnings("ignore:MC results were ran with ")


def zscore(sample: np.ndarray, output: np.ndarray) -> int:
    """Top-level custom transform with the fixed native ABI."""
    output[:, :] = (sample - sample.mean(axis=0)) / sample.std(axis=0)
    return 0


def _raw_model_data_pipeline() -> MCPipeline:
    rng = np.random.default_rng(0)
    observables = rng.normal(size=(3, 20, 2))  # n_rep, T, k
    return MCPipeline(
        raw_model_data_step(
            "dat",
            observables=observables,
            observable_names=("y", "x"),
        ),
        jarque_bera_test_step("jb", source="dat", field="observables", column=0),
    )


def test_add_mc_ships_raw_model_data_member_and_loader_rehydrates(tmp_path) -> None:
    pipe = _raw_model_data_pipeline()
    expected = np.asarray(pipe.steps[0].kwargs["observables"], dtype=np.float64)

    target = (
        BundleBuilder(created_by="mc-test").add_mc(pipe).write(tmp_path / "raw.sdsge")
    )

    loaded = load_bundle(target)
    assert loaded.mc is not None
    # The parquet side-channel member exists and rehydrated under data_ref.
    assert any(m.kind == "mc_data" for m in loaded.manifest.members)

    # The member rehydrated into the rebuilt pipeline's datagen step.
    rebuilt = loaded.mc.pipeline
    np.testing.assert_allclose(rebuilt.steps[0].kwargs["observables"], expected)
    assert [s.step_type for s in rebuilt.steps] == [
        "raw_model_data",
        "jarque_bera",
    ]


def test_add_mc_ships_custom_op_member_and_loader_rebuilds(tmp_path) -> None:
    observables = np.random.default_rng(1).normal(size=(2, 15, 2))
    pipe = MCPipeline(
        raw_model_data_step(
            "dat", observables=observables, observable_names=("y", "x")
        ),
        transform_step(
            "z", zscore, source="dat", field="observables", output_shape=(15, 2)
        ),
        jarque_bera_test_step("jb", source="z", field="payload"),
    )

    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe)
        .write(tmp_path / "custom.sdsge")
    )

    loaded = load_bundle(target)
    assert loaded.mc is not None
    assert any(m.kind == "mc_func" for m in loaded.manifest.members)

    rebuilt = loaded.mc.pipeline
    z_step = {s.name: s for s in rebuilt.steps}["z"]
    assert z_step.step_type == "transform:custom"
    assert isinstance(z_step.func, NumbaCustomFunc)  # wrapped + source-carrying
    assert "zscore" in z_step.func.source


def test_add_mc_bundles_native_result_without_legacy_retention_flags(
    tmp_path,
) -> None:
    # Native results retain only declared output rows. There are no legacy
    # contexts or Python result objects to warn about when bundling.
    observables = np.random.default_rng(0).normal(size=(4, 20, 2))
    pipe = MCPipeline(
        raw_model_data_step(
            "dat", observables=observables, observable_names=("y", "x")
        ),
        jarque_bera_test_step("jb", source="dat", field="observables", column=0),
    )
    result = pipe.run(
        models={"reference": cast(SolvedModel, object())}, n_rep=4, verbosity=0
    )
    target = (
        BundleBuilder(created_by="mc-test")
        .add_mc(pipe, result=result)
        .write(tmp_path / "native-result.sdsge")
    )
    assert load_bundle(target).mc is not None


def test_transform_step_rejects_unshippable_custom_op() -> None:
    with pytest.raises(Exception, match="[Ll]ambda"):
        transform_step(
            "z",
            lambda sample, output: 0,
            source="dat",
            field="observables",
            output_shape=(5, 2),
        )
