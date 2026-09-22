"""Bundle round trips preserve pipelines with zero or multiple datagens."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bundle import BundleBuilder, load_bundle
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.step_factories import (
    add_payload_step,
    raw_model_data_step,
    simulation_step,
)


@pytest.fixture(scope="module")
def simulation_model() -> SolvedModel:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


@pytest.mark.parametrize("as_parquet", [True, False], ids=["parquet", "csv"])
@pytest.mark.parametrize(
    "n_simulations,n_raw",
    [(0, 0), (2, 0), (3, 0), (1, 1), (0, 2), (0, 3)],
    ids=[
        "no-datagen",
        "two-simulations",
        "three-simulations",
        "mixed",
        "two-raw",
        "three-raw",
    ],
)
def test_datagen_counts_roundtrip(
    n_simulations: int,
    n_raw: int,
    as_parquet: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    reference = request.getfixturevalue("simulation_model") if n_simulations else None
    payload = np.arange(12, dtype=np.float64).reshape(6, 2)
    # A source-free transform first also exercises the absence of a positional datagen.
    steps = [add_payload_step("payload", payload)]
    for i in range(n_simulations):
        T = 6 + i
        shocks = {
            name: np.full(T, 0.001 * (i + 1), dtype=np.float64)
            for name in reference.compiled.shock_names
        }
        steps.append(
            simulation_step(
                f"sim_{i}",
                target="reference",
                T=T,
                shocks=shocks,
                n_retain=2 if i == 0 else -1,
            )
        )
    for i in range(n_raw):
        data = np.arange(3 * (7 + i) * (i + 1), dtype=np.float64).reshape(
            3, 7 + i, i + 1
        )
        steps.append(
            raw_model_data_step(
                f"raw_{i}",
                observables=data,
                observable_names=tuple(f"obs_{j}" for j in range(i + 1)),
                n_retain=2 if i == 0 else -1,
            )
        )
    pipeline = MCPipeline(steps)
    result = pipeline.run(
        models={"reference": reference} if reference is not None else None,
        n_rep=3,
        n_jobs=1,
        verbosity=0,
        fail_fast=True,
    )
    assert not result.failures
    expected_names = {f"sim_{i}" for i in range(n_simulations)} | {
        f"raw_{i}" for i in range(n_raw)
    }
    assert set(result.datagen_outputs) == expected_names

    target = (
        BundleBuilder()
        .add_mc(pipeline, result=result, as_parquet=as_parquet)
        .write(tmp_path / "datagens.sdsge")
    )
    loaded = load_bundle(target)
    assert loaded.mc is not None
    restored = loaded.mc.result
    assert restored is not None
    assert set(restored.datagen_outputs) == expected_names
    assert restored.n_rep == result.n_rep
    assert restored.n_successful == result.n_successful
    assert restored.failures == result.failures
    assert [s.name for s in loaded.mc.pipeline.replication_steps] == [
        s.name for s in steps
    ]
    assert [s.step_type for s in loaded.mc.pipeline.replication_steps] == [
        s.step_type for s in steps
    ]
    for name, original in result.datagen_outputs.items():
        rebuilt = restored.datagen_outputs[name]
        for field in fields(original):
            before, after = getattr(original, field.name), getattr(rebuilt, field.name)
            if isinstance(before, np.ndarray):
                assert after.shape == before.shape
                assert after.dtype == before.dtype
                np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-14)
            else:
                assert after == before
    np.testing.assert_array_equal(
        restored.transform_outputs["payload"], result.transform_outputs["payload"]
    )
