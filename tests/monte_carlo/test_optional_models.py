"""Model roles are required only by steps that consume them."""

from typing import Literal

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.step_factories import (
    filter_step,
    jarque_bera_test_step,
    raw_model_data_step,
    simulation_step,
    standardize_step,
)


@pytest.fixture(scope="module")
def dgp_model() -> SolvedModel:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def test_model_free_pipeline_omits_reference() -> None:
    data = np.random.default_rng(42).normal(size=(100, 2))
    pipeline = MCPipeline(
        [
            jarque_bera_test_step("jb", source="scaled", field="payload", column=0),
            standardize_step("scaled", source="raw", field="observables"),
            raw_model_data_step("raw", observables=data),
        ]
    )

    result = pipeline.run(n_rep=2, n_jobs=1, verbosity=0, fail_fast=True)

    assert not result.failures
    assert result.n_successful == 2
    assert set(result.datagen_outputs) == {"raw"}
    assert np.isfinite(result.transform_outputs["scaled"]).all()


def test_dgp_only_simulation_and_filter_omit_reference(dgp_model: SolvedModel) -> None:
    pipeline = MCPipeline(
        [
            filter_step("kf", target="dgp", obs_source="sim", obs_field="observables"),
            simulation_step("sim", target="dgp", T=20),
        ]
    )

    result = pipeline.run(dgp=dgp_model, n_rep=2, n_jobs=1, verbosity=0, fail_fast=True)

    assert not result.failures
    assert result.n_successful == 2
    assert result.filter_outputs["kf"].x_filt.shape[:2] == (2, 20)
    assert np.isfinite(result.filter_outputs["kf"].x_filt).all()


@pytest.mark.parametrize("target", ["reference", "dgp"])
@pytest.mark.parametrize("kind", ["simulation", "filter"])
def test_missing_target_model_raises_intentional_error(
    target: Literal["reference", "dgp"], kind: str
) -> None:
    if kind == "simulation":
        pipeline = MCPipeline([simulation_step("sim", target=target, T=20)])
        name = "sim"
    else:
        pipeline = MCPipeline(
            [
                raw_model_data_step("raw", observables=np.zeros((20, 1))),
                filter_step(
                    "kf", target=target, obs_source="raw", obs_field="observables"
                ),
            ]
        )
        name = "kf"

    with pytest.raises(
        ValueError, match=f"Step '{name}' requires its target model '{target}'"
    ):
        pipeline.run(n_rep=2, n_jobs=1, verbosity=0, fail_fast=True)
