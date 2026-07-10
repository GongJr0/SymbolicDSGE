from __future__ import annotations

from pathlib import Path

import numpy as np

from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.bundle.manifest import SimSpec
from SymbolicDSGE.bundle.parquet import collapse_columns, from_parquet_columns
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult
from SymbolicDSGE.estimation.spec import (
    EstimationParameterSpec,
    EstimationSpec,
    MCMCResultMeta,
)
from SymbolicDSGE.monte_carlo.spec import NodeSpec, PipelineSpec

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")


def _estimation_spec() -> EstimationSpec:
    return EstimationSpec(
        method="mcmc",
        parameters=[
            EstimationParameterSpec(name="beta", initial=0.99, estimate=True),
            EstimationParameterSpec(name="sigma", initial=1.0, estimate=True),
        ],
        observables=["Infl", "Rate"],
    )


def test_full_bundle_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((20, 2))
    posterior = {
        "samples": rng.standard_normal((50, 2)),
        "logpost": rng.standard_normal(50),
    }
    result_meta = MCMCResultMeta(
        param_names=["beta", "sigma"],
        accept_rate=0.31,
        n_draws=50,
        burn_in=10,
        thin=1,
    )
    pipeline = PipelineSpec(
        nodes=[NodeSpec(id="n1", step_type="simulation", name="sim", params={"T": 50})]
    )

    builder = (
        BundleBuilder(created_by="test-suite")
        .add_model(
            "reference",
            _MODEL_YAML,
            compile_kwargs={},
        )
        .add_estimation(
            _estimation_spec(),
            result=result_meta,
            observed=observed,
            observable_names=["Infl", "Rate"],
            posterior=posterior,
        )
        .add_mc(pipeline)
        .add_raw_data("series", "a,b\n1,2.5\n3,4.5\n")
        .set_simulation(
            "reference",
            SimSpec(
                T=8,
                shocks={
                    "u": {
                        "dist": "norm",
                        "multivar": False,
                        "seed": 42,
                        "dist_args": [],
                        "dist_kwargs": {"loc": 0.0},
                    }
                },
            ),
        )
    )
    target = builder.write(tmp_path / "model.sdsge")

    loaded = build_from(target)

    # model rebuilt and usable
    assert isinstance(loaded.reference, SolvedModel)
    assert loaded.dgp is None
    sim = loaded.reference.sim(8)
    assert sim["_X"].shape[0] == 9

    # estimation
    assert loaded.estimation is not None
    assert loaded.estimation.spec.method == "mcmc"
    assert [p.name for p in loaded.estimation.spec.parameters] == ["beta", "sigma"]
    assert isinstance(loaded.estimation.result, MCMCResult)
    assert loaded.estimation.result.accept_rate == 0.31
    assert loaded.estimation.observed is not None
    np.testing.assert_allclose(loaded.estimation.observed, observed)
    assert loaded.estimation.posterior is not None
    np.testing.assert_allclose(
        loaded.estimation.posterior["samples"], posterior["samples"]
    )
    np.testing.assert_allclose(
        loaded.estimation.posterior["logpost"], posterior["logpost"]
    )

    # monte carlo
    assert loaded.mc is not None
    assert loaded.mc.spec.nodes[0].step_type == "simulation"
    assert loaded.mc.document is None  # no result attached
    assert loaded.mc.wire() is None

    # simulation prefill
    assert loaded.simulation is not None
    assert loaded.simulation["reference"].T == 8
    assert loaded.simulation["reference"].shocks["u"]["seed"] == 42

    # manifest integrity
    assert loaded.manifest.created_by == "test-suite"
    assert set(loaded.manifest.checksums) == {m.path for m in loaded.manifest.members}


def test_add_estimation_accepts_live_mcmc_result() -> None:
    import json

    from SymbolicDSGE.estimation.results import MCMCResult

    rng = np.random.default_rng(1)
    mcmc = MCMCResult(
        param_names=["a", "b"],
        samples=rng.standard_normal((5, 2)),
        logpost_trace=rng.standard_normal(5),
        accept_rate=np.float64(0.4),
        n_draws=5,
        burn_in=0,
        thin=1,
    )
    spec = EstimationSpec.from_targets(["a", "b"], method="mcmc")

    builder = BundleBuilder().add_estimation(spec, result=mcmc)
    _, files = builder.build()

    # live result projected to meta, tagged mcmc
    payload = json.loads(files["estimation/result.json"])
    assert payload["type"] == "mcmc"
    assert payload["data"]["accept_rate"] == 0.4
    # posterior auto-extracted from the live result (not passed explicitly)
    assert "estimation/posterior.parquet" in files
    posterior = collapse_columns(
        from_parquet_columns(files["estimation/posterior.parquet"])
    )
    np.testing.assert_allclose(posterior["samples"], mcmc.samples)
    np.testing.assert_allclose(posterior["logpost"], mcmc.logpost_trace)


def test_raw_data_member_round_trips(tmp_path: Path) -> None:
    builder = BundleBuilder().add_raw_data("series", "a,b\n1,2.5\n3,4.5\n")
    _, files = builder.build()
    columns = collapse_columns(from_parquet_columns(files["data/series.parquet"]))
    assert list(columns["a"]) == [1, 3]
    np.testing.assert_allclose(columns["b"].astype(float), [2.5, 4.5])


def test_csv_passthrough_member(tmp_path: Path) -> None:
    builder = BundleBuilder().add_raw_data("series", "a,b\n1,2\n", as_parquet=False)
    manifest, files = builder.build()
    assert "data/series.csv" in files
    assert manifest.members[0].format == "csv"
