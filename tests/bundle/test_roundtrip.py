from __future__ import annotations

from pathlib import Path

import numpy as np

from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.bundle.parquet import collapse_columns, from_parquet_columns
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult
from SymbolicDSGE.estimation.spec import EstimatorParams, EstimatorSpec
from SymbolicDSGE.monte_carlo.spec import NodeSpec, PipelineSpec

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")


def _estimation_spec(y, *, names=("beta", "sigma")) -> EstimatorSpec:
    return EstimatorSpec(
        y=np.asarray(y).tolist(),
        params=EstimatorParams(
            observables=["Infl", "Rate"],
            filter_mode="linear",
            P0=None,
            R=None,
            estimated_params=list(names),
            priors=None,
            ss_seed=None,
            x0=None,
            jitter=0.0,
            symmetrize=True,
            joseph_cov=True,
        ),
    )


def test_full_bundle_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((20, 2))
    posterior = {
        "samples": rng.standard_normal((50, 2)),
        "logpost": rng.standard_normal(50),
        "logjac": rng.standard_normal(50),
    }
    result = MCMCResult(
        param_names=["beta", "sigma"],
        samples=posterior["samples"],
        logpost_trace=posterior["logpost"],
        logjac_trace=posterior["logjac"],
        accept_rate=np.float64(0.31),
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
        .add_estimation(_estimation_spec(observed), result=result)
        .add_mc(pipeline)
        .add_raw_data("series", "a,b\n1,2.5\n3,4.5\n")
        .set_simulation(
            "reference",
            T=8,
            shocks={"u": Shock(dist="norm", seed=42, dist_kwargs={"loc": 0.0})},
        )
    )
    target = builder.write(tmp_path / "model.sdsge")

    loaded = build_from(target)

    # model rebuilt and usable
    assert isinstance(loaded.reference, SolvedModel)
    assert loaded.dgp is None
    sim = loaded.reference.sim(8)
    assert sim.X.shape[0] == 8

    # estimation
    assert loaded.estimation is not None
    assert loaded.estimation.spec.params["estimated_params"] == ["beta", "sigma"]
    assert loaded.estimation.spec.params["observables"] == ["Infl", "Rate"]
    assert isinstance(loaded.estimation.result, MCMCResult)
    assert loaded.estimation.result.accept_rate == 0.31
    np.testing.assert_allclose(np.asarray(loaded.estimation.spec.y), observed)
    np.testing.assert_allclose(loaded.estimation.result.samples, posterior["samples"])
    np.testing.assert_allclose(
        loaded.estimation.result.logpost_trace, posterior["logpost"]
    )
    np.testing.assert_allclose(
        loaded.estimation.result.logjac_trace, posterior["logjac"]
    )

    # monte carlo
    assert loaded.mc is not None
    assert loaded.mc.spec.nodes[0].step_type == "simulation"
    assert loaded.mc.document is None  # no result attached
    assert loaded.mc.wire() is None

    # simulation prefill
    assert loaded.simulation is not None
    prefill = loaded.simulation["reference"]
    assert prefill["T"] == 8
    assert prefill["shocks"]["u"].seed == 42

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
        logjac_trace=rng.standard_normal(5),
        accept_rate=np.float64(0.4),
        n_draws=5,
        burn_in=0,
        thin=1,
    )
    spec = _estimation_spec(rng.standard_normal((4, 2)), names=("a", "b"))

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
