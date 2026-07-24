"""Tests for ``serve_from``, the workspace preload, and the unified emitter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.bundle.manifest import SimSpec
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult, MLEResult, MAPResult
from SymbolicDSGE.estimation.spec import (
    EstimationParameterSpec,
    EstimationSpec,
    MCMCResultMeta,
)
from SymbolicDSGE.monte_carlo.spec import NodeSpec, PipelineSpec
from SymbolicDSGE.ui import build_workspace, create_app, serve_from
from SymbolicDSGE.ui.estimation import emit_estimation_wire, serialize_estimation_result
from SymbolicDSGE.ui.session import UISession, Workspace

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")


# -- helpers ----------------------------------------------------------------


def _solved_test_model() -> SolvedModel:
    parser = ModelParser.from_string(_MODEL_YAML)
    model, kalman = parser.get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def _estimation_spec() -> EstimationSpec:
    return EstimationSpec(
        method="mcmc",
        parameters=[
            EstimationParameterSpec(name="beta", initial=0.99, estimate=True),
            EstimationParameterSpec(name="sigma", initial=1.0, estimate=True),
        ],
        observables=["Infl", "Rate"],
    )


def _hydrated_bundle(tmp_path: Path) -> Path:
    """Build a bundle that hits every preload slot (estimation+mc+sim)."""
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((10, 2))
    posterior = {
        "samples": rng.standard_normal((20, 2)),
        "logpost": rng.standard_normal(20),
    }
    result_meta = MCMCResultMeta(
        param_names=["beta", "sigma"],
        accept_rate=0.33,
        n_draws=20,
        burn_in=5,
        thin=1,
    )
    pipeline = PipelineSpec(
        nodes=[NodeSpec(id="n1", step_type="simulation", name="sim", params={"T": 20})]
    )
    sim_spec = SimSpec(
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
    )

    return (
        BundleBuilder(created_by="serve-test")
        .add_model("reference", _MODEL_YAML, compile_kwargs={})
        .add_estimation(
            _estimation_spec(),
            result=result_meta,
            observed=observed,
            observable_names=["Infl", "Rate"],
            posterior=posterior,
        )
        .add_mc(pipeline)
        .set_simulation("reference", sim_spec)
        .write(tmp_path / "hydrate.sdsge")
    )


# -- emit_estimation_wire parity --------------------------------------------


def test_emit_wire_mle_result() -> None:
    theta = {"beta": 0.99, "rho": 0.8}
    res = MLEResult(
        x=np.array(list(theta.values())),
        theta={k: np.float64(v) for k, v in theta.items()},
        success=True,
        message="ok",
        fun=np.float64(-12.3),
        nfev=42,
        nit=15,
        optimizer_config={},
        loglik=np.float64(-10.0),
    )
    wire = emit_estimation_wire(res)
    assert wire["theta"] == {"beta": 0.99, "rho": 0.8}
    assert (wire["fun"], wire["nfev"], wire["nit"]) == (-12.3, 42, 15)
    assert wire["loglik"] == -10.0
    assert wire["success"] is True and wire["message"] == "ok"


def test_emit_wire_mcmc_meta_plus_traces_matches_live_result() -> None:
    rng = np.random.default_rng(7)
    samples = rng.standard_normal((30, 2))
    logpost = rng.standard_normal(30)
    live = MCMCResult(
        param_names=["beta", "sigma"],
        samples=samples,
        logpost_trace=logpost,
        accept_rate=np.float64(0.31),
        n_draws=30,
        burn_in=5,
        thin=1,
    )
    meta = MCMCResultMeta(
        param_names=["beta", "sigma"],
        accept_rate=0.31,
        n_draws=30,
        burn_in=5,
        thin=1,
    )
    traces = {"samples": samples, "logpost_trace": logpost}
    assert emit_estimation_wire(live) == emit_estimation_wire(meta, traces=traces)


def test_emit_wire_mcmc_meta_requires_traces() -> None:
    meta = MCMCResultMeta(
        param_names=["beta"], accept_rate=0.3, n_draws=10, burn_in=0, thin=1
    )
    with pytest.raises(ValueError, match="samples"):
        emit_estimation_wire(meta)


def test_serialize_estimation_result_shim_delegates() -> None:
    res = MAPResult(
        x=np.array([1.0]),
        theta={"a": np.float64(1.0)},
        success=False,
        message="x",
        fun=np.float64(0.0),
        nfev=1,
        nit=None,
        optimizer_config={},
        logpost=np.float64(0.0),
        logprior=np.float64(0.0),
    )
    assert serialize_estimation_result(res) == emit_estimation_wire(res)


# -- Workspace + session summary -------------------------------------------


def test_session_summary_carries_empty_workspace_by_default() -> None:
    client = TestClient(create_app())
    payload = client.get("/api/session").json()
    assert payload["workspace"] == {}


def test_session_summary_surfaces_workspace_preload() -> None:
    workspace = Workspace(
        estimation={"kind": "mcmc", "param_names": ["beta"]},
        estimation_spec={"method": "mcmc"},
        mc={"kind": "mc"},
        mc_pipeline={"nodes": []},
        simulation={"T": 8},
    )
    client = TestClient(create_app(workspace=workspace))
    payload = client.get("/api/session").json()["workspace"]
    assert payload["estimation"]["kind"] == "mcmc"
    assert payload["estimation_spec"]["method"] == "mcmc"
    assert payload["mc"]["kind"] == "mc"
    assert payload["mc_pipeline"] == {"nodes": []}
    assert payload["simulation"] == {"T": 8}


def test_session_summary_drops_unset_workspace_slots() -> None:
    workspace = Workspace(simulation={"T": 5})  # only simulation populated
    client = TestClient(create_app(workspace=workspace))
    payload = client.get("/api/session").json()["workspace"]
    assert payload == {"simulation": {"T": 5}}


# -- build_workspace from a LoadedBundle -----------------------------------


def test_build_workspace_populates_all_slots(tmp_path: Path) -> None:
    loaded = build_from(_hydrated_bundle(tmp_path))
    ws = build_workspace(loaded)
    assert ws.estimation is not None and ws.estimation["kind"] == "mcmc"
    assert ws.estimation["param_names"] == ["beta", "sigma"]
    # bulk traces survived round-trip into the wire dict
    assert len(ws.estimation["samples"]["beta"]) == 20
    assert ws.estimation_spec is not None
    assert ws.estimation_spec["method"] == "mcmc"
    assert ws.mc_pipeline is not None
    assert ws.mc is None  # no MC result was attached at build time
    assert ws.simulation is not None
    assert ws.simulation["reference"]["T"] == 8
    assert ws.simulation["reference"]["shocks"]["u"]["seed"] == 42


# -- serve_from dispatch ---------------------------------------------------


def test_serve_from_none_calls_run_server_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("SymbolicDSGE.ui.cli.run_server", fake_run_server)
    serve_from(source=None, open_browser=False, port=12345)
    assert captured["port"] == 12345
    assert "reference" not in captured  # not forwarded when source is None
    assert "workspace" not in captured


def test_serve_from_solved_model_preloads_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("SymbolicDSGE.ui.cli.run_server", fake_run_server)
    solved = _solved_test_model()
    serve_from(source=solved, open_browser=False)
    assert captured["reference"] is solved
    assert captured.get("workspace") is None


def test_serve_from_bundle_path_hydrates_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("SymbolicDSGE.ui.cli.run_server", fake_run_server)
    bundle = _hydrated_bundle(tmp_path)
    serve_from(source=bundle, open_browser=False)
    assert isinstance(captured["reference"], SolvedModel)
    assert captured["dgp"] is None
    assert isinstance(captured["workspace"], Workspace)
    assert captured["workspace"].estimation is not None
    assert captured["workspace"].simulation is not None


def test_serve_from_rejects_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bundle path"):
        serve_from(source=tmp_path / "nope.sdsge")


# -- CLI argparse ---------------------------------------------------------


def test_cli_main_with_bundle_delegates_to_serve_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_serve_from(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("SymbolicDSGE.ui.serve.serve_from", fake_serve_from)
    bundle = _hydrated_bundle(tmp_path)
    from SymbolicDSGE.ui.cli import main

    main([str(bundle), "--no-browser", "--port", "9999"])
    assert captured["source"] == bundle
    assert captured["port"] == 9999
    assert captured["open_browser"] is False


def test_cli_main_without_bundle_passes_none_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_serve_from(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("SymbolicDSGE.ui.serve.serve_from", fake_serve_from)
    from SymbolicDSGE.ui.cli import main

    main(["--no-browser"])
    assert captured["source"] is None


def test_cli_main_rejects_missing_bundle_path(tmp_path: Path) -> None:
    from SymbolicDSGE.ui.cli import main

    with pytest.raises(SystemExit, match="bundle path"):
        main([str(tmp_path / "missing.sdsge"), "--no-browser"])
