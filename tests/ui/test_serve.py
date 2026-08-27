"""Tests for ``serve_from``, the workspace preload, and the unified emitter."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.monte_carlo.builder import build_pipeline
from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.bundle.manifest import SimSpec
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult, MLEResult, MAPResult
from SymbolicDSGE.estimation.spec import (
    EstimatorParams,
    EstimatorSpec,
)
from SymbolicDSGE.monte_carlo.spec import NodeSpec, PipelineSpec
from SymbolicDSGE.ui import build_workspace, create_app, serve_from
from SymbolicDSGE.ui.estimation import (
    build_estimation_prefill,
    emit_estimation_wire,
    serialize_estimation_result,
)
from SymbolicDSGE.ui.session import TabState, UISession, Workspace

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")


# -- helpers ----------------------------------------------------------------


def _solved_test_model() -> SolvedModel:
    parser = ModelParser.from_string(_MODEL_YAML)
    model, kalman = parser.get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def _estimation_spec(y) -> EstimatorSpec:
    return EstimatorSpec(
        y=np.asarray(y).tolist(),
        params=EstimatorParams(
            observables=["Infl", "Rate"],
            filter_mode="linear",
            P0=None,
            R=None,
            estimated_params=["beta", "sigma"],
            priors=None,
            ss_seed=None,
            x0=None,
            jitter=0.0,
            symmetrize=True,
            joseph_cov=True,
        ),
    )


def _hydrated_bundle(tmp_path: Path) -> Path:
    """Build a bundle that hits every preload slot (estimation+mc+sim)."""
    rng = np.random.default_rng(0)
    observed = rng.standard_normal((10, 2))
    posterior = {
        "samples": rng.standard_normal((20, 2)),
        "logpost": rng.standard_normal(20),
        "logjac": rng.standard_normal(20),
    }
    result = MCMCResult(
        param_names=["beta", "sigma"],
        samples=posterior["samples"],
        logpost_trace=posterior["logpost"],
        logjac_trace=posterior["logjac"],
        accept_rate=np.float64(0.33),
        n_draws=20,
        burn_in=5,
        thin=1,
    )
    pipeline = PipelineSpec(
        nodes=[NodeSpec(id="n1", step_type="simulation", name="sim", params={"T": 20})],
        edges=[],
        postprocs=[],
    )
    sim_spec = SimSpec(
        T=8,
        shocks={
            "e_u": {
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
        .add_estimation(_estimator(observed), result=result)
        .add_mc(build_pipeline(pipeline))
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
    assert wire["x"] == [0.99, 0.8]
    # A run that computed no covariance carries none, and reports OK for it.
    assert wire["vcov"] is None and wire["se"] is None and wire["cov_status"] == 0


def test_emit_wire_rebuilds_the_result_it_came_from() -> None:
    """The wire is the full result, not a display projection of one.

    ``x`` is the optimum and ``vcov`` is what ``cov=True`` paid for on the way
    there; with ``optimizer_config`` they are what ``from_spec`` reads back, so
    the workspace slot a bundle takes is not lossy.
    """
    original = MLEResult(
        x=np.array([0.98, 0.5]),
        theta={"beta": np.float64(0.98), "rho": np.float64(0.5)},
        success=True,
        message="ok",
        fun=np.float64(1.0),
        nfev=4,
        nit=2,
        optimizer_config={"method": "L-BFGS-B", "options": {"maxiter": 15000}},
        vcov=np.array([[1e-4, 0.0], [0.0, 4e-4]]),
        se={"beta": np.float64(0.01), "rho": np.float64(0.02)},
        cov_status=0,
        loglik=np.float64(-1.0),
    )

    rebuilt = MLEResult.from_spec(emit_estimation_wire(original))

    np.testing.assert_allclose(rebuilt.x, original.x)
    assert rebuilt.vcov is not None
    np.testing.assert_allclose(rebuilt.vcov, original.vcov)
    assert rebuilt.optimizer_config == original.optimizer_config
    assert rebuilt.se == original.se
    assert rebuilt.theta == original.theta
    assert rebuilt.loglik == original.loglik


def test_emit_wire_survives_a_covariance_that_failed() -> None:
    """A non-SPD Hessian leaves NaN throughout, which strict JSON rejects.

    Nulling it is what keeps one bad entry from costing the whole payload, and
    the nulls read back as the NaN they stood for.
    """
    failed = MLEResult(
        x=np.array([0.98]),
        theta={"beta": np.float64(0.98)},
        success=True,
        message="ok",
        fun=np.float64(1.0),
        nfev=4,
        nit=2,
        optimizer_config={},
        vcov=np.full((1, 1), np.nan),
        se={"beta": np.float64(np.nan)},
        cov_status=-1802,
        loglik=np.float64(-1.0),
    )

    wire = emit_estimation_wire(failed)

    assert wire["vcov"] == [[None]] and wire["se"] == {"beta": None}
    rebuilt = MLEResult.from_spec(wire)
    assert rebuilt.vcov is not None and bool(np.all(np.isnan(rebuilt.vcov)))
    assert rebuilt.se is not None and np.isnan(rebuilt.se["beta"])


def test_emit_wire_carries_standard_errors_and_nulls_non_finite() -> None:
    """A NaN standard error reaches the wire as ``null``.

    ``sdsge_fill_se`` leaves NaN where the covariance failed and where a
    diagonal variance came out negative, and the JSON encoder rejects NaN, so
    the emitter is the only place that can render it.
    """
    res = MLEResult(
        x=np.array([0.99, 0.8]),
        theta={"beta": np.float64(0.99), "rho": np.float64(0.8)},
        success=True,
        message="ok",
        fun=np.float64(-12.3),
        nfev=42,
        nit=15,
        optimizer_config={},
        se={"beta": np.float64(0.01), "rho": np.float64(np.nan)},
        cov_status=-1802,
        loglik=np.float64(-10.0),
    )

    wire = emit_estimation_wire(res)

    assert wire["se"] == {"beta": 0.01, "rho": None}
    assert wire["cov_status"] == -1802


def test_emit_wire_mcmc_meta_plus_traces_matches_live_result() -> None:
    rng = np.random.default_rng(7)
    samples = rng.standard_normal((30, 2))
    logpost = (rng.standard_normal(30),)
    logjac = rng.standard_normal(30)
    live = MCMCResult(
        param_names=["beta", "sigma"],
        samples=samples,
        logpost_trace=logpost,
        logjac_trace=logjac,
        accept_rate=np.float64(0.31),
        n_draws=30,
        burn_in=5,
        thin=1,
    )
    # a result rebuilt from a bundle carries its own bulk columns, so the wire is
    # identical whether the traces are supplied separately or read off the result
    traces = {"samples": samples, "logpost_trace": logpost, "logjac_trace": logjac}
    assert emit_estimation_wire(live) == emit_estimation_wire(live, traces=traces)


def test_emit_wire_mcmc_carries_sampler_config() -> None:
    """The sampler's call arguments, as optimizer_config is for a point run."""
    config = {"adapt": True, "adapt_start": 100, "random_state": 7}
    live = MCMCResult(
        param_names=["beta"],
        samples=np.zeros((3, 1)),
        logpost_trace=np.zeros(3),
        logjac_trace=np.zeros(3),
        accept_rate=np.float64(0.4),
        n_draws=3,
        burn_in=1,
        thin=1,
        sampler_config=config,
    )

    assert emit_estimation_wire(live)["sampler_config"] == config


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
        estimation=TabState(
            spec={"y": [[1.0]], "params": {"estimated_params": ["beta"]}},
            result={"kind": "mcmc", "param_names": ["beta"]},
            view={"method": "mcmc"},
        ),
        mc=TabState(spec={"nodes": []}, result={"kind": "mc"}),
        simulation={"reference": TabState(spec={"T": 8})},
    )
    client = TestClient(create_app(workspace=workspace))
    payload = client.get("/api/session").json()["workspace"]
    assert payload["estimation"]["spec"]["params"]["estimated_params"] == ["beta"]
    assert payload["estimation"]["result"]["kind"] == "mcmc"
    assert payload["estimation"]["view"]["method"] == "mcmc"
    assert payload["mc"] == {"spec": {"nodes": []}, "result": {"kind": "mc"}}
    assert payload["simulation"] == {"reference": {"spec": {"T": 8}}}


def test_session_summary_drops_unfilled_tab_slots() -> None:
    """A tab reports only the slots something filled, and vanishes with none."""
    workspace = Workspace(estimation=TabState(view={"method": "mle"}))
    client = TestClient(create_app(workspace=workspace))

    payload = client.get("/api/session").json()["workspace"]

    assert payload == {"estimation": {"view": {"method": "mle"}}}


def test_session_summary_drops_unset_workspace_slots() -> None:
    # Only simulation populated, and only its spec within that.
    workspace = Workspace(simulation={"reference": TabState(spec={"T": 5})})
    client = TestClient(create_app(workspace=workspace))
    payload = client.get("/api/session").json()["workspace"]
    assert payload == {"simulation": {"reference": {"spec": {"T": 5}}}}


# -- build_workspace from a LoadedBundle -----------------------------------


def test_build_workspace_populates_all_slots(tmp_path: Path) -> None:
    loaded = build_from(_hydrated_bundle(tmp_path))
    ws = build_workspace(loaded)

    # The bundle's own two members, carried over untouched by the GUI shape.
    assert ws.estimation.spec is not None
    assert ws.estimation.spec["params"]["estimated_params"] == ["beta", "sigma"]
    assert len(ws.estimation.spec["y"]) == 10
    assert ws.estimation.result is not None
    assert ws.estimation.result["param_names"] == ["beta", "sigma"]
    # bulk traces survived round-trip into the wire dict
    assert len(ws.estimation.result["samples"]["beta"]) == 20

    # The view is the pair projected into the form's own shape, per role.
    assert ws.estimation.view is not None
    view = ws.estimation.view["reference"]
    assert view["method"] == "mcmc"  # inferred from the result type
    rows = {row["name"]: row for row in view["parameters"]}
    assert rows["beta"]["estimate"] and rows["sigma"]["estimate"]
    # Observed data arrives already split into the form's per-column text.
    assert view["observables"] == "Infl, Rate"
    assert len(view["dataVectors"]["Infl"].splitlines()) == 10
    # The run's own settings come back, so re-running reproduces it.
    assert (view["nDraws"], view["burnIn"], view["thin"]) == (20, 5, 1)

    # A pipeline with no result: the spec is the only evidence of an MC run in
    # the bundle, so it has to carry enough for the canvas to draw the graph.
    assert ws.mc.spec is not None
    # The slot holds the live pipeline's own spec, which keys nodes by step name.
    assert [node["name"] for node in ws.mc.spec["nodes"]] == ["sim"]
    assert ws.mc.spec["edges"] == []
    assert ws.mc.result is None  # no MC result was attached at build time
    assert ws.mc.view is None  # a bundle stores the pipeline, not the canvas
    assert ws.simulation["reference"].spec is not None
    assert ws.simulation["reference"].spec["T"] == 8
    assert ws.simulation["reference"].spec["shocks"]["e_u"]["seed"] == 42


def test_prefill_restores_the_settings_a_run_was_made_with() -> None:
    """A bundled experiment has to re-run as it ran, not at form defaults.

    Includes the options the form renders no control for: leaving those to
    default would silently substitute a different run behind an unchanged
    screen.
    """
    parser = ModelParser.from_string(_MODEL_YAML)
    model, kalman = parser.get_all()
    compiled = DSGESolver(model, kalman).compile()
    result = MAPResult(
        x=np.array([0.97]),
        theta={"beta": np.float64(0.97)},
        success=True,
        message="ok",
        fun=np.float64(1.0),
        nfev=9,
        nit=3,
        optimizer_config={
            "theta0": [0.93],
            "method": "Nelder-Mead",
            "bounds": [[0.9, 0.999]],
            "options": {
                "maxiter": 250,
                "xatol": 1e-7,
                "jacobian": True,
                "cov": False,
                "cov_fd_step_scale": 2.5,
            },
        },
        logpost=np.float64(-1.0),
        logprior=np.float64(-0.1),
    )
    spec = EstimatorSpec(
        y=[[1.0, 2.0]],
        params=_estimation_spec([[1.0, 2.0]]).params | {"estimated_params": ["beta"]},
    )

    view = build_estimation_prefill(spec, result, compiled)

    assert view["method"] == "map"
    assert view["optimizer"] == "Nelder-Mead"
    assert (view["maxIter"], view["xatol"]) == (250, 1e-7)
    # Rendered no control on a fresh form; carried anyway.
    assert view["jacobian"] is True
    assert view["cov"] is False
    assert view["covFdStepScale"] == 2.5
    # The run's own starting point, not the model's calibration.
    beta = next(row for row in view["parameters"] if row["name"] == "beta")
    assert beta["initial"] == 0.93
    assert (beta["lower"], beta["upper"]) == (0.9, 0.999)
    # A key the run never recorded stays absent, for the form to default.
    assert "factr" not in view and "pgtol" not in view


def test_build_workspace_keeps_gui_shape_out_of_the_bundle_slot(
    tmp_path: Path,
) -> None:
    """``spec`` carries no field the form invented.

    The view holds per-row ``estimate``/``initial``/``lower``/``upper`` and an
    inferred ``method``, none of which an ``EstimatorSpec`` has. Keeping them
    out is what lets a bundle write take this slot as it stands.
    """
    ws = build_workspace(build_from(_hydrated_bundle(tmp_path)))

    assert ws.estimation.spec is not None
    assert set(ws.estimation.spec) == {"y", "params"}
    assert "method" not in ws.estimation.spec["params"]
    assert "parameters" not in ws.estimation.spec["params"]


def test_workspace_spec_slot_goes_straight_into_a_bundle(tmp_path: Path) -> None:
    """The point of the split: no projection back out of the GUI shape.

    ``spec`` is already the constructor form a bundle stores, so writing one
    is reading the slot and handing it over. If the GUI shape had leaked in,
    this would need a reverse mapping to strip it.
    """
    ws = build_workspace(build_from(_hydrated_bundle(tmp_path)))
    assert ws.estimation.spec is not None

    written = (
        BundleBuilder(created_by="round-trip")
        .add_model("reference", _MODEL_YAML, compile_kwargs={})
        .add_estimation(
            Estimator.from_spec(
                EstimatorSpec(
                    y=ws.estimation.spec["y"], params=ws.estimation.spec["params"]
                ),
                _compiled_reference(),
            )
        )
        .write(tmp_path / "round-trip.sdsge")
    )

    reloaded = build_from(written)
    assert reloaded.estimation is not None
    assert reloaded.estimation.estimator.estimated_params == ["beta", "sigma"]
    assert len(reloaded.estimation.estimator.y) == 10


# -- bundled simulation replay ---------------------------------------------


def test_bundled_simulation_replays_into_an_output(tmp_path: Path) -> None:
    """A stored spec becomes the output it stands for.

    A bundle keeps no simulation results, and the Outputs tab's only controls
    are spec fields, so without this there is nothing on screen to show a
    simulation is in the bundle at all.
    """
    loaded = build_from(_hydrated_bundle(tmp_path))
    app = create_app(
        reference=loaded.reference,
        dgp=loaded.dgp,
        workspace=build_workspace(loaded),
    )
    client = TestClient(app)

    body = client.get("/api/session").json()

    simulation = body["workspace"]["simulation"]["reference"]
    assert simulation["spec"]["T"] == 8
    result = simulation["result"]
    assert result["kind"] == "sim" and result["T"] == 8
    assert {"Infl", "Rate"} <= {series["name"] for series in result["series"]}
    # The replay lands in the tab's own result slot, which is where a run lives.
    assert simulation["result"]["role"] == "reference"


def test_bundled_simulation_replay_reproduces_rather_than_redraws(
    tmp_path: Path,
) -> None:
    """The spec pins the seed, so two replays of it agree."""
    loaded = build_from(_hydrated_bundle(tmp_path))

    first = create_app(reference=loaded.reference, workspace=build_workspace(loaded))
    second = create_app(reference=loaded.reference, workspace=build_workspace(loaded))

    def series(app: Any) -> Any:
        payload = TestClient(app).get("/api/session").json()
        return payload["workspace"]["simulation"]["reference"]["result"]["series"]

    assert series(first) == series(second)


def test_a_simulation_that_cannot_replay_leaves_the_session_usable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One bad spec must not cost the tabs that had nothing to do with it."""
    loaded = build_from(_hydrated_bundle(tmp_path))
    workspace = build_workspace(loaded)
    assert workspace.simulation["reference"].spec is not None
    workspace.simulation["reference"].spec["shocks"] = {
        "not_a_shock": {
            "dist": "norm",
            "multivar": False,
            "seed": 1,
            "dist_args": [],
            "dist_kwargs": {},
        }
    }

    app = create_app(reference=loaded.reference, workspace=workspace)

    assert "could not replay" in capsys.readouterr().out
    payload = TestClient(app).get("/api/session").json()
    # The spec survives for inspection, the result is simply absent, and the
    # estimation tab is untouched by any of it.
    assert "result" not in payload["workspace"]["simulation"]["reference"]
    assert payload["workspace"]["estimation"]["result"] is not None


# -- workspace view updates ------------------------------------------------


def test_workspace_view_round_trips_through_the_session() -> None:
    """What the client PUTs is what a reload reads back.

    This is the whole restore mechanism: the process outlives the refresh, so
    the view returns from server memory with nothing kept on the client.
    """
    client = TestClient(create_app())
    view = {"method": "mcmc", "nDraws": 4000, "dataVectors": {"Infl": "1 2"}}

    ack = client.put("/api/session/workspace", json={"tab": "estimation", "view": view})

    assert ack.status_code == 200 and ack.json() == {"tab": "estimation"}
    reread = client.get("/api/session").json()["workspace"]
    assert reread["estimation"] == {"view": view}


def test_workspace_view_is_held_verbatim() -> None:
    """The server holds the view without modelling it, so a new control lands."""
    client = TestClient(create_app())
    view = {"a_control_python_never_heard_of": [1, {"nested": True}]}

    client.put("/api/session/workspace", json={"tab": "mc", "view": view})

    assert client.get("/api/session").json()["workspace"]["mc"]["view"] == view


def test_workspace_view_cannot_write_the_bundle_bound_slots() -> None:
    """A client naming ``spec`` or ``result`` is rejected, not partly obeyed."""
    client = TestClient(create_app())

    refused = client.put(
        "/api/session/workspace",
        json={"tab": "estimation", "view": {}, "spec": {"y": []}},
    )

    assert refused.status_code == 422


def test_workspace_view_clears_when_set_to_null() -> None:
    client = TestClient(create_app())
    client.put("/api/session/workspace", json={"tab": "estimation", "view": {"a": 1}})

    client.put("/api/session/workspace", json={"tab": "estimation", "view": None})

    assert client.get("/api/session").json()["workspace"] == {}


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
    # A model handed over in process has no origin to cite.
    assert captured.get("source") is None


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
    assert captured["source"] == str(bundle)


def test_preloaded_model_reports_its_source_and_yaml(tmp_path: Path) -> None:
    """A bundle launch identifies the model by path and opens on its YAML.

    Both ride the summary the GUI already reads: ``source`` distinguishes one
    preloaded model from another, and ``raw_yaml`` is what the Builder tab
    seeds its editor from.
    """
    bundle = _hydrated_bundle(tmp_path)
    loaded = build_from(bundle)
    app = create_app(
        reference=loaded.reference,
        dgp=loaded.dgp,
        workspace=build_workspace(loaded),
        source=str(bundle),
    )

    reference = TestClient(app).get("/api/session").json()["models"]["reference"]

    assert reference["name"] == "TEST"
    assert reference["source"] == str(bundle)
    assert reference["raw_yaml"].startswith('name: "TEST"')


def test_in_process_model_leaves_source_unset(tmp_path: Path) -> None:
    """``SolvedModel.serve()`` names no origin, so the GUI shows none."""
    app = create_app(reference=_solved_test_model())

    reference = TestClient(app).get("/api/session").json()["models"]["reference"]

    assert reference["source"] is None
    # The YAML still rides the config, so the Builder tab is not left blank.
    assert reference["raw_yaml"].startswith('name: "TEST"')


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


@cache
def _compiled_reference() -> Any:
    """The compiled ``MODELS/test.yaml`` a loaded reference model comes back as."""
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    return DSGESolver(model, kalman).compile()


def _estimator(y: Any) -> Estimator:
    """A live estimator over the bundled model, in the shape a loader rebuilds.

    ``MODELS/test.yaml`` declares no ``kalman:`` section, so ``R`` is passed
    explicitly; without one the estimator a bundle describes cannot be built.
    """
    return Estimator(
        compiled=_compiled_reference(),
        y=np.asarray(y, dtype=np.float64),
        observables=["Infl", "Rate"],
        estimated_params=["beta", "sigma"],
        R=np.eye(2) * 1e-4,
    )


def _estimation_source(spec: EstimatorSpec) -> Any:
    """Stands in for the live estimator: ``add_estimation`` asks only for its spec.

    These tests are about how a spec is encoded into members, not about building
    an estimator, so they hand over the spec without the model behind it.
    """
    return SimpleNamespace(to_spec=lambda: spec)
