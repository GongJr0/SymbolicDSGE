"""Loader coverage driven by a real .sdsge fixture + targeted rebuild branches."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE.bundle import loader as L
from SymbolicDSGE.bundle.loader import (
    LoadedBundle,
    LoadedEstimation,
    LoadedMC,
    build_from,
)
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult, MAPResult, OptimizationResult

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bundle_fixture.sdsge"


def test_build_from_fixture_end_to_end():
    loaded = build_from(FIXTURE)
    assert isinstance(loaded, LoadedBundle)
    # models re-parsed + re-solved
    assert isinstance(loaded.reference, SolvedModel)
    assert isinstance(loaded.dgp, SolvedModel)
    # estimation: MCMC result rebuilt from metadata + posterior traces
    assert isinstance(loaded.estimation, LoadedEstimation)
    assert isinstance(loaded.estimation.result, MCMCResult)
    assert loaded.estimation.observed is not None
    assert loaded.estimation.observed.ndim == 2
    assert loaded.estimation.posterior is not None
    # monte carlo: runnable pipeline + document + traces
    assert isinstance(loaded.mc, LoadedMC)
    assert loaded.mc.pipeline is not None


_OPT_META = {
    "x": [1.0, -2.0],
    "theta": {"a": 1.0, "b": -2.0},
    "success": True,
    "message": "converged",
    "fun": 1.5,
    "nfev": 7,
    "nit": 3,
    "optimizer_config": {"method": "L-BFGS-B"},
    "logpost": -1.5,
    "logprior": -0.5,
}

_MCMC_META = {
    "param_names": ["a"],
    "accept_rate": 0.4,
    "n_draws": 3,
    "burn_in": 1,
    "thin": 1,
    "sampler_config": {"adapt": True},
}


def test_rebuild_optimization_result():
    res = MAPResult.from_dict(_OPT_META)
    assert isinstance(res, (MAPResult, OptimizationResult))
    assert res.theta["a"] == pytest.approx(1.0)
    assert np.allclose(res.x, [1.0, -2.0])
    assert res.nfev == 7


def test_rebuild_mcmc_result_requires_traces():
    with pytest.raises(ValueError, match="requires an 'estimation_trace'"):
        L._rebuild_mcmc_result(_MCMC_META, None)
    with pytest.raises(ValueError, match="requires an 'estimation_trace'"):
        L._rebuild_mcmc_result(_MCMC_META, {"samples": np.zeros((3, 1))})  # no logpost


def test_rebuild_mcmc_result_ok():
    posterior = {
        "samples": np.zeros((3, 1), dtype=np.float64),
        "logpost": np.zeros(3, dtype=np.float64),
    }
    res = L._rebuild_mcmc_result(_MCMC_META, posterior)
    assert isinstance(res, MCMCResult)
    assert res.n_draws == 3 and res.thin == 1


def test_load_estimation_optimization_result_dispatch():
    # A non-mcmc estimation_result routes through _rebuild_optimization_result,
    # with no estimation_data / estimation_trace members present.
    spec_member = SimpleNamespace(path="spec.json")
    result_member = SimpleNamespace(path="result.json")

    def members_by_kind(kind):
        if kind == "estimation_spec":
            return [spec_member]
        if kind == "estimation_result":
            return [result_member]
        return []

    manifest = SimpleNamespace(members_by_kind=members_by_kind)
    spec_json = (
        '{"method": "mle", '
        '"parameters": [{"name": "a", "initial": 0.0, "estimate": true}]}'
    )
    result_json = json.dumps({"type": "map", "data": _OPT_META})

    archive = SimpleNamespace(
        read_text=lambda path: spec_json if path == "spec.json" else result_json
    )

    loaded = L._load_estimation(archive, manifest)
    assert isinstance(loaded.result, OptimizationResult)
    assert loaded.observed is None
    assert loaded.posterior is None


def test_load_mc_postproc_nan_fallback(monkeypatch):
    member = SimpleNamespace(path="pp", options={"name": "art", "shape": [2, 2]})
    manifest = SimpleNamespace(
        members_by_kind=lambda kind: [member] if kind == "mc_postproc" else []
    )
    archive = SimpleNamespace(read=lambda path: b"")

    def _raise(raw, shapes):
        raise KeyError("a")

    monkeypatch.setattr(L, "arrays_from_parquet", _raise)
    out = L._load_mc_postproc(archive, manifest)
    assert out["art"].shape == (2, 2)
    assert np.all(np.isnan(out["art"]))
