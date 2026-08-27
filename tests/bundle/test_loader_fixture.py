"""Loader coverage driven by a real .sdsge fixture + targeted rebuild branches."""

from __future__ import annotations

import json
from functools import cache
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
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult, MAPResult, OptimizationResult

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bundle_fixture.sdsge"


@cache
def _compiled_reference():
    """A real compiled model for the estimator the loader now builds eagerly.

    The archive and manifest below stay mocked, since the branch under test is
    the loader's result dispatch. The model does not: ``_load_estimation``
    constructs the estimator the spec describes, so a stand-in that cannot be
    bound to would test a load that never happens.
    """
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    return DSGESolver(model, kalman).compile()


def test_build_from_fixture_end_to_end():
    loaded = build_from(FIXTURE)
    assert isinstance(loaded, LoadedBundle)
    # models re-parsed + re-solved
    assert isinstance(loaded.reference, SolvedModel)
    assert isinstance(loaded.dgp, SolvedModel)
    # estimation: MCMC result rebuilt from metadata + posterior traces
    assert isinstance(loaded.estimation, LoadedEstimation)
    assert isinstance(loaded.estimation.result, MCMCResult)
    y = np.asarray(loaded.estimation.estimator.y)
    assert y.ndim == 2
    assert loaded.estimation.result.samples.ndim == 2
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
    res = MAPResult.from_spec(_OPT_META)
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
        "logjac": np.zeros(3, dtype=np.float64),
    }
    res = L._rebuild_mcmc_result(_MCMC_META, posterior)
    assert isinstance(res, MCMCResult)
    assert res.n_draws == 3 and res.thin == 1


def test_load_estimation_optimization_result_dispatch():
    # A non-mcmc estimation_result routes through _rebuild_optimization_result,
    # with no estimation_data / estimation_trace members present.
    spec_member = SimpleNamespace(path="spec.json")
    result_member = SimpleNamespace(path="result.json")

    data_member = SimpleNamespace(path="observed.csv", format="csv", columns=None)

    def members_by_kind(kind):
        if kind == "estimation_spec":
            return [spec_member]
        if kind == "estimation_result":
            return [result_member]
        if kind == "estimation_data":
            return [data_member]
        return []

    manifest = SimpleNamespace(members_by_kind=members_by_kind)
    spec_json = json.dumps(
        {
            "observables": ["Infl", "Rate"],
            "filter_mode": "linear",
            "P0": None,
            "R": [[1e-4, 0.0], [0.0, 1e-4]],
            "estimated_params": ["beta"],
            "priors": None,
            "ss_seed": None,
            "x0": None,
            "jitter": 0.0,
            "symmetrize": True,
            "joseph_cov": True,
        }
    )
    result_json = json.dumps({"type": "map", "data": _OPT_META})

    archive = SimpleNamespace(
        read_text=lambda path: spec_json if path == "spec.json" else result_json,
        read=lambda path: b"y.0,y.1\n1.0,2.0\n3.0,4.0\n",
    )
    reference = SimpleNamespace(compiled=_compiled_reference())

    loaded = L._load_estimation(archive, manifest, reference)
    assert isinstance(loaded.result, OptimizationResult)
    np.testing.assert_allclose(np.asarray(loaded.estimator.y), [[1.0, 2.0], [3.0, 4.0]])

    # An estimation section is not loadable without a model to bind to, nor
    # without the data the estimator conditions on.
    with pytest.raises(ValueError, match="no reference model"):
        L._load_estimation(archive, manifest, None)

    bare = SimpleNamespace(
        members_by_kind=lambda kind: [spec_member] if kind == "estimation_spec" else []
    )
    with pytest.raises(ValueError, match="estimation_data"):
        L._load_estimation(archive, bare, reference)


def test_dropped_column_reads_back_as_the_authors_nans():
    # An all-null float column carries no values, so Parquet drops it. The shape
    # its step's meta recorded is what rebuilds it, NaN-filled as authored.
    out = L._mc_array({}, "art.value", (2, 2))
    assert out.shape == (2, 2)
    assert np.all(np.isnan(out))
