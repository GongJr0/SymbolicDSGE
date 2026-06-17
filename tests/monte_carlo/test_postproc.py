from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline, MCStep, OpType, Raw, Summary
from SymbolicDSGE.monte_carlo.operations.core import raw_data_step
from SymbolicDSGE.monte_carlo.operations.tests import jarque_bera_test_step
from SymbolicDSGE.monte_carlo.operations.transforms import standardize_step

_REFERENCE = cast(SolvedModel, object())


def _observables(n_rep: int = 4, T: int = 40, k: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_rep, T, k))


def _run(steps: list[MCStep], *, n_rep: int = 4, fail_fast: bool = True):
    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(n_rep), observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
            *steps,
        ]
    )
    return pipeline.run(
        reference=_REFERENCE, n_rep=n_rep, fail_fast=fail_fast, verbosity=0
    )


def _postproc(name: str, func: Any, **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.POSTPROC, func=func, kwargs=kwargs)


def test_postproc_runs_once_and_wraps_a_bare_scalar() -> None:
    calls: list[int] = []

    def op(*, traces, reference, dgp):
        calls.append(1)
        return 0.75

    result = _run([_postproc("probe", op)], n_rep=5)

    assert len(calls) == 1  # once per run(), not per replication
    artifact = result.postproc["probe"]
    assert isinstance(artifact, Summary) and artifact.value == 0.75


def test_no_postproc_yields_empty_bucket() -> None:
    assert _run([]).postproc == {}


def test_traces_expose_test_pvals_with_n_successful_length() -> None:
    captured: dict[str, Any] = {}

    def op(*, traces, reference, dgp):
        captured["keys"] = set(traces)
        captured["pval"] = traces["test.jb.pval"]
        return Summary(value=float(traces["test.jb.pval"].mean()))

    result = _run([_postproc("probe", op)], n_rep=6)

    assert "test.jb.pval" in captured["keys"]
    assert captured["pval"].shape == (6,)
    assert isinstance(result.postproc["probe"], Summary)


def test_mapping_return_yields_namespaced_artifacts() -> None:
    def op(*, traces, reference, dgp):
        return {"sel": Raw(np.zeros(3)), "pcs": Summary(0.4)}

    result = _run([_postproc("m", op)])

    assert isinstance(result.postproc["m.sel"], Raw)
    assert isinstance(result.postproc["m.pcs"], Summary)
    assert result.postproc["m.pcs"].value == 0.4


def test_bare_ndarray_wraps_as_raw() -> None:
    def op(*, traces, reference, dgp):
        return np.arange(3.0)

    artifact = _run([_postproc("r", op)]).postproc["r"]
    assert isinstance(artifact, Raw)
    np.testing.assert_array_equal(artifact.value, np.arange(3.0))


def test_postproc_failure_respects_fail_fast() -> None:
    def boom(*, traces, reference, dgp):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        _run([_postproc("b", boom)], fail_fast=True)

    result = _run([_postproc("b", boom)], fail_fast=False)
    assert result.succeeded is False
    assert any(f.step_name == "b" and f.rep_idx == -1 for f in result.failures)
    assert "b" not in result.postproc


def test_postproc_receives_step_kwargs_and_can_colstack_traces() -> None:
    # PCS-style: stack two tests' p-value traces (R x 2), argmin per row, compare
    # to the expected index, mean = correct-selection rate.
    def pcs(*, traces, reference, dgp, expected):
        matrix = np.column_stack([traces["test.jb0.pval"], traces["test.jb1.pval"]])
        selected = matrix.argmin(axis=1)
        return Summary(value=float((selected == expected).mean()))

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(8), observable_names=("y", "x")),
            jarque_bera_test_step("jb0", source="observables", column=0),
            jarque_bera_test_step("jb1", source="observables", column=1),
            _postproc("pcs", pcs, expected=0),
        ]
    )
    result = pipeline.run(reference=_REFERENCE, n_rep=8, verbosity=0)

    value = result.postproc["pcs"].value
    assert isinstance(value, float) and 0.0 <= value <= 1.0


def test_transform_payloads_are_stacked_into_traces() -> None:
    captured: dict[str, Any] = {}

    def op(*, traces, reference, dgp):
        captured["has_payload"] = "payload.s" in traces
        captured["shape"] = traces.get("payload.s", np.empty(0)).shape
        return 1.0

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(5), observable_names=("y", "x")),
            standardize_step("s", source="observables"),
            _postproc("p", op),
        ]
    )
    pipeline.run(reference=_REFERENCE, n_rep=5, verbosity=0)

    assert captured["has_payload"] is True
    assert captured["shape"][0] == 5  # stacked over replications


def test_postproc_step_is_excluded_from_per_rep_step_counts() -> None:
    result = _run([_postproc("probe", lambda *, traces, reference, dgp: 1.0)], n_rep=7)
    # per-rep steps run 7 times; the postproc runs exactly once.
    assert result.step_counts["jb"] == 7
    assert result.step_counts["probe"] == 1


def test_kde_builtin_runs_and_returns_raw_curve() -> None:
    from SymbolicDSGE.monte_carlo.operations.postproc import kde_step

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(12), observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
            kde_step("density", trace="test.jb.statistic", grid_points=64),
        ]
    )
    result = pipeline.run(reference=_REFERENCE, n_rep=12, verbosity=0)

    art = result.postproc["density"]
    assert isinstance(art, Raw)
    assert art.value.shape == (64, 2)  # (x, density)
