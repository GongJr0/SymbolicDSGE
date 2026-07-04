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
        ],
        steps,
    )
    return pipeline.run(
        reference=_REFERENCE, n_rep=n_rep, fail_fast=fail_fast, verbosity=0
    )


def _postproc(name: str, func: Any, **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.POSTPROC, func=func, kwargs=kwargs)


def test_postproc_runs_once_and_stores_bare_scalar() -> None:
    calls: list[int] = []

    def op(*, traces):
        calls.append(1)
        return 0.75

    result = _run([_postproc("probe", op)], n_rep=5)

    assert len(calls) == 1  # once per run(), not per replication
    # The op's return is stored verbatim -- a plain value, no wrapper in-memory.
    assert result.postproc["probe"] == 0.75


def test_no_postproc_yields_empty_bucket() -> None:
    assert _run([]).postproc == {}


def test_pcs_end_to_end_scalar_and_selection_vector() -> None:
    # PCS (Probability of Correct Selection): across reps, the test with the
    # smallest p-value is "selected"; PCS is the rate at which that index matches
    # the expected one. A worked example, not a built-in — its meaning depends on
    # the upstream tests, so the engine doesn't define it.
    from SymbolicDSGE.monte_carlo.serialize import result_traces

    n_rep = 6
    pval_keys = ["test.jb_y.pval", "test.jb_x.pval"]

    def selection(*, traces, expected):
        mat = np.column_stack([traces[k] for k in pval_keys])
        return Raw((mat.argmin(axis=1) == expected).astype(float))

    def pcs(*, traces, expected):
        mat = np.column_stack([traces[k] for k in pval_keys])
        return float((mat.argmin(axis=1) == expected).mean())

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(n_rep), observable_names=("y", "x")),
            jarque_bera_test_step("jb_y", source="observables", column=0),
            jarque_bera_test_step("jb_x", source="observables", column=1),
        ],
        [
            _postproc("sel", selection, expected=0),
            _postproc("pcs", pcs, expected=0),
        ],
    )
    result = pipeline.run(reference=_REFERENCE, n_rep=n_rep, verbosity=0)

    # Recompute independently from the stored traces (same arrays the op saw).
    mat = np.column_stack([result_traces(result)[k] for k in pval_keys])
    indicator = (mat.argmin(axis=1) == 0).astype(float)
    expected_pcs = float(indicator.mean())

    # `pcs` returns a plain float; `sel` returns a Raw (both stored verbatim).
    assert result.postproc["pcs"] == pytest.approx(expected_pcs)
    np.testing.assert_array_equal(result.postproc["sel"].value, indicator)
    # PCS is exactly the across-rep mean = sum / len of the 0/1 selection vector.
    assert result.postproc["pcs"] == pytest.approx(indicator.sum() / len(indicator))


def test_traces_expose_test_pvals_with_n_successful_length() -> None:
    captured: dict[str, Any] = {}

    def op(*, traces):
        captured["keys"] = set(traces)
        captured["pval"] = traces["test.jb.pval"]
        return Summary(value=float(traces["test.jb.pval"].mean()))

    result = _run([_postproc("probe", op)], n_rep=6)

    assert "test.jb.pval" in captured["keys"]
    assert captured["pval"].shape == (6,)
    assert isinstance(result.postproc["probe"], Summary)


def test_mapping_return_is_stored_nested_and_namespaced_on_serialize() -> None:
    from SymbolicDSGE.monte_carlo.serialize import serialize_pipeline_result

    def op(*, traces):
        return {"sel": Raw(np.zeros(3)), "pcs": Summary(0.4)}

    result = _run([_postproc("m", op)])

    # In-memory: the op's return (a mapping) is stored verbatim under its name.
    assert isinstance(result.postproc["m"]["sel"], Raw)
    assert result.postproc["m"]["pcs"].value == 0.4
    # On serialize, a mapping fans out into namespaced wire entries.
    wire = serialize_pipeline_result(result, run_id="t")["postproc"]
    assert set(wire) == {"m.sel", "m.pcs"}
    assert wire["m.pcs"]["value"] == 0.4


def test_bare_ndarray_stored_verbatim() -> None:
    def op(*, traces):
        return np.arange(3.0)

    stored = _run([_postproc("r", op)]).postproc["r"]
    assert isinstance(stored, np.ndarray)
    np.testing.assert_array_equal(stored, np.arange(3.0))


def test_postproc_failure_respects_fail_fast() -> None:
    def boom(*, traces):
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
    def pcs(*, traces, expected):
        matrix = np.column_stack([traces["test.jb0.pval"], traces["test.jb1.pval"]])
        selected = matrix.argmin(axis=1)
        return Summary(value=float((selected == expected).mean()))

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(8), observable_names=("y", "x")),
            jarque_bera_test_step("jb0", source="observables", column=0),
            jarque_bera_test_step("jb1", source="observables", column=1),
        ],
        [_postproc("pcs", pcs, expected=0)],
    )
    result = pipeline.run(reference=_REFERENCE, n_rep=8, verbosity=0)

    value = result.postproc["pcs"].value
    assert isinstance(value, float) and 0.0 <= value <= 1.0


def test_transform_payloads_are_stacked_into_traces() -> None:
    captured: dict[str, Any] = {}

    def op(*, traces):
        captured["has_payload"] = "payload.s" in traces
        captured["shape"] = traces.get("payload.s", np.empty(0)).shape
        return 1.0

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(5), observable_names=("y", "x")),
            standardize_step("s", source="observables"),
        ],
        [_postproc("p", op)],
    )
    pipeline.run(reference=_REFERENCE, n_rep=5, verbosity=0)

    assert captured["has_payload"] is True
    assert captured["shape"][0] == 5  # stacked over replications


def test_postproc_step_is_excluded_from_per_rep_step_counts() -> None:
    result = _run([_postproc("probe", lambda *, traces: 1.0)], n_rep=7)
    # per-rep steps run 7 times and carry it/s rates; the postproc runs once and
    # is timed separately (runtime only, never in the per-rep step maps).
    assert result.meta.step_counts["jb"] == 7
    assert "probe" not in result.meta.step_counts
    assert "probe" not in result.meta.step_elapsed_s
    assert "probe" not in result.meta.step_it_s
    assert "probe" in result.meta.postproc_elapsed_s
    assert result.meta.postproc_total_s == pytest.approx(
        result.meta.postproc_elapsed_s["probe"]
    )


def test_perf_report_separates_postproc_runtime_from_it_s() -> None:
    result = _run([_postproc("pp", lambda *, traces: 1.0)], n_rep=5)

    # verbosity=1: pipeline it/s, then postproc *total runtime* (no it/s for it).
    lines: list[str] = []
    result.report_performance(print_func=lines.append)
    assert lines[0].endswith(" it/s.")
    assert lines[1].startswith("Post-processing concluded successfully in ")
    assert lines[1].endswith("s.") and "it/s" not in lines[1]

    # verbosity=2: per-rep steps as it/s; postproc in its own section as runtime.
    lines.clear()
    result.report_step_performance(print_func=lines.append)
    assert lines[0].startswith("MC run concluded")
    assert any("jb" in line and line.endswith(" it/s.") for line in lines)
    assert any("Post-processing Report" in line for line in lines)
    pp_line = next(line for line in lines if line.strip().startswith("pp:"))
    assert pp_line.endswith("s.") and "it/s" not in pp_line


def test_kde_builtin_runs_and_returns_curve_and_descriptives() -> None:
    import pandas as pd

    from SymbolicDSGE.monte_carlo.operations.postproc import kde_step

    pipeline = MCPipeline(
        [
            raw_data_step(observables=_observables(12), observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
        ],
        [kde_step("density", trace="test.jb.statistic", grid_points=64)],
    )
    result = pipeline.run(reference=_REFERENCE, n_rep=12, verbosity=0)

    # kde returns a mapping; stored verbatim, so its entries nest under "density".
    curve = result.postproc["density"]["curve"]
    assert isinstance(curve, Raw)
    assert curve.value.shape == (64, 2)  # (x, density)

    desc = result.postproc["density"]["descriptives"]
    assert isinstance(desc, Summary) and desc.render == "table"
    assert isinstance(desc.value, pd.DataFrame)
    assert list(desc.value["statistic"]) == [
        "count",
        "mean",
        "std",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    ]
    assert desc.value.loc[0, "value"] == 12.0  # count == n_rep


def test_runtime_traces_match_available_registry() -> None:
    # The static registry (#179) must equal the keys a run actually produces.
    from SymbolicDSGE.monte_carlo.operations.postproc import postproc_step
    from SymbolicDSGE.monte_carlo.traces import available_traces

    captured: dict[str, set] = {}

    def probe(*, traces):
        captured["keys"] = set(traces)
        return 0.0

    pipe = MCPipeline(
        [
            raw_data_step(
                "dat", observables=_observables(5), observable_names=("y", "x")
            ),
            jarque_bera_test_step("jb", source="observables", column=0),
            standardize_step("s", source="observables"),
        ],
        [postproc_step("probe", probe)],
    )
    pipe.run(reference=_REFERENCE, n_rep=5, verbosity=0)

    assert captured["keys"] == set(available_traces(pipe.to_spec()))
