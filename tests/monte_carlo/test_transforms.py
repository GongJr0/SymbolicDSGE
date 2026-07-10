"""Tests for the built-in transform ops and their graph wiring.

Two layers:

* Unit tests that drive each ``run_*`` transform directly with a small
  hand-built :class:`MCContext` — fast and deterministic.
* Graph tests that exercise the full validate -> compile path through
  :func:`validate_pipeline_spec` / :func:`build_pipeline`, including the
  transform auto-binding rules and chain ordering.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE.monte_carlo import (
    TRANSFORM_STEP_TYPES,
    build_pipeline,
    validate_pipeline_spec,
)
from SymbolicDSGE.monte_carlo.operations.core import simulation_step
from SymbolicDSGE.monte_carlo.operations.transforms import (
    diff_step,
    log_diff_step,
    log_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    standardize_step,
)
from SymbolicDSGE.monte_carlo.mc_constructs import MCContext, MCData, OpType
from SymbolicDSGE.monte_carlo.operations.transforms.ops import (
    run_diff,
    run_log,
    run_log_diff,
    run_rolling_mean,
    run_rolling_std,
    run_rolling_var,
    run_standardize,
)
from SymbolicDSGE.monte_carlo.spec import EdgeSpec, NodeSpec, PipelineSpec

# ---- fixtures ------------------------------------------------------------


class _FakeSolvedModel(SimpleNamespace):
    """Stand-in for a SolvedModel; only the attributes used by transforms."""

    def __init__(self) -> None:
        super().__init__(config=SimpleNamespace(shock_map={}))


def _context(observables: np.ndarray) -> MCContext:
    return MCContext(
        rep_idx=0,
        reference=_FakeSolvedModel(),
        dgp=None,
        data=MCData(states=None, observables=observables),
    )


# ---- unit tests for the ops ---------------------------------------------


def test_run_standardize_zero_centers_and_unit_scales_per_column() -> None:
    obs = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=np.float64
    )
    out = run_standardize(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
    )
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(out[:, 0], out[:, 1])  # affine-equivalent columns


def test_run_standardize_zero_std_column_returns_zeros_safely() -> None:
    obs = np.array([[1.0, 7.0], [1.0, 8.0], [1.0, 9.0]], dtype=np.float64)
    out = run_standardize(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
    )
    np.testing.assert_allclose(out[:, 0], 0.0)
    assert np.isfinite(out).all()


def test_run_log_applies_elementwise_with_offset() -> None:
    obs = np.array([[1.0, np.e], [np.e**2, np.e**3]], dtype=np.float64)
    out = run_log(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
        offset=0.0,
    )
    np.testing.assert_allclose(out, [[0.0, 1.0], [2.0, 3.0]])


def test_run_log_diff_drops_one_row_and_returns_log_returns() -> None:
    obs = np.array([[1.0], [2.0], [4.0], [8.0]], dtype=np.float64)
    out = run_log_diff(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
    )
    np.testing.assert_allclose(out, np.full((3, 1), np.log(2.0)))


def test_run_diff_supports_higher_order() -> None:
    obs = np.array([[1.0], [3.0], [9.0], [27.0]], dtype=np.float64)
    out = run_diff(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
        order=2,
    )
    # 1st diff: [2, 6, 18]; 2nd diff: [4, 12].
    np.testing.assert_allclose(out, [[4.0], [12.0]])


def test_run_diff_rejects_non_positive_order() -> None:
    with pytest.raises(ValueError, match="order must be at least 1"):
        run_diff(
            context=_context(np.zeros((3, 1))),
            reference=_FakeSolvedModel(),
            dgp=None,
            rep_idx=0,
            sample=np.zeros((3, 1), dtype=np.float64),
            order=0,
        )


def test_run_rolling_mean_window_3() -> None:
    obs = np.arange(6.0, dtype=np.float64).reshape(6, 1)
    out = run_rolling_mean(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
        window=3,
    )
    np.testing.assert_allclose(out, [[1.0], [2.0], [3.0], [4.0]])


def test_run_rolling_std_and_var_window_3() -> None:
    obs = np.arange(6.0, dtype=np.float64).reshape(6, 1)
    std_out = run_rolling_std(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
        window=3,
    )
    var_out = run_rolling_var(
        context=_context(obs),
        reference=_FakeSolvedModel(),
        dgp=None,
        rep_idx=0,
        sample=obs,
        window=3,
    )
    np.testing.assert_allclose(std_out**2, var_out)


def test_rolling_window_rejects_window_larger_than_input() -> None:
    obs = np.zeros((3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds input length"):
        run_rolling_mean(
            context=_context(obs),
            reference=_FakeSolvedModel(),
            dgp=None,
            rep_idx=0,
            sample=obs,
            window=10,
        )


# ---- factory smoke tests -------------------------------------------------


@pytest.mark.parametrize(
    "factory, kwargs, expected_runner_kwargs",
    [
        (
            standardize_step,
            {"source": "datagen", "field": "observables"},
            {"ddof": 0},
        ),
        (
            log_step,
            {"source": "datagen", "field": "observables", "offset": 1.0},
            {"offset": 1.0},
        ),
        (
            log_diff_step,
            {"source": "datagen", "field": "observables"},
            {"offset": 0.0},
        ),
        (
            diff_step,
            {"source": "datagen", "field": "observables", "order": 1},
            {"order": 1},
        ),
        (
            rolling_mean_step,
            {"source": "datagen", "field": "observables", "window": 5},
            {"window": 5},
        ),
        (
            rolling_std_step,
            {
                "source": "datagen",
                "field": "observables",
                "window": 5,
            },
            {"window": 5, "ddof": 0},
        ),
        (
            rolling_var_step,
            {"source": "datagen", "field": "observables", "window": 5},
            {"window": 5, "ddof": 0},
        ),
    ],
)
def test_transform_factories_produce_transform_mcstep(
    factory: object, kwargs: dict, expected_runner_kwargs: dict
) -> None:
    step = factory("step_a", **kwargs)  # type: ignore[operator]
    assert step.op_type is OpType.TRANSFORM
    assert step.name == "step_a"
    assert dict(step.kwargs) == expected_runner_kwargs
    assert len(step.source_args) == 1


def test_transform_step_types_set_matches_catalog() -> None:
    assert TRANSFORM_STEP_TYPES == frozenset(
        {
            "standardize",
            "log",
            "log_diff",
            "diff",
            "rolling_mean",
            "rolling_std",
            "rolling_var",
        }
    )


# ---- graph validation / compilation -------------------------------------


def _sim_node() -> NodeSpec:
    return NodeSpec(
        id="sim",
        step_type="simulation",
        name="datagen",
        params={"T": 50, "observables": True},
    )


def test_validate_orders_transform_between_filter_and_terminal() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="lg",
                step_type="log",
                name="log_obs",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="jb",
                step_type="jarque_bera",
                name="normality",
                params={"source": "log_obs", "field": "payload"},
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="lg"),
            EdgeSpec(source="lg", target="jb"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    assert [node.name for node in ordered] == ["datagen", "log_obs", "normality"]
    terminal = ordered[-1]
    assert terminal.params["source"] == "log_obs"
    assert terminal.params["field"] == "payload"


def test_chained_transforms_are_topologically_ordered() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            # Note: spec order intentionally reversed so it's NOT trivially sorted.
            NodeSpec(
                id="b",
                step_type="standardize",
                name="standardize_log",
                params={"source": "log_obs", "field": "payload"},
            ),
            NodeSpec(
                id="a",
                step_type="log",
                name="log_obs",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="t",
                step_type="jarque_bera",
                name="normality",
                params={"source": "standardize_log", "field": "payload"},
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="a"),
            EdgeSpec(source="a", target="b"),
            EdgeSpec(source="b", target="t"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    names = [node.name for node in ordered]
    assert names == ["datagen", "log_obs", "standardize_log", "normality"]
    standardize = next(n for n in ordered if n.name == "standardize_log")
    terminal = ordered[-1]
    assert standardize.params["source"] == "log_obs"
    assert standardize.params["field"] == "payload"
    assert terminal.params["source"] == "standardize_log"
    assert terminal.params["field"] == "payload"


def test_terminal_with_transform_parent_keeps_explicit_source() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="lg",
                step_type="log",
                name="log_obs",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="jb",
                step_type="jarque_bera",
                name="normality",
                params={"source": "datagen", "field": "observables"},
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="lg"),
            EdgeSpec(source="lg", target="jb"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    terminal = ordered[-1]
    assert terminal.params["source"] == "datagen"
    assert terminal.params["field"] == "observables"


def test_multi_input_consumer_without_payload_leg_reads_declared_sources() -> None:
    # A multi-input node may sit downstream of a transform without consuming its
    # payload. It reads the sources its legs declare.
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="d",
                step_type="diff",
                name="diff_obs",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="bp",
                step_type="breusch_pagan",
                name="bp",
                params={
                    "residuals_source": "datagen",
                    "residuals_field": "observables",
                    "X_source": "datagen",
                    "X_field": "observables",
                },
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="d"),
            EdgeSpec(source="d", target="bp"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    bp = ordered[-1]
    assert bp.params["residuals_source"] == "datagen"
    assert bp.params["residuals_field"] == "observables"
    assert bp.params["X_source"] == "datagen"
    assert bp.params["X_field"] == "observables"


def test_multi_input_consumer_with_explicit_payload_source() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="d",
                step_type="diff",
                name="diff_obs",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="bp",
                step_type="breusch_pagan",
                name="bp",
                params={
                    "residuals_source": "diff_obs",
                    "residuals_field": "payload",
                    "X_source": "datagen",
                    "X_field": "observables",
                },
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="d"),
            EdgeSpec(source="d", target="bp"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    bp = ordered[-1]
    assert bp.params["residuals_source"] == "diff_obs"
    assert bp.params["residuals_field"] == "payload"
    assert bp.params["X_source"] == "datagen"
    assert bp.params["X_field"] == "observables"


def test_payload_source_with_dangling_producer_is_rejected() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="jb",
                step_type="jarque_bera",
                name="normality",
                params={"source": "nonexistent", "field": "payload"},
            ),
        ],
        edges=[EdgeSpec(source="sim", target="jb")],
    )
    with pytest.raises(ValueError, match="requires prior source"):
        validate_pipeline_spec(spec, has_reference=True, has_dgp=True)


def test_transform_cannot_link_from_terminal() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="jb",
                step_type="jarque_bera",
                name="normality",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="lg",
                step_type="log",
                name="log_after",
                params={"source": "datagen", "field": "observables"},
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="jb"),
            EdgeSpec(source="jb", target="lg"),
        ],
    )
    with pytest.raises(ValueError, match="Terminal step .* cannot link"):
        validate_pipeline_spec(spec, has_reference=True, has_dgp=True)


def test_transform_fans_out_to_multiple_downstream_chains() -> None:
    """One transform feeds two independent downstream chains.

    Shape: Sim to Standardize to {RollingMean to Wald(avg),
                                  RollingVar to Wald(var)}

    Each consumer takes Standardize as its single parent; Standardize has two
    outgoing edges. Each downstream step names the producer it reads from.
    """
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="std",
                step_type="standardize",
                name="standardize",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="rmean",
                step_type="rolling_mean",
                name="rmean",
                params={"source": "standardize", "field": "payload", "window": 3},
            ),
            NodeSpec(
                id="rvar",
                step_type="rolling_var",
                name="rvar",
                params={"source": "standardize", "field": "payload", "window": 3},
            ),
            NodeSpec(
                id="wmean",
                step_type="wald",
                name="wald_mean",
                params={
                    "kind": "mean",
                    "source": "rmean",
                    "field": "payload",
                    "target_vector": [0.0],
                    "burn_in": 0,
                    "alpha": 0.05,
                },
            ),
            NodeSpec(
                id="wvar",
                step_type="wald",
                name="wald_var",
                params={
                    "kind": "mean",
                    "source": "rvar",
                    "field": "payload",
                    "target_vector": [1.0],
                    "burn_in": 0,
                    "alpha": 0.05,
                },
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="std"),
            EdgeSpec(source="std", target="rmean"),
            EdgeSpec(source="std", target="rvar"),
            EdgeSpec(source="rmean", target="wmean"),
            EdgeSpec(source="rvar", target="wvar"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    names = [node.name for node in ordered]
    # standardize before the two rolling steps; both rolling steps before the
    # two terminals. The relative order within a layer follows spec order.
    assert names[0] == "datagen"
    assert names[1] == "standardize"
    assert set(names[2:4]) == {"rmean", "rvar"}
    assert set(names[4:6]) == {"wald_mean", "wald_var"}

    # Each rolling step's source is standardize's payload.
    rmean = next(n for n in ordered if n.name == "rmean")
    rvar = next(n for n in ordered if n.name == "rvar")
    assert rmean.params["source"] == "standardize"
    assert rmean.params["field"] == "payload"
    assert rvar.params["source"] == "standardize"
    assert rvar.params["field"] == "payload"

    # Each Wald reads its immediate rolling parent, not standardize.
    wmean = next(n for n in ordered if n.name == "wald_mean")
    wvar = next(n for n in ordered if n.name == "wald_var")
    assert wmean.params["source"] == "rmean"
    assert wmean.params["field"] == "payload"
    assert wvar.params["source"] == "rvar"
    assert wvar.params["field"] == "payload"

    # Catalog-driven compile succeeds with the bound params.
    pipeline = build_pipeline(ordered)
    assert [step.name for step in pipeline.per_rep_steps] == names


def test_terminal_can_read_an_earlier_transform_via_explicit_source() -> None:
    """A consumer can reference any earlier transform's payload by name."""
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="std",
                step_type="standardize",
                name="standardize",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="rm",
                step_type="rolling_mean",
                name="rmean",
                params={"source": "standardize", "field": "payload", "window": 3},
            ),
            NodeSpec(
                id="jb",
                step_type="jarque_bera",
                name="normality_on_std",
                # Override: read standardize directly, not rmean.
                params={"source": "standardize", "field": "payload"},
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="std"),
            EdgeSpec(source="std", target="rm"),
            EdgeSpec(source="rm", target="jb"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    jb = next(n for n in ordered if n.name == "normality_on_std")
    assert jb.params["source"] == "standardize"
    assert jb.params["field"] == "payload"


def test_step_kinds_match_catalog() -> None:
    """Every GUI-catalog step kind must be a valid spec `STEP_KIND`; drift would
    silently reject perfectly valid bundles at load time. `STEP_KINDS` may be a
    strict superset: serialization-only datagens (e.g. ``raw_model_data``) are valid
    spec kinds but carry no GUI-authorable `StepDefinition`."""
    from SymbolicDSGE.monte_carlo.catalog import STEP_CATALOG
    from SymbolicDSGE.monte_carlo.spec import STEP_KINDS

    assert frozenset(STEP_CATALOG.keys()) <= STEP_KINDS
    assert STEP_KINDS - frozenset(STEP_CATALOG.keys()) == {
        "raw_model_data",
        "transform:custom",
        "postproc:custom",
    }


def test_transform_pipeline_round_trips_through_bundle(tmp_path) -> None:
    """Authoring a transform-containing pipeline and re-opening it from a
    bundle preserves every node and its bound params."""
    import pathlib

    from SymbolicDSGE import BundleBuilder, load_bundle

    yaml_text = pathlib.Path("MODELS/test.yaml").read_text(encoding="utf-8")
    pipeline = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="std",
                step_type="standardize",
                name="standardize",
                params={"source": "datagen", "field": "observables"},
            ),
            NodeSpec(
                id="rm",
                step_type="rolling_mean",
                name="rmean",
                params={"source": "standardize", "field": "payload", "window": 3},
            ),
            NodeSpec(
                id="wm",
                step_type="wald",
                name="wald_mean",
                params={
                    "kind": "mean",
                    "source": "rmean",
                    "field": "payload",
                    "target_vector": [0.0],
                },
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="std"),
            EdgeSpec(source="std", target="rm"),
            EdgeSpec(source="rm", target="wm"),
        ],
    )

    target = (
        BundleBuilder(created_by="tx-test")
        .add_model("reference", yaml_text, compile_kwargs={})
        .add_mc(pipeline)
        .write(tmp_path / "tx.sdsge")
    )

    loaded = load_bundle(target)
    assert loaded.mc is not None
    restored_step_types = [n.step_type for n in loaded.mc.spec.nodes]
    assert restored_step_types == [
        "simulation",
        "standardize",
        "rolling_mean",
        "wald",
    ]
    # Validation against the restored spec still passes (no drift between the
    # spec's Literal and the catalog at load time).
    ordered, _ = validate_pipeline_spec(
        loaded.mc.spec, has_reference=True, has_dgp=True
    )
    assert [node.name for node in ordered] == [
        "datagen",
        "standardize",
        "rmean",
        "wald_mean",
    ]


def test_build_pipeline_emits_transform_mcstep_with_bound_params() -> None:
    spec = PipelineSpec(
        nodes=[
            _sim_node(),
            NodeSpec(
                id="rm",
                step_type="rolling_mean",
                name="rmean",
                params={"source": "datagen", "field": "observables", "window": 3},
            ),
            NodeSpec(
                id="jb",
                step_type="jarque_bera",
                name="normality",
                params={"source": "rmean", "field": "payload"},
            ),
        ],
        edges=[
            EdgeSpec(source="sim", target="rm"),
            EdgeSpec(source="rm", target="jb"),
        ],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    pipeline = build_pipeline(ordered)
    names = [step.name for step in pipeline.per_rep_steps]
    assert names == ["datagen", "rmean", "normality"]
    rm_step = pipeline.per_rep_steps[1]
    assert rm_step.op_type is OpType.TRANSFORM
    assert rm_step.kwargs["window"] == 3
    jb_step = pipeline.per_rep_steps[2]
    assert jb_step.source_args[0].source_step == "rmean"
