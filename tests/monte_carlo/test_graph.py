from __future__ import annotations

import pytest

from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.operations.core import (
    reference_filter_step,
    simulation_step,
)
from SymbolicDSGE.monte_carlo.operations.tests import (
    breusch_pagan_test_step,
    jarque_bera_test_step,
    wald_test_step,
)
from SymbolicDSGE.monte_carlo.operations.transforms import standardize_step


def test_linear_pipeline_graph_root_leaf_and_edges() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            reference_filter_step("filter"),
            wald_test_step("w", source="std_innov"),
        ]
    )
    g = pipe.graph

    assert g.root == "dgp"
    assert list(g.nodes) == ["dgp", "filter", "w"]
    assert g.nodes["dgp"].is_root and not g.nodes["dgp"].inputs
    assert g.nodes["filter"].parents == ("dgp",)
    assert g.nodes["w"].parents == ("filter",)
    assert g.nodes["w"].is_leaf
    assert [n.name for n in g.leaves] == ["w"]
    assert g.edges() == [("dgp", "filter"), ("filter", "w")]


def test_op_default_channel_is_resolved_from_catalog() -> None:
    # `wald_test_step("w")` carries no `source`; its catalogue default is
    # "std_innov", which must resolve to the filter producer.
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            reference_filter_step("filter"),
            wald_test_step("w"),
        ]
    )
    edge = pipe.graph.nodes["w"].inputs[0]
    assert edge.channel == "std_innov"
    assert edge.producer == "filter"


def test_branching_filter_has_two_children() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            reference_filter_step("filter"),
            wald_test_step("w", source="std_innov"),
            jarque_bera_test_step("jb", source="innov"),
        ]
    )
    g = pipe.graph
    assert set(g.nodes["filter"].children) == {"w", "jb"}
    assert {n.name for n in g.leaves} == {"w", "jb"}


def test_transform_chain_resolves_payload_producers() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            standardize_step("s1", source="observables"),
            standardize_step("s2", source="payload", payload_key="s1"),
            jarque_bera_test_step("jb", source="payload", payload_key="s2"),
        ]
    )
    g = pipe.graph
    assert g.nodes["s1"].parents == ("dgp",)
    assert g.nodes["s2"].parents == ("s1",)
    assert g.nodes["jb"].parents == ("s2",)
    assert g.nodes["jb"].is_leaf


def test_multi_input_step_records_both_producers() -> None:
    # Breusch-Pagan reads residual (filter) + regressors (datagen).
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            reference_filter_step("filter"),
            breusch_pagan_test_step(
                "bp", residual_source="std_innov", X_source="observables"
            ),
        ]
    )
    bp = pipe.graph.nodes["bp"]
    assert set(bp.parents) == {"filter", "dgp"}
    channels = {e.role: (e.producer, e.channel) for e in bp.inputs}
    assert channels["residual_source"] == ("filter", "std_innov")
    assert channels["X_source"] == ("dgp", "observables")
    # the filter producer is preferred as the structural edge
    assert bp.primary_parent == "filter"


def test_graph_is_cached() -> None:
    pipe = MCPipeline([simulation_step("dgp", T=4)])
    assert pipe.graph is pipe.graph


def test_unknown_payload_producer_raises() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            jarque_bera_test_step("jb", source="payload", payload_key="ghost"),
        ]
    )
    with pytest.raises(ValueError, match="unknown producer 'ghost'"):
        _ = pipe.graph


def test_forward_reference_raises() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            jarque_bera_test_step("jb", source="payload", payload_key="s1"),
            standardize_step("s1", source="observables"),
        ]
    )
    with pytest.raises(ValueError, match="does not appear earlier"):
        _ = pipe.graph
