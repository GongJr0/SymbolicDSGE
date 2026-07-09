from __future__ import annotations

import numpy as np
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
            wald_test_step(
                "w",
                source="filter",
                field="std_innov",
                target=np.zeros(1, dtype=np.float64),
            ),
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


def test_source_is_required_for_source_consumers() -> None:
    with pytest.raises(TypeError, match="source"):
        wald_test_step("w")


def test_branching_filter_has_two_children() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            reference_filter_step("filter"),
            wald_test_step(
                "w",
                source="filter",
                field="std_innov",
                target=np.zeros(1, dtype=np.float64),
            ),
            jarque_bera_test_step("jb", source="filter", field="innov"),
        ]
    )
    g = pipe.graph
    assert set(g.nodes["filter"].children) == {"w", "jb"}
    assert {n.name for n in g.leaves} == {"w", "jb"}


def test_transform_chain_resolves_payload_producers() -> None:
    pipe = MCPipeline(
        [
            simulation_step("dgp", T=8),
            standardize_step("s1", source="dgp", field="observables"),
            standardize_step("s2", source="s1", field="payload"),
            jarque_bera_test_step("jb", source="s2", field="payload"),
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
                "bp",
                residuals_source="filter",
                residuals_field="std_innov",
                X_source="dgp",
                X_field="observables",
            ),
        ]
    )
    bp = pipe.graph.nodes["bp"]
    assert set(bp.parents) == {"filter", "dgp"}
    channels = {e.role: (e.producer, e.channel) for e in bp.inputs}
    assert channels["residuals"] == ("filter", "std_innov")
    assert channels["X"] == ("dgp", "observables")
    # the filter producer is preferred as the structural edge
    assert bp.primary_parent == "filter"


def test_graph_is_cached() -> None:
    pipe = MCPipeline([simulation_step("dgp", T=4)])
    assert pipe.graph is pipe.graph


def test_unknown_payload_producer_raises() -> None:
    with pytest.raises(ValueError, match="unknown producer 'ghost'"):
        MCPipeline(
            [
                simulation_step("dgp", T=8),
                jarque_bera_test_step("jb", source="ghost", field="payload"),
            ]
        )


def test_forward_reference_raises() -> None:
    with pytest.raises(ValueError, match="does not appear earlier"):
        MCPipeline(
            [
                simulation_step("dgp", T=8),
                jarque_bera_test_step("jb", source="s1", field="payload"),
                standardize_step("s1", source="dgp", field="observables"),
            ]
        )
