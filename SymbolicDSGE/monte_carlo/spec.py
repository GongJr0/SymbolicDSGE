"""Serializable Monte Carlo pipeline specification (graph form).

Stdlib dataclasses. The core ``monte_carlo`` module must stay pydantic-free
(pydantic is only present transitively under the ``[ui]`` extra). The UI keeps its
pydantic request models and converts via :meth:`PipelineSpec.from_dict`. This is the
text representation a ``.sdsge`` bundle stores for the MC pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, get_args

MCStepKind = Literal[
    # datagen / filter
    "simulation",
    "raw_model_data",
    "filter",
    # terminal: tests
    "wald",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "cusum",
    "cusumsq",
    "chow",
    # terminal: regression
    "regression",
    # transforms
    "standardize",
    "log",
    "log_diff",
    "diff",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "payload",
    # post-processing (post-loop ops over across-rep traces)
    "kde",
    # custom (user-supplied ops, shipped as cloudpickle bundle members); the
    # prefix records the op role since a custom op may be a transform or a postproc.
    "transform:custom",
    "postproc:custom",
]

#: Authoritative set of valid step-type strings. Must agree with the keys of
#: :data:`SymbolicDSGE.monte_carlo.catalog.STEP_CATALOG`. There's a regression
#: test in ``tests/monte_carlo/test_catalog_builder.py`` that enforces parity.
STEP_KINDS: frozenset[str] = frozenset(get_args(MCStepKind))

#: Post-loop step kinds. A postproc is a *terminal reduction* over the assembled
#: across-rep traces, not a graph node. It lives in ``PipelineSpec.postprocs``,
#: never in ``nodes``. Keep in sync with ``catalog.POSTPROC_STEP_TYPES`` + the
#: custom postproc kind (guarded by the catalog parity test).
PostprocStepKind = Literal["kde", "postproc:custom"]
POSTPROC_KINDS: frozenset[str] = frozenset(get_args(PostprocStepKind))

#: Per-replication step kinds (everything that is an actual graph node).
PER_REP_KINDS: frozenset[str] = STEP_KINDS - POSTPROC_KINDS


class NodeSpec(TypedDict):
    id: str
    step_type: str
    name: str
    params: dict[str, Any]


class EdgeSpec(TypedDict):
    source: str
    target: str


class PostprocSpec(TypedDict):
    """A post-loop op: a named, typed, parameterized terminal reduction over the
    assembled across-rep traces. Deliberately *not* a graph node. It has no
    ``id`` and no edges; its inputs are trace keys carried in ``params``.
    """

    name: str
    step_type: str
    params: dict[str, Any]


class PipelineSpec(TypedDict):
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    #: Post-loop ops, run once over the assembled traces. Kept separate from the
    #: per-rep DAG (``nodes``/``edges``). They are not graph participants.
    postprocs: list[PostprocSpec]
