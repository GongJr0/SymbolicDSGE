"""Serializable Monte Carlo pipeline specification (graph form)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping
from typing import Any, Literal, Sequence, TypedDict, get_args
from numpy.typing import NDArray
from numpy import float64, int_

NDF = NDArray[float64]
NDI = NDArray[int_]

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


class MCTestResultMeta(TypedDict):
    test_name: str
    dist: str
    df: Any
    pval_method: str
    alpha: float
    n_retained: int
    n_rep: int


@dataclass(slots=True)
class MCTestResultSpec:
    meta: MCTestResultMeta
    statistic_trace: NDF
    _raw_status: NDI
    retained_reps: NDI


class MCRegressionResultMeta(TypedDict):
    kind: str
    variables: Sequence[str]
    n_retained: int
    n_rep: int
    n: int
    k: int


@dataclass(slots=True)
class MCRegressionResultSpec:
    meta: MCRegressionResultMeta
    coef_trace: NDF
    ssr_trace: NDF
    sst_trace: NDF
    retained_reps: NDI
    _raw_status: NDI
    _se_trace: NDF | None = None
