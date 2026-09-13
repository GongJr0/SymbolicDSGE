"""Serializable Monte Carlo pipeline specification."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from collections.abc import Callable, Mapping
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

#: Authoritative set of valid step-type strings. Every kind here must appear in
#: :data:`OP_TYPES`; ``tests/monte_carlo/test_from_spec.py`` enforces the parity.
STEP_KINDS: frozenset[str] = frozenset(get_args(MCStepKind))

#: Post-loop step kinds. A postproc is a *terminal reduction* over the assembled
#: across-rep traces, so it lives in :attr:`PipelineSpec.postproc_steps`, never in
#: :attr:`PipelineSpec.replication_steps`.
PostprocStepKind = Literal["kde", "postproc:custom"]
POSTPROC_KINDS: frozenset[str] = frozenset(get_args(PostprocStepKind))


#: The op kind each step kind is. This is what a step *is*, not how a form renders
#: it, so it stays here beside the rest of the kind taxonomy. A client declares a
#: step's ``op_type``; native lowering dispatches on it and then on the step kind
#: within it, so a pair this map does not name has no branch to land in.
OP_TYPES: dict[str, str] = {
    "simulation": "datagen",
    "raw_model_data": "datagen",
    "filter": "filter",
    "payload": "transform",
    "standardize": "transform",
    "log": "transform",
    "log_diff": "transform",
    "diff": "transform",
    "rolling_mean": "transform",
    "rolling_std": "transform",
    "rolling_var": "transform",
    "transform:custom": "transform",
    "wald": "test",
    "ljung_box": "test",
    "jarque_bera": "test",
    "breusch_pagan": "test",
    "breusch_godfrey": "test",
    "cusum": "test",
    "cusumsq": "test",
    "chow": "test",
    "regression": "regression",
    "kde": "postproc",
    "postproc:custom": "postproc",
}


class SourceSpec(TypedDict):
    """Serialized form of a :class:`SourceArgs`, one key per constructor argument.

    ``columns`` is a list of explicit indices, an object carrying a slice's
    ``start``/``stop``/``step``, or null for every column.
    """

    arg: str
    source_step: str
    field: str
    columns: list[int] | dict[str, int | None] | None
    burn_in: int


class StepMeta(TypedDict):
    """The JSON-ready half of a :class:`StepSpec`.

    Everything an :class:`MCStep` holds that travels as data. A callable and any
    bulk arrays ride the :class:`StepSpec` around it instead.
    """

    name: str
    op_type: str
    step_type: str | None
    kwargs: dict[str, Any]
    source_args: list[SourceSpec]
    n_retain: int


@dataclass(slots=True)
class StepSpec:
    """Serialized form of an :class:`MCStep`: its meta, plus what JSON cannot hold.

    Attributes
    ----------
    meta : StepMeta
        The step as data. This is what a bundle writes to its spec member.
    func : Callable[..., Any] | None
        The step's callable, for the kinds that carry one. A bundle ships it as
        its own member and hands it back here on load.
    arrays : dict[str, NDArray[Any]]
        Bulk array kwargs, lifted out of ``meta["kwargs"]`` under the names they
        were passed with. A bundle ships these as their own members too.
    """

    meta: StepMeta
    func: Callable[..., Any] | None = None
    arrays: dict[str, NDArray[Any]] = field(default_factory=dict)


def pipeline_meta(spec: "PipelineSpec") -> "PipelineMeta":
    """A pipeline spec as the document a bundle writes.

    Drops what JSON cannot hold: each step's callable and bulk arrays ride their
    own members, so only the metas travel here.
    """
    return PipelineMeta(
        replication_steps=[step.meta for step in spec.replication_steps],
        postproc_steps=[step.meta for step in spec.postproc_steps],
    )


class PipelineMeta(TypedDict):
    replication_steps: list[StepMeta]
    postproc_steps: list[StepMeta]


@dataclass(slots=True)
class PipelineSpec:
    """Serializable spec for a :class:`MCPipeline`.

    Attributes
    ----------
    replication_steps : list[StepSpec]
        The steps run once per replication, each with its own kwargs, source
        bindings, and whatever side channels its meta could not hold.
    postproc_steps : list[StepSpec]
        Post-loop ops, run once over the assembled traces. A separate terminal
        phase, kept out of the per-replication steps.
    """

    replication_steps: list[StepSpec]
    postproc_steps: list[StepSpec]


class MCDataGenResultMeta(TypedDict):
    n_rep: int
    n_retained: int
    var_names: Sequence[str]
    shock_names: Sequence[str]
    observable_names: Sequence[str]
    shapes: Mapping[str, Sequence[int]]


class MCFilterResultMeta(TypedDict):
    n_rep: int
    n_retained: int
    filter_mode: str
    shapes: Mapping[str, Sequence[int]]


class MCTestResultMeta(TypedDict):
    """Reference-distribution and accounting metadata for a serialized test result.

    Attributes
    ----------
    test_name : str
        Name of the test the step ran.
    dist : str
        Reference distribution the p-values were read against.
    df : Any
        Degrees of freedom of the reference distribution, in whatever shape the
        test reports them.
    pval_method : str
        How the p-value was derived from the statistic.
    alpha : float
        Significance level recorded with the run.
    n_retained : int
        Number of replications whose output the step's arena kept.
    n_rep : int
        Total number of replications.
    """

    test_name: str
    dist: str
    df: Any
    pval_method: str
    alpha: float
    n_retained: int
    n_rep: int


@dataclass(slots=True)
class MCTestResultSpec:
    """Serializable form of a Monte Carlo test result.

    Attributes
    ----------
    meta : MCTestResultMeta
        Reference-distribution and accounting metadata.
    statistic_trace : NDF
        Test statistic per retained replication.
    retained_reps : NDI
        Indices of the retained replications relative to the complete run.
    """

    meta: MCTestResultMeta
    statistic_trace: NDF
    _raw_status: NDI
    retained_reps: NDI


class MCRegressionResultMeta(TypedDict):
    """Shape and accounting metadata for a serialized regression result.

    Attributes
    ----------
    kind : str
        Regression method the step ran.
    variables : Sequence[str]
        Shared variable ordering across replications.
    n_retained : int
        Number of replications whose output the step's arena kept.
    n_rep : int
        Total number of replications.
    n : int
        Shared number of observations per replication.
    k : int
        Shared number of design columns.
    """

    kind: str
    variables: Sequence[str]
    n_retained: int
    n_rep: int
    n: int
    k: int


@dataclass(slots=True)
class MCRegressionResultSpec:
    """Serializable form of a Monte Carlo regression result.

    Attributes
    ----------
    meta : MCRegressionResultMeta
        Shape and accounting metadata.
    coef_trace : NDF
        Coefficients stacked by retained replication.
    ssr_trace : NDF
        Per-replication sum of squared residuals.
    sst_trace : NDF
        Per-replication total sum of squares.
    retained_reps : NDI
        Indices of the retained replications relative to the complete run.
    """

    meta: MCRegressionResultMeta
    coef_trace: NDF
    ssr_trace: NDF
    sst_trace: NDF
    retained_reps: NDI
    _raw_status: NDI
    _se_trace: NDF | None = None


class MCFailureSpec(TypedDict):
    rep_idx: int
    step_name: str
    error_type: str
    message: str


class MCRunMeta(TypedDict):
    """A run's own metadata, independent of any one step.

    Carries what the run recorded, never what it can recompute: the throughput
    rates, ``succeeded``, and the failed-step tallies are all properties derived
    from the timings and ``failures`` beside them.
    """

    n_rep: int
    n_successful: int
    n_retained_by_step: Mapping[str, int]
    elapsed_s: float
    step_elapsed_s: Mapping[str, float]
    step_counts: Mapping[str, int]
    step_failures: Mapping[str, int]
    postproc_elapsed_s: Mapping[str, float]
    failures: list[MCFailureSpec]
    run_config: Mapping[str, Any]


class MCTransformResultMeta(TypedDict):
    shape: list[int]


class MCPostprocResultMeta(TypedDict):
    shape: list[int] | None
    summary: Any
