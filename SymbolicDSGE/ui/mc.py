from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np

from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import (
    MCPipeline,
    MCPipelineResult,
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    reference_filter_step,
    regression_step,
    simulation_step,
    wald_test_step,
)
from SymbolicDSGE.monte_carlo.serialize import (
    serialize_pipeline_result as serialize_pipeline_result,
)

from .mc_schemas import MCNodeSpec, MCPipelineSpec

INPUT_SOURCES = [
    "states",
    "observables",
    "x_pred",
    "x_filt",
    "y_pred",
    "y_filt",
    "innov",
    "std_innov",
]
FILTER_SOURCES = {"x_pred", "x_filt", "y_pred", "y_filt", "innov", "std_innov"}
TERMINAL_STEP_TYPES = {
    "wald",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "cusum",
    "cusumsq",
    "chow",
    "regression",
}


def mc_catalog() -> dict[str, Any]:
    return {
        "steps": [
            _step_catalog(
                "simulation",
                "DGP Simulation",
                "datagen",
                "Generate one sample from the DGP model.",
                [
                    _field("T", "Periods", "number", 100, required=True, minimum=1),
                    _field("observables", "Observables", "boolean", True),
                    _field("shock_scale", "Shock scale", "number", 1.0),
                    _field("seed_increment", "Seed increment", "text", "auto"),
                    _field(
                        "distribution",
                        "Distribution",
                        "select",
                        "norm",
                        options=["norm", "t", "uni"],
                    ),
                    _field("seed", "Initial seed", "number", 0),
                    _field("loc", "Location", "number", 0.0),
                    _field("df", "Degrees of freedom", "number", 5.0),
                ],
            ),
            _step_catalog(
                "filter",
                "Reference Filter",
                "filter",
                "Filter generated observables through the reference model.",
                [
                    _field(
                        "filter_mode",
                        "Mode",
                        "select",
                        "linear",
                        options=["linear", "extended"],
                    ),
                    _field("return_shocks", "Return shocks", "boolean", False),
                    _field("estimate_R_diag", "Estimate R diagonal", "boolean", False),
                    _field("R_scale", "R scale", "number", 1.0),
                ],
            ),
            _step_catalog(
                "wald",
                "Wald Test",
                "wald",
                "Run a HAC Wald diagnostic on a selected source.",
                [
                    _field(
                        "source", "Source", "select", "std_innov", options=INPUT_SOURCES
                    ),
                    _field(
                        "kind",
                        "Moment",
                        "select",
                        "mean",
                        options=["mean", "covariance", "second_moment"],
                    ),
                    _field(
                        "target_vector",
                        "Target vector",
                        "number_list",
                        [0.0],
                        required=True,
                        when=["mean"],
                    ),
                    _field(
                        "target_matrix",
                        "Target matrix",
                        "number_matrix",
                        [[1.0]],
                        required=True,
                        when=["covariance", "second_moment"],
                    ),
                    _field("columns", "Columns", "number_list", []),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field(
                        "kernel",
                        "Kernel",
                        "select",
                        "bartlett",
                        options=["bartlett", "parzen", "qs"],
                    ),
                    _field("bandwidth", "Bandwidth", "text", "auto"),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "ljung_box",
                "Ljung-Box Test",
                "ljung_box",
                "Test one selected series for serial correlation.",
                [
                    _field(
                        "source", "Source", "select", "std_innov", options=INPUT_SOURCES
                    ),
                    _field("column", "Column", "number_list", []),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("lags", "Lags", "number", 10, minimum=1),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "jarque_bera",
                "Jarque-Bera Test",
                "jarque_bera",
                "Test one selected series for normality.",
                [
                    _field(
                        "source", "Source", "select", "std_innov", options=INPUT_SOURCES
                    ),
                    _field("column", "Column", "number_list", []),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "breusch_pagan",
                "Breusch-Pagan Test",
                "breusch_pagan",
                "Test residual variance against selected regressors.",
                [
                    _field(
                        "residual_source",
                        "Residual source",
                        "select",
                        "std_innov",
                        options=INPUT_SOURCES,
                    ),
                    _field(
                        "X_source",
                        "Regressor source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field("residual_col", "Residual column", "number_list", [0]),
                    _field("X_columns", "Regressor columns", "number_list", [0]),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("robust", "Robust (Koenker)", "boolean", False),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "breusch_godfrey",
                "Breusch-Godfrey Test",
                "breusch_godfrey",
                "Test residuals for serial correlation up to a given lag order.",
                [
                    _field(
                        "residual_source",
                        "Residual source",
                        "select",
                        "std_innov",
                        options=INPUT_SOURCES,
                    ),
                    _field(
                        "X_source",
                        "Regressor source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field("residual_col", "Residual column", "number_list", [0]),
                    _field("X_columns", "Regressor columns", "number_list", [0]),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("lags", "Lags", "number", 1, minimum=1),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "cusum",
                "CUSUM Test",
                "cusum",
                "Test regression coefficients for stability via recursive residuals.",
                [
                    _field(
                        "y_source",
                        "Response source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field(
                        "x_source",
                        "Regressor source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field("y_column", "Response column", "number_list", [0]),
                    _field("X_columns", "Regressor columns", "number_list", [1]),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "cusumsq",
                "CUSUM of Squares Test",
                "cusumsq",
                "Test regression variance stability via squared recursive residuals.",
                [
                    _field(
                        "y_source",
                        "Response source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field(
                        "x_source",
                        "Regressor source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field("y_column", "Response column", "number_list", [0]),
                    _field("X_columns", "Regressor columns", "number_list", [1]),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "chow",
                "Chow Test",
                "chow",
                "Test for a structural break in regression coefficients at a "
                "known break point.",
                [
                    _field(
                        "y_source",
                        "Response source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field(
                        "x_source",
                        "Regressor source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field("y_column", "Response column", "number_list", [0]),
                    _field("X_columns", "Regressor columns", "number_list", [1]),
                    _field(
                        "t_break",
                        "Break point",
                        "number",
                        10,
                        required=True,
                        minimum=1,
                    ),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("alpha", "Alpha", "number", 0.05),
                ],
            ),
            _step_catalog(
                "regression",
                "Regression",
                "regression",
                "Fit a linear regression in each replication.",
                [
                    _field(
                        "kind",
                        "Kind",
                        "select",
                        "ols",
                        options=[
                            "ols",
                            "ridge",
                            "ridge_gs",
                            "lasso",
                            "lasso_gs",
                            "elastic_net",
                            "elastic_net_gs",
                        ],
                    ),
                    _field(
                        "y_source",
                        "Response source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field(
                        "X_source",
                        "Design source",
                        "select",
                        "observables",
                        options=INPUT_SOURCES,
                    ),
                    _field("y_column", "Response column", "number_list", [0]),
                    _field("X_columns", "Design columns", "number_list", [1]),
                    _field("intercept", "Intercept", "boolean", True),
                    _field("burn_in", "Burn-in", "number", 0, minimum=0),
                    _field("drop_initial", "Drop initial", "boolean", False),
                    _field("variables", "Variable names", "text_list", []),
                    _field(
                        "alpha",
                        "Alpha",
                        "number",
                        0.5,
                        when=["ridge", "lasso", "elastic_net"],
                    ),
                    _field(
                        "l1_ratio",
                        "L1 ratio",
                        "number",
                        0.5,
                        when=["elastic_net", "elastic_net_gs"],
                    ),
                    _field(
                        "start",
                        "Grid start",
                        "number",
                        0.01,
                        when=["ridge_gs", "lasso_gs", "elastic_net_gs"],
                    ),
                    _field(
                        "stop",
                        "Grid stop",
                        "number",
                        2.0,
                        when=["ridge_gs", "lasso_gs", "elastic_net_gs"],
                    ),
                    _field(
                        "num",
                        "Grid points",
                        "number",
                        20,
                        when=["ridge_gs", "lasso_gs", "elastic_net_gs"],
                    ),
                    _field(
                        "criterion",
                        "Criterion",
                        "select",
                        "loss",
                        options=["aic", "bic", "loss"],
                        when=["ridge_gs", "elastic_net_gs"],
                    ),
                    _field(
                        "max_iter",
                        "Max iterations",
                        "number",
                        1000,
                        when=["lasso", "lasso_gs", "elastic_net", "elastic_net_gs"],
                    ),
                    _field(
                        "tol",
                        "Tolerance",
                        "number",
                        1e-10,
                        when=["lasso", "lasso_gs", "elastic_net", "elastic_net_gs"],
                    ),
                ],
            ),
        ]
    }


def validate_pipeline_spec(
    spec: MCPipelineSpec,
    *,
    has_reference: bool,
    has_dgp: bool,
) -> list[MCNodeSpec]:
    nodes = {node.id: node for node in spec.nodes}
    if len(nodes) != len(spec.nodes):
        raise ValueError("Pipeline node IDs must be unique.")
    names = [node.name for node in spec.nodes]
    if len(set(names)) != len(names):
        raise ValueError("Pipeline step names must be unique.")

    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    seen_edges: set[tuple[str, str]] = set()
    for edge in spec.edges:
        pair = (edge.source, edge.target)
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError("Pipeline edges must reference existing nodes.")
        if edge.source == edge.target:
            raise ValueError("Pipeline steps cannot connect to themselves.")
        if pair in seen_edges:
            raise ValueError("Pipeline edges must be unique.")
        source = nodes[edge.source]
        target = nodes[edge.target]
        if source.step_type in TERMINAL_STEP_TYPES:
            raise ValueError(
                f"Terminal step '{source.name}' cannot link to another step."
            )
        if target.step_type == "simulation":
            raise ValueError("The simulation step cannot have an incoming link.")
        if target.step_type == "filter" and source.step_type != "simulation":
            raise ValueError("Filter steps must link directly from simulation.")
        if target.step_type in TERMINAL_STEP_TYPES and source.step_type not in {
            "simulation",
            "filter",
        }:
            raise ValueError(
                "Tests and regressions must link from simulation or a filter."
            )
        seen_edges.add(pair)
        outgoing[edge.source].append(edge.target)
        incoming[edge.target].append(edge.source)

    simulations = [node for node in spec.nodes if node.step_type == "simulation"]
    if len(simulations) != 1:
        raise ValueError("Pipeline supports exactly one simulation step.")
    simulation = simulations[0]
    if incoming[simulation.id]:
        raise ValueError("The simulation step cannot have an incoming link.")
    for node in spec.nodes:
        if node.id == simulation.id:
            continue
        if len(incoming[node.id]) != 1:
            raise ValueError(
                f"Step '{node.name}' must have exactly one incoming dependency link."
            )
        if node.step_type in TERMINAL_STEP_TYPES and outgoing[node.id]:
            raise ValueError(f"Terminal step '{node.name}' cannot link forward.")

    if not has_reference:
        raise ValueError("A solved reference model is required.")
    if not has_dgp:
        raise ValueError("A solved DGP model is required by the simulation step.")

    ordered = [
        simulation,
        *(node for node in spec.nodes if node.step_type == "filter"),
        *(node for node in spec.nodes if node.step_type in TERMINAL_STEP_TYPES),
    ]
    bound: list[MCNodeSpec] = []
    prior_names: set[str] = set()
    for node in ordered:
        parent = nodes[incoming[node.id][0]] if incoming[node.id] else None
        bound_node = _bind_graph_dependency(node, parent, simulation)
        _validate_dependency(bound_node, prior_names)
        bound.append(bound_node)
        prior_names.add(bound_node.name)
    return bound


def build_pipeline(
    ordered: Sequence[MCNodeSpec],
    *,
    dgp: SolvedModel,
) -> MCPipeline:
    steps = []
    for node in ordered:
        params = _clean_params(node.params)
        if node.step_type == "simulation":
            params["seed_increment"] = _integer_or_keyword(
                params.get("seed_increment", "auto"),
                keywords={"auto"},
                field="seed_increment",
            )
            params["shocks"] = _build_generated_shocks(dgp, params)
            steps.append(simulation_step(name=node.name, **params))
        elif node.step_type == "filter":
            params.pop("filter_key", None)
            steps.append(reference_filter_step(name=node.name, **params))
        elif node.step_type == "wald":
            kind = str(params.get("kind", "mean"))
            target_key = "target_vector" if kind == "mean" else "target_matrix"
            params["target"] = np.asarray(params.pop(target_key, []), dtype=np.float64)
            params["bandwidth"] = _integer_or_keyword(
                params.get("bandwidth", "auto"),
                keywords={"andrews", "wooldridge", "auto"},
                field="bandwidth",
            )
            params.pop(
                "target_matrix" if target_key == "target_vector" else "target_vector",
                None,
            )
            steps.append(wald_test_step(node.name, **params))
        elif node.step_type == "ljung_box":
            steps.append(ljung_box_test_step(node.name, **params))
        elif node.step_type == "jarque_bera":
            steps.append(jarque_bera_test_step(node.name, **params))
        elif node.step_type == "breusch_pagan":
            steps.append(breusch_pagan_test_step(node.name, **params))
        elif node.step_type == "breusch_godfrey":
            steps.append(breusch_godfrey_test_step(node.name, **params))
        elif node.step_type == "cusum":
            steps.append(cusum_test_step(node.name, **params))
        elif node.step_type == "cusumsq":
            steps.append(cusumsq_test_step(node.name, **params))
        elif node.step_type == "chow":
            steps.append(chow_test_step(node.name, **params))
        elif node.step_type == "regression":
            steps.append(regression_step(node.name, **_regression_params(params)))
        else:
            raise ValueError(f"Unsupported MC step type: {node.step_type}")
    return MCPipeline(steps)


def run_pipeline(
    spec: MCPipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
) -> MCPipelineResult:
    ordered = validate_pipeline_spec(
        spec,
        has_reference=reference is not None,
        has_dgp=dgp is not None,
    )
    assert reference is not None
    assert dgp is not None
    pipeline = build_pipeline(ordered, dgp=dgp)
    return pipeline.run(
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        retain_payloads=False,
        retain_test_results=False,
        retain_contexts=True,
        fail_fast=fail_fast,
        verbosity=0,
    )


def _step_catalog(
    step_type: str,
    title: str,
    default_name: str,
    description: str,
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "step_type": step_type,
        "title": title,
        "default_name": default_name,
        "description": description,
        "fields": fields,
    }


def _field(
    key: str,
    label: str,
    field_type: str,
    default: Any,
    *,
    required: bool = False,
    options: Sequence[str] | None = None,
    minimum: float | None = None,
    when: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "type": field_type,
        "default": default,
        "required": required,
        "options": list(options or ()),
        "minimum": minimum,
        "when": list(when or ()),
    }


def _validate_dependency(node: MCNodeSpec, prior_names: set[str]) -> None:
    params = node.params
    if node.step_type == "filter":
        return
    sources = [
        params.get(key)
        for key in ("source", "residual_source", "y_source", "X_source", "x_source")
        if params.get(key) is not None
    ]
    if any(source in FILTER_SOURCES for source in sources):
        filter_key = str(params.get("filter_key", "filter"))
        if filter_key not in prior_names:
            raise ValueError(
                f"Step '{node.name}' requires prior filter payload '{filter_key}'."
            )
    payload_keys = [
        params.get(key)
        for key in (
            "payload_key",
            "residual_payload_key",
            "y_payload_key",
            "x_payload_key",
        )
        if params.get(key)
    ]
    for payload_key in payload_keys:
        if str(payload_key) not in prior_names:
            raise ValueError(
                f"Step '{node.name}' requires prior payload '{payload_key}'."
            )


def _bind_graph_dependency(
    node: MCNodeSpec,
    parent: MCNodeSpec | None,
    simulation: MCNodeSpec,
) -> MCNodeSpec:
    params = dict(node.params)
    if node.step_type == "filter":
        if not bool(simulation.params.get("observables", True)):
            raise ValueError("Filter steps require simulation observables.")
    elif node.step_type in TERMINAL_STEP_TYPES:
        sources = [
            params.get(key)
            for key in ("source", "residual_source", "y_source", "X_source", "x_source")
            if params.get(key) is not None
        ]
        if any(source == "payload" for source in sources):
            raise ValueError("Payload sources are not supported by the UI builder.")
        if parent is not None and parent.step_type == "filter":
            params["filter_key"] = parent.name
        elif any(source in FILTER_SOURCES for source in sources):
            raise ValueError(
                f"Step '{node.name}' uses filter output and must link from a filter."
            )
    return node.model_copy(update={"params": params})


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value == "" or value == [] or value is None:
            continue
        out[key] = value
    return out


def _integer_or_keyword(
    value: Any,
    *,
    keywords: set[str],
    field: str,
) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in keywords:
        return text
    try:
        return int(text)
    except ValueError as exc:
        options = ", ".join(sorted(keywords))
        raise ValueError(f"{field} must be an integer or one of: {options}.") from exc


def _build_generated_shocks(
    dgp: SolvedModel,
    params: dict[str, Any],
) -> Mapping[str, Shock] | None:
    T = int(params["T"])
    distribution_value = str(params.pop("distribution", "norm"))
    if distribution_value not in {"norm", "t", "uni"}:
        raise ValueError(f"Unsupported shock distribution: {distribution_value}")
    distribution = cast(Literal["norm", "t", "uni"], distribution_value)
    seed_value = params.pop("seed", 0)
    seed = None if seed_value is None else int(seed_value)
    loc = float(params.pop("loc", 0.0))
    df = float(params.pop("df", 5.0))
    targets = [str(target) for target in dgp.config.shock_map.values()]
    if not targets:
        return None

    if distribution in {"norm", "t"} and len(targets) > 1:
        kwargs: dict[str, Any] = (
            {"loc": [loc] * len(targets), "df": df}
            if distribution == "t"
            else {"mean": [loc] * len(targets)}
        )
        return {
            ",".join(targets): Shock(
                T=T,
                dist=distribution,
                multivar=True,
                seed=seed,
                dist_kwargs=kwargs,
            )
        }

    shocks: dict[str, Shock] = {}
    for index, target in enumerate(targets):
        kwargs = {"loc": loc}
        if distribution == "t":
            kwargs["df"] = df
        shocks[target] = Shock(
            T=T,
            dist=distribution,
            multivar=False,
            seed=None if seed is None else seed + index,
            dist_kwargs=kwargs,
        )
    return shocks


def _regression_params(params: dict[str, Any]) -> dict[str, Any]:
    kind = str(params.get("kind", "ols"))
    allowed_by_kind = {
        "ols": set(),
        "ridge": {"alpha"},
        "ridge_gs": {"start", "stop", "num", "criterion"},
        "lasso": {"alpha", "max_iter", "tol"},
        "lasso_gs": {"start", "stop", "num", "max_iter", "tol"},
        "elastic_net": {"alpha", "l1_ratio", "max_iter", "tol"},
        "elastic_net_gs": {
            "start",
            "stop",
            "num",
            "l1_ratio",
            "criterion",
            "max_iter",
            "tol",
        },
    }
    conditional = {
        "alpha",
        "l1_ratio",
        "start",
        "stop",
        "num",
        "criterion",
        "max_iter",
        "tol",
    }
    allowed = allowed_by_kind.get(kind)
    if allowed is None:
        raise ValueError(f"Unsupported regression kind: {kind}")
    return {
        key: value
        for key, value in params.items()
        if key not in conditional or key in allowed
    }
