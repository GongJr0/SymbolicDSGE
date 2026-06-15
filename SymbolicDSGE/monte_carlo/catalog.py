"""Concrete Monte-Carlo step catalogue (core, UI-independent).

Each built-in step kind is described once by a :class:`StepDefinition` that bundles
its form metadata (:class:`FieldSpec`), its graph role, the built-in factory that
turns parameters into an :class:`MCStep`, and an optional parameter-compile hook
(shock generation, Wald target shaping, regression kwarg filtering). This single
registry replaces both the old ``ui.mc.mc_catalog`` literal and the ``build_pipeline``
``if/elif`` dispatch, so a pipeline can be compiled and run without the ``[ui]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np

from ..core.shock_generators import Shock
from .config import (
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    diff_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    log_diff_step,
    log_step,
    reference_filter_step,
    regression_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    simulation_step,
    standardize_step,
    wald_test_step,
)
from .mc_constructs import MCStep

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

#: Series a diagnostic/regression step may read from an upstream context.
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
#: Sources produced only by a filter step (require an upstream filter link).
FILTER_SOURCES = {"x_pred", "x_filt", "y_pred", "y_filt", "innov", "std_innov"}

StepRole = Literal["datagen", "filter", "transform", "terminal"]
CompileParams = Callable[[dict[str, Any], "SolvedModel"], dict[str, Any]]


@dataclass(frozen=True)
class FieldSpec:
    """One configurable parameter of a step (drives the GUI form + defaults)."""

    key: str
    label: str
    type: str
    default: Any
    required: bool = False
    options: tuple[str, ...] = ()
    minimum: float | None = None
    when: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "required": self.required,
            "options": list(self.options),
            "minimum": self.minimum,
            "when": list(self.when),
        }


@dataclass(frozen=True)
class StepDefinition:
    """Everything the library knows about one built-in MC step kind."""

    step_type: str
    title: str
    default_name: str
    description: str
    op_role: StepRole
    factory: Callable[..., MCStep]
    fields: tuple[FieldSpec, ...] = ()
    compile_params: CompileParams | None = None

    @property
    def is_terminal(self) -> bool:
        return self.op_role == "terminal"

    @property
    def is_transform(self) -> bool:
        return self.op_role == "transform"

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type,
            "title": self.title,
            "default_name": self.default_name,
            "description": self.description,
            "fields": [spec.to_dict() for spec in self.fields],
        }

    def build(self, name: str, params: dict[str, Any], dgp: SolvedModel) -> MCStep:
        """Compile cleaned ``params`` into an :class:`MCStep` via the factory."""
        if self.compile_params is not None:
            params = self.compile_params(params, dgp)
        return self.factory(name=name, **params)


# --------------------------------------------------------------------------- #
# Parameter-compile helpers (shared by the per-step hooks below).
# --------------------------------------------------------------------------- #


def _integer_or_keyword(
    value: Any, *, keywords: set[str], field_name: str
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
        raise ValueError(
            f"{field_name} must be an integer or one of: {options}."
        ) from exc


def _build_generated_shocks(
    dgp: SolvedModel, params: dict[str, Any]
) -> dict[str, Shock] | None:
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


_REGRESSION_ALLOWED_BY_KIND: dict[str, set[str]] = {
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
_REGRESSION_CONDITIONAL = {
    "alpha",
    "l1_ratio",
    "start",
    "stop",
    "num",
    "criterion",
    "max_iter",
    "tol",
}


def _regression_params(params: dict[str, Any]) -> dict[str, Any]:
    kind = str(params.get("kind", "ols"))
    allowed = _REGRESSION_ALLOWED_BY_KIND.get(kind)
    if allowed is None:
        raise ValueError(f"Unsupported regression kind: {kind}")
    return {
        key: value
        for key, value in params.items()
        if key not in _REGRESSION_CONDITIONAL or key in allowed
    }


def _compile_simulation(params: dict[str, Any], dgp: SolvedModel) -> dict[str, Any]:
    params = dict(params)
    params["seed_increment"] = _integer_or_keyword(
        params.get("seed_increment", "auto"),
        keywords={"auto"},
        field_name="seed_increment",
    )
    params["shocks"] = _build_generated_shocks(dgp, params)
    return params


def _compile_filter(params: dict[str, Any], dgp: SolvedModel) -> dict[str, Any]:
    params = dict(params)
    params.pop("filter_key", None)
    return params


def _compile_wald(params: dict[str, Any], dgp: SolvedModel) -> dict[str, Any]:
    params = dict(params)
    kind = str(params.get("kind", "mean"))
    target_key = "target_vector" if kind == "mean" else "target_matrix"
    params["target"] = np.asarray(params.pop(target_key, []), dtype=np.float64)
    params["bandwidth"] = _integer_or_keyword(
        params.get("bandwidth", "auto"),
        keywords={"andrews", "wooldridge", "auto"},
        field_name="bandwidth",
    )
    params.pop(
        "target_matrix" if target_key == "target_vector" else "target_vector",
        None,
    )
    return params


def _compile_regression(params: dict[str, Any], dgp: SolvedModel) -> dict[str, Any]:
    return _regression_params(params)


# --------------------------------------------------------------------------- #
# The catalogue.
# --------------------------------------------------------------------------- #


def _source_field(key: str, label: str, default: str = "std_innov") -> FieldSpec:
    return FieldSpec(key, label, "select", default, options=tuple(INPUT_SOURCES))


_STEP_DEFINITIONS: tuple[StepDefinition, ...] = (
    StepDefinition(
        step_type="simulation",
        title="DGP Simulation",
        default_name="datagen",
        description="Generate one sample from the DGP model.",
        op_role="datagen",
        factory=simulation_step,
        compile_params=_compile_simulation,
        fields=(
            FieldSpec("T", "Periods", "number", 100, required=True, minimum=1),
            FieldSpec("observables", "Observables", "boolean", True),
            FieldSpec("shock_scale", "Shock scale", "number", 1.0),
            FieldSpec("seed_increment", "Seed increment", "text", "auto"),
            FieldSpec(
                "distribution",
                "Distribution",
                "select",
                "norm",
                options=("norm", "t", "uni"),
            ),
            FieldSpec("seed", "Initial seed", "number", 0),
            FieldSpec("loc", "Location", "number", 0.0),
            FieldSpec("df", "Degrees of freedom", "number", 5.0),
        ),
    ),
    StepDefinition(
        step_type="filter",
        title="Reference Filter",
        default_name="filter",
        description="Filter generated observables through the reference model.",
        op_role="filter",
        factory=reference_filter_step,
        compile_params=_compile_filter,
        fields=(
            FieldSpec(
                "filter_mode",
                "Mode",
                "select",
                "linear",
                options=("linear", "extended"),
            ),
            FieldSpec("return_shocks", "Return shocks", "boolean", False),
            FieldSpec("estimate_R_diag", "Estimate R diagonal", "boolean", False),
            FieldSpec("R_scale", "R scale", "number", 1.0),
        ),
    ),
    StepDefinition(
        step_type="wald",
        title="Wald Test",
        default_name="wald",
        description="Run a HAC Wald diagnostic on a selected source.",
        op_role="terminal",
        factory=wald_test_step,
        compile_params=_compile_wald,
        fields=(
            _source_field("source", "Source"),
            FieldSpec(
                "kind",
                "Moment",
                "select",
                "mean",
                options=("mean", "covariance", "second_moment"),
            ),
            FieldSpec(
                "target_vector",
                "Target vector",
                "number_list",
                [0.0],
                required=True,
                when=("mean",),
            ),
            FieldSpec(
                "target_matrix",
                "Target matrix",
                "number_matrix",
                [[1.0]],
                required=True,
                when=("covariance", "second_moment"),
            ),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec(
                "kernel",
                "Kernel",
                "select",
                "bartlett",
                options=("bartlett", "parzen", "qs"),
            ),
            FieldSpec("bandwidth", "Bandwidth", "text", "auto"),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="ljung_box",
        title="Ljung-Box Test",
        default_name="ljung_box",
        description="Test one selected series for serial correlation.",
        op_role="terminal",
        factory=ljung_box_test_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("column", "Column", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("lags", "Lags", "number", 10, minimum=1),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="jarque_bera",
        title="Jarque-Bera Test",
        default_name="jarque_bera",
        description="Test one selected series for normality.",
        op_role="terminal",
        factory=jarque_bera_test_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("column", "Column", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="breusch_pagan",
        title="Breusch-Pagan Test",
        default_name="breusch_pagan",
        description="Test residual variance against selected regressors.",
        op_role="terminal",
        factory=breusch_pagan_test_step,
        fields=(
            _source_field("residual_source", "Residual source"),
            _source_field("X_source", "Regressor source", "observables"),
            FieldSpec("residual_col", "Residual column", "number_list", [0]),
            FieldSpec("X_columns", "Regressor columns", "number_list", [0]),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("robust", "Robust (Koenker)", "boolean", False),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="breusch_godfrey",
        title="Breusch-Godfrey Test",
        default_name="breusch_godfrey",
        description="Test residuals for serial correlation up to a given lag order.",
        op_role="terminal",
        factory=breusch_godfrey_test_step,
        fields=(
            _source_field("residual_source", "Residual source"),
            _source_field("X_source", "Regressor source", "observables"),
            FieldSpec("residual_col", "Residual column", "number_list", [0]),
            FieldSpec("X_columns", "Regressor columns", "number_list", [0]),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("lags", "Lags", "number", 1, minimum=1),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="cusum",
        title="CUSUM Test",
        default_name="cusum",
        description="Test regression coefficients for stability via recursive residuals.",
        op_role="terminal",
        factory=cusum_test_step,
        fields=(
            _source_field("y_source", "Response source", "observables"),
            _source_field("x_source", "Regressor source", "observables"),
            FieldSpec("y_column", "Response column", "number_list", [0]),
            FieldSpec("X_columns", "Regressor columns", "number_list", [1]),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="cusumsq",
        title="CUSUM of Squares Test",
        default_name="cusumsq",
        description="Test regression variance stability via squared recursive residuals.",
        op_role="terminal",
        factory=cusumsq_test_step,
        fields=(
            _source_field("y_source", "Response source", "observables"),
            _source_field("x_source", "Regressor source", "observables"),
            FieldSpec("y_column", "Response column", "number_list", [0]),
            FieldSpec("X_columns", "Regressor columns", "number_list", [1]),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="chow",
        title="Chow Test",
        default_name="chow",
        description=(
            "Test for a structural break in regression coefficients at a "
            "known break point."
        ),
        op_role="terminal",
        factory=chow_test_step,
        fields=(
            _source_field("y_source", "Response source", "observables"),
            _source_field("x_source", "Regressor source", "observables"),
            FieldSpec("y_column", "Response column", "number_list", [0]),
            FieldSpec("X_columns", "Regressor columns", "number_list", [1]),
            FieldSpec("t_break", "Break point", "number", 10, required=True, minimum=1),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("alpha", "Alpha", "number", 0.05),
        ),
    ),
    StepDefinition(
        step_type="regression",
        title="Regression",
        default_name="regression",
        description="Fit a linear regression in each replication.",
        op_role="terminal",
        factory=regression_step,
        compile_params=_compile_regression,
        fields=(
            FieldSpec(
                "kind",
                "Kind",
                "select",
                "ols",
                options=(
                    "ols",
                    "ridge",
                    "ridge_gs",
                    "lasso",
                    "lasso_gs",
                    "elastic_net",
                    "elastic_net_gs",
                ),
            ),
            _source_field("y_source", "Response source", "observables"),
            _source_field("X_source", "Design source", "observables"),
            FieldSpec("y_column", "Response column", "number_list", [0]),
            FieldSpec("X_columns", "Design columns", "number_list", [1]),
            FieldSpec("intercept", "Intercept", "boolean", True),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("variables", "Variable names", "text_list", []),
            FieldSpec(
                "alpha",
                "Alpha",
                "number",
                0.5,
                when=("ridge", "lasso", "elastic_net"),
            ),
            FieldSpec(
                "l1_ratio",
                "L1 ratio",
                "number",
                0.5,
                when=("elastic_net", "elastic_net_gs"),
            ),
            FieldSpec(
                "start",
                "Grid start",
                "number",
                0.01,
                when=("ridge_gs", "lasso_gs", "elastic_net_gs"),
            ),
            FieldSpec(
                "stop",
                "Grid stop",
                "number",
                2.0,
                when=("ridge_gs", "lasso_gs", "elastic_net_gs"),
            ),
            FieldSpec(
                "num",
                "Grid points",
                "number",
                20,
                when=("ridge_gs", "lasso_gs", "elastic_net_gs"),
            ),
            FieldSpec(
                "criterion",
                "Criterion",
                "select",
                "loss",
                options=("aic", "bic", "loss"),
                when=("ridge_gs", "elastic_net_gs"),
            ),
            FieldSpec(
                "max_iter",
                "Max iterations",
                "number",
                1000,
                when=("lasso", "lasso_gs", "elastic_net", "elastic_net_gs"),
            ),
            FieldSpec(
                "tol",
                "Tolerance",
                "number",
                1e-10,
                when=("lasso", "lasso_gs", "elastic_net", "elastic_net_gs"),
            ),
        ),
    ),
    # ----- transforms -----
    StepDefinition(
        step_type="standardize",
        title="Standardize",
        default_name="standardize",
        description=(
            "Per-column z-score: (x - mean) / std. Columns with zero std "
            "pass through as zeros."
        ),
        op_role="transform",
        factory=standardize_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("ddof", "Degrees of freedom", "number", 0, minimum=0),
        ),
    ),
    StepDefinition(
        step_type="log",
        title="Log",
        default_name="log",
        description="Elementwise log(x + offset). Offset handles inputs that touch zero.",
        op_role="transform",
        factory=log_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("offset", "Offset", "number", 0.0),
        ),
    ),
    StepDefinition(
        step_type="log_diff",
        title="Log Difference",
        default_name="log_diff",
        description=(
            "One-period log differences along the time axis. Output is one "
            "row shorter than the input."
        ),
        op_role="transform",
        factory=log_diff_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("offset", "Offset", "number", 0.0),
        ),
    ),
    StepDefinition(
        step_type="diff",
        title="Difference",
        default_name="diff",
        description="Repeated np.diff along the time axis (order-th difference).",
        op_role="transform",
        factory=diff_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("order", "Order", "number", 1, minimum=1),
        ),
    ),
    StepDefinition(
        step_type="rolling_mean",
        title="Rolling Mean",
        default_name="rolling_mean",
        description="Trailing rolling mean over the time axis.",
        op_role="transform",
        factory=rolling_mean_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("window", "Window", "number", 10, required=True, minimum=1),
        ),
    ),
    StepDefinition(
        step_type="rolling_std",
        title="Rolling Std",
        default_name="rolling_std",
        description="Trailing rolling standard deviation over the time axis.",
        op_role="transform",
        factory=rolling_std_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("window", "Window", "number", 10, required=True, minimum=1),
            FieldSpec("ddof", "Degrees of freedom", "number", 0, minimum=0),
        ),
    ),
    StepDefinition(
        step_type="rolling_var",
        title="Rolling Variance",
        default_name="rolling_var",
        description="Trailing rolling variance over the time axis.",
        op_role="transform",
        factory=rolling_var_step,
        fields=(
            _source_field("source", "Source"),
            FieldSpec("columns", "Columns", "number_list", []),
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
            FieldSpec("window", "Window", "number", 10, required=True, minimum=1),
            FieldSpec("ddof", "Degrees of freedom", "number", 0, minimum=0),
        ),
    ),
)

#: Step kind -> definition, insertion order matching the GUI catalogue payload.
STEP_CATALOG: dict[str, StepDefinition] = {
    definition.step_type: definition for definition in _STEP_DEFINITIONS
}

#: Step kinds that terminate a branch (tests + regression).
TERMINAL_STEP_TYPES: frozenset[str] = frozenset(
    step_type
    for step_type, definition in STEP_CATALOG.items()
    if definition.is_terminal
)

#: Step kinds that produce a single ndarray payload downstream nodes can chain
#: from (one-input-one-output graph middle nodes).
TRANSFORM_STEP_TYPES: frozenset[str] = frozenset(
    step_type
    for step_type, definition in STEP_CATALOG.items()
    if definition.is_transform
)


def catalog_payload() -> dict[str, Any]:
    """The JSON-serializable step catalogue consumed by the GUI form builder."""
    return {
        "steps": [definition.catalog_entry() for definition in STEP_CATALOG.values()]
    }
