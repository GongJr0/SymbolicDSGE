"""Concrete Monte-Carlo step catalogue (core, UI-independent).

Each built-in step kind is described once by a :class:`StepDefinition` that bundles
its form metadata (:class:`FieldSpec`), its graph role, the built-in factory that
turns parameters into an :class:`MCStep`, and an optional parameter-compile hook
(shock generation, Wald target shaping, regression kwarg filtering). This single
registry replaces both the old ``ui.mc.mc_catalog`` literal and the ``build_pipeline``
``if/elif`` dispatch, so a pipeline can be compiled and run without the ``[ui]`` extra.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, cast

import numpy as np

from ..core.shock_generators import Shock
from .operations.core import reference_filter_step, simulation_step
from .operations.postproc import kde_step
from .operations.regressions import regression_step
from .operations.tests import (
    breusch_godfrey_test_step,
    breusch_pagan_test_step,
    chow_test_step,
    cusum_test_step,
    cusumsq_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    wald_test_step,
)
from .operations.transforms import (
    diff_step,
    log_diff_step,
    log_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    standardize_step,
)
from .mc_constructs import ARRAY_SOURCE_FIELDS, FILTER_SOURCE_FIELDS, MCStep

#: Array-valued sources exposed by the built-in diagnostic and regression ops.
INPUT_SOURCES = list(ARRAY_SOURCE_FIELDS)
FILTER_OUTPUT_SOURCES = tuple(
    source for source in INPUT_SOURCES if source not in {"states", "observables"}
)
#: Sources produced only by a filter step (require an upstream filter link).
FILTER_SOURCES = set(FILTER_SOURCE_FIELDS)

StepRole = Literal["datagen", "filter", "transform", "terminal", "postproc"]
CompileParams = Callable[[dict[str, Any]], dict[str, Any]]


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


class SourceBinding(NamedTuple):
    arg: str
    source_key: str
    field_key: str
    columns_key: str
    label: str
    source_default: str = "datagen"
    field_default: str = "observables"
    columns_label: str = "Columns"
    columns_default: tuple[int, ...] = ()


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
    source_bindings: tuple[SourceBinding, ...] = ()
    compile_params: CompileParams | None = None

    def __post_init__(self) -> None:
        if self.source_bindings:
            object.__setattr__(
                self,
                "fields",
                (*_source_binding_fields(self.source_bindings), *self.fields),
            )

    @property
    def is_terminal(self) -> bool:
        return self.op_role == "terminal"

    @property
    def is_transform(self) -> bool:
        return self.op_role == "transform"

    @property
    def category(self) -> str:
        """Palette grouping for the GUI step selector tabs.

        Derived from ``op_role`` (so new steps group automatically): datagen and
        filter are ``"core"``, transforms ``"transforms"``, post-loop ops
        ``"postproc"``, the regression step ``"regressions"``, and the remaining
        terminals (tests) ``"tests"``.
        """
        if self.op_role in ("datagen", "filter"):
            return "core"
        if self.op_role == "transform":
            return "transforms"
        if self.op_role == "postproc":
            return "postproc"
        if self.step_type == "regression":
            return "regressions"
        return "tests"

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type,
            "title": self.title,
            "default_name": self.default_name,
            "description": self.description,
            "category": self.category,
            "fields": [spec.to_dict() for spec in self.fields],
        }

    def build(self, name: str, params: dict[str, Any]) -> MCStep:
        """Compile cleaned ``params`` into an :class:`MCStep` via the factory.

        No model is consulted: every step compiles purely from its parameters
        (simulation shocks come from the explicit registry, not a model)."""
        if self.compile_params is not None:
            params = self.compile_params(params)
        return self.factory(name=name, **params)


# Parameter-compile helpers (shared by the per-step hooks below).


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


def _shock_for(
    vars_: list[str],
    *,
    dist: str = "norm",
    loc: float = 0.0,
    df: float = 5.0,
    seed: int | None = None,
) -> Shock:
    """Build one :class:`Shock` for a shock-registry entry.

    ``multivar`` is derived from the selection: a joint shock when more than one
    variable is chosen. Uniform is univariate only (no joint uniform is
    implemented), so a ``uni`` entry must select exactly one variable. No model
    is consulted; the variables are exactly what the user chose.
    """
    n = len(vars_)
    multivar = n > 1
    if dist == "uni" and multivar:
        raise ValueError(
            "A 'uni' shock is univariate; select exactly one variable per "
            "uniform entry (use separate entries for independent uniform shocks)."
        )
    if dist == "norm":
        dist_kwargs: dict[str, Any] = {"mean": [loc] * n} if multivar else {"loc": loc}
    elif dist == "t":
        dist_kwargs = (
            {"loc": [loc] * n, "df": df} if multivar else {"loc": loc, "df": df}
        )
    elif dist == "uni":
        dist_kwargs = {"loc": loc}
    else:
        raise ValueError(f"Unsupported shock distribution: {dist!r}")
    return Shock(
        dist=cast(Literal["norm", "t", "uni"], dist),
        multivar=multivar,
        seed=seed,
        dist_kwargs=dist_kwargs,
    )


def _shocks_from_registry(
    registry: list[dict[str, Any]],
) -> dict[str, Shock] | None:
    """Compile an explicit shock registry into a ``{key: Shock}`` mapping.

    Each entry is ``{vars, dist, loc, df, seed}``; the key is the joined variable
    names. No model is read: the variables are the user's explicit selection.
    """
    shocks: dict[str, Shock] = {}
    for entry in registry:
        vars_ = [str(v) for v in entry["vars"]]
        if not vars_:
            raise ValueError(
                "Each shock registry entry must select at least one variable."
            )
        key = ",".join(vars_)
        if key in shocks:
            raise ValueError(f"Duplicate shock entry for {key!r} in the registry.")
        seed = None
        if "seed" in entry and entry["seed"] is not None:
            seed = int(entry["seed"])
        kwargs: dict[str, Any] = {}
        if "dist" in entry:
            kwargs["dist"] = str(entry["dist"])
        if "loc" in entry:
            kwargs["loc"] = float(entry["loc"])
        if "df" in entry:
            kwargs["df"] = float(entry["df"])
        shocks[key] = _shock_for(vars_, seed=seed, **kwargs)
    return shocks or None


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
    if "kind" not in params:
        return {
            key: value
            for key, value in params.items()
            if key not in _REGRESSION_CONDITIONAL
        }
    kind = str(params["kind"])
    if kind not in _REGRESSION_ALLOWED_BY_KIND:
        raise ValueError(f"Unsupported regression kind: {kind}")
    allowed = _REGRESSION_ALLOWED_BY_KIND[kind]
    return {
        key: value
        for key, value in params.items()
        if key not in _REGRESSION_CONDITIONAL or key in allowed
    }


def _coerce_shock_mapping(value: Any) -> dict[str, Shock]:
    """Normalize a ``shocks`` mapping of live :class:`Shock` / serialized dicts.

    Library-authored pipelines carry explicit :class:`Shock` instances; a spec
    loaded from a bundle carries their :meth:`Shock.to_dict` form. Both compile
    to the same live mapping the factory stores.
    """
    if not isinstance(value, Mapping):
        raise TypeError("simulation 'shocks' must be a mapping of name -> Shock.")
    out: dict[str, Shock] = {}
    for key, shock in value.items():
        if isinstance(shock, Shock):
            out[str(key)] = shock
        elif isinstance(shock, Mapping):
            out[str(key)] = Shock.from_dict(shock)
        else:
            raise TypeError(
                f"shocks[{key!r}] must be a Shock or a serialized shock dict."
            )
    return out


def _compile_simulation(params: dict[str, Any]) -> dict[str, Any]:
    # No model is consulted. Shocks are either explicit (a ``{key: Shock}``
    # mapping from library authoring) or come from the explicit registry the
    # user authored.
    params = dict(params)
    if "seed_increment" in params:
        params["seed_increment"] = _integer_or_keyword(
            params["seed_increment"],
            keywords={"auto"},
            field_name="seed_increment",
        )
    registry = None
    if "shock_registry" in params:
        registry = params.pop("shock_registry")
    if "shocks" in params and params["shocks"] is not None:
        params["shocks"] = _coerce_shock_mapping(params["shocks"])
    elif registry:
        params["shocks"] = _shocks_from_registry(registry)
    # Empty registry and no explicit shocks -> deterministic sim; the op treats
    # ``shocks=None`` as a zero shock matrix.
    return params


def _compile_filter(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)


def _compile_wald(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    if "target_vector" in params:
        params["target"] = np.asarray(params.pop("target_vector"), dtype=np.float64)
        params.pop("target_matrix", None)
    elif "target_matrix" in params:
        params["target"] = np.asarray(params.pop("target_matrix"), dtype=np.float64)
    if "bandwidth" in params:
        params["bandwidth"] = _integer_or_keyword(
            params["bandwidth"],
            keywords={"andrews", "wooldridge", "auto"},
            field_name="bandwidth",
        )
    return params


def _compile_regression(params: dict[str, Any]) -> dict[str, Any]:
    return _regression_params(params)


# The catalogue.


def _source_step(key: str, label: str, default: str = "datagen") -> FieldSpec:
    return FieldSpec(key, label, "text", default, required=True)


def _source_field(key: str, label: str, default: str = "observables") -> FieldSpec:
    return FieldSpec(key, label, "select", default, options=tuple(INPUT_SOURCES))


def _source_binding_fields(
    bindings: tuple[SourceBinding, ...],
) -> tuple[FieldSpec, ...]:
    fields: list[FieldSpec] = []
    for binding in bindings:
        fields.extend(
            (
                _source_step(
                    binding.source_key,
                    f"{binding.label} step",
                    binding.source_default,
                ),
                _source_field(
                    binding.field_key,
                    f"{binding.label} field",
                    binding.field_default,
                ),
                FieldSpec(
                    binding.columns_key,
                    binding.columns_label,
                    "number_list",
                    list(binding.columns_default),
                ),
            )
        )
    fields.extend(
        (
            FieldSpec("burn_in", "Burn-in", "number", 0, minimum=0),
            FieldSpec("drop_initial", "Drop initial", "boolean", False),
        )
    )
    return tuple(fields)


WALD_SOURCE = SourceBinding(
    "sample",
    "source",
    "field",
    "columns",
    "Source",
    source_default="filter",
    field_default="std_innov",
)
SAMPLE_SOURCE = SourceBinding("sample", "source", "field", "columns", "Source")
SAMPLE_COLUMN_SOURCE = SourceBinding(
    "sample", "source", "field", "column", "Source", columns_label="Column"
)
RESIDUAL_SOURCE = SourceBinding(
    "residuals",
    "residuals_source",
    "residuals_field",
    "residual_col",
    "Residual",
    columns_label="Residual column",
    columns_default=(0,),
)
X_REGRESSOR_SOURCE = SourceBinding(
    "X",
    "X_source",
    "X_field",
    "X_columns",
    "Regressor",
    columns_label="Regressor columns",
    columns_default=(0,),
)
Y_SOURCE = SourceBinding(
    "y",
    "y_source",
    "y_field",
    "y_column",
    "Response",
    columns_label="Response column",
    columns_default=(0,),
)
X_REGRESSION_SOURCE = SourceBinding(
    "X",
    "X_source",
    "X_field",
    "X_columns",
    "Regressor",
    columns_label="Regressor columns",
    columns_default=(1,),
)
X_DESIGN_SOURCE = SourceBinding(
    "X",
    "X_source",
    "X_field",
    "X_columns",
    "Design",
    columns_label="Design columns",
    columns_default=(1,),
)


_STEP_DEFINITIONS: tuple[StepDefinition, ...] = (
    StepDefinition(
        step_type="simulation",
        title="Simulation",
        default_name="datagen",
        description="Generate one sample by simulating a solved model (DGP or reference).",
        op_role="datagen",
        factory=simulation_step,
        compile_params=_compile_simulation,
        fields=(
            FieldSpec(
                "target", "Simulate", "select", "dgp", options=("dgp", "reference")
            ),
            FieldSpec("T", "Periods", "number", 100, required=True, minimum=1),
            FieldSpec("observables", "Observables", "boolean", True),
            FieldSpec("shock_scale", "Shock scale", "number", 1.0),
            FieldSpec("seed_increment", "Seed increment", "text", "auto"),
            FieldSpec("shock_registry", "Shocks", "shock_registry", []),
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
                options=("linear", "extended", "unscented"),
            ),
            FieldSpec("return_shocks", "Return shocks", "boolean", False),
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
        source_bindings=(WALD_SOURCE,),
        fields=(
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
        source_bindings=(SAMPLE_COLUMN_SOURCE,),
        fields=(
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
        source_bindings=(SAMPLE_COLUMN_SOURCE,),
        fields=(FieldSpec("alpha", "Alpha", "number", 0.05),),
    ),
    StepDefinition(
        step_type="breusch_pagan",
        title="Breusch-Pagan Test",
        default_name="breusch_pagan",
        description="Test residual variance against selected regressors.",
        op_role="terminal",
        factory=breusch_pagan_test_step,
        source_bindings=(RESIDUAL_SOURCE, X_REGRESSOR_SOURCE),
        fields=(
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
        source_bindings=(RESIDUAL_SOURCE, X_REGRESSOR_SOURCE),
        fields=(
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
        source_bindings=(Y_SOURCE, X_REGRESSION_SOURCE),
        fields=(FieldSpec("alpha", "Alpha", "number", 0.05),),
    ),
    StepDefinition(
        step_type="cusumsq",
        title="CUSUM of Squares Test",
        default_name="cusumsq",
        description="Test regression variance stability via squared recursive residuals.",
        op_role="terminal",
        factory=cusumsq_test_step,
        source_bindings=(Y_SOURCE, X_REGRESSION_SOURCE),
        fields=(FieldSpec("alpha", "Alpha", "number", 0.05),),
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
        source_bindings=(Y_SOURCE, X_REGRESSION_SOURCE),
        fields=(
            FieldSpec("t_break", "Break point", "number", 10, required=True, minimum=1),
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
        source_bindings=(Y_SOURCE, X_DESIGN_SOURCE),
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
            FieldSpec("intercept", "Intercept", "boolean", True),
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
    # Transforms.
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
        source_bindings=(SAMPLE_SOURCE,),
        fields=(FieldSpec("ddof", "Degrees of freedom", "number", 0, minimum=0),),
    ),
    StepDefinition(
        step_type="log",
        title="Log",
        default_name="log",
        description="Elementwise log(x + offset). Offset handles inputs that touch zero.",
        op_role="transform",
        factory=log_step,
        source_bindings=(SAMPLE_SOURCE,),
        fields=(FieldSpec("offset", "Offset", "number", 0.0),),
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
        source_bindings=(SAMPLE_SOURCE,),
        fields=(FieldSpec("offset", "Offset", "number", 0.0),),
    ),
    StepDefinition(
        step_type="diff",
        title="Difference",
        default_name="diff",
        description="Repeated np.diff along the time axis (order-th difference).",
        op_role="transform",
        factory=diff_step,
        source_bindings=(SAMPLE_SOURCE,),
        fields=(FieldSpec("order", "Order", "number", 1, minimum=1),),
    ),
    StepDefinition(
        step_type="rolling_mean",
        title="Rolling Mean",
        default_name="rolling_mean",
        description="Trailing rolling mean over the time axis.",
        op_role="transform",
        factory=rolling_mean_step,
        source_bindings=(SAMPLE_SOURCE,),
        fields=(FieldSpec("window", "Window", "number", 10, required=True, minimum=1),),
    ),
    StepDefinition(
        step_type="rolling_std",
        title="Rolling Std",
        default_name="rolling_std",
        description="Trailing rolling standard deviation over the time axis.",
        op_role="transform",
        factory=rolling_std_step,
        source_bindings=(SAMPLE_SOURCE,),
        fields=(
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
        source_bindings=(SAMPLE_SOURCE,),
        fields=(
            FieldSpec("window", "Window", "number", 10, required=True, minimum=1),
            FieldSpec("ddof", "Degrees of freedom", "number", 0, minimum=0),
        ),
    ),
    # Post-processing.
    StepDefinition(
        step_type="kde",
        title="KDE",
        default_name="kde",
        description=(
            "Gaussian kernel density estimate of an across-replication trace; "
            "returns the raw (x, density) curve."
        ),
        op_role="postproc",
        factory=kde_step,
        fields=(
            # A ``"trace"`` field references an across-rep trace key (e.g.
            # "test.<name>.statistic"); validated against the pipeline's producible
            # traces, and (GUI, #184) offered from that registry.
            FieldSpec("trace", "Trace", "trace", "", required=True),
            FieldSpec("bandwidth", "Bandwidth", "text", "scott"),
            FieldSpec("grid_points", "Grid points", "number", 200, minimum=2),
            FieldSpec("kernel", "Kernel", "select", "gaussian", options=("gaussian",)),
        ),
    ),
)

#: Step kind -> definition, insertion order matching the GUI catalogue payload.
STEP_CATALOG: dict[str, StepDefinition] = {
    definition.step_type: definition for definition in _STEP_DEFINITIONS
}

#: Step kinds that act as the pipeline's single root datagen. ``simulation`` is
#: catalogue-driven (GUI-authorable); ``raw_model_data`` ships pre-computed
#: arrays as a bundle member and has no GUI ``StepDefinition``.
DATAGEN_STEP_TYPES: frozenset[str] = frozenset({"simulation", "raw_model_data"})

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

#: Post-loop step kinds (run once after the replication loop over the assembled
#: across-rep traces).
POSTPROC_STEP_TYPES: frozenset[str] = frozenset(
    step_type
    for step_type, definition in STEP_CATALOG.items()
    if definition.op_role == "postproc"
)


def catalog_payload() -> dict[str, Any]:
    """The JSON-serializable step catalogue consumed by the GUI form builder."""
    return {
        "steps": [definition.catalog_entry() for definition in STEP_CATALOG.values()]
    }
