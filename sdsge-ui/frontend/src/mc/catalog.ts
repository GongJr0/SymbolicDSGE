// The Monte Carlo step catalogue: what each step kind is called, what it reads,
// and what its form collects.
//
// This is form metadata, so it lives here rather than on the server. A node is
// posted with its op kind and its source legs already resolved, which is why a
// leg declares both the fields it renders and the keys those fields split back
// out under. Nothing here is validated client-side; every value is checked
// again during lowering, the last point before the native kernels.

import type {
  MCFieldSpec,
  MCFieldType,
  MCSourceSpec,
  MCStepCatalogItem,
  MCStepCategory,
  MCStepType,
} from "../types";

export type MCOpType =
  | "datagen"
  | "filter"
  | "transform"
  | "test"
  | "regression"
  | "postproc";

// One source leg of a step: the arg it binds, and the flat form keys its three
// fields occupy. A step with one leg spells them bare; several legs prefix each
// by its arg so they stay distinguishable. A leg taking exactly one column says
// so in its field name.
export interface MCSourceLeg {
  arg: string;
  sourceKey: string;
  fieldKey: string;
  columnsKey: string;
  fields: MCFieldSpec[];
}

export interface MCStepDefinition extends MCStepCatalogItem {
  opType: MCOpType;
  legs: MCSourceLeg[];
}

interface FieldOptions {
  required?: boolean;
  options?: readonly string[];
  minimum?: number;
  when?: readonly string[];
}

interface LegOptions {
  source?: string;
  field?: string;
  columns?: number[];
  columnsLabel?: string;
  single?: boolean;
}

interface StepBody {
  title: string;
  defaultName: string;
  description: string;
  opType: MCOpType;
  legs: MCSourceLeg[];
  fields: MCFieldSpec[];
}

function field(
  key: string,
  label: string,
  type: MCFieldType,
  value: unknown,
  opts: FieldOptions = {},
): MCFieldSpec {
  return {
    key,
    label,
    type,
    default: value,
    required: opts.required ?? false,
    options: [...(opts.options ?? [])],
    minimum: opts.minimum ?? null,
    when: [...(opts.when ?? [])],
  };
}

function leg(
  arg: string,
  label: string,
  solo: boolean,
  opts: LegOptions = {},
): MCSourceLeg {
  const prefix = solo ? "" : `${arg}_`;
  const sourceKey = `${prefix}source`;
  const fieldKey = `${prefix}field`;
  const columnsKey = `${prefix}${opts.single ? "column" : "columns"}`;
  return {
    arg,
    sourceKey,
    fieldKey,
    columnsKey,
    fields: [
      field(sourceKey, `${label} step`, "text", opts.source ?? "datagen", {
        required: true,
      }),
      field(fieldKey, `${label} field`, "select", opts.field ?? "observables", {
        options: INPUT_SOURCES,
      }),
      field(columnsKey, opts.columnsLabel ?? "Columns", "number_list", [
        ...(opts.columns ?? []),
      ]),
    ],
  };
}

// A step's palette tab follows its op kind, with the two terminal kinds split
// so a regression does not land among the tests.
const CATEGORY: Record<MCOpType, MCStepCategory> = {
  datagen: "core",
  filter: "core",
  transform: "transforms",
  test: "tests",
  regression: "regressions",
  postproc: "postproc",
};

// The window is one per step, not one per leg: a step's sources are read over
// the same rows, and every multi-source op requires them to align.
function step(stepType: MCStepType, body: StepBody): MCStepDefinition {
  const windowed =
    body.legs.length > 0
      ? [
          field("burn_in", "Burn-in", "number", 0, { minimum: 0 }),
          field("drop_initial", "Drop initial", "boolean", false),
        ]
      : [];
  return {
    step_type: stepType,
    title: body.title,
    default_name: body.defaultName,
    description: body.description,
    category: CATEGORY[body.opType],
    opType: body.opType,
    legs: body.legs,
    fields: [...body.legs.flatMap((l) => l.fields), ...windowed, ...body.fields],
  };
}

export const INPUT_SOURCES = ["states", "observables", "x_pred", "x_filt", "x1_pred", "x2_pred", "x1_filt", "x2_filt", "y_pred", "y_filt", "innov", "std_innov", "eps_hat"] as const;

const DEFINITIONS: MCStepDefinition[] = [
  step("simulation", {
    title: "Simulation",
    defaultName: "datagen",
    description: "Generate one sample by simulating a solved model (DGP or reference).",
    opType: "datagen",
    legs: [],
    fields: [
      field("target", "Simulate", "select", "dgp", { options: ["dgp", "reference"] }),
      field("T", "Periods", "number", 100, { required: true, minimum: 1 }),
      field("observables", "Observables", "boolean", true),
      field("shock_scale", "Shock scale", "number", 1.0),
      field("x0", "Initial state", "number_list", null),
      field("shock_registry", "Shocks", "shock_registry", []),
    ],
  }),
  step("filter", {
    title: "Reference Filter",
    defaultName: "filter",
    description: "Filter generated observables through the reference model.",
    opType: "filter",
    legs: [],
    fields: [
      field("filter_mode", "Mode", "select", "linear", { options: ["linear", "extended", "unscented"] }),
      field("return_shocks", "Return shocks", "boolean", false),
    ],
  }),
  step("wald", {
    title: "Wald Test",
    defaultName: "wald",
    description: "Run a HAC Wald diagnostic on a selected source.",
    opType: "test",
    legs: [
      leg("sample", "Source", true, { source: "filter", field: "std_innov" }),
    ],
    fields: [
      field("kind", "Moment", "select", "mean", { options: ["mean", "covariance", "second_moment"] }),
      field("target_vector", "Target vector", "number_list", [0.0], { required: true, when: ["mean"] }),
      field("target_matrix", "Target matrix", "number_matrix", [[1.0]], { required: true, when: ["covariance", "second_moment"] }),
      field("kernel", "Kernel", "select", "bartlett", { options: ["bartlett", "parzen", "qs"] }),
      field("bandwidth", "Bandwidth", "text", "auto"),
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("ljung_box", {
    title: "Ljung-Box Test",
    defaultName: "ljung_box",
    description: "Test one selected series for serial correlation.",
    opType: "test",
    legs: [
      leg("sample", "Source", true, { columnsLabel: "Column", single: true }),
    ],
    fields: [
      field("lags", "Lags", "number", 10, { minimum: 1 }),
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("jarque_bera", {
    title: "Jarque-Bera Test",
    defaultName: "jarque_bera",
    description: "Test one selected series for normality.",
    opType: "test",
    legs: [
      leg("sample", "Source", true, { columnsLabel: "Column", single: true }),
    ],
    fields: [
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("breusch_pagan", {
    title: "Breusch-Pagan Test",
    defaultName: "breusch_pagan",
    description: "Test residual variance against selected regressors.",
    opType: "test",
    legs: [
      leg("residuals", "Residual", false, { columns: [0], columnsLabel: "Residual column", single: true }),
      leg("X", "Regressor", false, { columns: [0], columnsLabel: "Regressor columns" }),
    ],
    fields: [
      field("robust", "Robust (Koenker)", "boolean", false),
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("breusch_godfrey", {
    title: "Breusch-Godfrey Test",
    defaultName: "breusch_godfrey",
    description: "Test residuals for serial correlation up to a given lag order.",
    opType: "test",
    legs: [
      leg("residuals", "Residual", false, { columns: [0], columnsLabel: "Residual column", single: true }),
      leg("X", "Regressor", false, { columns: [0], columnsLabel: "Regressor columns" }),
    ],
    fields: [
      field("lags", "Lags", "number", 1, { minimum: 1 }),
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("cusum", {
    title: "CUSUM Test",
    defaultName: "cusum",
    description: "Test regression coefficients for stability via recursive residuals.",
    opType: "test",
    legs: [
      leg("y", "Response", false, { columns: [0], columnsLabel: "Response column", single: true }),
      leg("X", "Regressor", false, { columns: [1], columnsLabel: "Regressor columns" }),
    ],
    fields: [
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("cusumsq", {
    title: "CUSUM of Squares Test",
    defaultName: "cusumsq",
    description: "Test regression variance stability via squared recursive residuals.",
    opType: "test",
    legs: [
      leg("y", "Response", false, { columns: [0], columnsLabel: "Response column", single: true }),
      leg("X", "Regressor", false, { columns: [1], columnsLabel: "Regressor columns" }),
    ],
    fields: [
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("chow", {
    title: "Chow Test",
    defaultName: "chow",
    description: "Test for a structural break in regression coefficients at a known break point.",
    opType: "test",
    legs: [
      leg("y", "Response", false, { columns: [0], columnsLabel: "Response column", single: true }),
      leg("X", "Regressor", false, { columns: [1], columnsLabel: "Regressor columns" }),
    ],
    fields: [
      field("t_break", "Break point", "number", 10, { required: true, minimum: 1 }),
      field("alpha", "Alpha", "number", 0.05),
    ],
  }),
  step("regression", {
    title: "Regression",
    defaultName: "regression",
    description: "Fit a linear regression in each replication.",
    opType: "regression",
    legs: [
      leg("y", "Response", false, { columns: [0], columnsLabel: "Response column", single: true }),
      leg("X", "Design", false, { columns: [1], columnsLabel: "Design columns" }),
    ],
    fields: [
      field("kind", "Kind", "select", "ols", { options: ["ols", "ridge", "ridge_gs", "lasso", "lasso_gs", "elastic_net", "elastic_net_gs"] }),
      field("intercept", "Intercept", "boolean", true),
      field("variables", "Variable names", "text_list", []),
      field("alpha", "Alpha", "number", 0.5, { when: ["ridge", "lasso", "elastic_net"] }),
      field("l1_ratio", "L1 ratio", "number", 0.5, { when: ["elastic_net", "elastic_net_gs"] }),
      field("start", "Grid start", "number", 0.01, { when: ["ridge_gs", "lasso_gs", "elastic_net_gs"] }),
      field("stop", "Grid stop", "number", 2.0, { when: ["ridge_gs", "lasso_gs", "elastic_net_gs"] }),
      field("num", "Grid points", "number", 20, { when: ["ridge_gs", "lasso_gs", "elastic_net_gs"] }),
      field("criterion", "Criterion", "select", "loss", { options: ["aic", "bic", "loss"], when: ["ridge_gs", "elastic_net_gs"] }),
      field("max_iter", "Max iterations", "number", 1000, { when: ["lasso", "lasso_gs", "elastic_net", "elastic_net_gs"] }),
      field("tol", "Tolerance", "number", 1e-10, { when: ["lasso", "lasso_gs", "elastic_net", "elastic_net_gs"] }),
    ],
  }),
  step("standardize", {
    title: "Standardize",
    defaultName: "standardize",
    description: "Per-column z-score: (x - mean) / std. Columns with zero std pass through as zeros.",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("ddof", "Degrees of freedom", "number", 0, { minimum: 0 }),
    ],
  }),
  step("log", {
    title: "Log",
    defaultName: "log",
    description: "Elementwise log(x + offset). Offset handles inputs that touch zero.",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("offset", "Offset", "number", 0.0),
    ],
  }),
  step("log_diff", {
    title: "Log Difference",
    defaultName: "log_diff",
    description: "One-period log differences along the time axis. Output is one row shorter than the input.",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("offset", "Offset", "number", 0.0),
    ],
  }),
  step("diff", {
    title: "Difference",
    defaultName: "diff",
    description: "Repeated np.diff along the time axis (order-th difference).",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("order", "Order", "number", 1, { minimum: 1 }),
    ],
  }),
  step("rolling_mean", {
    title: "Rolling Mean",
    defaultName: "rolling_mean",
    description: "Trailing rolling mean over the time axis.",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("window", "Window", "number", 10, { required: true, minimum: 1 }),
    ],
  }),
  step("rolling_std", {
    title: "Rolling Std",
    defaultName: "rolling_std",
    description: "Trailing rolling standard deviation over the time axis.",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("window", "Window", "number", 10, { required: true, minimum: 1 }),
      field("ddof", "Degrees of freedom", "number", 0, { minimum: 0 }),
    ],
  }),
  step("rolling_var", {
    title: "Rolling Variance",
    defaultName: "rolling_var",
    description: "Trailing rolling variance over the time axis.",
    opType: "transform",
    legs: [
      leg("sample", "Source", true),
    ],
    fields: [
      field("window", "Window", "number", 10, { required: true, minimum: 1 }),
      field("ddof", "Degrees of freedom", "number", 0, { minimum: 0 }),
    ],
  }),
  step("kde", {
    title: "KDE",
    defaultName: "kde",
    description: "Gaussian kernel density estimate of an across-replication trace; returns the raw (x, density) curve.",
    opType: "postproc",
    legs: [],
    fields: [
      field("trace", "Trace", "trace", "", { required: true }),
      field("bandwidth", "Bandwidth", "text", "scott"),
      field("grid_points", "Grid points", "number", 200, { minimum: 2 }),
      field("kernel", "Kernel", "select", "gaussian", { options: ["gaussian"] }),
    ],
  }),
];

// The two custom kinds carry no field-authored parameters, so they were never
// in the server payload and are declared here beside everything else. A custom
// transform still reads one source; a custom postproc reads traces by key from
// inside its own body.
const CUSTOM_DEFINITIONS: MCStepDefinition[] = [
  step("transform:custom", {
    title: "Custom Op",
    defaultName: "custom_op",
    description: "User-defined Numba transform, run once per replication.",
    opType: "transform",
    legs: [leg("sample", "Input", true)],
    fields: [
      field("output_shape", "Output shape", "number_list", [1, 1], {
        required: true,
        minimum: 0,
      }),
    ],
  }),
  step("postproc:custom", {
    title: "Custom Postproc",
    defaultName: "postproc_op",
    description: "User-defined post-loop summary op over the across-rep traces.",
    opType: "postproc",
    legs: [],
    fields: [],
  }),
];

export const MC_CATALOG: MCStepDefinition[] = [
  ...DEFINITIONS,
  ...CUSTOM_DEFINITIONS,
];

const BY_STEP_TYPE = new Map<string, MCStepDefinition>(
  MC_CATALOG.map((definition) => [definition.step_type, definition]),
);

export function stepDefinition(
  stepType: string,
): MCStepDefinition | undefined {
  return BY_STEP_TYPE.get(stepType);
}

// Lift a node's source legs out of the flat form params, in declaration order.
// The window is shared, so each leg carries the step's single burn-in.
export function splitSources(
  definition: MCStepDefinition,
  params: Record<string, unknown>,
): MCSourceSpec[] {
  const burnIn = Number(params.burn_in ?? 0) || 0;
  const dropInitial = Boolean(params.drop_initial ?? false);
  return definition.legs.map((l) => ({
    arg: l.arg,
    source_step: String(params[l.sourceKey] ?? ""),
    field: String(params[l.fieldKey] ?? ""),
    columns: asColumns(params[l.columnsKey]),
    burn_in: burnIn,
    drop_initial: dropInitial,
  }));
}

// The keys a node's params drop once its legs and window have been lifted out.
export function sourceParamKeys(definition: MCStepDefinition): string[] {
  const keys = definition.legs.flatMap((l) => [
    l.sourceKey,
    l.fieldKey,
    l.columnsKey,
  ]);
  return definition.legs.length > 0
    ? [...keys, "burn_in", "drop_initial"]
    : keys;
}

function asColumns(value: unknown): number[] | null {
  if (value === null || value === undefined || value === "") return null;
  const list = Array.isArray(value) ? value : [value];
  const columns = list.map(Number).filter((n) => Number.isFinite(n));
  return columns.length > 0 ? columns : null;
}
