export type Role = "reference" | "dgp";

export interface ArrayEnvelope {
  dtype: "float64";
  shape: number[];
  order: "C";
  data_b64: string;
}

export interface NamedArray {
  name: string;
  array: ArrayEnvelope;
}

export interface ShockSpec {
  shock: string;
  target: string;
  std_param: string | null;
  std_value: number | null;
}

export interface ShockCorrSpec {
  pair: string[];
  key: string;
  corr_param: string;
  corr_value: number | null;
}

export type ShockDistribution = "norm" | "t" | "uni";

export interface ShockGeneration {
  dist: ShockDistribution;
  seed: number | null;
  loc: number;
  df: number;
}

export interface ShockParamUpdate {
  std: Record<string, number>;
  corr: Record<string, number>;
}

export interface ModelSummary {
  role: Role;
  loaded: boolean;
  solved: boolean;
  source?: string;
  raw_yaml?: string;
  name?: string;
  variables?: string[];
  observables?: string[];
  parameters?: string[];
  parameter_values?: Record<string, number>;
  shock_specs?: ShockSpec[];
  shock_corr_specs?: ShockCorrSpec[];
  n_state?: number;
  n_exog?: number;
  A_shape?: number[];
  B_shape?: number[];
  has_kalman?: boolean;
}

export interface SessionSummary {
  models: Record<Role, ModelSummary>;
  runs: Array<{ run_id: string; kind: string; role: Role }>;
}

export type FunctionKind = "array" | "figure";

export interface FunctionRecord {
  name: string;
  kind: FunctionKind;
  source: string;
}

export interface FigureResult {
  name: string;
  image_b64?: string;
  error?: string;
}

export interface SimResult {
  run_id: string;
  kind: "sim";
  role: Role;
  T: number;
  observables: boolean;
  series: NamedArray[];
  figures?: FigureResult[];
}

export type EstimationMethod = "mle" | "map" | "mcmc";

export interface EstimationCatalog {
  distributions: Record<string, Record<string, number | null>>;
  transforms: Record<string, Record<string, number | null>>;
  optimizer_methods: string[];
  posterior_points: string[];
}

export interface EstimationPriorSpec {
  distribution: string;
  parameters: Record<string, number>;
  transform: string;
  transform_kwargs: Record<string, number>;
}

export interface EstimationParameterSpec {
  name: string;
  estimate: boolean;
  initial: number;
  lower: number | null;
  upper: number | null;
  prior: EstimationPriorSpec | null;
}

export interface EstimationRunRequest {
  role: Role;
  method: EstimationMethod;
  y: number[][];
  observables: string[] | null;
  parameters: EstimationParameterSpec[];
  method_kwargs: Record<string, unknown>;
  compile_kwargs: Record<string, unknown>;
  steady_state: number[] | null;
  posterior_point: string;
  estimate_and_solve: boolean;
}

export interface EstimationRunResult {
  run_id: string;
  kind: "estimation";
  role: Role;
  method: EstimationMethod;
  solved: boolean;
  result: {
    kind: EstimationMethod;
    success?: boolean;
    message?: string;
    theta?: Record<string, number>;
    fun?: number;
    loglik?: number;
    logprior?: number;
    logpost?: number;
    nfev?: number;
    nit?: number | null;
    param_names?: string[];
    posterior_mean?: Record<string, number>;
    posterior_std?: Record<string, number>;
    samples?: Record<string, number[]>;
    logpost_trace?: number[];
    accept_rate?: number;
    n_draws?: number;
    burn_in?: number;
    thin?: number;
    logpost_mean?: number;
    logpost_min?: number;
    logpost_max?: number;
  };
}

export type MCStepType =
  | "simulation"
  | "filter"
  | "wald"
  | "ljung_box"
  | "jarque_bera"
  | "breusch_pagan"
  | "breusch_godfrey"
  | "cusum"
  | "cusumsq"
  | "chow"
  | "regression"
  | "standardize"
  | "log"
  | "log_diff"
  | "diff"
  | "rolling_mean"
  | "rolling_std"
  | "rolling_var"
  | "kde"
  | "transform:custom"
  | "postproc:custom";

export type MCStepCategory =
  | "core"
  | "transforms"
  | "tests"
  | "regressions"
  | "postproc";

export type MCFieldType =
  | "text"
  | "number"
  | "boolean"
  | "select"
  | "trace"
  | "number_list"
  | "number_matrix"
  | "text_list"
  | "shock_registry";

// One entry in a simulation step's shock registry: an explicit, free-form shock
// over a chosen set of the target model's exogenous variables. `vars.length > 1`
// is a joint (multivar) shock; the joined names form the registry key.
export interface ShockRegistryEntry {
  vars: string[];
  dist: ShockDistribution;
  loc: number;
  df: number;
  seed: number | null;
}

export interface MCFieldSpec {
  key: string;
  label: string;
  type: MCFieldType;
  default: unknown;
  required: boolean;
  options: string[];
  minimum: number | null;
  when: string[];
}

export interface MCStepCatalogItem {
  step_type: MCStepType;
  title: string;
  default_name: string;
  description: string;
  category: MCStepCategory;
  fields: MCFieldSpec[];
}

export interface MCCatalog {
  steps: MCStepCatalogItem[];
}

export interface MCNodeSpec {
  id: string;
  step_type: MCStepType;
  name: string;
  params: Record<string, unknown>;
}

export interface MCEdgeSpec {
  source: string;
  target: string;
}

// A post-loop op. Not a graph node -- no `id`/edges; it references producers by
// trace key in `params` and runs once over the assembled traces.
export interface MCPostprocSpec {
  step_type: MCStepType;
  name: string;
  params: Record<string, unknown>;
}

export interface MCPipelineSpec {
  nodes: MCNodeSpec[];
  edges: MCEdgeSpec[];
  postprocs: MCPostprocSpec[];
}

export interface MCTraceSummary {
  n: number;
  n_finite: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  q025: number | null;
  q975: number | null;
}

export interface MCDataSummary {
  n_rep: number;
  shape: number[];
  n_values: number;
  n_finite: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

export interface MCTestSummary {
  test_name: string;
  n: number;
  alpha: number;
  distribution: string;
  df: number | Array<number | null> | null;
  pval_method: string;
  mean_statistic: number;
  mean_pval: number;
  rejection_rate: number;
  statistic_se: number | null;
  pval_se: number | null;
  statistic_ci: Array<number | null>;
  rejection_ci: Array<number | null>;
  statistic_trace: Array<number | null>;
  pval_trace: Array<number | null>;
  status_trace: number[];
  status_counts: Record<string, number>;
  statistic_summary: MCTraceSummary;
  pval_summary: MCTraceSummary;
}

export interface MCRegressionSummary {
  variables: string[];
  n_rep: number;
  n: number;
  k: number;
  coef_trace: Array<Array<number | null>>;
  r2_trace: Array<number | null>;
  status_trace: number[];
  status_counts: Record<string, number>;
  coefficient_summaries: Array<MCTraceSummary & { variable: string }>;
  metrics: Record<string, MCTraceSummary>;
  ols: null | {
    mean_standard_errors: Array<number | null>;
    mean_t_statistics: Array<number | null>;
    mean_pvalues: Array<number | null>;
    mean_partial_r2: Array<number | null>;
    f_statistic: MCTraceSummary;
    f_pvalue: MCTraceSummary;
  };
}

export interface MCPipelineResult {
  run_id: string;
  kind: "mc";
  n_rep: number;
  n_successful: number;
  succeeded: boolean;
  elapsed_s: number;
  it_s: number;
  step_elapsed_s: Record<string, number>;
  step_it_s: Record<string, number>;
  step_counts: Record<string, number>;
  step_failures: Record<string, number>;
  failures: Array<{
    rep_idx: number;
    step_name: string;
    error_type: string;
    message: string;
  }>;
  test_summaries: Record<string, MCTestSummary>;
  regression_summaries: Record<string, MCRegressionSummary>;
  data_summaries: Record<string, MCDataSummary>;
  postproc?: Record<string, MCPostprocArtifact>;
}

// A post-loop (POSTPROC) artifact, one per summary surface. `scalar` carries an
// inline `value`; `array` an `value` (1-D or N-D nested arrays); `table` a
// columnar `data` map plus `columns`/`dtypes`/`index` metadata.
export interface MCPostprocArtifact {
  kind: "summary" | "raw";
  artifact: "scalar" | "array" | "table";
  title?: string | null;
  render?: string;
  value?: unknown;
  shape?: number[];
  columns?: string[];
  dtypes?: Record<string, string>;
  index?: { kind: string; name: string | null };
  data?: Record<string, unknown[]>;
}
