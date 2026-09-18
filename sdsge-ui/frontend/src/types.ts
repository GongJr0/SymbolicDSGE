export type Role = "reference" | "dgp";

export interface ArrayEnvelope {
  shape: number[];
  data_b64: string;
}

export interface NamedArray {
  name: string;
  array: ArrayEnvelope;
}

export type ShockDistribution = "norm" | "t" | "uni";

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
  shocks?: string[];
  n_state?: number;
  n_exog?: number;
  A_shape?: number[];
  B_shape?: number[];
  has_kalman?: boolean;
}

export type WorkspaceTab = "estimation" | "mc";

/** The estimation tab as it stands on screen.
 *
 * Client-owned and held verbatim by the server, so a new control lands here
 * without a backend change. Carries no result: that is the tab's `result`
 * slot, which only a run fills.
 */
export interface MapOptions {
  method?: string;
  m?: number;
  maxiter?: number;
  maxfun?: number;
  maxls?: number;
  factr?: number;
  pgtol?: number;
  fd_step?: number;
  xatol?: number;
  fatol?: number;
  // Carried through untouched; the sampler's own tab renders no bounds table.
  bounds?: Array<[number | null, number | null]> | null;
}

export interface EstimationViewState {
  routine: EstimationMethod;
  parameters: EstimationParameterSpec[];
  selected: string | null;
  observables: string;
  dataVectors: Record<string, string>;
  // Every knob below is spelled the way `Estimator.mle`/`mcmc` takes it, so the
  // form posts them without translating and a run restores them without a map.
  optimizer: string;
  maxiter: number;
  maxfun: number;
  m: number;
  maxls: number;
  factr: number;
  pgtol: number;
  fd_step: number;
  xatol: number;
  fatol: number;
  n_draws: number;
  burn_in: number;
  thin: number;
  random_state: number;
  proposal_scale: number;
  adapt: boolean;
  adapt_start: number;
  adapt_epsilon: number;
  posterior_point: string;
  // No control on a fresh form. A bundle whose run set one away from its
  // default reveals it, so what is on screen is what will run.
  cov: boolean;
  jacobian: boolean;
  compute_map: boolean;
  cov_fd_step_scale: number;
  cov_fd_absolute_floor: number;
  // The MAP presolve's own optimizer options, passed to the sampler verbatim.
  // Null until touched, which leaves the estimator on its own defaults. Every
  // field is optional: a run records only what it was given.
  map_options: MapOptions | null;
  // Restored and re-posted, but too structured for a scalar control.
  proposal_cov: number[][] | null;
  modeFolded: boolean;
}

/** The MC tab as it stands on screen.
 *
 * Carries `pipeline` as well as the layout: an edited graph that has not been
 * run has no `spec` yet, since the server only writes that slot from a run.
 */
export interface MCViewState {
  pipeline: MCPipelineSpec;
  positions: Record<string, { x: number; y: number }>;
  /** The canvas connections, including any drawn before their leg was bound.
   *  A bound one is recoverable from the pipeline's `source_args`; an unbound
   *  one exists nowhere else, and it is what makes its producer selectable. */
  edges: MCEdgeSpec[];
  n_rep: number;
  n_jobs: number | null;
  verbosity: number;
  fail_fast: boolean;
}

/** One tab's slots on the session.
 *
 * `spec` and `result` are server-written, from a run or the bundle the UI was
 * launched with; `view` is the only slot the client writes. Each is absent
 * until something fills it.
 */
export interface WorkspaceTabState<TSpec, TResult, TView> {
  spec?: TSpec;
  result?: TResult;
  view?: TView;
}

/** An `EstimatorSpec` as a bundle stores it: observed data plus the
 * estimator's construction arguments, with nothing the form added. */
export interface EstimatorSpecWire {
  y: number[][];
  params: Record<string, unknown>;
}

/** Everything a reload repaints from.
 *
 * Lives in the server process, which the refresh does not restart, so this is
 * the whole restore mechanism: nothing is kept on the client.
 */
/** Estimation views keyed by role: the form is per-role, the tab's other two
 * slots are not (a bundle holds one estimation, against the reference model).
 * Partial per role so a bundle can fill only what it knows and the form merges
 * the rest from its own defaults. */
export type EstimationViewsByRole = Partial<
  Record<Role, Partial<EstimationViewState>>
>;

export interface SessionWorkspace {
  estimation?: WorkspaceTabState<
    EstimatorSpecWire,
    EstimationRunResult["result"],
    EstimationViewsByRole
  >;
  mc?: WorkspaceTabState<MCPipelineSpec, MCPipelineResult, MCViewState>;
  // Per role, and with no view: the Outputs tab renders only `T` and the
  // observables toggle, both of them fields of the spec itself.
  simulation?: Partial<Record<Role, WorkspaceTabState<SimSpecWire, SimResult, never>>>;
}

export interface SessionSummary {
  models: Record<Role, ModelSummary>;
  workspace: SessionWorkspace;
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

/** One drawn shock family: `ShockParameters` as the library spells it. */
export interface DrawnShock {
  // The shocks this entry drives. An entry names one or more of them, which a
  // JSON object cannot be keyed by, so each entry carries its own targets and a
  // spec travels as a list.
  target: string[];
  dist: string;
  seed: number | null;
  dist_kwargs: Record<string, unknown>;
}

/** One supplied path: `ShockPathParameters`, an array of shape (T, width). */
export interface PathShock {
  target: string[];
  path: number[][];
}

export type ShockEntry = DrawnShock | PathShock;

/** A `SimSpec` as a bundle stores it, and as a run posts it. The tab's controls
 * are a subset of its fields, which is why the simulation slot carries no
 * separate view. */
export interface SimSpecWire {
  T: number;
  x0: Record<string, number> | number[] | null;
  observables: boolean;
  shock_scale: number;
  /** One self-describing entry per spec entry, each carrying its own `target`. */
  shocks: ShockEntry[] | null;
}

export interface SimResult {
  kind: "sim";
  role: Role;
  T: number;
  observables: boolean;
  series: NamedArray[];
  figures?: FigureResult[];
  // Transforms that produced no series, and why. A failure is otherwise
  // indistinguishable from the submit not having worked.
  transform_errors?: Array<{ name: string; error: string }>;
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
  /** Which estimation to run. Named as the library names it, which keeps it
   *  distinct from `method_kwargs.method`, the optimizer an mle or map run
   *  hands to the solver. */
  routine: EstimationMethod;
  y: number[][];
  observables: string[] | null;
  parameters: EstimationParameterSpec[];
  method_kwargs: Record<string, unknown>;
  compile_kwargs: Record<string, unknown>;
  ss_seed: number[] | null;
  posterior_point: string;
  estimate_and_solve: boolean;
}

/** A result as the wire carries it: what the run produced, with none of the
 * session framing (`role`, `solved`) the run envelope adds. This is
 * the shape the workspace's `result` slot holds. */
export type EstimationResultWire = EstimationRunResult["result"];

export interface EstimationRunResult {
  kind: "estimation";
  role: Role;
  routine: EstimationMethod;
  solved: boolean;
  result: {
    // Opt results (mle/map) carry no inner kind; only the mcmc wire sets it.
    kind?: EstimationMethod;
    success?: boolean;
    message?: string;
    // The optimum, in the unconstrained space the optimizer moved through.
    // `theta` is the same point in the model's own parameters.
    x?: Array<number | null>;
    theta?: Record<string, number | null>;
    fun?: number | null;
    // Asymptotic covariance at the optimum, ordered like `x`. Null throughout
    // when the Hessian there was not positive definite; absent when the run
    // computed no covariance at all.
    vcov?: Array<Array<number | null>> | null;
    // Keyed like `theta`; an entry is null where the covariance gave no
    // finite standard error.
    se?: Record<string, number | null> | null;
    cov_status?: number;
    // The call arguments the run was made with, for reproducing it.
    optimizer_config?: Record<string, unknown>;
    sampler_config?: Record<string, unknown>;
    loglik?: number;
    logprior?: number;
    logpost?: number;
    nfev?: number;
    nit?: number | null;
    param_names?: string[];
    posterior_mean?: Record<string, number>;
    samples?: Record<string, number[]>;
    logpost_trace?: number[];
    logjac_trace?: number[];
    accept_rate?: number | null;
    n_draws?: number;
    burn_in?: number;
    thin?: number;
    logpost_mean?: number | null;
    logpost_min?: number | null;
    logpost_max?: number | null;
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
  | "payload"
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

// One entry in a simulation step's shock registry, in the form the panel edits.
// The two kinds mirror the two the spec carries, and are told apart the same
// way: a drawn entry names a family, a path entry supplies the array. Both name
// their own innovations under `target`, which is the spec's own field; two
// entries may not name the same selection.

/** A family drawn over the chosen innovations. `target.length > 1` is one joint
 *  (multivar) shock, which is what lets the calibrated correlations apply. */
export interface DrawnRegistryEntry {
  kind: "drawn";
  target: string[];
  dist: ShockDistribution;
  loc: number[];
  df: number;
  seed: number | null;
}

/** An array supplied for the chosen innovations, held as the `(T, width)`
 *  matrix the spec wants rather than as the text some editor typed. */
export interface PathRegistryEntry {
  kind: "path";
  target: string[];
  path: number[][];
}

export type ShockRegistryEntry = DrawnRegistryEntry | PathRegistryEntry;

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

// One authored source leg of a step. `column_selector` and `row_start` are
// derived server-side, so only the fields an author sets travel.
export interface MCSourceSpec {
  arg: string;
  source_step: string;
  field: string;
  columns: number[] | null;
  burn_in: number;
}

export interface MCStepSpec {
  name: string;
  op_type: string;
  step_type: MCStepType;
  kwargs: Record<string, unknown>;
  source_args: MCSourceSpec[];
  n_retain: number;
  /** Custom-op source. Compiled server-side into the step's callable, so it is
   *  a sibling of `kwargs` and never one of them. */
  code?: string;
}

export interface MCEdgeSpec {
  source: string;
  target: string;
}


export interface MCPipelineSpec {
  replication_steps: MCStepSpec[];
  postproc_steps: MCStepSpec[];
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

export interface MCTestSummary {
  test_name: string;
  n_rep: number;
  n_retained: number;
  retained_reps: number[];
  alpha: number;
  distribution: string;
  df: number | Array<number | null> | null;
  pval_method: string;
  mean_statistic: number;
  mean_pval: number;
  rejection_rate: number;
  statistic_se: number | null;
  pval_se: number | null;
  rejection_rate_se: number | null;
  statistic_ci: Array<number | null>;
  pval_ci: Array<number | null>;
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
  n_retained: number;
  retained_reps: number[];
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
  kind: "mc";
  n_rep: number;
  n_retained_by_step: Record<string, number>;
  n_successful: number;
  succeeded: boolean;
  elapsed_s: number;
  it_s: number;
  step_elapsed_s: Record<string, number>;
  step_it_s: Record<string, number>;
  step_worker_it_s: Record<string, number>;
  step_wall_it_s: Record<string, number>;
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
