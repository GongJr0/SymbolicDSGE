---
tags:
    - doc
---
# Core Containers

```python
@dataclass(frozen=True)
class MCStep(
    name: str,
    op_type: OpType,
    func: Callable[..., Any],
    kwargs: Mapping[str, Any] = {},
    source_args: tuple[SourceArgs, ...] = (),
    store_key: str | None = None,
    step_type: str | None = None,
)
```

`MCStep` describes one operation in the pipeline. Most users should create steps through the factories under `SymbolicDSGE.monte_carlo.operations`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Unique step name. Test steps use this as the key in `MCPipelineResult.test_summaries`. |
| op_type | `#!python OpType` | Operation type: `DATAGEN`, `TRANSFORM`, `FILTER`, `TEST`, `REGRESSION`, or `POSTPROC`. |
| func | `#!python Callable` | Callable executed by the pipeline. |
| kwargs | `#!python Mapping[str, Any]` | Keyword arguments stored with the step and passed into `func`. |
| source_args | `#!python tuple[SourceArgs, ...]` | Compiled source selections resolved when the pipeline is built. |
| store_key | `#!python str | None` | Optional payload key. If omitted, `name` is used. Transform `store_key` values may be used as source names downstream. |
| step_type | `#!python str | None` | Serializable step kind stamped by the factory, for example `"wald"`, `"simulation"`, `"standardize"`, `"transform:custom"`, or `"postproc:custom"`. `None` is reserved for hand-built steps that cannot be projected to a `PipelineSpec`. |

???+ note "Factory groups"
    Step factories are organized by operation group:

    - `SymbolicDSGE.monte_carlo.operations.core`
    - `SymbolicDSGE.monte_carlo.operations.tests`
    - `SymbolicDSGE.monte_carlo.operations.regressions`
    - `SymbolicDSGE.monte_carlo.operations.transforms`

&nbsp;

```python
class MCData(NamedTuple):
    states: ndarray | None = None,
    observables: ndarray | None = None,
    n_exog: int = -1,
    raw: Mapping[str, ndarray] = {},
    observable_names: tuple[str, ...] = (),
)
```

`MCData` is the standard per-replication data payload. State-only and observable-only payloads are supported, but downstream steps may require one or the other.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| states | `#!python ndarray | None` | Simulated or supplied state matrix. |
| observables | `#!python ndarray | None` | Simulated or supplied observable matrix. |
| n_exog | `#!python int` | Number of exogenous shocks when known. |
| raw | `#!python Mapping[str, ndarray]` | Additional raw arrays, usually from `SolvedModel.sim(...)`. |
| observable_names | `#!python tuple[str, ...]` | Observable column names used by the reference filter when explicit names are not supplied. |

&nbsp;

```python
@dataclass(frozen=True)
class SourceArgs(
    arg: str,
    source_step: str,
    source_idx: int,
    source_kind: int,
    field: str,
    field_idx: int,
    columns: int | Sequence[int] | slice | ndarray | None = None,
    column_selector: Sequence[int] | slice = slice(None),
    row_start: int = 0,
    burn_in: int = 0,
    drop_initial: bool = False,
)
```

`SourceArgs` is the compiled source selector used by built-in transforms, tests, and regressions. Factories create it from public `source` and `field` arguments.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| arg | `#!python str` | Runner keyword populated with the selected array, such as `"sample"`, `"y"`, or `"X"`. |
| source_step | `#!python str` | Producer step name after pipeline binding. |
| source_idx | `#!python int` | Producer position in the per replication payload slot list. |
| source_kind | `#!python int` | Internal source class: data, transform payload, or filter output. |
| field | `#!python str` | Field read from the producer, such as `"observables"`, `"std_innov"`, or `"payload"`. |
| field_idx | `#!python int` | Positional field index used inside the runner. |
| columns | `#!python int | Sequence[int] | slice | ndarray | None` | Author supplied column selector. |
| column_selector | `#!python Sequence[int] | slice` | Normalized selector used by the runner. |
| row_start | `#!python int` | First selected row after applying `burn_in` and `drop_initial`. |
| burn_in | `#!python int` | Number of leading rows to drop. |
| drop_initial | `#!python bool` | If `True` and `burn_in` is zero, start at row `1`. |

???+ warning "Source fields"
    Source fields are tied to the producer type. Data steps expose `states` and `observables`; transform steps expose `payload`; filter steps expose raw filter fields such as `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov`, `eps_hat`, `x1_pred`, `x2_pred`, `x1_filt`, and `x2_filt`. Built-in array consumers expect the selected field to resolve to a 2D numeric array.

&nbsp;

```python
@dataclass
class MCContext(
    rep_idx: int,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    data: MCData | None = None,
    payload_slots: list[Any] = [],
    payloads: dict[str, Any] = {},
    results: dict[str, TestResult] = {},
    regressions: dict[str, RegressionResult] = {},
)
```

`MCContext` is the mutable object passed through a single replication. Transform, filter, test, and regression operations receive it as `context`. Postproc operations run after the replication loop and receive the assembled `traces` mapping instead.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| rep_idx | `#!python int` | Replication index. |
| reference | `#!python SolvedModel` | Reference model for filtering and reference target simulation. |
| dgp | `#!python SolvedModel | None` | DGP model for DGP target simulation. |
| data | `#!python MCData | None` | Current replication data, populated by the data step. |
| payload_slots | `#!python list[Any]` | Ordered producer outputs used by compiled source selectors. |
| payloads | `#!python dict[str, Any]` | Outputs keyed by step output key for user inspection and custom operations. |
| results | `#!python dict[str, TestResult]` | Scalar test results keyed by test step name. |
| regressions | `#!python dict[str, RegressionResult]` | Regression results keyed by regression step name. |

__Methods:__

| __Name__ | __Description__ |
|:---------|----------------:|
| `require_data()` | Return `data` or raise if no data-generation step has populated it. |
| `require_payload(key)` | Return a payload by key or raise if the key is missing. |

&nbsp;

```python
@dataclass(frozen=True)
class MCFailure(
    rep_idx: int,
    step_name: str,
    error_type: str,
    message: str,
)
```

`MCFailure` records one collected replication failure when `MCPipeline.run(..., fail_fast=False)`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| rep_idx | `#!python int` | Replication index that failed. |
| step_name | `#!python str` | Step executing when the failure occurred. |
| error_type | `#!python str` | Exception type name. |
| message | `#!python str` | Exception message. |

&nbsp;

```python
@dataclass(frozen=True)
class MCMeta(
    n_rep: int,
    payloads_retained: bool,
    test_results_retained: bool,
    contexts_retained: bool,
    elapsed_s: float = 0.0,
    step_elapsed_s: Mapping[str, float] = {},
    step_counts: Mapping[str, int] = {},
    step_failures: Mapping[str, int] = {},
    postproc_elapsed_s: Mapping[str, float] = {},
    failed_steps: dict[str, int] = {},
    failed_postprocs: set[str] = set(),
)
```

`MCMeta` stores run accounting and performance counters.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| n_rep | `#!python int` | Requested replication count. |
| payloads_retained | `#!python bool` | Whether per replication payload dictionaries were retained. |
| test_results_retained | `#!python bool` | Whether scalar test results were retained. |
| contexts_retained | `#!python bool` | Whether full contexts were retained. |
| elapsed_s | `#!python float` | Wall time for the replication loop. |
| step_elapsed_s | `#!python Mapping[str, float]` | Wall time by per replication step. |
| step_counts | `#!python Mapping[str, int]` | Attempted calls by per replication step. |
| step_failures | `#!python Mapping[str, int]` | Collected failures by per replication step. |
| postproc_elapsed_s | `#!python Mapping[str, float]` | Wall time by post-loop step. |
| failed_steps | `#!python dict[str, int]` | Collected per replication failures by step. |
| failed_postprocs | `#!python set[str]` | Post-loop steps that failed. |
| it_s | `#!python float` | Replications attempted per replication loop second. |
| step_it_s | `#!python Mapping[str, float]` | Step calls attempted per step wall second. |
| postproc_total_s | `#!python float` | Total post-loop wall time. |
| steps_success | `#!python bool` | `True` when no per replication failures were collected. |
| postproc_success | `#!python bool` | `True` when no post-loop failures were collected. |

&nbsp;

```python
@dataclass(frozen=True)
class MCPipelineResult(
    meta: MCMeta,
    n_rep: int,
    n_successful: int,
    test_summaries: Mapping[str, MCResult],
    test_results: Mapping[str, tuple[TestResult, ...]] | None,
    payloads: tuple[Mapping[str, Any], ...] | None,
    contexts: tuple[MCContext, ...] | None,
    failures: tuple[MCFailure, ...] = (),
    regression_summaries: Mapping[str, MCRegressionResult] = {},
    postproc: Mapping[str, Any] = {},
)
```

`MCPipelineResult` is the aggregate return object from `MCPipeline.run(...)`.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| meta | `#!python MCMeta` | Run metadata and performance counters. |
| n_rep | `#!python int` | Requested replication count. |
| n_successful | `#!python int` | Number of completed replications. |
| test_summaries | `#!python Mapping[str, MCResult]` | Per-test aggregate result containers. |
| test_results | `#!python Mapping[str, tuple[TestResult, ...]] | None` | Optional scalar per-replication test results. |
| payloads | `#!python tuple[Mapping[str, Any], ...] | None` | Optional per-replication payload dictionaries. |
| contexts | `#!python tuple[MCContext, ...] | None` | Optional full contexts. |
| failures | `#!python tuple[MCFailure, ...]` | Failures collected when `fail_fast=False`. |
| regression_summaries | `#!python Mapping[str, MCRegressionResult]` | Per-regression aggregate result containers. |
| postproc | `#!python Mapping[str, Any]` | Post-loop artifacts keyed by step name or nested artifact key. |
| succeeded | `#!python bool` | `True` when no failures were collected. |
| statistic_traces | `#!python Mapping[str, ndarray]` | Shortcut for each test summary's statistic trace. |
| pval_traces | `#!python Mapping[str, ndarray]` | Shortcut for each test summary's p-value trace. |
| test_status_traces | `#!python Mapping[str, tuple[TestStatus, ...]]` | Shortcut for each test summary's status trace. |
| rejection_traces | `#!python Mapping[str, ndarray]` | Boolean rejection trace for each test summary. |
| coefficient_traces | `#!python Mapping[str, ndarray]` | Shortcut for each regression summary's coefficient trace. |
| regression_status_traces | `#!python Mapping[str, tuple[RegressionStatus, ...]]` | Shortcut for each regression summary's status trace. |
| `report_performance()` | `#!python None` | Print the aggregate pipeline throughput report. |
| `report_step_performance()` | `#!python None` | Print one throughput report line per pipeline step. |

???+ note "P-Value Evaluation"
    Scalar `TestResult` objects produced inside Monte Carlo Wald steps defer p-value and frozen-distribution construction until `pval`, `frozen_dist`, or `compute_pval()` is accessed. Aggregate `MCResult` objects compute vectorized p-values when `MCPipelineResult.test_summaries` is built.
