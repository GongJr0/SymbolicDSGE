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
    func: Callable[..., Any] | None = None,
    kwargs: Mapping[str, Any] = {},
    source_args: tuple[SourceArgs, ...] = (),
    step_type: str | None = None,
    n_retain: int = -1,
)
```

`MCStep` describes one operation in the pipeline. Most users should create steps through the factories in `SymbolicDSGE.monte_carlo.step_factories`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Unique step name. Test steps use this as the key in `MCPipelineResult.test_summaries`, and every producer derives its across-replication trace keys from it. |
| op_type | `#!python OpType` | Operation type: `DATAGEN`, `TRANSFORM`, `FILTER`, `TEST`, `REGRESSION`, or `POSTPROC`. |
| func | `#!python Callable | None` | Callable carried by the step. Built-in per-replication steps run as native kernels and leave this `None`; custom transforms carry a `NumbaCustomFunc` and post-loop ops carry a Python callable. |
| kwargs | `#!python Mapping[str, Any]` | Keyword arguments stored with the step. Per-replication kernels read them when the run is lowered; post-loop ops receive them at call time. |
| source_args | `#!python tuple[SourceArgs, ...]` | Compiled source selections resolved when the pipeline is built. |
| step_type | `#!python str | None` | Serializable step kind stamped by the factory, for example `"wald"`, `"simulation"`, `"standardize"`, `"transform:custom"`, or `"postproc:custom"`. `None` is reserved for hand-built steps that cannot be projected to a `PipelineSpec`. |
| n_retain | `#!python int` | Number of replications whose output is retained for this step. `-1` retains all `n_rep` replications. A non-negative value sizes the step's arena to that many rows, filled from an evenly spaced subset of replication indices. It may not exceed `n_rep`. |

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python .to_spec()` | `#!python StepSpec` | The step as data, with its callable and its bulk arrays carried beside it. |
| `#!python MCStep.from_spec(spec)` | `#!python MCStep` | Rebuild a step from the form `to_spec()` records. |

???+ note "Factory module"
    All step factories live in `SymbolicDSGE.monte_carlo.step_factories`: data generation, filtering, transforms, tests, regressions, and post-processing.

???+ warning "What `to_spec()` refuses"
    Only what cannot travel as data: a replication step carrying a `func` that is not a custom transform, and a `shocks` entry that is a callable. Everything else projects, including a `None` `step_type`, which no pipeline can run anyway. Post-loop steps are exempt from the `func` rule, since every post-loop kind either names its own callable or ships one.

???+ note "`shocks`"
    A kwarg named `shocks` holding a mapping is read as one shock spec per name, whichever step carries it. Each entry is either a generator spec, which serializes itself, or a shock path, which travels as nested lists and comes back as an array. `raw_model_data` passes `shocks` as a bare array rather than a mapping, so it rides `StepSpec.arrays` like the other bulk kwargs.

&nbsp;

```python
@dataclass(frozen=True, slots=True)
class SourceArgs(
    arg: str,
    source_step: str,
    field: str,
    columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
)
```

`SourceArgs` is the compiled source selector used by transforms, tests, and regressions. Factories create it from public `source` and `field` arguments, and the native lowering layer resolves it to concrete buffer offsets before the run starts.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| arg | `#!python str` | Role the selected array fills, such as `"sample"`, `"y"`, or `"X"`. |
| source_step | `#!python str` | Producer step name after pipeline binding. |
| field | `#!python str` | Field read from the producer, such as `"observables"`, `"std_innov"`, or `"payload"`. |
| columns | `#!python int | Sequence[int] | slice | ndarray | None` | Author supplied column selector, normalized to a tuple of ints or a slice at construction. |
| burn_in | `#!python int` | Number of leading rows to drop. |
| column_selector | `#!python Sequence[int] | slice` | Selected columns, with an unset `columns` standing for all of them. |

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python .to_spec()` | `#!python SourceSpec` | The selector as plain data, keyed by constructor argument. |
| `#!python SourceArgs.from_spec(spec)` | `#!python SourceArgs` | Rebuild a selector from the form `to_spec()` records. |

???+ note "Column selectors in a spec"
    `SourceSpec.columns` is a list of ints for explicit indices. A slice carries no width to resolve against, so it travels as an object of its own `start`, `stop`, and `step`, read back by name. The differing JSON types are what keep the two forms apart.

???+ warning "Source fields"
    Source fields are tied to the producer type. Data steps expose `states` and `observables`; transform steps expose `payload`; filter steps expose raw filter fields such as `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov`, `eps_hat`, `x1_pred`, `x2_pred`, `x1_filt`, and `x2_filt`. Array consumers expect the selected field to resolve to a 2D numeric array.

&nbsp;

```python
@dataclass(frozen=True, slots=True)
class MCFailure(
    rep_idx: int,
    step_name: str,
    error_type: str,
    message: str,
)
```

`MCFailure` records one collected replication failure when `MCPipeline.run(..., fail_fast=False)`. Post-loop failures are recorded with `rep_idx = -1`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| rep_idx | `#!python int` | Replication index that failed, or `-1` for a post-loop step. |
| step_name | `#!python str` | Step executing when the failure occurred. |
| error_type | `#!python str` | Exception type name. |
| message | `#!python str` | Exception message. |

&nbsp;

```python
@dataclass(frozen=True)
class MCMeta(
    n_rep: int,
    n_retained_by_step: Mapping[str, int],
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
| n_retained_by_step | `#!python Mapping[str, int]` | Replications whose output was retained, by producer step. |
| elapsed_s | `#!python float` | Wall time for the replication loop alone, excluding post-loop aggregation and postproc. |
| step_elapsed_s | `#!python Mapping[str, float]` | Accumulated worker seconds by per-replication step. Populated only when the run is started with `verbosity=2`. |
| step_counts | `#!python Mapping[str, int]` | Attempted calls by per-replication step. |
| step_failures | `#!python Mapping[str, int]` | Collected failures by per-replication step. |
| postproc_elapsed_s | `#!python Mapping[str, float]` | Wall time by post-loop step. |
| failed_steps | `#!python dict[str, int]` | Collected per-replication failures by step. |
| failed_postprocs | `#!python set[str]` | Post-loop steps that failed. |
| it_s | `#!python float` | Replications attempted per replication loop second. |
| step_worker_it_s | `#!python Mapping[str, float]` | Exclusive per-step throughput against accumulated worker seconds. |
| step_wall_it_s | `#!python Mapping[str, float]` | Per-step throughput against the replication loop's wall time. |
| step_it_s | `#!python Mapping[str, float]` | Alias for `step_worker_it_s`. |
| postproc_total_s | `#!python float` | Total post-loop wall time. |
| steps_success | `#!python bool` | `True` when no per-replication failures were collected. |
| postproc_success | `#!python bool` | `True` when no post-loop failures were collected. |

&nbsp;

```python
@dataclass(frozen=True)
class MCPipelineResult(
    meta: MCMeta,
    n_rep: int,
    n_successful: int,
    datagen_outputs: MCDataGenResult,
    filter_outputs: Mapping[str, MCFilterResult] = {},
    transform_outputs: Mapping[str, ndarray] = {},
    test_summaries: Mapping[str, MCTestResult] = {},
    regression_summaries: Mapping[str, MCRegressionResult] = {},
    failures: tuple[MCFailure, ...] = (),
    postproc: Mapping[str, Artifact] = {},
    run_config: Mapping[str, Any] = {},
)
```

`MCPipelineResult` is the aggregate return object from `MCPipeline.run(...)`.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| meta | `#!python MCMeta` | Run metadata and performance counters. |
| n_rep | `#!python int` | Requested replication count. |
| n_successful | `#!python int` | Number of completed replications. |
| datagen_outputs | `#!python MCDataGenResult` | The single datagen step's retained states, shocks and observables. A field the step never produced is reported as a `NaN` block rather than omitted. |
| filter_outputs | `#!python Mapping[str, MCFilterResult]` | Per-filter retained output, keyed by step name. |
| transform_outputs | `#!python Mapping[str, ndarray]` | Retained transform output stacked across replications, keyed by step name, each shaped `(n_retained, *output_shape)`. |
| test_summaries | `#!python Mapping[str, MCTestResult]` | Per-test aggregate result containers. |
| regression_summaries | `#!python Mapping[str, MCRegressionResult]` | Per-regression aggregate result containers. |
| failures | `#!python tuple[MCFailure, ...]` | Failures collected when `fail_fast=False`. |
| postproc | `#!python Mapping[str, Artifact]` | Post-loop artifacts keyed by step name. Values are `Artifact` instances holding optional `Raw` and `Summary` data. |
| succeeded | `#!python bool` | `True` when no per-replication or post-loop failures were collected. |
| statistic_traces | `#!python Mapping[str, ndarray]` | Shortcut for each test summary's statistic trace. |
| pval_traces | `#!python Mapping[str, ndarray]` | Shortcut for each test summary's p-value trace. |
| test_status_traces | `#!python Mapping[str, tuple[TestStatus, ...]]` | Shortcut for each test summary's status trace. |
| rejection_traces | `#!python Mapping[str, ndarray]` | Boolean rejection trace for each test summary. |
| coefficient_traces | `#!python Mapping[str, ndarray]` | Shortcut for each regression summary's coefficient trace. |
| regression_status_traces | `#!python Mapping[str, tuple[RegressionStatus, ...]]` | Shortcut for each regression summary's status trace. |
| run_config | `#!python dict[str, Any]` | Run configuration used to produce this result. |
