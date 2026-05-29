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
    store_key: str | None = None,
)
```

`MCStep` describes one operation in the pipeline. Most users should create steps through `simulation_step`, `raw_data_step`, `transform_step`, `reference_filter_step`, and `wald_test_step`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Unique step name. Test steps use this as the key in `MCPipelineResult.summaries`. |
| op_type | `#!python OpType` | Operation type: `DATAGEN`, `TRANSFORM`, `FILTER`, `TEST`, or `POSTPROC`. |
| func | `#!python Callable` | Callable executed by the pipeline. |
| kwargs | `#!python Mapping[str, Any]` | Keyword arguments stored with the step and passed into `func`. |
| store_key | `#!python str | None` | Optional payload key. If omitted, `name` is used. |

&nbsp;

```python
@dataclass(frozen=True)
class MCData(
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
@dataclass
class MCContext(
    rep_idx: int,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    data: MCData | None = None,
    payloads: dict[str, Any] = {},
    results: dict[str, TestResult] = {},
)
```

`MCContext` is the mutable object passed through a single replication. Custom transform, filter, test, and post-processing operations receive it as `context`.

__Methods:__

| __Name__ | __Description__ |
|:---------|----------------:|
| `require_data()` | Return `data` or raise if no data-generation step has populated it. |
| `require_payload(key)` | Return a payload by key or raise if the key is missing. |

&nbsp;

```python
@dataclass(frozen=True)
class MCPipelineResult(
    n_rep: int,
    n_successful: int,
    summaries: Mapping[str, MCResult],
    test_results: Mapping[str, tuple[TestResult, ...]] | None,
    payloads: tuple[Mapping[str, Any], ...] | None,
    contexts: tuple[MCContext, ...] | None,
    failures: tuple[MCFailure, ...] = (),
)
```

`MCPipelineResult` is the aggregate return object from `MCPipeline.run(...)`.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| n_rep | `#!python int` | Requested replication count. |
| n_successful | `#!python int` | Number of completed replications. |
| summaries | `#!python Mapping[str, MCResult]` | Per-test aggregate result containers. |
| test_results | `#!python Mapping[str, tuple[TestResult, ...]] | None` | Optional scalar per-replication test results. |
| payloads | `#!python tuple[Mapping[str, Any], ...] | None` | Optional per-replication payload dictionaries. |
| contexts | `#!python tuple[MCContext, ...] | None` | Optional full contexts. |
| failures | `#!python tuple[MCFailure, ...]` | Failures collected when `fail_fast=False`. |
| succeeded | `#!python bool` | `True` when no failures were collected. |
| statistic_traces | `#!python Mapping[str, ndarray]` | Shortcut for each summary's statistic trace. |
| pval_traces | `#!python Mapping[str, ndarray]` | Shortcut for each summary's p-value trace. |
| rejection_traces | `#!python Mapping[str, ndarray]` | Boolean rejection trace for each summary. |

???+ note "P-Value Evaluation"
    Scalar `TestResult` objects produced inside Monte Carlo Wald steps defer p-value and frozen-distribution construction until `pval`, `frozen_dist`, or `compute_pval()` is accessed. Aggregate `MCResult` objects compute vectorized p-values when `MCPipelineResult.summaries` is built.
