---
tags:
    - doc
---
# Result Access

`MCPipelineResult.test_summaries` maps each test step name to an `MCTestResult` aggregate.

`MCPipelineResult.regression_summaries` maps each regression step name to an `MCRegressionResult` aggregate.

__Summary Fields and Methods:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| statistic_trace | `#!python ndarray` | Test statistic from each retained replication. |
| pval_trace | `#!python ndarray` | Vectorized p-values for `statistic_trace`. |
| n_rep | `#!python int` | Requested replication count. |
| n_retained | `#!python int` | Number of replications whose output the step's arena kept. |
| retained_reps | `#!python ndarray` | Replication indices behind each row of the traces, so a trace entry can be mapped back to its replication. |
| mean_statistic | `#!python float64` | Mean test statistic over retained replications. |
| mean_pval | `#!python float64` | Mean p-value over retained replications. |
| rejection_rate | `#!python float64` | Share of p-values below `alpha`. |
| pval_confidence_interval(...) | `#!python tuple[float64, float64]` | Confidence interval for the rejection rate. |
| statistic_confidence_interval(...) | `#!python tuple[float64, float64]` | Confidence interval for the mean test statistic. |

`MCRegressionResult` carries the same retention fields alongside `coef_trace`, `ssr_trace`, `sst_trace`, and, for OLS, a standard-error trace.

__Transform Output:__

`MCPipelineResult.transform_outputs` maps each transform step name to its output stacked across retained replications, shaped `(n_retained, *output_shape)`. A transform writing `(T, p)` per replication appears as `(n_retained, T, p)`. The mapping is empty (`{}`) when the pipeline has no transform steps. Retention follows the step's `n_retain`, and the producing step's `retained_reps` records which replications the rows came from.

These are the same arrays post-loop ops receive under the `payload.<name>` trace keys below, so a value read here needs no post-processing step to reach it.

__Across-Replication Traces:__

Every producer's stacked output is addressable by a trace key. Post-loop ops receive these keys in their `traces` mapping, and `available_traces(spec)` enumerates them from a spec alone, before a run.

| __Producer__ | __Keys__ |
|:-------------|---------:|
| test steps | `test.<name>.{statistic, pval, status}` |
| regression steps | `regression.<name>.{coef, ssr, sst, r2, status, se (OLS only)}` |
| transform steps | `payload.<name>` |

Data-generation, filter, and post-processing steps emit no consumable trace.

__Post-Loop Artifacts:__

`MCPipelineResult.postproc` maps each post-loop step to its returned artifacts, keyed by step name. Values are `Summary` (renderable: scalar, table, or small array) or `Raw` (bulk numeric data).

__Performance Reporting:__

| __Name__ | __Description__ |
|:---------|----------------:|
| `MCPipelineResult.report_performance()` | Print the aggregate pipeline throughput report. |
| `MCPipelineResult.report_step_performance()` | Print one throughput report line per pipeline step. |
| `MCPipelineResult.meta` | `MCMeta` object containing elapsed time, step timings, retention counts, failure counts, and throughput properties. |

???+ warning "Retention and Memory Use"
    Retained output is allocated up front, so a step's arena is sized before the loop runs. A transform retaining every replication of a large array is the dominant cost in a big run. Cap it with `n_retain` on the step when only the aggregate summaries matter; the run then keeps an evenly spaced subset of replications, and `retained_reps` records which ones.
