---
tags:
    - doc
---
# Monte Carlo Results

```python
@dataclass(frozen=True)
class MCDataGenResult(
    n_rep: int,
    n_retained: int,
    retained_reps: ndarray,
    var_names: Sequence[str],
    X: ndarray,
    shock_names: Sequence[str],
    eps: ndarray,
    observable_names: Sequence[str] = (),
    y: ndarray = np.empty((0, 0, 0)),
    _regimes: ndarray | None = None,
    _diagnostics: Sequence[OccBinDiagnostics] | None = None,
)
```

`MCDataGenResult` stacks the `SimResult` a datagen step produces on each replication into one container, adding a leading replication axis to every path. A pipeline has exactly one datagen step, so `MCPipelineResult.datagen_outputs` holds the container itself rather than a mapping.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| n_rep | `#!python int` | Total number of replications. |
| n_retained | `#!python int` | Number of replications whose output the step's arena kept. |
| retained_reps | `#!python ndarray` | Replication indices behind each row of the stacked paths, so a row can be mapped back to its replication. |
| var_names | `#!python Sequence[str]` | Names of the state-path columns, in compiled canonical order. |
| X | `#!python ndarray` | Full state paths. Shape `(n_retained, T, n_var)`. |
| shock_names | `#!python Sequence[str]` | Names of the shock columns, in canonical order. |
| eps | `#!python ndarray` | Shock paths. Shape `(n_retained, T, n_shock)`. |
| observable_names | `#!python Sequence[str]` | Names of the observable-path columns, in observable order. Empty when the step produced no observables. |
| y | `#!python ndarray` | Observable paths. Shape `(n_retained, T, n_obs)`. |
| states | `#!python dict[str, ndarray]` | State paths keyed by `var_names`. Each value is a `(n_retained, T)` column view of `X`. |
| shocks | `#!python dict[str, ndarray]` | Shock paths keyed by `shock_names`. Each value is a `(n_retained, T)` column view of `eps`. |
| observables | `#!python dict[str, ndarray]` | Observable paths keyed by `observable_names`. Each value is a `(n_retained, T)` column view of `y`. |
| regimes | `#!python ndarray` | Accepted regime guess per period, per replication. Shape `(n_retained, T, H)`, where `H` is the longest check-ahead horizon any period used and column 0 is the regime realized at that date. |
| diagnostics | `#!python Sequence[OccBinDiagnostics]` | Each replication's per-period convergence record behind `regimes`. |
| `replication(idx)` | `#!python SimResult` | One retained replication's paths as a single-run result. |

???+ note "Retained indexing"
    `idx` counts the retained replications, which are stored compactly, so it is not the replication's index in the run. `retained_reps[idx]` is that index; the two coincide only when the run retained everything.

???+ note "Fields the step never produced"
    A datagen step declares what it writes: raw data may carry any of states, shocks and observables, and a simulation omits observables when it was built without them. An undeclared field still reports a block, filled with `NaN` at the width its names imply.

???+ warning "Piecewise Simulation"
    Datagen steps for MC pipelines currently do not support piecewise simulation. A OccBin model will use the reference regime (state where no constraint binds) for all periods. `regimes` and `diagnostics` currently raise `AttributeError` on access to reflect that.

&nbsp;

```python
@dataclass(frozen=True)
class MCFilterResult(
    n_rep: int,
    n_retained: int,
    retained_reps: ndarray,
    filter_mode: str,
    x_pred: ndarray,
    x_filt: ndarray,
    P_pred: ndarray,
    P_filt: ndarray,
    y_pred: ndarray,
    y_filt: ndarray,
    S: ndarray,
    innov: ndarray,
    std_innov: ndarray,
    loglik: ndarray,
    eps_hat: ndarray | None = None,
    _x1_pred: ndarray | None = None,
    _x2_pred: ndarray | None = None,
    _x1_filt: ndarray | None = None,
    _x2_filt: ndarray | None = None,
)
```

`MCFilterResult` stacks the `FilterResult` or `UnscentedFilterResult` a filter step produces on each replication into one container, adding a leading replication axis to every history. `MCPipelineResult.filter_outputs` maps each filter step name to one.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| n_rep | `#!python int` | Total number of replications. |
| n_retained | `#!python int` | Number of replications whose output the step's arena kept. |
| retained_reps | `#!python ndarray` | Replication indices behind each row of the stacked histories. |
| filter_mode | `#!python str` | Kernel the step ran: `"linear"`, `"extended"` or `"unscented"`. |
| x_pred | `#!python ndarray` | Predicted $x$ states over time. Shape `(n_retained, T, n_var)`. |
| x_filt | `#!python ndarray` | Filtered $x$ states over time. Shape `(n_retained, T, n_var)`. |
| P_pred | `#!python ndarray` | Predicted state covariance $P$ over time. Shape `(n_retained, T, n_var, n_var)`. |
| P_filt | `#!python ndarray` | Filtered state covariance $P$ over time. Shape `(n_retained, T, n_var, n_var)`. |
| y_pred | `#!python ndarray` | Predicted observables over time. Shape `(n_retained, T, n_obs)`. |
| y_filt | `#!python ndarray` | Filtered observables over time. Shape `(n_retained, T, n_obs)`. |
| S | `#!python ndarray` | Innovation covariance over time. Shape `(n_retained, T, n_obs, n_obs)`. |
| innov | `#!python ndarray` | Observable innovations $y_t - y_{t\mid t-1}$. Shape `(n_retained, T, n_obs)`. |
| std_innov | `#!python ndarray` | Innovations standardized by their covariance. Shape `(n_retained, T, n_obs)`. |
| loglik | `#!python ndarray` | Per-replication log likelihood ($\boldsymbol{\ell}$) of measurements. Shape `(n_retained,)`. |
| eps_hat | `#!python ndarray \| None` | Conditional estimates of structural shocks given observed data, shape `(n_retained, T, n_shock)`. `None` when the step ran without `return_shocks`. |
| `replication(idx)` | `#!python FilterResult \| UnscentedFilterResult` | One retained replication's filter output as a single-run result. |

__Unscented-Only Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x1_pred | `#!python ndarray` | First-order component of the predicted state over time. Shape `(n_retained, T, n_state)`. |
| x2_pred | `#!python ndarray` | Second-order component of the predicted state over time. Shape `(n_retained, T, n_state)`. |
| x1_filt | `#!python ndarray` | First-order component of the filtered state over time. Shape `(n_retained, T, n_state)`. |
| x2_filt | `#!python ndarray` | Second-order component of the filtered state over time. Shape `(n_retained, T, n_state)`. |

???+ note "Retained indexing"
    `idx` counts the retained replications, which are stored compactly, so it is not the replication's index in the run. `retained_reps[idx]` is that index; the two coincide only when the run retained everything.

???+ note "Unscented shapes and fields"
    An unscented step tracks a pruned state, so its covariance histories carry the stacked first- and second-order state: `P_pred` and `P_filt` are shaped `(n_retained, T, 2 * n_state, 2 * n_state)`. Shock recovery is not supported for the UKF, so `eps_hat` is `None`. The four pruned-state fields are only recorded under `filter_mode="unscented"`; reading one off a linear or extended result raises `AttributeError`.

&nbsp;

```python
@dataclass(frozen=True)
class MCTestResult(
    test_name: str,
    dist: ReferenceDistribution,
    df: DistributionParameter | Sequence[DistributionParameter],
    pval_method: PvalMethod,
    alpha: float64 | float,
    statistic_trace: ndarray,
    n_retained: int,
    retained_reps: ndarray,
    n_rep: int,
    _raw_status: ndarray,
)
```

`MCTestResult` aggregates the per-replication statistic a Monte Carlo diagnostic-test step produces. `MCPipelineResult.test_summaries` maps each test step name to one.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| test_name | `#!python str` | Name of the diagnostic test behind the statistic. |
| dist | `#!python ReferenceDistribution` | Reference distribution the statistic is read against. |
| df | `#!python DistributionParameter \| Sequence[DistributionParameter]` | Degrees of freedom parameterizing `dist`. |
| pval_method | `#!python PvalMethod` | Tail convention turning a statistic into a p-value. |
| alpha | `#!python float64 \| float` | Test size the rejection rate counts against. |
| frozen_dist | `#!python FrozenDistribution` | `dist` frozen at `df`, shared by every p-value in the run. |
| statistic_trace | `#!python ndarray` | Test statistic from each retained replication. Shape `(n_retained,)`. |
| pval_trace | `#!python ndarray` | Vectorized p-values for `statistic_trace`. |
| status_trace | `#!python tuple[TestStatus, ...]` | Test status for each retained replication. |
| n_rep | `#!python int` | Total number of replications. |
| n_retained | `#!python int` | Number of replications whose output the step's arena kept. |
| retained_reps | `#!python ndarray` | Replication indices behind each row of the traces, so a trace entry can be mapped back to its replication. |
| mean_statistic | `#!python float64` | Mean test statistic across retained replications. |
| statistic_se | `#!python float64` | Monte Carlo standard error of `mean_statistic`. |
| mean_pval | `#!python float64` | Mean p-value across retained replications. |
| pval_se | `#!python float64` | Monte Carlo standard error of `mean_pval`. |
| rejection_rate | `#!python float64` | Share of retained replications rejecting at `alpha`. |
| rejection_rate_se | `#!python float64` | Binomial standard error of `rejection_rate`. |
| `summary()` | `#!python pandas.DataFrame` | One row, indexed by `test_name`: `statistic`, `statistic_se`, `pval`, `reject_rate`. |
| `intervals(confidence_level=0.95, t_interval=False, wilson=True)` | `#!python pandas.DataFrame` | Bounds for every quantity `summary()` reports, indexed by `quantity`. |
| `statistic_confidence_interval(confidence_level=0.95, t_interval=False)` | `#!python tuple[float64, float64]` | Interval for `mean_statistic`. |
| `pval_confidence_interval(confidence_level=0.95, t_interval=False)` | `#!python tuple[float64, float64]` | Interval for `mean_pval`. |
| `rejection_rate_confidence_interval(confidence_level=0.95, wilson=True)` | `#!python tuple[float64, float64]` | Interval for `rejection_rate`. |

???+ note "Interval kinds"
    `statistic` and `pval` intervals come off the spread of their trace, normal by default and Student-t under `t_interval`. `reject_rate` is a proportion and takes a Wilson interval unless `wilson` is off. `pval` and `reject_rate` bounds clamp to `[0, 1]`.

???+ warning "Monte Carlo standard errors need two replications"
    `statistic_se` and `pval_se` are the spread of a trace around its own mean, so a step retaining a single replication reports `NaN` for both, and every interval built on them is `NaN` wide.

&nbsp;

```python
@dataclass(frozen=True)
class MCRegressionResult(
    kind: Literal["ols", "ridge", "lasso", "elastic_net"],
    variables: list[str],
    coef_trace: ndarray,
    ssr_trace: ndarray,
    sst_trace: ndarray,
    _se_trace: ndarray | None,
    n_retained: int,
    retained_reps: ndarray,
    n_rep: int,
    n: int,
    k: int,
    _raw_status: ndarray,
)
```

`MCRegressionResult` aggregates the per-replication fits a Monte Carlo regression step produces. `MCPipelineResult.regression_summaries` maps each regression step name to one.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| kind | `#!python Literal["ols", "ridge", "lasso", "elastic_net"]` | Regression method. |
| variables | `#!python list[str]` | Shared variable ordering across replications. |
| coef_trace | `#!python ndarray` | Coefficients stacked by retained replication. Shape `(n_retained, k)`. |
| coefficients | `#!python ndarray` | Alias for `coef_trace`. |
| status_trace | `#!python tuple[RegressionStatus, ...]` | Solver status for each retained replication. |
| n_rep | `#!python int` | Total number of replications. |
| n_retained | `#!python int` | Number of replications whose output the step's arena kept. |
| retained_reps | `#!python ndarray` | Indices of retained replications relative to the complete run. |
| n | `#!python int` | Shared number of observations per replication. |
| k | `#!python int` | Shared number of design columns. |
| ssr_trace | `#!python ndarray` | Per-replication SSR values. |
| sst_trace | `#!python ndarray` | Per-replication SST values. |
| mse_trace | `#!python ndarray` | Per-replication MSE values. |
| rmse_trace | `#!python ndarray` | Per-replication RMSE values. |
| r2_trace | `#!python ndarray` | Per-replication R-squared values. |
| r2_adj_trace | `#!python ndarray` | Per-replication adjusted R-squared values. |
| mean_coef | `#!python ndarray` | Mean coefficient across retained replications. Shape `(k,)`. |
| coef_se | `#!python ndarray` | Monte Carlo standard error of `mean_coef`. Not a per-fit standard error; those are `se_trace`. |
| `summary(alpha=0.05)` | `#!python pandas.DataFrame` | One row per variable: `coef`, `coef_se`, `t_stat`, `pval`, `reject_rate`. |
| `intervals(alpha=0.05, confidence_level=0.95, t_interval=False, wilson=True)` | `#!python pandas.DataFrame` | Bounds for every quantity `summary()` reports, indexed by `(variable, quantity)`. |
| `trace_frame(alpha=0.05)` | `#!python pandas.DataFrame` | The retained replications in long form, one row per replication and variable. |

__OLS-Only Aggregate Diagnostics:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| se_trace | `#!python ndarray` | OLS standard-error trace. |
| t_stat_trace | `#!python ndarray` | OLS t-statistic trace. |
| partial_r2_trace | `#!python ndarray` | OLS partial R-squared trace. |
| pval_trace | `#!python ndarray` | OLS coefficient p-value trace. |
| F_stat_trace | `#!python ndarray` | OLS F-statistic trace. |
| F_pval_trace | `#!python ndarray` | OLS F-test p-value trace. |
| mean_t_stat | `#!python ndarray` | Mean t-statistic across retained replications. |
| t_stat_se | `#!python ndarray` | Monte Carlo standard error of `mean_t_stat`. |
| mean_pval | `#!python ndarray` | Mean coefficient p-value across retained replications. |
| pval_se | `#!python ndarray` | Monte Carlo standard error of `mean_pval`. |
| `rejection_rate(alpha=0.05)` | `#!python ndarray` | Share of retained replications rejecting at `alpha`, per coefficient. |
| `rejection_rate_se(alpha=0.05)` | `#!python ndarray` | Binomial standard error of `rejection_rate`. |
| `F_test(alpha=0.05)` | `#!python MCTestResult` | Aggregate F-test result container. |

???+ note "OLS-Specific Diagnostics"
    OLS aggregate diagnostics require every stored result to be an `OLSResult`. Ridge, lasso, and elastic-net aggregates have no unbiased standard errors, so `se_trace` warns and every quantity derived from it is `NaN`. `summary()` and `intervals()` keep those columns and fill them with `NaN` rather than dropping them.

???+ note "Interval kinds"
    `coef`, `t_stat`, and `pval` intervals come off the spread of their trace, normal by default and Student-t under `t_interval`. `reject_rate` is a proportion and takes a Wilson interval unless `wilson` is off. `pval` and `reject_rate` bounds clamp to `[0, 1]`.

&nbsp;

__Transform Output:__

`MCPipelineResult.transform_outputs` maps each transform step name to its output stacked across retained replications, shaped `(n_retained, *output_shape)`. A transform writing `(T, p)` per replication appears as `(n_retained, T, p)`. The mapping is empty (`{}`) when the pipeline has no transform steps. Retention follows the step's `n_retain`.

These are the same arrays post-loop ops receive under the `payload.<name>` trace keys below, so a value read here needs no post-processing step to reach it.

__Across-Replication Traces:__

Every producer's stacked output is addressable by a trace key. Post-loop ops receive these keys in their `traces` mapping, and `available_traces(pipeline)` enumerates them from a spec alone, before a run.

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
