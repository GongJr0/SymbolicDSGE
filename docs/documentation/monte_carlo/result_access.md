---
tags:
    - doc
---
# Result Access

`MCPipelineResult.test_summaries` maps each test step name to an `MCResult` aggregate.

`MCPipelineResult.regression_summaries` maps each regression step name to an `MCRegressionResult` aggregate.

__Summary Fields and Methods:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| statistic_trace | `#!python ndarray` | Test statistic from each successful replication. |
| pval_trace | `#!python ndarray` | Vectorized p-values for `statistic_trace`. |
| mean_statistic | `#!python float64` | Mean test statistic over successful replications. |
| mean_pval | `#!python float64` | Mean p-value over successful replications. |
| rejection_rate | `#!python float64` | Share of p-values below `alpha`. |
| pval_confidence_interval(...) | `#!python tuple[float64, float64]` | Confidence interval for the rejection rate. |
| statistic_confidence_interval(...) | `#!python tuple[float64, float64]` | Confidence interval for the mean test statistic. |

If `retain_test_results=True`, `MCPipelineResult.test_results` stores scalar per-replication `TestResult` objects keyed by test step name.

???+ warning "Retention and Memory Use"
    `retain_contexts=True` and `retain_payloads=True` can store large arrays from every successful replication. For large Monte Carlo runs, prefer retaining aggregate summaries and scalar test results unless full per-replication payloads are needed for debugging or downstream analysis.
