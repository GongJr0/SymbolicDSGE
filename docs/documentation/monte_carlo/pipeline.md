---
tags:
    - doc
---
# MCPipeline

```python
class MCPipeline(steps: Sequence[MCStep])
```

`MCPipeline` stores the ordered operations executed inside each Monte Carlo replication.

__Contract:__

| __Rule__ | __Description__ |
|:---------|----------------:|
| One data-generation step | The first step must have `op_type=OpType.DATAGEN`. Later steps cannot generate data once `DATAGEN` is performed.|
| Unique step names | Names are used as payload and result keys. |
| Per-replication payloads | Steps may pass results through `MCContext.payloads` inside the same replication. Aggregate summaries are created after all successful replications finish. |

__Methods:__

```python
MCPipeline.run(
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None = None,
    n_rep: int,
    retain_payloads: bool = True,
    retain_test_results: bool = True,
    retain_contexts: bool = False,
    fail_fast: bool = True,
) -> MCPipelineResult
```

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| reference | Reference `SolvedModel` used by reference-side operations such as Kalman filtering. |
| dgp | Optional DGP `SolvedModel`. Required by `simulation_step`, but not required by `raw_data_step`. |
| n_rep | Number of Monte Carlo replications. |
| retain_payloads | Store each successful replication's payload dictionary in the result container. |
| retain_test_results | Store scalar `TestResult` objects from each successful replication. |
| retain_contexts | Store full `MCContext` objects. This is useful for debugging and memory-heavy for large runs. |
| fail_fast | If `True`, raise on the first failed replication. If `False`, collect `MCFailure` entries and summarize successful replications. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python MCPipelineResult` | Aggregate container with test summaries, optional per-replication payloads, optional scalar test results, optional contexts, and failures. |
