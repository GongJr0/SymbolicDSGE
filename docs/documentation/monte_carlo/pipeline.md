---
tags:
    - doc
---
# MCPipeline

```python
class MCPipeline(
    per_rep_steps: Sequence[MCStep],
    postproc_steps: Sequence[MCStep] = (),
)
```

`MCPipeline` holds two step lists: `per_rep_steps` (the dependency DAG executed inside every replication) and `postproc_steps` (post-loop ops run **once** after the loop, over the assembled across-rep traces). The two are separate because a postproc is a terminal reduction, not a graph node.

__Contract:__

| __Rule__ | __Description__ |
|:---------|----------------:|
| One data-generation step | `per_rep_steps[0]` must have `op_type=OpType.DATAGEN`. Later per-rep steps cannot generate data once `DATAGEN` is performed.|
| Postproc list is post-loop only | `postproc_steps` may contain only `OpType.POSTPROC` steps; per-rep steps may not. |
| Unique step names | Names are used as payload/result/artifact keys, unique across both lists. |
| Per-replication payloads | Per-rep steps pass results through `MCContext.payloads` inside the same replication. Aggregate summaries + postprocs run after all successful replications finish. |

__Methods:__

```python
MCPipeline.graph -> PipelineGraph
```

Return the cached dependency graph inferred from `per_rep_steps` (postprocs are not graph participants). The graph records structural edges used by serialization, including filter dependencies and payload-producing transform/custom steps.

```python
MCPipeline.to_spec() -> PipelineSpec
```

Serialize the live pipeline into the graph-form `PipelineSpec`. Bulk side channels are referenced by key: `raw_data` arrays and custom callables are written as separate bundle members by `BundleBuilder.add_mc(...)`.

`PipelineSpec` is the archive and UI representation. A bundle loaded back into Python reconstructs a live `LoadedMC.pipeline`.

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
    verbosity: int = 1,
) -> MCPipelineResult
```

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| reference | Reference `SolvedModel` used by reference-side operations such as Kalman filtering. |
| dgp | Optional DGP `SolvedModel`. Required by `simulation_step` when it targets the DGP (`target="dgp"`, the default); not required for `target="reference"` or `raw_data_step`. |
| n_rep | Number of Monte Carlo replications. |
| retain_payloads | Store each successful replication's payload dictionary in the result container. |
| retain_test_results | Store scalar `TestResult` objects from each successful replication. |
| retain_contexts | Store full `MCContext` objects. This is useful for debugging and memory-heavy for large runs. |
| fail_fast | If `True`, raise on the first failed replication. If `False`, collect `MCFailure` entries and summarize successful replications. |
| verbosity | Performance-reporting level: `0` prints nothing, `1` prints one aggregate throughput line, and `2` prints one throughput line per step. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python MCPipelineResult` | Aggregate container with test summaries, optional per-replication payloads, optional scalar test results, optional contexts, and failures. |

???+ warning "Serializable steps"
    `to_spec()` requires each step to carry a `step_type`. Use the built-in factories rather than hand-building `MCStep` objects when the pipeline needs to enter a `.sdsge` bundle.
