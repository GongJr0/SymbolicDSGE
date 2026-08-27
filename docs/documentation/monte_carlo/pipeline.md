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

Per-replication steps execute in a native `nogil` loop. Building the pipeline resolves each step's producers, and lowering resolves the whole run into buffer arenas and native step descriptors before any replication starts.

__Contract:__

| __Rule__ | __Description__ |
|:---------|----------------:|
| One data-generation step | `per_rep_steps` must have exactly one step with `op_type=OpType.DATAGEN`. Later per-rep steps cannot generate data once `DATAGEN` is performed. |
| Postproc list is post-loop only | `postproc_steps` may contain only `OpType.POSTPROC` steps; per-rep steps may not. |
| Unique step names | Names are used as result and trace keys, unique across both lists. |
| Reserved characters | Step names cannot contain the characters `.`, `:`, `\`, `/`. |
| Post-loop inputs are traces | Postprocs do not see individual replications. They receive the assembled `traces` mapping built after every replication finishes. |

```python
MCPipeline.run(
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None = None,
    n_rep: int,
    fail_fast: bool = True,
    verbosity: int = 1,
    n_jobs: int | None = None,
    check_memory_availability: bool = True,
) -> MCPipelineResult
```

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| reference | Reference `SolvedModel` used by reference-side operations such as Kalman filtering. |
| dgp | Optional DGP `SolvedModel`. Required by `simulation_step` when it targets the DGP (`target="dgp"`, the default); not required for `target="reference"` or `raw_model_data_step`. |
| n_rep | Number of Monte Carlo replications. |
| fail_fast | If `True`, raise on the first failed replication. If `False`, collect `MCFailure` entries and summarize successful replications. |
| verbosity | Performance-reporting level: `0` prints nothing, `1` prints one aggregate throughput line, and `2` enables native per-step profiling and prints one throughput line per step. |
| n_jobs | Worker count for the native loop, resolved joblib-style: `None` uses one worker, a positive value is taken literally, and a negative value means `cpu_count + 1 + n_jobs`. `0` is rejected. |
| check_memory_availability | If `True`, check if there is enough memory available before starting the pipeline. Warns when a run will not fit physical RAM and raises when total available RAM + swap cannot contain the run. |
__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python MCPipelineResult` | Aggregate container with test summaries, regression summaries, post-loop artifacts, run metadata, and failures. |

???+ note "Step timings need verbosity 2"
    `MCMeta.step_elapsed_s`, `step_counts`, and `step_failures` are populated only when the run is started with `verbosity=2`, since per-step profiling instruments the native loop. At lower verbosity they are empty mappings.

???+ warning "Serializable steps"
    Use the factories in `SymbolicDSGE.monte_carlo.step_factories` rather than hand-building `MCStep` objects when the pipeline needs to enter a `.sdsge` bundle.
    Custom transforms `NumbaCustomFunc` and postprocs `PandasCustomFunc` are the only bundle-safe custom operations.
