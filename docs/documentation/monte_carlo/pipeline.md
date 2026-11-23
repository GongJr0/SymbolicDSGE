---
tags:
    - doc
---
# MCPipeline

```python
class MCPipeline(*steps: MCStep)
```

`MCPipeline` receives `MCStep` objects as arguments and constructs an execution plan (the dependency DAG executed inside every replication). Each step is ordered such that it's sources are availbe as that step executes. The order of specification does not matter, and the constructor will raise when a dependency cycle or otherwise unsatisfiable specification is presented.

The pipeline can be though of as a declarative configuration that feeds the native experiment runner, the Python-side only resolves the graph and constructd the context for the native execution engine to consume.

__Contract:__

| __Rule__ | __Description__ |
|:---------|----------------:|
| Unique step names | Names are used as result and trace keys. |
| Reserved characters | Step names cannot contain the characters `.`, `:`, `\`, `/`. |

```python
MCPipeline.run(
    models: dict[str, SolvedModel] | None = None,
    *,
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
| models | Named collection of `SolvedModel` object that steps can refer to. Optional for runs that do not require a model. |
| n_rep | Number of Monte Carlo replications. |
| fail_fast | If `True`, raise on the first failed replication. If `False`, collect `MCFailure` entries and summarize successful replications. |
| verbosity | Performance-reporting level: `0` prints nothing, `1` prints one aggregate throughput line, and `2` enables native per-step profiling and prints one throughput line per step. |
| n_jobs | Worker count for the native loop, resolved joblib-style: `None` uses one worker, a positive value is taken literally, and a negative value means `cpu_count + 1 + n_jobs`. `0` is rejected. |
| check_memory_availability | If `True`, check if there is enough memory available before starting the pipeline. Warns when a run will not fit physical RAM and raises when total available RAM + swap cannot contain the run. |
__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python MCPipelineResult` | Aggregate container with test summaries, regression summaries, run metadata, and failures. |

???+ note "Step timings need verbosity 2"
    `MCMeta.step_elapsed_s`, `step_counts`, and `step_failures` are populated only when the run is started with `verbosity=2`, since per-step profiling instruments the native loop. At lower verbosity they are empty mappings.

???+ warning "Serializable steps"
    Use the factories in `SymbolicDSGE.monte_carlo.step_factories` rather than hand-building `MCStep` objects when the pipeline needs to enter a `.sdsge` bundle.
    Custom transforms (`NumbaCustomFunc`, functions are auto-converted) are the only bundle-safe custom operations.
