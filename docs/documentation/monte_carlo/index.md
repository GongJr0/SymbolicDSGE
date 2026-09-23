---
tags:
    - doc
---
# Monte Carlo

The `monte_carlo` module provides a bounded pipeline for repeated simulation, filtering, transformation, and diagnostic testing. The main use case is to treat one `SolvedModel` as the data-generating process (DGP), treat another `SolvedModel` as the reference model, and aggregate diagnostic test results over independent replications.

The replication loop is native. Building a pipeline resolves the step graph, lowering resolves it into buffer arenas and native step descriptors, and the loop itself then runs without holding the GIL, across as many workers as `n_jobs` requests. Nothing in the per-replication path calls back into Python except a custom transform, which is compiled by Numba and invoked through a pointer ABI.

???+ tip "Model and data members"
    The pipeline can take any number of models into its `models` mapping.
    Native lowering makes model calibrations immutable, so studies targeting
    multiple calibrations of a single model need to separate each calibration into its own `SolvedModel` object.

    Alternatively, the `raw_model_data_step` can be used to add numerical simulation outputs with no model needed.
    For arbitrary raw data, the `add_payload_step` can be used. These steps allow the Monte Carlo pipeline to be
    used for arbitrary statistical testing and regression studies beyond the DSGE context.

## Pipeline Exports

| Export | Purpose |
|:---:|:---:|
| `MCPipeline` | Runnable pipeline object. Loaded bundles reconstruct this directly at `LoadedMC.pipeline`. |
| `MCStep` / `OpType` | The step container and its operation-role enum. Prefer the factories in `SymbolicDSGE.monte_carlo.step_factories` over hand-building steps. |
| `custom_transform` / `NumbaCustomFunc` | Author a per-replication custom transform. The function is compiled by Numba and called from the native loop. |
| `pandas_operation` / `PandasCustomFunc` | Author a post-loop custom op. Runs once in Python, may build a DataFrame. |
| `Summary` / `Raw` | Return wrappers a post-loop op uses to tag an output as renderable or as bulk data. |
| `replication_shocks` | Generate a shock sequence for a single replication of a simulation step. |

???+ tip "Bundle integration"
    Both `MCPipeline` and `MCPipelineResult` are directly accessible from a loaded bundle containing them.

    ```python
    from SymbolicDSGE import load_bundle

    loaded = load_bundle("experiment-1.sdsge")
    pipeline = loaded.mc.pipeline
    result = loaded.mc.result
    ```
