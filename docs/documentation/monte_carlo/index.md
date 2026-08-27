---
tags:
    - doc
---
# Monte Carlo

The `monte_carlo` module provides a bounded pipeline for repeated simulation, filtering, transformation, and diagnostic testing. The main use case is to treat one `SolvedModel` as the data-generating process (DGP), treat another `SolvedModel` as the reference model, and aggregate diagnostic test results over independent replications.

The replication loop is native. Building a pipeline resolves the step graph, lowering resolves it into buffer arenas and native step descriptors, and the loop itself then runs without holding the GIL, across as many workers as `n_jobs` requests. Nothing in the per-replication path calls back into Python except a custom transform, which is compiled by Numba and invoked through a pointer ABI.

???+ info "Reference and DGP Roles"
    The built-in simulation step draws data from the `dgp` by default, or from the `reference` model when configured with `target="reference"` (a size study, vs. a misspecification study against a distinct DGP).

## Pipeline Exports

| Export | Purpose |
|:---:|:---:|
| `MCPipeline` | Runnable pipeline object. Loaded bundles reconstruct this directly at `LoadedMC.pipeline`. |
| `MCStep` / `OpType` | The step container and its operation-role enum. Prefer the factories in `SymbolicDSGE.monte_carlo.step_factories` over hand-building steps. |
| `custom_transform` / `NumbaCustomFunc` | Author a per-replication custom transform. The function is compiled by Numba and called from the native loop. |
| `pandas_operation` / `PandasCustomFunc` | Author a post-loop custom op. Runs once in Python, may build a DataFrame. |
| `Summary` / `Raw` | Return wrappers a post-loop op uses to tag an output as renderable or as bulk data. |


???+ tip "Bundle integration"
    Both `MCPipeline` and `MCPipelineResult` are directly accessible from a loaded bundle containing them.

    ```python
    from SymbolicDSGE import load_bundle

    loaded = load_bundle("experiment-1.sdsge")
    pipeline = loaded.mc.pipeline
    result = loaded.mc.result
    ```