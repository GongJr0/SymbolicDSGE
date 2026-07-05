---
tags:
    - doc
---
# Monte Carlo

The `monte_carlo` module provides a bounded pipeline for repeated simulation, filtering, transformation, and diagnostic testing. The main use case is to treat one `SolvedModel` as the data-generating process (DGP), treat another `SolvedModel` as the reference model, and aggregate diagnostic test results over independent replications.

???+ info "Reference and DGP Roles"
    The built-in simulation step draws data from the `dgp` by default, or from the `reference` model when configured with `target="reference"` (a size study, vs. a misspecification study against a distinct DGP). The built-in filtering step then runs `reference.kalman(...)` on the generated observables.

## Pipeline and Spec Exports

The live `MCPipeline` is the normal in-code object. `PipelineSpec` is the portable graph form used by bundle serialization, the GUI backend, and callers that need to validate or compile a stored graph.

| Export | Purpose |
| --- | --- |
| `MCPipeline` | Runnable pipeline object. Loaded bundles reconstruct this directly at `LoadedMC.pipeline`. |
| `PipelineSpec` / `NodeSpec` / `EdgeSpec` / `PostprocSpec` | Plain dataclass specification. `nodes` and `edges` are the per-replication DAG; `postprocs` is the post-loop phase. Serialized to JSON inside a bundle's `montecarlo/pipeline.json`. |
| `validate_pipeline_spec(spec, *, has_reference, has_dgp)` | Topological validation against the step-kind sets and catalog metadata; returns `(ordered per-rep nodes, postprocs)` when well-formed. |
| `build_pipeline(ordered, postprocs=(), *, resources=None)` | Compile validated per-rep nodes and postprocs into an `MCPipeline` ready to run. `resources` reattaches raw-data arrays and custom callables referenced by a serialized spec. |
| `run_pipeline(spec, *, reference, dgp, n_rep, fail_fast, resources=None)` | Validate, compile, and run a `PipelineSpec`; returns `MCPipelineResult`. Use this for explicit spec workflows. |

```python
from SymbolicDSGE import load_bundle

loaded = load_bundle("experiment-1.sdsge")
result = loaded.mc.pipeline.run(
    reference=loaded.reference,
    dgp=loaded.dgp,
    n_rep=500,
    fail_fast=True,
)
```

???+ tip "Bundle integration"
    A loaded `LoadedMC.pipeline` is a runnable `MCPipeline` rebuilt from the stored spec and resources. `LoadedMC.spec` remains available for UI rendering and archive inspection.

## Step catalog

`STEP_CATALOG` is the registry for catalog-backed built-ins and the GUI step palette. Resource-backed node kinds such as `raw_data`, `transform:custom`, and `postproc:custom` reattach large arrays or callables through the `resources` mapping when a serialized pipeline is compiled.

| Name | Purpose |
| --- | --- |
| `STEP_CATALOG` | Mapping from `step_type` (string) to `StepDefinition`. |
| `StepDefinition` | Per-step metadata: human label, parameter `FieldSpec` list, operation role, category, factory, and optional parameter compile hook. |
| `FieldSpec` | One parameter on a step: name, type, default, validation hints. Drives the GUI form generation. |
| `DATAGEN_STEP_TYPES` | Catalog-local step-kind set for valid datagen roots: `"simulation"` and `"raw_data"`. |
| `TRANSFORM_STEP_TYPES` | Catalog-local step-kind set for catalog-backed transforms. |
| `TERMINAL_STEP_TYPES` | Catalog-local step-kind set for test/regression summaries. Terminal steps cannot link forward. |
| `POSTPROC_STEP_TYPES` | Catalog-local step-kind set for post-loop ops (e.g. `kde`) run once after the replication loop. |
| `catalog_payload()` | JSON-safe rendering of the catalog for the GUI / external consumers. |

???+ note "Step-kind sets"
    The step-kind sets are implementation metadata in `SymbolicDSGE.monte_carlo.catalog`. They describe compiler behavior but are not the primary user-facing import surface.
