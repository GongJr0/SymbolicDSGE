---
tags:
    - doc
---
# Monte Carlo

The `monte_carlo` module provides a bounded pipeline for repeated simulation, filtering, transformation, and diagnostic testing. The main use case is to treat one `SolvedModel` as the data-generating process (DGP), treat another `SolvedModel` as the reference model, and aggregate diagnostic test results over independent replications.

???+ info "Reference and DGP Roles"
    The built-in simulation step draws data from `dgp`. The built-in filtering step then runs `reference.kalman(...)` on the generated observables. The reference model is not simulated by the built-in DGP pipeline.

## Spec and runner exports

The serializable pipeline spec and the runner that consumes it live in the core module — no `[ui]` extra is required to validate, compile, or run a pipeline. The same entry points back the GUI and the `.sdsge` bundle.

| Export | Purpose |
| --- | --- |
| `PipelineSpec` / `NodeSpec` / `EdgeSpec` | Pydantic-free graph specification. Serialized to JSON inside a bundle's `montecarlo/pipeline.json`. |
| `validate_pipeline_spec(spec, *, has_reference, has_dgp)` | Topological validation against the step-kind sets and catalog metadata; returns the ordered node list when the graph is well-formed. |
| `build_pipeline(ordered_nodes, *, dgp=None, resources=None)` | Compile validated nodes into an `MCPipeline` ready to run. `resources` reattaches raw-data arrays and custom callables referenced by a bundle spec. |
| `run_pipeline(spec, *, reference, dgp, n_rep, fail_fast, resources=None)` | One-shot validate + compile + run; returns `MCPipelineResult`. |

```python
from SymbolicDSGE import load_bundle
from SymbolicDSGE.monte_carlo import run_pipeline

loaded = load_bundle("experiment-1.sdsge")
result = run_pipeline(
    loaded.mc.spec,
    reference=loaded.reference,
    dgp=loaded.dgp,
    n_rep=500,
    fail_fast=True,
    resources=loaded.mc.resources,
)
```

???+ tip "Bundle integration"
    A loaded `LoadedMC.spec` is a `PipelineSpec`; `LoadedMC.resources` carries the side-channel arrays/callables needed by `raw_data` and `custom` nodes. See the [Bundle Loading Guide](../../guides/bundle_loading_guide.md#re-run-a-monte-carlo-pipeline-from-a-loaded-bundle) for the end-to-end flow.

## Step catalog

`STEP_CATALOG` is the registry for catalog-backed built-ins and the GUI step palette. Resource-backed node kinds such as `raw_data` and `custom` reattach large arrays or callables through the `resources` seam when a bundled pipeline is loaded.

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
