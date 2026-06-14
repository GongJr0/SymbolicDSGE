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
| `validate_pipeline_spec(spec, *, has_reference, has_dgp)` | Topological validation against [`STEP_CATALOG`](#step-catalog); returns the ordered node list when the graph is well-formed. |
| `build_pipeline(ordered_nodes, *, dgp)` | Compile validated nodes into an `MCPipeline` ready to run. |
| `run_pipeline(spec, *, reference, dgp, n_rep, fail_fast)` | One-shot validate + compile + run; returns `MCPipelineResult`. |

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
)
```

???+ tip "Bundle integration"
    A loaded `LoadedMC.spec` is a `PipelineSpec` ready for these entry points. See the [Bundle Loading Guide](../../guides/bundle_loading_guide.md#re-run-a-monte-carlo-pipeline-from-a-loaded-bundle) for the end-to-end flow.

## Step catalog

`STEP_CATALOG` is the registry that drives validation, compilation, and the GUI step palette. Compilation is catalog-driven — there is no per-step branching in `build_pipeline`.

| Export | Purpose |
| --- | --- |
| `STEP_CATALOG` | Mapping from `step_type` (string) to `StepDefinition`. |
| `StepDefinition` | Per-step metadata: human label, parameter `FieldSpec` list, source/output flags, terminal flag. |
| `FieldSpec` | One parameter on a step: name, type, default, validation hints. Drives the GUI form generation. |
| `TERMINAL_STEP_TYPES` | Set of step types that emit a result column (tests, regressions). Validation enforces that every pipeline ends in one. |
| `catalog_payload()` | JSON-safe rendering of the catalog for the GUI / external consumers. |
