---
tags:
    - doc
---
# LoadedBundle

```python
@dataclass
class LoadedBundle()
```

`LoadedBundle` is the return value of `load_bundle` (and the underlying `bundle.loader.build_from`). Each field is `None` when the corresponding component is absent from the archive.

`LoadedBundle` is re-exported at `SymbolicDSGE` root.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| manifest | `#!python Manifest` | The full bundle manifest, including member inventory, checksums, and inline `SimSpec`. |
| reference | `#!python SolvedModel | None` | Re-solved reference model, or `None` if the bundle has no `reference.yaml`. |
| dgp | `#!python SolvedModel | None` | Re-solved DGP model, or `None` if absent. |
| estimation | `#!python LoadedEstimation | None` | Estimation artifacts, or `None` if `estimation/` was not in the bundle. |
| mc | `#!python LoadedMC | None` | Monte Carlo artifacts, or `None` if `montecarlo/` was not in the bundle. |
| simulation | `#!python dict[str, dict[str, Any]] | None` | Simulation prefills keyed by role, or `None` if none set. |

???+ info "Re-solving on load"
    `reference` and `dgp` are reconstructed by re-parsing the embedded YAML and re-running `DSGESolver.compile(**compile_kwargs).solve(**solve_kwargs)` with the kwargs recorded at compile time. The receiver does not need the original parser state.

## `LoadedEstimation`

```python
@dataclass
class LoadedEstimation()
```

__Fields:__

| __Name__  | __Type__ | __Description__ |
|:----------|:--------:|----------------:|
| estimator | `#!python Estimator` | The live estimator object lazily rebuilt from the stored spec and the reference model. Always present when `LoadedEstimation` is. | 
| result    | `#!python MLEResult | MAPResult | MCMCResult | None` | The reconstructed result object. |
| spec      | `#!python EstimatorSpec` | Parameter specification for `Estimator`. Always present when `LoadedEstimation` is. |

???+ tip "Re-running a loaded estimation"
    Result classes carry their options as a dictionary. `optimizer_config` in `MLEResult`/`MAPResult` and `sampler_config` in `MCMCResult`.
    Unpack the options into the relevant method. Example: `estimator.mle(**res.optimizer_config)`.

## `LoadedMC`

```python
@dataclass
class LoadedMC()
```

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| spec | `#!python PipelineSpec` | The stored pipeline graph. Always present when `LoadedMC` is. Kept for UI rendering, archive inspection, and explicit compile workflows. |
| pipeline | `#!python MCPipeline` | Runnable pipeline rebuilt from `spec` and `resources` during load. |
| document | `#!python dict[str, Any] | None` | Trace-free run document (test/regression summaries, timing, etc.). |
| traces | `#!python dict[str, NDArray] | None` | Bulk trace columns keyed by `test.<name>.{statistic,pval,status}` / `regression.<name>.{coef,r2,status}`. |
| resources | `#!python dict[str, Any]` | Restored side-channel objects referenced by the spec, including raw-data arrays and custom callables. |
| postproc_arrays | `#!python dict[str, NDArray]` | Bulk postproc ndarray artifacts keyed by artifact name. |
| postproc_tables | `#!python dict[str, dict[str, list[Any]]]` | Tabular postproc artifacts restored as column dictionaries keyed by artifact name. |

__Methods:__

```python
LoadedMC.wire(
) -> dict[str, Any] | None
```

Re-merge `document`, `traces`, `postproc_arrays`, and `postproc_tables` into the canonical UI wire shape. Returns `None` when either `document` or `traces` is missing.

???+ info "When `wire()` returns `None`"
    A bundle authored with the pipeline spec only (no completed run attached) carries neither `document` nor `traces`. `wire()` reports `None` so callers can distinguish "no run available" from a run with empty traces.

???+ tip "Re-running a loaded pipeline"
    Call `#!python loaded.mc.pipeline.run(reference=loaded.reference, dgp=loaded.dgp, n_rep=..., fail_fast=...)`. The loader already rebuilt the pipeline from the stored spec and resources.

## Example

```python
from SymbolicDSGE import load_bundle

loaded = load_bundle("experiment-1.sdsge")

# Use the re-solved reference model directly.
if loaded.reference is not None:
    sim = loaded.reference.sim(T=25, observables=True)

# Inspect the estimation tab.
if loaded.estimation is not None:
    est = loaded.estimation.estimator
    res = loaded.estimation.result
    print(res.x)

# Manifest is always present.
print(loaded.manifest.created_by, loaded.manifest.created_at)
```

## See also

- [`load_bundle`](load_bundle.md): the constructor.
- [`Manifest`](Manifest.md): the archive index.
- [Bundle Loading Guide](../../guides/bundle_loading_guide.md): end-to-end walkthrough.
