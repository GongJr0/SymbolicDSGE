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
| reference | `#!python SolvedModel \| None` | Re-solved reference model, or `None` if the bundle has no `reference.yaml`. |
| dgp | `#!python SolvedModel \| None` | Re-solved DGP model, or `None` if absent. |
| estimation | `#!python LoadedEstimation \| None` | Estimation artifacts, or `None` if `estimation/` was not in the bundle. |
| mc | `#!python LoadedMC \| None` | Monte Carlo artifacts, or `None` if `montecarlo/` was not in the bundle. |
| simulation | `#!python SimSpec \| None` | Simulation prefill, or `None` if not set. |

???+ info "Re-solving on load"
    `reference` and `dgp` are reconstructed by re-parsing the embedded YAML and re-running `DSGESolver.compile(**compile_kwargs).solve(**solve_kwargs)` with the kwargs recorded at compile time. The receiver does not need the original parser state.

## `LoadedEstimation`

```python
@dataclass
class LoadedEstimation()
```

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| spec | `#!python EstimationSpec` | The text-only run specification. Always present when `LoadedEstimation` is. |
| result | `#!python OptimizationResult \| MCMCResult \| None` | The reconstructed first-class result: an `OptimizationResult` (MLE/MAP) or `MCMCResult` (MCMC), rebuilt from the stored metadata and, for MCMC, the `posterior` traces. `None` when the bundle carries no estimation result. |
| observed | `#!python NDArray[np.float64] \| None` | Observed `y` matrix shaped `(n, k)`, reconstructed from the CSV or Parquet member. |
| posterior | `#!python dict[str, NDArray] \| None` | MCMC posterior columns, conventionally `{"samples": (n_draws, n_params), "logpost": (n_draws,)}`. |

???+ note "Results Come Back Reconstructed"
    `result` is the same first-class object the run produced. The loader rebuilds it from the stored metadata and, for MCMC, the `posterior` traces. If an estimation was bundled with results, read `loaded.estimation.result` directly. The raw `posterior` columns stay on `LoadedEstimation.posterior` for callers that want the arrays.

???+ tip "Re-running a loaded estimation"
    `spec.to_estimator_inputs()` lowers the spec to an [`EstimatorInputs`](index.md#estimation-spec-and-result-types) with `estimated_params`, `theta0`, `priors` as built `Prior` objects, and `bounds`. Pass those values to `#!python DSGESolver.estimate(...)` when a bundle stored only the spec.

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
| document | `#!python dict[str, Any] \| None` | Trace-free run document (test/regression summaries, timing, etc.). |
| traces | `#!python dict[str, NDArray] \| None` | Bulk trace columns keyed by `test.<name>.{statistic,pval,status}` / `regression.<name>.{coef,r2,status}`. |
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
    print(loaded.estimation.spec.method)
    if loaded.estimation.posterior is not None:
        samples = loaded.estimation.posterior["samples"]
        print(samples.shape)

# Manifest is always present.
print(loaded.manifest.created_by, loaded.manifest.created_at)
```

## See also

- [`load_bundle`](load_bundle.md): the constructor.
- [`Manifest`](Manifest.md): the archive index.
- [Bundle Loading Guide](../../guides/bundle_loading_guide.md): end-to-end walkthrough.
