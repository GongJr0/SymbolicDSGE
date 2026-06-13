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
| result | `#!python OptimizationResultMeta \| MCMCResultMeta \| None` | Result metadata (scalar slice). Discriminated by `"type": "mcmc" \| "optimization"` in the embedded JSON. |
| observed | `#!python NDArray[np.float64] \| None` | Observed `y` matrix shaped `(n, k)`, reconstructed from the CSV or Parquet member. |
| posterior | `#!python dict[str, NDArray] \| None` | MCMC posterior columns — `{"samples": (n_draws, n_params), "logpost": (n_draws,)}` by convention. |

???+ note "Pairing metadata with traces"
    `MCMCResultMeta` carries only the scalar slice. To reconstruct the full sampling diagnostics or repaint the GUI, pair the metadata with `posterior` — both come back as part of the same `LoadedEstimation` so callers don't have to track member paths.

???+ tip "Re-running a loaded estimation"
    `spec.to_estimator_inputs()` lowers the spec to an [`EstimatorInputs`](index.md#estimation-spec-and-result-types) — `estimated_params`, `theta0`, `priors` (built `Prior` objects), and `bounds` — directly feedable to `#!python DSGESolver.estimate(...)`. The lowering lives in the core library, so a loaded estimation can be re-run without the `[ui]` extra. See the [Bundle Loading Guide](../../guides/bundle_loading_guide.md#re-run-an-estimation-from-a-loaded-bundle).

## `LoadedMC`

```python
@dataclass
class LoadedMC()
```

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| spec | `#!python PipelineSpec` | The pipeline graph (nodes + edges). Always present when `LoadedMC` is. |
| document | `#!python dict[str, Any] \| None` | Trace-free run document (test/regression summaries, timing, etc.). |
| traces | `#!python dict[str, NDArray] \| None` | Bulk trace columns keyed by `test.<name>.{statistic,pval,status}` / `regression.<name>.{coef,r2,status}`. |

__Methods:__

```python
LoadedMC.wire(
) -> dict[str, Any] | None
```

Re-merge `document` and `traces` into the canonical UI wire shape (the same dict an in-process run would emit). Returns `None` when either side is missing.

???+ info "When `wire()` returns `None`"
    A bundle authored with the pipeline spec only (no completed run attached) carries neither `document` nor `traces`. `wire()` reports `None` so callers can distinguish "no run available" from a run with empty traces.

???+ tip "Re-running a loaded pipeline"
    `#!python from SymbolicDSGE.monte_carlo import run_pipeline` runs `spec` against the loaded models without the `[ui]` extra. See [Monte Carlo > Overview](../monte_carlo/index.md) for the core-side runner exports and the [Bundle Loading Guide](../../guides/bundle_loading_guide.md#re-run-a-monte-carlo-pipeline-from-a-loaded-bundle).

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

- [`load_bundle`](load_bundle.md) — the constructor.
- [`Manifest`](Manifest.md) — the archive index.
- [Bundle Loading Guide](../../guides/bundle_loading_guide.md) — end-to-end walkthrough.
