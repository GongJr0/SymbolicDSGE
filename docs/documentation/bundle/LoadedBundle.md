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

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| manifest | `#!python Manifest` | The full bundle manifest, including member inventory, checksums, and inline `SimSpec`. |
| reference | `#!python SolvedModel | None` | Re-solved reference model, or `None` if the bundle has no `reference.yaml`. |
| dgp | `#!python SolvedModel | None` | Re-solved DGP model, or `None` if absent. |
| estimation | `#!python LoadedEstimation | None` | Estimation artifacts, or `None` if `estimation/` was not in the bundle. |
| mc | `#!python LoadedMC | None` | Monte Carlo artifacts, or `None` if `montecarlo/` was not in the bundle. |
| simulation | `#!python dict[str, SimSpec] | None` | Simulation prefills keyed by role, or `None` if none set. |

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
| pipeline | `#!python MCPipeline` | Runnable pipeline. |
| result | `#!python MCPipelineResult | None` | The reconstructed result object, when available. |

???+ tip "Re-running a loaded pipeline"
    A result is already complete with all traces and postproc artifacts the author retained.
    Alternatively, the results can be reproduced by calling `pipeline.run(reference, dgp, **result.run_config)` with the bundled reference and DGP models.

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
