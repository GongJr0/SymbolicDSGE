---
tags:
    - doc
---
# load_bundle

```python
def load_bundle(
    path: str | Path,
) -> LoadedBundle
```

Open a `.sdsge` archive and reconstruct its components. 
`load_bundle` is re-exported at `SymbolicDSGE` root.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| path | Path to the `.sdsge` file. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python LoadedBundle` | Container with `manifest`, `models`, `estimation` artifacts, `mc` artifacts, and `simulation` prefills. See [`LoadedBundle`](LoadedBundle.md). |

__Behavior:__

1. Read `manifest.json` and validate `sdsge_version`.
2. Walk every member declared in the manifest:
    - YAML models are re-parsed, re-compiled, and solved using stored `compile_kwargs` / `solve_kwargs`.
    - Estimation specs are parsed back into dataclasses, and bundled results are reconstructed as live `MLEResult`, `MAPResult`, or `MCMCResult` objects.
    - Monte Carlo pipeline specs are parsed, resources are restored, and `LoadedMC.pipeline` is rebuilt as a live `MCPipeline`.
    - CSV / Parquet members are dispatched by `Member.format` and decoded into numpy arrays.
3. Prefills populate `LoadedBundle.simulation` as a `{model: SimulationPrefill}` map.

## Example

```python
from SymbolicDSGE import load_bundle

loaded = load_bundle("experiment-1.sdsge")

print(loaded.manifest.created_by)
if loaded.models is not None:
    print(loaded.models["reference"].is_determinate)
```

## In-code authoring from a `SolvedModel`

- [`SolvedModel.save_sdsge`](../SolvedModel.md#solvedModelSaveSdsge)
- [`SolvedModel.to_bundle_builder`](../SolvedModel.md#solvedModelToBundleBuilder)

## See also

- [`BundleBuilder`](BundleBuilder.md): assemble bundles from code.
- [`LoadedBundle`](LoadedBundle.md): the return type.
- [`sdsge-decompile`](../../portable_experiments/sdsge-decompile.md): the CLI counterpart.
- [Bundle Loading Guide](../../guides/bundle_loading_guide.md): full walkthrough.
