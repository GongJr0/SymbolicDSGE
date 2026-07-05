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

Open a `.sdsge` archive and reconstruct its components. Equivalent to `SymbolicDSGE.bundle.build_from`; `load_bundle` is the friendlier top-level alias.

`load_bundle` is re-exported at `SymbolicDSGE` root.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| path | Path to the `.sdsge` file. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python LoadedBundle` | Container with `manifest`, `reference`/`dgp` (re-solved models), `estimation` artifacts, `mc` artifacts, and the optional `simulation` prefill. See [`LoadedBundle`](LoadedBundle.md). |

__Behavior:__

1. Read `manifest.json` and validate `sdsge_version`.
2. Walk every member declared in the manifest:
    - YAML models are re-parsed, re-compiled, and solved using stored `compile_kwargs` / `solve_kwargs`.
    - Estimation specs are parsed back into dataclasses, and bundled results are reconstructed as live `OptimizationResult` or `MCMCResult` objects.
    - Monte Carlo pipeline specs are parsed, resources are restored, and `LoadedMC.pipeline` is rebuilt as a live `MCPipeline`.
    - CSV / Parquet members are dispatched by `Member.format` and decoded into numpy arrays.
3. Inline `SimSpec` is read from the manifest itself.

???+ info "Format Agnostic Reader"
    `load_bundle` dispatches each tabular member on `Member.format`. A hand-zipped CSV-only bundle and a CLI-built Parquet bundle both load through the same path. No `[bundle]` extra is required because `parquet-engine` is a regular dependency.

???+ warning "Model Reconstruction Cost"
    Loading runs the full parse, compile, and solve pipeline once per model member. For large models or scripts that open many bundles in a loop, cache the `LoadedBundle` rather than reopening.

## Example

```python
from SymbolicDSGE import load_bundle

loaded = load_bundle("experiment-1.sdsge")

print(loaded.manifest.created_by)
if loaded.reference is not None:
    print(loaded.reference.policy.stab == 0)
```

## In-code authoring counterpart

```python
SolvedModel.save_sdsge(
    path: str | Path,
    *,
    yaml_text: str | None = None, # (1)!
    role: str = "reference",
    compile_kwargs: Mapping[str, Any] | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
) -> Path
```

1. Override the YAML embedded in the bundle. Defaults to `compiled.config.source_yaml` (populated by `ModelParser`).

Convenience wrapper around [`SolvedModel.to_bundle_builder`](../SolvedModel.md) for the model-only case. For bundles that include estimation / MC / sim members, call `to_bundle_builder` and chain the additions.

```python
SolvedModel.to_bundle_builder(
    *,
    yaml_text: str | None = None,
    role: str = "reference",
    compile_kwargs: Mapping[str, Any] | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> BundleBuilder
```

Return a `BundleBuilder` pre-seeded with the model's YAML. Chain estimation / MC / simulation members, then call `.write()` to materialize the archive.

???+ warning "Source YAML required"
    `to_bundle_builder` raises `ValueError` if `compiled.config.source_yaml` is `None` and no `yaml_text=` override is given. `ModelParser(path)` and `ModelParser.from_string(text)` both populate `source_yaml` automatically; programmatically constructed `ModelConfig` instances do not.

## See also

- [`BundleBuilder`](BundleBuilder.md): assemble bundles from code.
- [`LoadedBundle`](LoadedBundle.md): the return type.
- [`sdsge-decompile`](../../portable_experiments/sdsge-decompile.md): the CLI counterpart.
- [Bundle Loading Guide](../../guides/bundle_loading_guide.md): full walkthrough.
