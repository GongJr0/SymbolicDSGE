---
tags:
    - doc
---
# BundleBuilder

```python
class BundleBuilder()
```

`BundleBuilder` is the fluent in-code assembler for `.sdsge` archives. Members are appended via `add_*` methods, then committed by `.write()` or returned in memory via `.build()`. Every `add_*` method returns `self` to support chaining.

`BundleBuilder` is re-exported at `SymbolicDSGE` root.

__Constructor:__

```python
BundleBuilder(
    *,
    created_by: str | None = None,
)
```

| __Name__ | __Description__ |
|:---------|----------------:|
| created_by | Manifest `created_by` string. Defaults to `"SymbolicDSGE <version>"`. |

&nbsp;

__Methods:__

```python
BundleBuilder.add_model(
    role: str,
    yaml_text: str,
    *,
    compile_kwargs: Mapping[str, Any] | None = None, # (1)!
    solve_kwargs: Mapping[str, Any] | None = None,   # (2)!
) -> BundleBuilder
```

1. Forwarded to `DSGESolver.compile(...)` when the loader rebuilds the `SolvedModel`.
2. Forwarded to `DSGESolver.solve(...)` at the same step.

Add a model configuration to the bundle. `role` is `"reference"` or `"dgp"`. `yaml_text` is the source YAML the loader will re-parse.

| __Name__ | __Description__ |
|:---------|----------------:|
| role | `"reference"` or `"dgp"`. At least one model is required for a valid bundle. |
| yaml_text | Source YAML text. Available on a parsed `ModelConfig` at `config.source_yaml`. |
| compile_kwargs | Compile kwargs the loader passes to `DSGESolver.compile`. |
| solve_kwargs | Solve kwargs the loader passes to `DSGESolver.solve`. |

&nbsp;

```python
BundleBuilder.add_raw_data(
    name: str,
    data: bytes | str,
    *,
    as_parquet: bool = True, # (1)!
) -> BundleBuilder
```

1. Pass `False` to embed the CSV verbatim (for hand-zip-friendly bundles).

Add a raw observable file. CSV input is re-encoded as Parquet by default.

| __Name__ | __Description__ |
|:---------|----------------:|
| name | Member stem — stored under `data/<name>.csv` or `data/<name>.parquet`. |
| data | CSV bytes or text. Parquet input should be added through [`add_member`](#raw-passthrough) instead. |
| as_parquet | When `True` the CSV is re-encoded as Parquet for size; when `False` it is stored verbatim. |

&nbsp;

```python
BundleBuilder.add_estimation(
    spec: EstimationSpec,
    *,
    result: OptimizationResultMeta | MCMCResultMeta | None = None,
    observed: NDArray[Any] | None = None,
    observable_names: list[str] | None = None,
    posterior: Mapping[str, NDArray[Any]] | None = None,
    as_parquet: bool = True,
) -> BundleBuilder
```

Add the estimation tab. `spec` is always written; the other arguments are conditional. With `as_parquet=False`, `observed` is stored as a semantic-header CSV (using `observable_names` as headers) and `posterior` is stored via mechanical `{name}.{j}` expansion.

| __Name__ | __Description__ |
|:---------|----------------:|
| spec | `EstimationSpec` for the run (method, parameters, observables, kwargs, posterior point). |
| result | Result metadata — either `OptimizationResultMeta` (MLE/MAP) or `MCMCResultMeta` (MCMC). Bulk MCMC traces ride alongside via `posterior`. |
| observed | Observed `y` matrix shaped `(n, k)`. |
| observable_names | List of `k` observable names matching the matrix columns. Stored on the manifest member for semantic-header CSV authoring. |
| posterior | MCMC posterior columns — typically `{"samples": (n_draws, n_params), "logpost": (n_draws,)}`. Either `logpost` or `logpost_trace` is accepted as the bulk-log key. |
| as_parquet | When `False` the bulk members are written as CSV instead of Parquet. |

???+ warning "Observable name validation"
    `observable_names` must match the model's `observables` in count **and** order. Mismatch raises at compile time with an actionable message. See [`sdsge-compile` validation](../../portable_experiments/sdsge-compile.md#validation) for the details.

&nbsp;

```python
BundleBuilder.add_mc(
    pipeline: PipelineSpec,
    *,
    result: MCPipelineResult | None = None,
    run_id: str = "",
    as_parquet: bool = True,
) -> BundleBuilder
```

Add the Monte Carlo tab. The pipeline spec is always written; an attached `result` is split into a trace-free document (JSON) plus a trace member (Parquet by default, CSV when `as_parquet=False`).

| __Name__ | __Description__ |
|:---------|----------------:|
| pipeline | `PipelineSpec` describing the MC graph. |
| result | Optional live `MCPipelineResult`; the builder splits the document from the bulk traces internally. |
| run_id | Identifier embedded in the result document. |
| as_parquet | When `False` the trace member is written as CSV. |

&nbsp;

```python
BundleBuilder.set_simulation(
    simulation: SimSpec,
) -> BundleBuilder
```

Attach the simulation prefill. `SimSpec` rides inline in the manifest rather than as its own member.

&nbsp;

```python
BundleBuilder.add_member(
    member: Member,
    data: bytes,
) -> BundleBuilder
```

Low-level passthrough — append a pre-encoded member at its declared path. Used by `sdsge-compile` to embed Parquet `data/` files verbatim and to stage pre-split MC result + traces pairs.

???+ note "When to use `add_member` directly"
    Prefer the typed `add_*` methods. Reach for `add_member` only when the member bytes are already in their final form and one of the higher-level methods would re-encode them.

&nbsp;

```python
BundleBuilder.write(
    path: str | Path,
) -> Path
```

Materialize the bundle to disk and return the written path. Equivalent to `write_bundle(path, builder.manifest(), files)`.

&nbsp;

```python
BundleBuilder.build(
) -> tuple[Manifest, dict[str, bytes]]
```

Return the in-memory `(manifest, files)` pair instead of writing — useful for tests and for callers that handle the I/O themselves.

&nbsp;

```python
BundleBuilder.manifest(
) -> Manifest
```

Return a `Manifest` describing the currently-accumulated members. Each call regenerates `created_at` and re-computes SHA-256 checksums over the staged bytes.

## Example

```python
from SymbolicDSGE import (
    BundleBuilder,
    DSGESolver,
    ModelParser,
)

parser = ModelParser("MODELS/POST82.yaml") # (1)!
model, kalman = parser.get_all()
solver = DSGESolver(model, kalman)
sol = solver.solve(solver.compile())

bundle_path = (
    BundleBuilder(created_by="experiment-1") # (2)!
    .add_model(
        "reference",
        model.source_yaml, # (3)!
        compile_kwargs={"linearize": False},
    )
    .write("experiment-1.sdsge")
)
```

1. `ModelParser` populates `ModelConfig.source_yaml` automatically.
2. `created_by` is purely metadata. Defaults to `"SymbolicDSGE <version>"` when omitted.
3. The loader re-parses this YAML and re-solves with the recorded `compile_kwargs`.

## See also

- [`sdsge-compile`](../../portable_experiments/sdsge-compile.md) — the CLI equivalent.
- [Bundle Authoring Guide](../../guides/bundle_authoring_guide.md) — full walkthrough.
- [`SolvedModel.save_sdsge`](../SolvedModel.md) and [`SolvedModel.to_bundle_builder`](../SolvedModel.md) — convenience wrappers for the model-only case.
