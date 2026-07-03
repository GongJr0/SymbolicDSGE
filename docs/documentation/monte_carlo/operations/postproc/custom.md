---
tags:
    - doc
---
# Custom Post-processing Steps

```python
postproc_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep
```

`postproc_step` creates a post-loop (`OpType.POSTPROC`) operation. It lives under `SymbolicDSGE.monte_carlo.operations.postproc`.

Unlike a per-replication step, a post-processing op runs **once** after the replication loop completes, over the assembled across-replication `traces` registry:

```python
func(
    *,
    traces: Mapping[str, np.ndarray],  # across-rep trace keys -> stacked arrays
    reference: SolvedModel,
    dgp: SolvedModel | None,
    **kwargs,
) -> Summary | Raw | Mapping[str, Any]
```

`traces` is keyed by across-replication trace name (e.g. `"test.<name>.statistic"`, `"test.<name>.pval"`, `"regression.<name>.coef"`); see [Result Access](../../result_access.md) for the available keys. The op reads whichever traces it needs and returns one or more tagged artifacts.

## Return artifacts

Import from `SymbolicDSGE.monte_carlo`:

| __Type__ | __Signature__ | __Handling__ |
|:---------|:--------------|-------------:|
| `Summary` | `#!python Summary(value, title=None, render="auto")` | Renderable result (scalar / table / small array / DataFrame) — gets its own summary surface. `render` ∈ `{"auto", "table", "scalar", "array"}`. |
| `Raw` | `#!python Raw(value: np.ndarray)` | Bulk numeric data kept as data (a trace member), not auto-rendered. |

An op may return a single artifact, a bare value (an `ndarray` becomes `Raw`, anything else becomes `Summary`), or a `Mapping` of named outputs to emit several at once. A single artifact is stored under `store_key or name`; a mapping is stored under `f"{name}.{key}"`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| name | Runtime step name and default artifact key. |
| func | Callable to execute once after the replication loop. |
| store_key | Optional override for the artifact key (single-artifact returns). |
| kwargs | Extra keyword arguments forwarded to `func`. |

???+ note "Bundled custom operations"
    In-process pipelines may use any callable. A pipeline written to an `.sdsge` bundle requires the callable to be a [`CustomFunc`](../../custom_ops.md); post-loop ops are wrapped under the **pandas** namespace, so a returned DataFrame's builder code may reference `pd` — use [`pandas_operation`](../../custom_ops.md#pandas_operation) (or pass a `PandasCustomFunc`). The bundle builder enforces and auto-wraps this at serialization time.
