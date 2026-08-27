---
tags:
    - doc
---
# Custom Post-processing Steps

```python
postproc_step(
    name: str,
    func: Callable[..., Any],
    **kwargs: Any,
) -> MCStep
```

`postproc_step` creates a post-loop (`OpType.POSTPROC`) operation. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

Unlike a per-replication step, a post-processing op runs **once** after the replication loop completes, over the assembled across-replication `traces` registry:

```python
func(
    *,
    traces: Mapping[str, np.ndarray],  # across-rep trace keys -> stacked arrays
    **kwargs,
) -> Summary | Raw | tuple[Summary, Raw] | tuple[Raw, Summary]
```

`traces` is keyed by across-replication trace name (`"test.<name>.statistic"`, `"test.<name>.pval"`, `"regression.<name>.coef"`, `"payload.<name>"`, and so on); see [Result Access](../../result_access.md) for the full key registry, or call `available_traces(pipeline)` to enumerate the keys a spec will produce. The op reads whichever traces it needs and returns one or more tagged artifacts.

## Return artifacts

Import from `SymbolicDSGE.monte_carlo`:

| __Type__ | __Signature__ | __Handling__ |
|:---------|:--------------|-------------:|
| `Summary` | `#!python Summary(value)` | Renderable result (scalar, table, small array, or DataFrame) with its own summary surface. |
| `Raw` | `#!python Raw(value: np.ndarray)` | Bulk numeric data kept as data (a trace member). |

???+ note "Return configuration"
    `Summary` and `Raw` can be used to organize the outputs of a post processing step.
    A common use case is to keep a raw data computation and descriptive summary without having to pick one or the other.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| name | Runtime step name and artifact key. |
| func | Callable to execute once after the replication loop. |
| kwargs | Extra keyword arguments forwarded to `func`. |

???+ note "Bundled custom operations"
    In-process pipelines may use any callable. A pipeline written to an `.sdsge` bundle requires the callable to be a [`CustomFunc`](../../custom_ops.md); post-loop ops are wrapped under the **pandas** namespace, so a returned DataFrame's builder code may reference `pd`. Use [`pandas_operation`](../../custom_ops.md#pandas_operation), or pass a `PandasCustomFunc`. The bundle builder enforces and auto-wraps this at serialization time.
