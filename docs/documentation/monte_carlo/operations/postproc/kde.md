---
tags:
    - doc
---

# KDE Post-processing

```python
kde_step(
    name: str,
    **kwargs: Any,
) -> MCStep
```

`kde_step` is a built-in post-loop (`OpType.POSTPROC`) operation that estimates a Gaussian kernel density of an across-replication trace. It lives under `SymbolicDSGE.monte_carlo.operations.postproc`.

It reads `traces[trace]` (a stacked across-replication array), estimates its density on a uniform grid spanning the finite data range, and emits two artifacts keyed under the step name:

- `#!python "<name>.curve"`: a `Raw` `N x 2` array of `(x, density)` for plotting.
- `#!python "<name>.descriptives"`: a `Summary` table of the trace's moments and quantiles (`count`, `mean`, `std`, `min`, `q25`, `median`, `q75`, `max`).

__Key Parameters:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| trace | required | Across-replication trace key, e.g. `"test.<name>.statistic"`. Validated against the pipeline's producible traces. |
| bandwidth | `"scott"` | `gaussian_kde` bandwidth: `"scott"`, `"silverman"`, or a float. |
| grid_points | `200` | Number of evaluation points on the density grid (min 2). |
| kernel | `"gaussian"` | Kernel family. Only `"gaussian"` is currently supported. |

The op needs at least two finite values in the selected trace. See [Result Access](../../result_access.md) for the trace keys a run produces.
