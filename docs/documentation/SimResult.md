---
tags:
    - doc
---
# SimResult

```python
@dataclass(frozen=True)
class SimResult(
    var_names: Sequence[str],
    X: ndarray,
    shock_names: Sequence[str],
    eps: ndarray,
    observable_names: Sequence[str] = (),
    y: ndarray | None = None,
)
```

The result returned by `SolvedModel.sim(...)` and `SolvedModel.irf(...)`. Simulation paths are in levels. Impulse responses are deviations from their zero-shock baseline.

## Fields

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `var_names` | `Sequence[str]` | Names of the state-path columns, in compiled canonical order. |
| `X` | `ndarray[float]`, shape `(T, n_var)` | Full state path. Each row is one simulated period. |
| `shock_names` | `Sequence[str]` | Names of the shock columns, in canonical order. |
| `eps` | `ndarray[float]`, shape `(T, n_shock)` | Shock path. Each row is one simulated period. |
| `observable_names` | `Sequence[str]` | Names of the observable-path columns, in observable order. Empty when observables were not requested. |
| `y` | `ndarray[float] | None`, shape `(T, n_obs)` | Observable path when requested with `observables=True`; otherwise `None`. |

## Named path views

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `states` | `dict[str, ndarray]` | State paths keyed by `var_names`. Each value is a column view of `X`. |
| `shocks` | `dict[str, ndarray]` | Shock paths keyed by `shock_names`. Each value is a column view of `eps`. |
| `observables` | `dict[str, ndarray]` | Observable paths keyed by `observable_names`. Each value is a column view of `y`. Raises `ValueError` when the simulation did not request observables. |

## Piecewise results

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `is_piecewise` | `bool` | `True` when the result came from an OccBin simulation. |
| `regimes` | `ndarray[int]`, shape `(T, H)` | Accepted regime guesses. Column `0` is the regime realized at each date; `H` is the largest look-ahead horizon used. Raises `ValueError` for a nonpiecewise result. |
| `diagnostics` | `OccBinDiagnostics` | Per-date convergence record. Raises `ValueError` for a nonpiecewise result. |

## OccBinDiagnostics

```python
@dataclass(frozen=True)
class OccBinDiagnostics(
    T_used: ndarray,
    iters: ndarray,
    max_err: ndarray,
    periodic: ndarray,
)
```

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `T_used` | `ndarray[int]`, shape `(T,)` | Look-ahead horizon accepted for each date. |
| `iters` | `ndarray[int]`, shape `(T,)` | Guess-and-verify passes used for each date. |
| `max_err` | `ndarray[float]`, shape `(T,)` | Largest remaining constraint violation in the accepted guess for each date. |
| `periodic` | `ndarray[int]`, shape `(T,)` | Nonzero where a periodic regime sequence was accepted. |
