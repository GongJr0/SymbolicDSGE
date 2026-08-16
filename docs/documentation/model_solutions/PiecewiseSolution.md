---
tags:
    - doc
---
# Piecewise Solution

```python
@dataclass(frozen=True)
class PiecewiseSolution(
    steady_state: ndarray,
    stab: int,
    eig: ndarray,
    order: int,
    a: ndarray,
    b: ndarray,
    c: ndarray,
    d: ndarray,
    cst: ndarray,
    ghx_ref: ndarray,
    ref: FirstOrderSolution,
)
```

Piecewise-linear policy data returned by the OccBin solver. `PiecewiseSolution` extends `BaseSolution`; all inherited fields are repeated here for reference. The solution stores a linear pencil for each binding regime and a first-order reference policy for the relaxed regime.

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `steady_state` | `ndarray[float]`, shape `(n_var,)` | Expansion point of the relaxed reference regime. |
| `stab` | `int` | Reference-regime stability indicator: `-1` means too few stable eigenvalues, `0` means the required number, and `1` means too many. |
| `eig` | `ndarray[complex]` | Eigenvalues of the relaxed reference regime. |
| `order` | `int` | Always `1`, because each regime is linear. |
| `a`, `b`, `c` | `ndarray[float]`, shape `(n_regime, n_var, n_var)` | Lead, current, and lag coefficients of each regime pencil. |
| `d` | `ndarray[float]`, shape `(n_regime, n_var, n_shock)` | Innovation coefficient of each regime pencil. |
| `cst` | `ndarray[float]`, shape `(n_regime, n_var)` | Constraint offset of each regime. It is zero in the relaxed reference regime. |
| `ghx_ref` | `ndarray[float]`, shape `(n_var, n_state)` | Full relaxed reference policy used as the terminal condition beyond the checked horizon. |
| `ref` | `FirstOrderSolution` | Relaxed reference policy, including its `p`, `f`, `A`, and `B` matrices. |

For regime `r`, the pencil is `a[r] E[y(t+1)] = b[r] y(t) + c[r] y(t-1) + d[r] ε(t) - cst[r]`. Regimes are indexed by their constraint-binding bitmask, and index `0` is the relaxed reference regime. `ghx_ref` stacks the reference solution's state and control rules, `ref.p` and `ref.f`.

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `A` | `ndarray[float]`, shape `(n_var, n_var)` | `ref.A`, the relaxed reference regime's state transition. |
| `B` | `ndarray[float]`, shape `(n_var, n_shock)` | `ref.B`, the relaxed reference regime's shock impact. |

These forward the reference regime's state space, so a consumer that reads a first-order `A`/`B` reaches it without unpacking `ref`. Filtering and estimation take this route: they describe the model with its constraints ignored, which is the reference regime by definition.
