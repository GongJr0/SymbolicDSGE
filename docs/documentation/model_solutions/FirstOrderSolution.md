---
tags:
    - doc
---
# First Order Solution

```python
@dataclass(frozen=True)
class FirstOrderSolution(
    steady_state: ndarray,
    stab: int,
    eig: ndarray,
    order: int,
    p: ndarray,
    f: ndarray,
    A: ndarray,
    B: ndarray,
)
```

First-order policy data returned by the Klein solver. `FirstOrderSolution` extends `BaseSolution`; all inherited fields are repeated here for reference. Its rule is `u(t) = f s(t)` and `s(t+1) = p s(t)`.

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `steady_state` | `ndarray[float]`, shape `(n_var,)` | Newton-resolved expansion point of the solution. |
| `stab` | `int` | Stability indicator: `-1` means too few stable eigenvalues, `0` means the required number, and `1` means too many. |
| `eig` | `ndarray[complex]` | Eigenvalues of the linearized system. |
| `order` | `int` | Always `1`. |
| `p` | `ndarray[float]`, shape `(n_state, n_state)` | State transition matrix. |
| `f` | `ndarray[float]`, shape `(n_control, n_state)` | Control policy matrix. |
| `A` | `ndarray[float]`, shape `(n_var, n_var)` | Full first-order state transition matrix in compiled variable order. |
| `B` | `ndarray[float]`, shape `(n_var, n_shock)` | Innovation impact matrix in compiled shock order. |

`p` and `f` use the model's state and control partition. `A` and `B` give the full affine state-space transition in compiled order: `x(t+1) = A x(t) + B ε(t+1)`.
