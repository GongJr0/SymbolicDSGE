---
tags:
    - doc
---
# Second Order Solution

```python
@dataclass(frozen=True)
class SecondOrderSolution(
    steady_state: ndarray,
    stab: int,
    eig: ndarray,
    order: int,
    p: ndarray,
    f: ndarray,
    A: ndarray,
    B: ndarray,
    gxx: ndarray,
    hxx: ndarray,
    gxu: ndarray,
    hxu: ndarray,
    guu: ndarray,
    huu: ndarray,
    gss: ndarray,
    hss: ndarray,
)
```

Second-order policy data returned by the perturbation solver. `SecondOrderSolution` extends `FirstOrderSolution`; all inherited fields are repeated here for reference. It adds the quadratic and risk corrections around the same steady state.

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| `steady_state` | `ndarray[float]`, shape `(n_var,)` | Newton-resolved expansion point of the solution. |
| `stab` | `int` | Stability indicator: `-1` means too few stable eigenvalues, `0` means the required number, and `1` means too many. |
| `eig` | `ndarray[complex]` | Eigenvalues of the linearized system. |
| `order` | `int` | Always `2`. |
| `p`, `hx` | `ndarray[float]`, shape `(n_state, n_state)` | First-order state transition. `hx` is the perturbation-notation alias of `p`. |
| `f`, `gx` | `ndarray[float]`, shape `(n_control, n_state)` | First-order control policy. `gx` is the perturbation-notation alias of `f`. |
| `A` | `ndarray[float]`, shape `(n_var, n_var)` | Full first-order state transition matrix in compiled variable order. |
| `B` | `ndarray[float]`, shape `(n_var, n_shock)` | Innovation impact matrix in compiled shock order. |
| `gxx` | `ndarray[float]`, shape `(n_control, n_state, n_state)` | Control correction for two state deviations. |
| `hxx` | `ndarray[float]`, shape `(n_state, n_state, n_state)` | State correction for two state deviations. |
| `gxu` | `ndarray[float]`, shape `(n_control, n_state, n_shock)` | Control correction for a state deviation and a shock. |
| `hxu` | `ndarray[float]`, shape `(n_state, n_state, n_shock)` | State correction for a state deviation and a shock. |
| `guu` | `ndarray[float]`, shape `(n_control, n_shock, n_shock)` | Control correction for two shocks. |
| `huu` | `ndarray[float]`, shape `(n_state, n_shock, n_shock)` | State correction for two shocks. |
| `gss` | `ndarray[float]`, shape `(n_control,)` | Risk correction scaled by shock variance for controls. |
| `hss` | `ndarray[float]`, shape `(n_state,)` | Risk correction scaled by shock variance for states. |
