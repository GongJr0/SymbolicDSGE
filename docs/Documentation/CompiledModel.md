---
tags:
    - doc
---
# CompiledModel

```python
@dataclass(frozen=True)
class CompiledModel()
```

`CompiledModel` contains the model components mapped to numeric/vectorized/lambdified counterparts.

```python
@dataclass(frozen=True)
class VariableLayout()
```

`VariableLayout` records the variable grouping inferred during compilation.

__Layout Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| declared_names | `#!python tuple[str, ...]` | Variables in config declaration order. |
| canonical_names | `#!python tuple[str, ...]` | Variables in compiled solver order. |
| exo_state_names | `#!python tuple[str, ...]` | Shock-map targets placed in the shocked/exogenous state block. |
| endo_state_names | `#!python tuple[str, ...]` | Unshocked variables inferred as states from model dynamics. |
| control_names | `#!python tuple[str, ...]` | Variables not inferred as states. |
| n_exog | `#!python int` | Number of shocked/exogenous state variables. |
| n_state | `#!python int` | Total number of state variables. |
| idx | `#!python dict[str, int]` | Variable-name to canonical index mapping. |

&nbsp;

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| config | `#!python ModelConfig` | Model config that was compiled. |
| kalman | `#!python KalmanConfig \| None` | Parsed Kalman config attached at compile time. |
| cur_syms | `#!python list[sympy.Symbol]` | Symbolized current-period state vector (`cur_*` symbols). |
| layout | `#!python VariableLayout` | Config-declared and compiler-inferred variable layout metadata. |
| var_names | `#!python list[str]` | Variables in compiled canonical order. |
| idx | `#!python dict[str, int]` | Variable-name to index mapping used by solver/simulation. |
| objective_eqs | `#!python list[sp.Expr]` | Solver targets in symbolic representation. (not used in the solver) |
| objective_funcs | `#!python list[Callable]` | Solver objectives as standalone `#!python Callable`s. |
| equations | `#!python Callable` | `#!python objective_funcs` compiled into a single `#!python Callable` target. (solver input) |
| calib_params | `#!python list[sympy.Symbol]` | Parameter symbols in canonical calibration order used by compiled functions. |
| observable_names | `#!python list[str]` | Observable variables as strings. |
| observable_eqs | `#!python list[sp.Expr]` | Measurement equations in symbolic representation. |
| observable_funcs | `#!python list[Callable]` | Measurement equations as `Callable`s. |
| observable_jacobian | `#!python Callable[..., np.ndarray]` | Compiled Jacobian function of measurement equations. |
| n_state | `#!python int` | Number of state variables in the compiled layout. |
| n_exog | `#!python int` | Number of shocked/exogenous state variables in the compiled layout. |

&nbsp;

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python .to_dict()` | `#!python dict` | `CompiledModel` in dictionary form. |
