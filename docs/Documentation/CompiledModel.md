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

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| config | `#!python ModelConfig` | Model config that was compiled. |
| var_names | `#!python list[str]` | Variables as strings. |
| objective_eqs | `#!python list[sp.Expr]` | Solver targets in symbolic representation. (not used in the solver) |
| objective_funcs | `#!python list[Callable]` | Solver objectives as standalone `#!python Callable`s. |
| equations | `#!python Callable` | `#!python objective_funcs` compiled into a single `#!python Callable` target. (solver input) |
| observable_names | `#!python list[str]` | Observable variables as strings. |
| observable_eqs | `#!python list[sp.Expr]` | Measurement equations in symbolic representation. |
| objective_funcs | `#!python list[Callable]` | Measurement equations as `Callable`s. |
| n_state | `#!python int` | Number of state variables. |
| n_exog | `#!python int` | Number of exogenous variables. |

&nbsp;

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python .to_dict()` | `#!python dict` | `CompiledModel` in dictionary form. |
