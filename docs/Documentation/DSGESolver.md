# DSGESolver

```python
class DSGESolver(model_config: ModelConfig, t: sp.Symbol = sp.Symbol('t', integer=True))
```

Class responsible for model compilation and solution.

__Attributes:__

| __Name__ | __Type__ | __Description__ |
|:--------:|:--------:|:---------------:|
| model_config | `#!python ModelConfig` | Configuration object to be compiled/solved. |
| t | `#!python sp.Symbol` | Time symbol used in model components. |

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:-------------:|:---------------:|:---------------:|
| `#!python .compile(variable_order: list[sp.Function], n_state: int = None, n_exog: int = None, params_order: list[str] = None)` | `#!python CompiledModel` | Produces a `#!python CompiledModel` object respecting the given orders. `#!python n_exog` and `#!python n_state` must be supplied. |
| `#!python .solve(compiled: CompiledModel, parameters:dict[str, float] = None, steady_state: ndarray[float] | dict[str, float] = None, log_linear: bool = False)` | `#!python SolvedModel` | Solves the given compiled model and returns a `#!python SolvedModel` object.|