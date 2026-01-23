---
tags:
    - doc
---
# DSGESolver

```python
class DSGESolver(model_config: ModelConfig, t: sp.Symbol = sp.Symbol('t', integer=True))
```

Class responsible for model compilation and solution.

__Attributes:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| model_config | `#!python ModelConfig` | Configuration object to be compiled/solved. |
| t | `#!python sp.Symbol` | Time symbol used in model components. |

__Methods:__

```python
DSGESolver.compile(
    variable_order: list[sp.Function], 
    n_state: int = None, 
    n_exog: int = None, 
    params_order: list[str] = None
    ) -> CompiledModel 
```

???+ warning "Variable Ordering Convention"
    The model expects the first `#!python n_exog` variables to be the exogenous components. Before solving the model either;
    
    - Ensure the variable ordering in the config file follows this convention.
    - Supply an order specification at compile time. 

??? info "Planned Changes"
    Current input constraints will be eliminated as `#! SymbolicDSGE` moves towards the beta releases. `n_exog` and `n_state` will be inferred through flags in the config; and variable ordering will be managed internally. 

Produces a `#!python CompiledModel` object respecting the given orders. `#!python n_exog` and `#!python n_state` must be supplied.
 
 __Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| variable_order | Custom ordering of variables if desired. |
| n_state | Number of state variables. |
| n_exog | Number of exogenous variables. |
| params_order | Custom ordering of model parameters if desired. |

 &nbsp;

__Returns:__
| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python CompiledModel` | Numerically compiled model components returned as an object. |




```python
SolvedModel.solve(
    compiled: CompiledModel, 
    parameters: dict[str, float] = None, 
    steady_state: ndarray[float] | dict[str, float] = None, 
    log_linear: bool = False
    ) -> SolvedModel
``` 

Solves the given compiled model and returns a `#!python SolvedModel` object.

 __Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| compiled | The `#!python CompiledModel` to solve. |
| parameters | parameter values as dict to override the calibration config. |
| steady_state | model variables' steady state. Defaults to zeroes. (often used in gap models) |
| log_linear | Indicates the model is in log-linear specification to the solver. |

 &nbsp;

__Returns:__
| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python SolvedModel` | Solved model object with relevant methods attached. |