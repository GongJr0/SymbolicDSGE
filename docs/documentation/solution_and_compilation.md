---
tags:
    - doc
---
# Model Solution and Compilation

```python
def compile_model(config: ModelConfig) -> CompiledModel
```
Compile a parsed config into a numerical representation.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| config | Parsed config object. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `CompiledModel` | Numericalized model holding auxiliary generation context and evaluable functions of model components. |

???+ note "Pre-compilation"
    Solving a model only requires the `ModelConfig` object.
    Pre-compilation via `compile_model` is useful for avoiding repeated compilation when the same config is used in multiple library components. Classes and functions that allow skipping compilation will advertise an optional `compiled` argument.
    Note that compilation is often orders of magnitude more expensive than the actual solver, it is recommended to avoid recompiling the model in any workflow that solves inside a loop.

    Native kernels, like estimation and Monte Carlo, are not subject to this runtime cost.

### Example

```python
from SymbolicDSGE import parse_config, compile_model
conf = parse_config("path/to/model.yaml")
compiled = compile_model(conf)
```

```python
def solve_model(
    config: ModelConfig,
    *,
    compiled: CompiledModel | None = None,
    parameters: dict[str, float] | None = None,
    ss_seed: dict[str, float] | list[float] | ndarray | None = None
    order: int = 1,
) -> SolvedModel[FirstOrderSolution | SecondOrderSolution | PiecewiseSolution]:
```
Solve a model specification and return the appropriate solution object.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| config | Parsed config object. |
| compiled | optional pre-compiled model. |
| parameters | optional parameter value overrides. |
| ss_seed | optional ss_seed override, if omitted as an argument and in the config, `ss_seed` will be zeros for every variable. |
| order | Perturbation order of the solution, models with constraints only support first order perturbations. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `SolvedModel` | A solved model ready for use in any library function. |

???+ note "Return Types per Specification"
    Each `T` for `SolvedModel[T]` has a concrete class, with `SolvedModel` being the base class. The types mostly differ the type of solution while methods remain stably accessible in every class. For combinations of `{order, has_constraints}` the returns are as follows:

    | order/has_constraints | `False` | `True` |
    | 1                     | FirstOrderSolvedModel | PiecewiseSolvedModel |
    | 2                     | SecondOrderSolvedModel | N/A |

### Example

```python
from SymbolicDSGE import parse_config, solve_model
conf = parse_config("path/to/model.yaml")
sol = solve_model(conf)
```
