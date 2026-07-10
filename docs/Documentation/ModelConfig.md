---
tags:
    - doc
---
# ModelConfig

```python
@dataclass
class ModelConfig()
```

`ModelConfig` stores the parsed model as `SymPy` objects and expressions.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Model name. |
| variables | `#!python Variables` | Parsed variable functions, steady state expressions, and linearization methods. |
| parameters | `#!python list[sp.Symbol]` | Model parameters as symbols. |
| shock_map | `#!python dict[sp.Symbol, sp.Symbol]` | Mapping from innovation symbols to their corresponding model variables. |
| observables | `#!python list[sp.Symbol]` | Observable variables as symbols. |
| equations | `#!python Equations` | `dataclass` containing model, constraint, observable equations, observable affinity flags, and observable Jacobian. |
| calibration | `#!python Calib` | `dataclass` of parameter calibrations plus shock standard deviation and correlation parameter mappings. |
| symbolically_linearized | `#!python bool` | Whether the config has already been symbolically linearized. |
| source_yaml | `#!python str | None` | Source YAML text retained for bundle round trips. |

## `Variables`

```python
@dataclass
class Variables()
```

Parsed variable metadata.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| variables | `#!python list[sp.Function]` | Variables as functions of time. |
| steady_state | `#!python FunctionGetterDict[Function, Expr | None]` | Steady state expression per variable, or `None`. |
| linearization | `#!python FunctionGetterDict[Function, LinearizationMethod]` | Linearization method per variable. |

## `Equations`

```python
@dataclass
class Equations()
```

Parsed model, constraint, and observable equations.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| model | `#!python list[sp.Eq]` | Model equations. |
| constraint | `#!python SymbolGetterDict[Symbol, dict[Relational | And | Or | Not, Expr]]` | Piecewise OBC map keyed by constrained variable. Each variable may carry multiple condition entries, each paired with an alternative expression. |
| observable | `#!python SymbolGetterDict[Symbol, Expr]` | Observable equations. |
| obs_is_affine | `#!python SymbolGetterDict[Symbol, bool]` | Whether each observable equation is affine in current state variables. |
| obs_jacobian | `#!python Matrix` | Observable Jacobian with respect to current state variables. |
