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
| parameters | `#!python list[sympy.Symbol]` | Model parameters as symbols. |
| shocks | `#!python list[sympy.Symbol]` | Innovation symbols, in declaration order, which is the shock column order. |
| observables | `#!python list[sympy.Symbol]` | Observable variables as symbols. |
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
| variables | `#!python list[sympy.Function]` | Variables as functions of time. |
| ss_seed | `#!python FunctionGetterDict[Expr | None]` | Newton seed for the steady-state solve, per variable. `None` seeds at zero. |
| linearization | `#!python FunctionGetterDict[LinearizationMethod]` | Linearization method per variable. |

## `Equations`

```python
@dataclass
class Equations()
```

Parsed model, constraint, regime, and observable equations.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| model | `#!python dict[str, sympy.Eq]` | Model equations keyed by equation name. |
| constraint | `#!python dict[str, Constraint | None]` | `bind`/`relax` pairs as `Constraint` objects. Keyed by the constraint name.|
| regime | `#!python RegimeGetterDict[Regime]` | Regime equations keyed by the constraint names that define the regime. Each `Regime` is a mapping of model equation name and the expression that replaces it when the regime binds. |
| observable | `#!python SymbolGetterDict[Expr]` | Observable equations. |
| obs_is_affine | `#!python SymbolGetterDict[bool]` | Whether each observable equation is affine in current state variables. |
