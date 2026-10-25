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
| variables | `#!python list[sympy.Function]` | Parsed variables (as functions of time). |
| ss_seed | `#!python FunctionGetterDict[Expr | None]` | Newton seed for the steady-state solve, per variable. `None` seeds at zero. |
| parameters | `#!python list[sympy.Symbol]` | Model parameters as symbols. |
| shocks | `#!python list[sympy.Symbol]` | Innovation symbols, in declaration order, which is the shock column order. |
| observables | `#!python list[sympy.Symbol]` | Observable variables as symbols. |
| equations | `#!python Equations` | `dataclass` containing model equations, constraints, regimes, observable equations, observable affinity flags. |
| calibration | `#!python Calib` | `dataclass` of parameter calibrations plus std/corr parameter mappings for shocks and measurement noise. |


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
| constraints | `#!python dict[str, Constraint | None]` | `bind`/`relax` pairs as `Constraint` objects. Keyed by the constraint name.|
| regimes | `#!python RegimeGetterDict[Regime]` | Regime equations keyed by the constraint names that define the regime. Each `Regime` is a mapping of model equation name and the expression that replaces it when the regime binds. |
| observables | `#!python SymbolGetterDict[Expr]` | Observable equations. |
| obs_is_affine | `#!python SymbolGetterDict[bool]` | Whether each observable equation is affine in current state variables. |
