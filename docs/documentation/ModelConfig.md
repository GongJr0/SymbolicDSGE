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
| name | `str` | Model name. |
| variables | `list[sympy.Function]` | Parsed variables (as functions of time). |
| ss_seed | `FunctionGetterDict[Expr | None]` | Newton seed for the steady-state solve, per variable. `None` seeds at zero. |
| parameters | `list[sympy.Symbol]` | Model parameters as symbols. |
| shocks | `list[sympy.Symbol]` | Innovation symbols, in declaration order, which is the shock column order. |
| observables | `list[sympy.Symbol]` | Observable variables as symbols. |
| equations | `Equations` | `dataclass` containing model equations, constraints, regimes, observable equations, observable affinity flags. |
| calibration | `Calib` | `dataclass` of parameter calibrations plus std/corr parameter mappings for shocks and measurement noise. |

__Methods:__

```python
@staticmethod
ModelConfig.parse_yaml(
    path: str | Path,
) -> ModelConfig
```

The same parse as [`parse_config`](#parse_config), reached from the config class.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| path | Path to the YAML config file. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `ModelConfig` | Parsed model config. |

&nbsp;


```python
ModelConfig.to_yaml(
    path: str | Path,
    *,
    digits: int | None = None, # (1)!
) -> None
```

1. Rounds the floats written to the file. It does not change the values held on the config.

Write this config to a YAML file.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| path | Output YAML path. Missing parent directories are created. |
| digits | Decimal places for plain floats. `None` writes them unrounded. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `None` | Writes to `path`. |

???+ info "What `digits` reaches"
    Calibration parameter values and numeric `ss_seed` entries. A symbolic `ss_seed` expression is printed as it stands.

???+ note "Writing mutated configs"
    `to_yaml` produces the exact representation of the config in memory.
    Edits made to a config after parsing (e.g. parameter estimation) will be reflected in the written file.

# Equations

```python
@dataclass
class Equations()
```

Parsed model, constraint, regime, and observable equations.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| model | `dict[str, sympy.Eq]` | Model equations keyed by equation name. |
| constraints | `dict[str, Constraint | None]` | `bind`/`relax` pairs as `Constraint` objects. Keyed by the constraint name.|
| regimes | `RegimeGetterDict[Regime]` | Regime equations keyed by the constraint names that define the regime. Each `Regime` is a mapping of model equation name and the expression that replaces it when the regime binds. |
| observables | `SymbolGetterDict[sympy.Expr]` | Observable equations. |
| obs_is_affine | `SymbolGetterDict[bool]` | Whether each observable equation is affine in current state variables. |

# Calib

```python
@dataclass
class Calib()
```

Calibration values for parameters and std/corr symbol mappings for shocks and measurement noise.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| parameters | `SymbolGetterDict[float]` | Parameter calibration values keyed by parameter symbol. |
| shocks | `CovarianceSpec` | `SymbolGetterDict[str]` for `std` parameters and `PairGetterDict[str | None]` for `corr` parameters. |
| measurement_noise | `CovarianceSpec | None` | Optional; `SymbolGetterDict[str]` for `std` parameters and `PairGetterDict[str | None]` for `corr` parameters. |

# parse_config

```python
def parse_config(
    path: str | Path,
) -> ModelConfig
```

Parse a YAML configuration file (see [configuration guide](../guides/model_config_guide.md)) into a `ModelConfig`.
`parse_config` is re-exported at `SymbolicDSGE` root.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| path | Path to the YAML config file. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `ModelConfig` | Config holding the model's symbolic equations, variables, parameters, shocks, and calibration. |

???+ warning "Calibration completeness"
    Parsing requires every declared parameter to carry a calibration value, and every parameter named in a shock or measurement noise section to be both declared and calibrated.

## Example

```python
from SymbolicDSGE import parse_config

model = parse_config("MODELS/POST82.yaml")
model.to_yaml("MODELS/POST82_emitted.yaml", digits=6)
```
