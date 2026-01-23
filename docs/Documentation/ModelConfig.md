---
tags:
    - doc
---
# ModelConfig

```python
@dataclass
class ModelConfig()
```

`ModelConfig` stores the parsed model as `SymPy` objects/expressions.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| name | `#!python str` | Model name. |
| variables | `#!python list[sp.Function]` | Variables as functions of time. |
| constrained | `#!python dict[Function, bool]` | Dictionary mapping functions to constraint status. |
| parameters | `#!python list[sp.Symbol]` | Model parameters as symbols. |
| shocks | `#!python list[sp.Symbol]` | Shock variables as symbols. |
| observables | `#!python list[sp.Symbol]` | Observable variables as symbols. |
| equations | `#!python Equations` | `dataclass` containing model, constraint, and observable equations.
| calibration | `#!python Calib` | `dataclass` of parameter calibrations and shock-to-sigma mappings. | 