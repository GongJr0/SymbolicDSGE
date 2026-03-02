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
| shock_map | `#!python dict[sp.Symbol, sp.Symbol]` | Mapping from innovation symbols to their corresponding model variables. |
| observables | `#!python list[sp.Symbol]` | Observable variables as symbols. |
| equations | `#!python Equations` | `dataclass` containing model, constraint, observable equations, observable-affinity flags, and observable Jacobian. |
| calibration | `#!python Calib` | `dataclass` of parameter calibrations plus shock standard-deviation/correlation parameter mappings. |
