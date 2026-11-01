---
tags:
    - doc
---

# Linearization

```python
from SymbolicDSGE.linearization import Linearizer, linearize_model
```

`linearization` is a minimal submodule providing symbolic linearization of models written in level form.
The module is written to be used as a tool for analysis, no library function depends on a model being symbolically linearized.

__Linearization methods per variable:__

| __Value__ | __Variable denotes__ | __Reconstruction__ |
|:----------|:--------------------:|-------------------:|
| `#!python "level"` | deviation from the point, $\hat{x} = x - \bar{x}$ | $\bar{x} + \hat{x}$ |
| `#!python "log"` | log deviation, $\hat{x} = \ln x - \ln \bar{x}$ | $\bar{x}\exp(\hat{x})$ |


???+ warning "Solving a point-linear model"
    For any model in levels, the steady state is a function of parameters.
    Linearizing a model at some parameter vector embeds that steady state into the model's equations irrecoverably.
    While it is possible to generate a `ModelConfig` from a point-linear view and solve it as a regular model, it is strongly discouraged.
    With steady states locked to zero and the model in deviation form, any workflow involving parameter draws (i.e. estimation) will remain unable to re-linearize the model numerically, likelihoods and parameter values will be affected and no error will be produced.

&nbsp;

## Linearizer

```python
class Linearizer(
    compiled: CompiledModel,
    methods: Mapping[str, Literal["log", "level"]] | None = None,
    *,
    default: Literal["log", "level"] = "level",
)
```

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| compiled | Compiled model to linearize. |
| methods | Method per variable. Omitted variables take `default`. `None` puts every variable on `default`. |
| default | Method for variables `methods` does not name. |

???+ note "Compiler-Minted Variables"
    Linearization occurs on the variables named in the model config.
    Compiler-generated auxiliary variables inherit their methods from the variable they are derived from.

&nbsp;

```python
Linearizer.at(params: Mapping[str, float] | None = None) -> PointLinearView
```

Linearize the model at the given parameterization.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| params | Parameter overrides on the model's calibration. `None` linearizes at the config calibration. |

&nbsp;

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python PointLinearView` | The model expanded about the steady state implied by `params`. |

???+ warning "A non-positive point under `log`"
    Variables with a `log` method must resolve to positive steady state values.
    `level` does not have this restriction; switch an offending variable to use `level` or a parameterization that yields a positive steady state.

&nbsp;

```python
linearize_model(
    compiled: CompiledModel,
    methods: Mapping[str, Literal["log", "level"]] | None = None,
    *,
    default: Literal["log", "level"] = "level",
) -> PointLinearView
```

Linearize the model at the calibration parameters.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| compiled | Compiled model to linearize. |
| methods | Method per variable. Omitted variables take `default`. `None` puts every variable on `default`. |
| default | Method for variables `methods` does not name. |

&nbsp;

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python PointLinearView` | The model expanded about the steady state implied by the config calibration. |

&nbsp;

## PointLinearView

```python
@dataclass(frozen=True)
class PointLinearView()
```

Container for linearized model equations and parameterization/steady state context.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| model | `#!python dict[str, Eq]` | Model equations in deviation form. |
| observables | `#!python SymbolGetterDict[Expr]` | Measurement equations, affine and in levels. |
| parameters | `#!python SymbolGetterDict[float]` | The parameterization the expansion was taken at. |
| steady_state | `#!python FunctionGetterDict[float]` | The expansion point. |

Parameters stay symbolic in the emitted equations.

&nbsp;

```python
PointLinearView.to_config() -> ModelConfig
```

The linearized model as a `ModelConfig`, which `solve_model` and `compile_model` accepts like any other.

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python ModelConfig` | A copy of the source config with the linearization applied. |

__Overrides:__

| __Field__ | __Becomes__ |
|:----------|------------:|
| `equations.model` | deviation form |
| `equations.observables` | affine form |
| `equations.obs_is_affine` | all `#!python True`, by construction |
| `calibration.parameters` | the view's parameterization |
| `ss_seed` | zeros; the deviation system's steady state is the origin |

Everything else is carried from the source config unchanged.

???+ warning "Estimating a point-linear config"
    `solve_model` and `compile_model` refuses a `PointLinearView`, and `to_config()` is the explicit way past that.
    Freezing the expansion at a steady state introduces a restriction original config does not have.
    Consequently, a pre-linearized levels model is accurate only at the specific parameterization producing the resolved steady state.
    Proceed with caution if any workflow mutating the parameterization is intended.

???+ warning "Constrained models"
    `to_config()` raises `#!python NotImplementedError` when the model declares constraints.
    Regime-switching models can have ambiguous interpretations under symbolic linearization and are not yet supported.
