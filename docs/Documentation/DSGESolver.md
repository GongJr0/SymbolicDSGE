---
tags:
    - doc
---
# DSGESolver

```python
class DSGESolver(model_config: ModelConfig, kalman_config: KalmanConfig)
```

Class responsible for model compilation and solution.

__Attributes:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| model_config | `#!python ModelConfig` | Configuration object to be compiled/solved. |
| kalman_config | `#!python KalmanConfig` | Kalman Filter configuration object. |
| t | `#!python sp.Symbol` | Time symbol used in model components. |

__Methods:__

```python
DSGESolver.compile(
    *,
    variable_order: list[sp.Function] | None = None,
    n_state: int = None,
    n_exog: int = None,
    params_order: list[str] | None = None
) -> CompiledModel
```

???+ warning "Variable Ordering Convention"
    The model expects the first `#!python n_exog` variables to be the exogenous components. Before solving the model either;

    - Ensure the variable ordering in the config file follows this convention.
    - Supply an order specification at compile time.

??? info "Planned Changes"
    Current input constraints will be eliminated as `#! SymbolicDSGE` moves towards the beta releases. `n_exog` and `n_state` will be inferred through flags in the config; and variable ordering will be managed internally.

Produces a `#!python CompiledModel` object respecting the given orders. `#!python n_exog` and `#!python n_state` must be supplied for the current `#!python linearsolve` backend.

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
DSGESolver.solve(
    compiled: CompiledModel,
    parameters: dict[str, float] = None,
    steady_state: ndarray[float] | dict[str, float] = None
) -> SolvedModel
```

Solves the given compiled model and returns a `#!python SolvedModel` object.

 __Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| compiled | The `#!python CompiledModel` to solve. |
| parameters | parameter values as dict to override the calibration config. |
| steady_state | model variables' steady state. Defaults to zeroes. (often used in gap models) |

 &nbsp;

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python SolvedModel` | Solved model object with relevant methods attached. |

&nbsp;

???+ note "Linearized Inputs"
    `DSGESolver.solve(...)` expects a compiled linearized model. If your original model is nonlinear, apply `#!python SymbolicDSGE.linearization.linearize_model(...)` before compilation.

```python
DSGESolver.estimate(
    *,
    compiled: CompiledModel,
    y: np.ndarray | pd.DataFrame,
    method: str = "mle",
    theta0: np.ndarray | Mapping[str, float] | None = None, # (1)!
    observables: list[str] | None = None,
    estimated_params: list[str] | None = None,
    priors: Mapping[str, Any] | None = None,
    steady_state: np.ndarray | dict[str, float] | None = None,
    x0: np.ndarray | None = None,
    p0_mode: str | None = None,
    p0_scale: float | None = None,
    jitter: float | None = None,
    symmetrize: bool | None = None,
    R: np.ndarray | None = None, # (2)!
    **method_kwargs: Any,
) -> MCMCResult | OptimizationResult
```

1. If `#!python theta0` is passed as a dictionary, it is reordered internally to the estimator's canonical parameter order.
2. If `#!python R` is not supplied, the estimator attempts to infer `R` from data before optimization/sampling (MAP on full `R`, with MLE fallback on failure).

???+ note Filter Mode
    Filter mode is inferred internally (`linear` if all selected measurement equations are affine, otherwise `extended`).

__Method kwargs:__

- `#!python method="mle"`: forwarded to `Estimator.mle(...)`
- `#!python method="map"`: forwarded to `Estimator.map(...)`
- `#!python method="mcmc"`: forwarded to `Estimator.mcmc(...)`

&nbsp;

```python
DSGESolver.estimate_and_solve(
    *,
    compiled: CompiledModel,
    y: np.ndarray | pd.DataFrame,
    method: str = "mle",
    theta0: np.ndarray | Mapping[str, float] | None = None,
    posterior_point: str = "mean",
    observables: list[str] | None = None,
    estimated_params: list[str] | None = None,
    priors: Mapping[str, Any] | None = None,
    steady_state: np.ndarray | dict[str, float] | None = None,
    x0: np.ndarray | None = None,
    p0_mode: str | None = None,
    p0_scale: float | None = None,
    jitter: float | None = None,
    symmetrize: bool | None = None,
    R: np.ndarray | None = None,
    **method_kwargs: Any,
) -> tuple[MCMCResult | OptimizerResult, SolvedModel]
```

Runs estimation and then solves the model at the estimated parameter point.

For `#!python method="mcmc"`, `#!python posterior_point` selects the parameter point used for solving:

- `#!python "mean"`: posterior sample mean
- `#!python "last"`: last retained sample
- `#!python "map"`: sample with the highest posterior likelihood
- `#!python "mode"`: equivalent to `"map"` by definition
