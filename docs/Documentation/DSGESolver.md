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
    variable_order: list[sp.Function | str] | None = None,
    n_state: int | None = None,
    n_exog: int | None = None,
    params_order: list[str] | None = None,
    linearize: bool = False,
) -> CompiledModel
```

???+ note "Inferred Variable Layout"
    `DSGESolver.compile(...)` infers the solver layout from the model config. Shock-map targets define the shocked/exogenous state block, dynamic equations define the remaining state variables, and the rest are treated as controls.

    If `#!python variable_order`, `#!python n_state`, or `#!python n_exog` are supplied, they are treated as explicit expectations. The compiler sanity-checks them against the config-derived layout and raises if they disagree.

Produces a `#!python CompiledModel` object using the inferred canonical variable layout unless an explicit, validated order is supplied.

 __Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| variable_order | Optional expected variable order. If supplied, it must agree with the config-derived state/exogenous grouping. |
| n_state | Optional expected number of state variables. Raises if it disagrees with inference. |
| n_exog | Optional expected number of shocked/exogenous state variables. Raises if it disagrees with inference. |
| params_order | Custom ordering of model parameters if desired. |
| linearize | Apply symbolic linearization to a copied model config before compilation. |

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
    `DSGESolver.solve(...)` expects a compiled linearized model. If your original model is nonlinear, pass `#!python linearize=True` to `#!python DSGESolver.compile(...)` or apply `#!python SymbolicDSGE.core.linearize_model(...)` before compilation.

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
