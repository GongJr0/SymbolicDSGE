---
tags:
    - doc
---
# Estimator

```python
from SymbolicDSGE import Estimator
```

`Estimator` exposes `mle`, `map`, and `mcmc` on top of a compiled DSGE model + Kalman likelihood.

???+ note "Recommended Entry"
    Most users should call [`DSGESolver.estimate`](./DSGESolver.md) or [`DSGESolver.estimate_and_solve`](./DSGESolver.md) instead of constructing `Estimator` directly.

## Constructor
```python
Estimator(
    *,
    solver: DSGESolver, # (1)!
    compiled: CompiledModel, # (2)!
    y: np.ndarray | pd.DataFrame, # (3)!
    observables: list[str] | None = None,
    estimated_params: Sequence[str] | None = None,
    priors: Mapping[str, Prior] | None = None, # (4)!
    steady_state: np.ndarray | dict[str, float] | None = None,
    log_linear: bool = False,
    x0: np.ndarray | None = None,
    p0_mode: str | None = None,
    p0_scale: float | None = None,
    jitter: float | None = None,
    symmetrize: bool | None = None,
    R: np.ndarray | None = None, # (5)!
)
```

1. Existing solver instance.
2. Compiled model from `DSGESolver.compile(...)`.
3. Measurement data for Kalman likelihood.
4. Required for `map(...)` and `mcmc(...)`.
5. Optional observation covariance override. If omitted through solver entrypoints, `R` can be inferred before estimation.

## Utility
```python
Estimator.make_prior(
    *,
    distribution: str, # (1)!
    parameters: dict[str, Any],
    transform: str, # (2)!
    transform_kwargs: dict[str, Any] | None = None,
) -> Prior
```

1. Distribution family string.
2. Transform method string.

Equivalent to `#!python SymbolicDSGE.bayesian.make_prior(...)`.

## Likelihood / Posterior Evaluation
```python
Estimator.theta0() -> np.ndarray
Estimator.loglik(theta: np.ndarray) -> float
Estimator.logprior(theta: np.ndarray) -> float
Estimator.logpost(theta: np.ndarray) -> float
```

???+ note "Optimization Space"
    `theta` is unconstrained internal space. Estimator applies prior transforms to map between unconstrained `theta` and constrained model parameters.

## MLE
```python
Estimator.mle(
    *,
    theta0: np.ndarray | None = None, # (1)!
    bounds: Sequence[tuple[float, float]] | None = None,
    method: str = "L-BFGS-B",
    options: Mapping[str, Any] | None = None,
) -> OptimizationResult
```

1. If `None`, uses transformed calibration defaults.

## MAP
```python
Estimator.map(
    *,
    theta0: np.ndarray | None = None, # (1)!
    bounds: Sequence[tuple[float, float]] | None = None,
    method: str = "L-BFGS-B",
    options: Mapping[str, Any] | None = None,
) -> OptimizationResult
```

1. Requires non-`None` priors at estimator construction.

## MCMC
```python
Estimator.mcmc(
    *,
    n_draws: int, # (1)!
    burn_in: int = 1000, # (2)!
    thin: int = 1, # (3)!
    theta0: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    adapt: bool = True, # (4)!
    adapt_start: int = 100,
    adapt_interval: int = 25,
    proposal_scale: float = 0.1,
    adapt_epsilon: float = 1e-8,
    update_R_in_iterations: bool = False, # (5)!
) -> MCMCResult
```

1. Number of retained posterior draws.
2. Number of initial iterations discarded.
3. Retain every `thin`-th iteration after burn-in.
4. Adaptive covariance updates are performed during burn-in only.
5. Rebuild `R` from current parameter draw when symbolic `R` metadata is available and relevant parameters are being estimated.

???+ note "Thinning Semantics"
    Thinning is applied after burn-in using `(t - burn_in) % thin == 0`.

???+ note "MCMC Sample Space"
    `MCMCResult.samples` are returned in constrained parameter space (parameter names), not raw unconstrained `theta`.

## Result Objects
### OptimizationResult
| __Field__ | __Type__ | __Description__ |
|:----------|:--------:|----------------:|
| kind | `#!python str` | `"mle"` or `"map"` |
| x | `#!python np.ndarray` | Optimized unconstrained vector |
| theta | `#!python dict[str, float]` | Optimized constrained parameters |
| success | `#!python bool` | Optimizer convergence flag |
| message | `#!python str` | Optimizer status message |
| fun | `#!python float` | Objective value at optimum |
| loglik | `#!python float` | Log-likelihood at optimum |
| logprior | `#!python float` | Log-prior at optimum |
| logpost | `#!python float` | Log-posterior at optimum |
| nfev | `#!python int` | Objective evaluations |
| nit | `#!python int | None` | Iterations |
| raw | `#!python scipy.optimize.OptimizeResult` | Raw scipy output |

### MCMCResult
| __Field__ | __Type__ | __Description__ |
|:----------|:--------:|----------------:|
| param_names | `#!python list[str]` | Parameter order for samples |
| samples | `#!python np.ndarray` | Retained posterior samples |
| logpost_trace | `#!python np.ndarray` | Posterior trace for retained samples |
| accept_rate | `#!python float` | Acceptance ratio |
| n_draws | `#!python int` | Retained draw count |
| burn_in | `#!python int` | Burn-in iterations |
| thin | `#!python int` | Thinning interval |

__Methods:__

```python
MCMCResult.hpd_intervals(
    alpha: float = 0.05, # (1)!
) -> dict[str, tuple[float, float]]
```

1. Significance level. Must satisfy `#!python 0 <= alpha < 1`; each interval covers approximately `#!python 1 - alpha` of the retained marginal draws.

Compute marginal highest-posterior-density (HPD) intervals for each parameter column.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| alpha | Significance level used to determine the empirical HPD coverage. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, tuple[float, float]]` | Mapping from parameter name to the shortest empirical marginal interval containing approximately `#!python 1 - alpha` of the retained posterior draws. |

&nbsp;

```python
MCMCResult.joint_hpd_set(
    alpha: float = 0.05, # (1)!
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]
```

1. Significance level. Must satisfy `#!python 0 <= alpha < 1`; the returned set covers at least `#!python 1 - alpha` of the retained joint draws.

Compute an empirical joint HPD set for the full parameter vector.

???+ note "Finite-Sample Joint HPD Approximation"
    Retained draws are ranked by `#!python logpost_trace` and all draws at or above the cutoff are included in the set. If multiple draws are tied at the boundary log-posterior, they are all retained, so the realized coverage can be slightly larger than `#!python 1 - alpha`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| alpha | Significance level used to determine the empirical HPD coverage. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python tuple[np.ndarray, np.ndarray, float, np.ndarray]` | Tuple `#!python (samples, logpost, threshold, indices)` where `samples` are the retained parameter vectors in the joint HPD set, `logpost` are their posterior values, `threshold` is the cutoff log-posterior, and `indices` are positions of the retained draws in the original stored chain. |

&nbsp;

```python
MCMCResult.posterior_kde_plot() -> None
```

Plot marginal posterior kernel-density estimates for each retained parameter column.

This is a quick visual diagnostic for posterior shape. It is useful for checking skewness, heavy tails, and obvious multimodality in the retained draws. A separate subplot is produced for each parameter and displayed immediately with `#!python matplotlib.pyplot.show()`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| None | This method uses the retained samples already stored on the result object. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python None` | Displays a Matplotlib figure of marginal KDE curves and returns nothing. |

&nbsp;

```python
MCMCResult.posterior_traces() -> None
```

Plot retained posterior draws for each parameter as trace diagnostics.

Trace plots are useful for checking whether the retained chain appears to mix well, whether it still shows drift, and whether particular parameters exhibit unusually persistent autocorrelation or regime changes. A separate subplot is produced for each parameter and displayed immediately with `#!python matplotlib.pyplot.show()`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| None | This method uses the retained samples already stored on the result object. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python None` | Displays a Matplotlib figure of per-parameter trace plots and returns nothing. |

&nbsp;

```python
MCMCResult.logpost_trace_plot() -> None
```

Plot the retained log-posterior sequence across MCMC iterations.

This diagnostic helps identify abrupt changes in posterior fit, long stretches of poor exploration, or chains that remain unstable even after burn-in and thinning have been applied. The plot is generated from `#!python MCMCResult.logpost_trace`, which stores one log-posterior value per retained draw, and is displayed immediately with `#!python matplotlib.pyplot.show()`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| None | This method uses the retained log-posterior trace already stored on the result object. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python None` | Displays a Matplotlib figure of the retained log-posterior trace and returns nothing. |
