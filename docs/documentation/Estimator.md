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
    filter_mode: str = "linear", # (6)!
    estimated_params: Sequence[str] | None = None,
    priors: Mapping[str, Prior] | None = None, # (4)!
    ss_seed: np.ndarray | dict[str, float] | None = None,
    x0: np.ndarray | None = None,
    jitter: float | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = True,
    R: np.ndarray | None = None, # (5)!
    P0: np.ndarray | None = None, # (7)!
)
```

1. Existing solver instance.
2. Compiled model from `DSGESolver.compile(...)`.
3. Measurement data for Kalman likelihood.
4. Required for `map(...)` and `mcmc(...)`.
5. Optional constant observation-covariance override. If omitted, `R` comes from the Kalman config: a fixed calibrated matrix, or rebuilt from the current parameters each evaluation when the model exposes symbolic `R` metadata.
6. Filter algorithm for the likelihood: `#!python "linear"`, `#!python "extended"` (EKF), or `#!python "unscented"` (UKF). Chosen explicitly, not inferred.
7. Optional initial state-covariance override. If omitted, `P0` comes from the Kalman config. Supply a full `(n_var, n_var)` matrix in compiled variable order; for `unscented` mode its state block is embedded automatically.

???+ info "Filter Initial Conditions"
    In linear and extended likelihoods, `x0` and `P0` are the prior mean and covariance for the first observation. In unscented likelihoods, they describe the state and covariance before the first observation.

`joseph_cov=True` uses the Joseph covariance update for linear and extended likelihoods. Set it to `False` for the simplified update, which is faster but less numerically robust. It does not affect unscented likelihoods.

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
    method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B", # (2)!
    m: int = 10, # (3)!
    maxiter: int = 15000,
    maxfun: int = 15000,
    maxls: int = 20,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    fd_step: float = 0.0,
    xatol: float = 1e-4, # (4)!
    fatol: float = 1e-4,
) -> OptimizationResult
```

1. If `None`, uses transformed calibration defaults.
2. Native optimizer. Only the two listed are supported; see the note below.
3. L-BFGS-B options (`m`, `maxiter`, `maxfun`, `maxls`, `factr`, `pgtol`, `fd_step`); ignored by Nelder-Mead.
4. Nelder-Mead options (`xatol`, `fatol`); ignored by L-BFGS-B.

## MAP

```python
Estimator.map(
    *,
    theta0: np.ndarray | None = None, # (1)!
    bounds: Sequence[tuple[float, float]] | None = None,
    method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B", # (2)!
    jacobian: bool = False,  # (3)!
    m: int = 10, # (4)!
    maxiter: int = 15000,
    maxfun: int = 15000,
    maxls: int = 20,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    fd_step: float = 0.0,
    xatol: float = 1e-4, # (5)!
    fatol: float = 1e-4,
) -> OptimizationResult
```

1. Requires non-`None` priors at estimator construction.
2. Native optimizer. Only the two listed are supported; see the note below.
3. Includes the jacobian term arising from the prior transformations of random variables.

3. L-BFGS-B options (`m`, `maxiter`, `maxfun`, `maxls`, `factr`, `pgtol`, `fd_step`); ignored by Nelder-Mead.
4. Nelder-Mead options (`xatol`, `fatol`); ignored by L-BFGS-B.

???+ note "On the Jacobian Parameter"
    When `False`, the MAP transformations are treated as simple coordinate changes, consistent with point estimation.
    When `True`, the MAP transformations treat theta as a random variable and include the jacobian term to perserve the proabilistic interpretation of the posterior.
    Practically, set `jacobian=True` when passing the MAP estimates to `Estimator.mcmc` as `theta0` for the proposal covariance.
    For one-off point estimation, `jacobian=False` correctly finds the mode of the posterior in constrained space (the parameters as specified), the transforms only serve to move where that search occurs.

???+ note "Native optimizer set"
    `mle` and `map` run entirely in the native backend, which ships a curated set of optimizers with no scipy fallback. Only `#!python "L-BFGS-B"` (default; quasi-Newton with a finite-difference gradient) and `#!python "Nelder-Mead"` (gradient-free) are supported; any other `method` raises. Each optimizer's tuning parameters are passed as explicit keyword arguments: the L-BFGS-B group (`m`, `maxiter`, `maxfun`, `maxls`, `factr`, `pgtol`, `fd_step`) and the Nelder-Mead group (`xatol`, `fatol`).

## MCMC

```python
Estimator.mcmc(
    *,
    n_draws: int, # (1)!
    burn_in: int = 1000, # (2)!
    thin: int = 1, # (3)!
    theta0: np.ndarray | None = None, # (4)!
    random_state: int | np.random.Generator | None = None,
    adapt: bool = True, 
    adapt_start: int = 100,
    proposal_scale: float = 0.1,
    adapt_epsilon: float = 1e-8,
    compute_map: bool = True,
    map_options: dict[str, Any] | None = None,  # (5)!
    proposal_cov: np.ndarray | None = None,  # (6)!
    cov_fd_step_scale: float = 1.0,
    cov_fd_absolute_floor: float = 0.1,
) -> MCMCResult
```

1. Number of retained posterior draws.
2. Number of initial iterations discarded.
3. Retain every `thin`-th iteration after burn-in.
4. `None` uses calibration defaults. When `compute_map=True`, the MAP estimate is computed first and used as the starting point for the chain.
The proposal covariance will use the MAP point to compute the Hessian if `compute_map=True`.
5. Passed to `map(...)` if `compute_map=True`. If `None`, defaults are used. Refer to [Estimator.map](#map) for the available options.
6. Proposal covariance for the sampler. It's mutually exclusive with `compute_map=True` since the MAP estimate is used as the differentiation point for the proposal covariance and the `theta0` a chain starts at.

???+ note "Thinning Semantics"
    Thinning is applied after burn-in using `(t - burn_in) % thin == 0`.

???+ note "MCMC Sample Space"
    `MCMCResult.samples` are returned in constrained parameter space (parameter names), not raw unconstrained `theta`.

## Result Objects

MLE and MAP return `SymbolicDSGE.OptimizationResult`, mapped from the native optimizer's result struct; scipy is not involved.

### OptimizationResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| kind | `#!python str` | `"mle"` or `"map"` |
| x | `#!python np.ndarray` | Optimized unconstrained vector |
| theta | `#!python dict[str, float]` | The estimated parameters at the optimum, in constrained space, keyed by `estimated_params`. Parameters that were not estimated are not included: they stay at the model's calibration. |
| success | `#!python bool` | Optimizer convergence flag |
| message | `#!python str` | Optimizer status message |
| fun | `#!python float` | Objective value at optimum |
| loglik | `#!python float` | Log-likelihood at optimum |
| logprior | `#!python float` | Log-prior at optimum |
| logpost | `#!python float` | Log-posterior at optimum |
| nfev | `#!python int` | Objective evaluations |
| nit | `#!python int | None` | Iterations |

### MCMCResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| param_names | `#!python list[str]` | Parameter order for samples |
| samples | `#!python np.ndarray` | Retained posterior samples |
| logpost_trace | `#!python np.ndarray` | Posterior trace for retained samples, as a density over the parameters |
| logjac_trace | `#!python np.ndarray` | Log-jacobian of the prior transforms at each retained draw; add it to `logpost_trace` for the unconstrained density the sampler walked |
| accept_rate | `#!python float` | Acceptance ratio |
| n_draws | `#!python int` | Retained draw count |
| burn_in | `#!python int` | Burn-in iterations |
| thin | `#!python int` | Thinning interval |

???+ tip "Projecting to bundle metadata"
    Both result classes expose a `#!python .to_meta()` method that returns the matching `#!python OptimizationResultMeta` / `#!python MCMCResultMeta` for `.sdsge` storage. `MCMCResult` additionally exposes `#!python .posterior_arrays()` returning `#!python {"samples": ..., "logpost": ..., "logjac": ...}`, the bulk dict the bundle expects as `posterior`. `#!python BundleBuilder.add_estimation(result=...)` accepts the live result directly and calls both for you. See [`BundleBuilder`](./bundle/BundleBuilder.md#bundlebuilderadd_estimation).

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
