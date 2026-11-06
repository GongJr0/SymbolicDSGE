---
tags:
    - doc
---
# Estimator

```python
from SymbolicDSGE import Estimator
```

```python
Estimator(
    *,
    config: ModelConfig
    y: np.ndarray | pd.DataFrame, # (1)!
    compiled: CompiledModel, # (2)!
    observables: list[str] | None = None,
    filter_mode: str = "linear", # (3)!
    estimated_params: Sequence[str] | Mapping[str, Prior] | None = None, # (4)!
    ss_seed: np.ndarray | dict[str, float] | None = None,
    jitter: float | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False,
    R: np.ndarray | None = None, # (5)!
    P0: np.ndarray | None = None, # (6)!
)
```

1. Measurement data for Kalman likelihood.
2. Compiled model from `compile_model(...)`.
3. Filter algorithm for the likelihood: `#!python "linear"`, `#!python "extended"` (EKF), or `#!python "unscented"` (UKF). Chosen explicitly, not inferred.
4. Parameter names to estimate, or a mapping from parameter name to `Prior` object. If `None`, all parameters are estimated with uniform priors. `map()` and `mcmc()` require priors to be specified and `mle()` ignores them.
5. Optional constant observation-covariance override. If omitted, `R` is built from the calibration's `measurement_noise` block.
6. Optional initial state-covariance override. If omitted, the stationary state-space covariance is used. Supply a full `(n_var, n_var)` matrix in compiled variable order; for `unscented` mode its state block is embedded automatically.

???+ info "Filter Initial Conditions"
    In linear and extended likelihoods, `P0` is the prior covariance for the first observation. In unscented likelihoods, it covariance before the first observation.

`joseph_cov=True` uses the Joseph covariance update for linear and extended likelihoods. `False` applies a simplw update, which is faster but less numerically robust.

## Likelihood / Posterior Evaluation

```python
Estimator.loglik(params: Mapping[str, float] | None = None) -> float
Estimator.logprior(params: Mapping[str, float] | None = None, jacobian: bool = False) -> float
Estimator.logpost(params: Mapping[str, float] | None = None, jacobian: bool = False) -> float
```

???+ note "Optimization Space"
    `params` are passed in constrained space (i.e. the values that estimation outputs and the model's calibration uses).
    If any `prior`s imply a transformation to unconstrained space, the said transformation is executed before the objective is evaluated.

???+ note "Log-Jacobian"
    `jacobian` controls whether the prior's transformation is treated as a coordinate change (`False`) or as a change of random variables (`True`).
    See the [map()](#map) note on the jacobian parameter for more details.

## MLE

```python
Estimator.mle(
    *,
    theta0: Mapping[str, float] | None = None, # (1)!
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
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
    cov: bool = True,  # (5)!
    cov_fd_step_scale: float = 1.0,
    cov_fd_absolute_floor: float = 0.1,
) -> MLERestult
```

1. If `None`, uses transformed calibration defaults.
2. Native optimizer. Only the two listed are supported; see the note below.
3. L-BFGS-B options (`m`, `maxiter`, `maxfun`, `maxls`, `factr`, `pgtol`, `fd_step`); ignored by Nelder-Mead.
4. Nelder-Mead options (`xatol`, `fatol`); ignored by L-BFGS-B.
5. Compute the covariance of the estimated parameters at the optimum by finite-difference Hessian of the log-likelihood.
If `False`, the returned result will not include a covariance.
Enabling this will also compute the standard errors of the estimated parameters, which are returned in the result object.


## MAP

```python
Estimator.map(
    *,
    theta0: Mapping[str, float] | None = None, # (1)!
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B", # (2)!
    jacobian: bool = False, # (3)!
    m: int = 10, # (4)!
    maxiter: int = 15000,
    maxfun: int = 15000,
    maxls: int = 20,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    fd_step: float = 0.0,
    xatol: float = 1e-4, # (5)!
    fatol: float = 1e-4,
    cov: bool = True, # (6)!
    cov_fd_step_scale: float = 1.0,
    cov_fd_absolute_floor: float = 0.1,
) -> MAPResult
```

1. Requires non-`None` priors at estimator construction.
2. Native optimizer. Only the two listed are supported; see the note below.
3. Includes the jacobian term arising from the prior transformations of random variables.
4. L-BFGS-B options (`m`, `maxiter`, `maxfun`, `maxls`, `factr`, `pgtol`, `fd_step`); ignored by Nelder-Mead.
5. Nelder-Mead options (`xatol`, `fatol`); ignored by L-BFGS-B.
6. Compute the covariance of the estimated parameters at the optimum by finite-difference Hessian of the log-likelihood.
If `False`, the returned result will not include a covariance.
Enabling this will also compute the standard errors of the estimated parameters, which are returned in the result object.

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
    theta0: Mapping[str, float] | None = None, # (4)!
    random_state: int | None = None,
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

### MLEResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x | `#!python np.ndarray` | Optimized unconstrained vector |
| theta | `#!python dict[str, float]` | The estimated parameters at the optimum, in constrained space, keyed by `estimated_params`. Parameters that were not estimated are not included: they stay at the model's calibration. |
| success | `#!python bool` | Optimizer convergence flag |
| message | `#!python str` | Optimizer status message |
| fun | `#!python float` | Objective value at optimum |
| loglik | `#!python float` | Log-likelihood at optimum |
| vcov | `#!python np.ndarray | None` | Covariance of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| se | `#!python dict[str, float] | None` | Standard errors of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| nfev | `#!python int` | Objective evaluations |
| nit | `#!python int | None` | Iterations |
| optimizer_config | `#!python dict[str, Any]` | Optimizer method, bounds, and options the run was called with |
| cov_status | `#!python int` | `#!python 0` when the covariance was produced, otherwise why it was not |

__Methods:__

```python
MLEResult.apply_to(
    config: ModelConfig,
) -> ModelConfig
```

Copy of `config` with the estimated parameters written in at their optimum values.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| config | Config to copy. Must declare every name in `theta`. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python ModelConfig` | Copy of `config` carrying `theta`. Parameters that were not estimated keep their calibration values. |

### MAPResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x | `#!python np.ndarray` | Optimized unconstrained vector |
| theta | `#!python dict[str, float]` | The estimated parameters at the optimum, in constrained space, keyed by `estimated_params`. Parameters that were not estimated are not included: they stay at the model's calibration. |
| success | `#!python bool` | Optimizer convergence flag |
| message | `#!python str` | Optimizer status message |
| fun | `#!python float` | Objective value at optimum |
| logprior | `#!python float` | Log-prior at optimum |
| logpost | `#!python float` | Log-posterior at optimum |
| vcov | `#!python np.ndarray | None` | Covariance of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| se | `#!python dict[str, float] | None` | Standard errors of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| nfev | `#!python int` | Objective evaluations |
| nit | `#!python int | None` | Iterations |
| optimizer_config | `#!python dict[str, Any]` | Optimizer method, bounds, and options the run was called with |
| cov_status | `#!python int` | `#!python 0` when the covariance was produced, otherwise why it was not |

__Methods:__

```python
MAPResult.apply_to(
    config: ModelConfig,
) -> ModelConfig
```

Identical to [`MLEResult.apply_to`](#mleresult).

### MCMCResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| param_names | `#!python tuple[str, ...]` | Parameter order for samples |
| samples | `#!python np.ndarray` | Retained posterior samples |
| logpost_trace | `#!python np.ndarray` | Posterior trace for retained samples, as a density over the parameters |
| logjac_trace | `#!python np.ndarray` | Log-jacobian of the prior transforms at each retained draw; add it to `logpost_trace` for the unconstrained density the sampler walked |
| accepted_burn_in | `#!python int` | Proposals accepted during burn-in |
| accepted_in_sample | `#!python int` | Proposals accepted after burn-in |
| burn_in_seconds | `#!python float` | Seconds spent on burn-in |
| in_sample_seconds | `#!python float` | Seconds spent on the retained phase |
| n_draws | `#!python int` | Retained draw count |
| burn_in | `#!python int` | Burn-in iterations |
| thin | `#!python int` | Thinning interval |
| sampler_config | `#!python dict[str, Any]` | The sampler configuration used to produce the result |

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| accept_rate | `#!python float` | Accepted proposals over every step the sampler took, burn-in included |
| accept_rate_burn_in | `#!python float` | Acceptance rate over burn-in alone, or NaN when `burn_in` is `#!python 0` |
| accept_rate_in_sample | `#!python float` | Acceptance rate over the retained phase alone |
| ess | `#!python dict[str, float]` | Effective sample size of each parameter's posterior mean, keyed by parameter name |
| ess_per_second | `#!python dict[str, float]` | `ess` divided by `in_sample_seconds` |
| mcse | `#!python dict[str, float]` | Monte Carlo standard error of each parameter's posterior mean, $\mathrm{sd}/\sqrt{\mathrm{ESS}}$ |

???+ note "Effective Sample Size"
    ESS is computed with the integrated autocorrelation time under Geyer's initial monotone sequence. Reported values are capped at `n_draws`. A parameter whose chain never moved, or whose autocorrelation has not decayed within the retained draws, reports NaN rather than raising, and that NaN carries through `ess_per_second`, `mcse`, and `mcse_quantile`.

__Methods:__

```python
MCMCResult.summary(
    alpha: float = 0.05, # (1)!
    digits: int | None = 3,
) -> pd.DataFrame
```

1. Significance level for the highest-posterior-density (HPD) intervals.

Return a summary table reporting posterior means, standard deviations, ESS, ESS/s, MCSE for the mean, and HPD intervals for each parameter.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| alpha | Significance level used to determine the empirical HPD coverage. |
| digits | Number of decimal places to round the summary table to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `pd.DataFrame` | Summary table with one row per parameter and columns for posterior mean, standard deviation, ESS, ESS/s, MCSE for the mean, and HPD intervals. |

```python
MCMCResult.hpd_intervals(
    alpha: float = 0.05, # (1)!
    digits: int | None = 3,
) -> dict[str, tuple[float, float]]
```

1. Significance level. Must satisfy `#!python 0 <= alpha < 1`; each interval covers approximately `#!python 1 - alpha` of the retained marginal draws.

Compute marginal highest-posterior-density (HPD) intervals for each parameter column.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| alpha | Significance level used to determine the empirical HPD coverage. |
| digits | Number of decimal places to round the HPD intervals to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, tuple[float, float]]` | Mapping from parameter name to the shortest empirical marginal interval containing approximately `#!python 1 - alpha` of the retained posterior draws. |

&nbsp;

```python
MCMCResult.joint_hpd_set(
    alpha: float = 0.05, # (1)!
    digits: int | None = 3,
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
| digits | Number of decimal places to round the retained draws to. `None` does not apply rounding. |
__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python tuple[np.ndarray, np.ndarray, float, np.ndarray]` | Tuple `#!python (samples, logpost, threshold, indices)` where `samples` are the retained parameter vectors in the joint HPD set, `logpost` are their posterior values, `threshold` is the cutoff log-posterior, and `indices` are positions of the retained draws in the original stored chain. |

&nbsp;

```python
MCMCResult.mcse_quantile(
    q: float, # (1)!
    digits: int | None = 3,
) -> dict[str, float]
```

1. Quantile whose error is reported. Must satisfy `#!python 0 < q < 1`.

Compute the Monte Carlo standard error of the `q` quantile for each parameter column.

???+ note "Indicator-Chain Estimator"
    The error is taken on the probability scale from the effective sample size of the indicator chain $\mathbf{1}\{x_t \le \hat{x}_q\}$, then carried back to the parameter scale through the empirical quantile function, which needs no density estimate at $\hat{x}_q$. Reading discrete order statistics leaves the result mildly conservative.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| q | Quantile whose Monte Carlo standard error is reported. |
| digits | Number of decimal places to round the MCSE to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, float]` | Mapping from parameter name to the Monte Carlo standard error of that parameter's `q` quantile, or NaN where the indicator chain has no usable effective sample size. |

&nbsp;

```python
MCMCResult.posterior_point(
    point: Literal["mean", "map", "last"] = "mean", # (1)!
    digits: int | None = 3,
) -> dict[str, float]
```

1. `#!python "mean"` averages the retained draws, `#!python "map"` takes the draw maximizing `logpost_trace`, and `#!python "last"` takes the final retained draw.

Reduce the retained posterior draws to a single parameter vector.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| point | Which summary of the retained draws to return. |
| digits | Number of decimal places to round the returned parameter vector to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, float]` | Parameter values at the selected point, keyed by parameter name. |

&nbsp;

```python
MCMCResult.apply_to(
    config: ModelConfig,
    point: Literal["mean", "map", "last"] = "mean",
) -> ModelConfig
```

Copy of `config` with a posterior point estimate written in.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| config | Config to copy. Must declare every name in `param_names`. |
| point | Which posterior summary to write in, as taken by `posterior_point`. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python ModelConfig` | Copy of `config` carrying the selected point. Parameters that were not estimated keep their calibration values. |

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
