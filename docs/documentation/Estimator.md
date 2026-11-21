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
    estimated_params: Sequence[str] | Mapping[str, Prior] | None = None, # (1)!
    y: np.ndarray | pd.DataFrame, # (2)!
    compiled: CompiledModel, # (3)!
    observables: list[str] | None = None,
    filter: Filter = LinearKalman(), # (4)!
    ss_seed: np.ndarray | dict[str, float] | None = None,
    R: np.ndarray | None = None, # (5)!
    P0: np.ndarray | None = None, # (6)!
)
```
1. Parameter names to estimate, or a mapping from parameter name to `Prior` object. If `None`, all parameters are estimated with uniform priors. `map()` and `mcmc()` require priors to be specified and `mle()` ignores them.
2. Measurement data for Kalman likelihood.
3. Compiled model from `compile_model(...)`.
4. Filter algorithm for the likelihood can be any of the [`Filter`](./filters/index.md) subclasses.
5. Optional constant observation-covariance override. If omitted, `R` is built from the calibration's `measurement_noise` block.
6. Optional initial state-covariance override. If omitted, the stationary state-space covariance is used. Supply a full `(n_var, n_var)` matrix in compiled variable order; for `unscented` mode its state block is embedded automatically.

???+ info "Filter Initial Conditions"
    In linear and extended likelihoods, `P0` is the prior covariance for the first observation. In unscented likelihoods, its the covariance before the first observation.

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
    `mle` and `map` run entirely in the native backend, which ships a curated set of optimizers with no scipy fallback. Only `"L-BFGS-B"` (default; quasi-Newton with a finite-difference gradient) and `"Nelder-Mead"` (gradient-free) are supported; any other `method` raises. Each optimizer's tuning parameters are passed as explicit keyword arguments: the L-BFGS-B group (`m`, `maxiter`, `maxfun`, `maxls`, `factr`, `pgtol`, `fd_step`) and the Nelder-Mead group (`xatol`, `fatol`).

## MCMC

```python
Estimator.mcmc(
    *,
    n_draws: int, # (1)!
    n_chains: int = 1,  # (2)!
    burn_in: int = 1000, # (3)!
    thin: int = 1, # (4)!
    theta0: Mapping[str, float] | None = None, # (5)!
    random_state: int | None = None,
    adapt: bool = True,
    adapt_start: int | None = None,  # (6)!
    adapt_end: int | Literal["burn_in"] | None = "burn_in",  # (7)!
    adapt_interval: int = 1, # (8)!
    adapt_epsilon: float = 1e-8,  # (9)!
    init_scale: float = 1, # (10)!
    proposal_scale: float = 1,  # (11)!
    compute_map: bool = True,
    map_options: dict[str, Any] | None = None,  # (12)!
    proposal_cov: np.ndarray | None = None,  # (13)!
    cov_fd_step_scale: float = 1.0,
    cov_fd_absolute_floor: float = 0.1,
    n_jobs: int | None = None,
) -> MCMCResult
```

1. Number of retained posterior draws.
2. Number of independent chains to run.
3. Number of initial iterations discarded.
4. Retain every `thin`-th iteration after burn-in.
5. `None` uses calibration defaults. When `compute_map=True`, the MAP estimate is computed first and used as the starting point for the chain.
The proposal covariance will use the MAP point to compute the Hessian if `compute_map=True`.
6. `None` defaults to half-way between `0` and `adapt_end` and cannot exceed `1000`.
Integer iteration indices must be greater than `0`, the adaptation algorithm requires at least one sample to be collected.
7. `None` continues adaptation until the end of the chain, `"burn_in"` stops adaptation
after the burn-in phase, and integer values are the iteration at which adaptation stops.
8. Number of iterations between adaptation steps. Must be at least 1.
9. Small constant added to the diagonal of the proposal covariance during adaptation to ensure positive definiteness.
10. Scaling factor for the proposal **standard deviation** used to sample initial theta points for multiple chains. The first chain starts at `theta0` or the MAP estimate; subsequent chains are a Gaussian random sample around that point with `init_scale * proposal_std`.
11. Scaling factor for proposal **standard deviation**: `2.38/sqrt(d) * proposal_scale`. The default of 1 corresponds to the standard "optimal" point for a Gaussian random-walk MH.
12. Passed to `map(...)` if `compute_map=True`. If `None`, defaults are used. Refer to [Estimator.map](#map) for the available options.
13. Proposal covariance for the sampler. It's mutually exclusive with `compute_map=True` since the MAP estimate is used as the differentiation point for the proposal covariance and the `theta0` a chain starts at.

???+ note "Thinning Semantics"
    Thinning is applied after burn-in using `(t - burn_in) % thin == 0`.

???+ note "MCMC Sample Space"
    `MCMCResult.samples` are returned in constrained parameter space (parameter names), not raw unconstrained `theta`.

## Result Objects

### MLEResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x | `np.ndarray` | Optimized unconstrained vector |
| theta | `dict[str, float]` | The estimated parameters at the optimum, in constrained space, keyed by `estimated_params`. Parameters that were not estimated are not included: they stay at the model's calibration. |
| success | `bool` | Optimizer convergence flag |
| message | `str` | Optimizer status message |
| fun | `float` | Objective value at optimum |
| loglik | `float` | Log-likelihood at optimum |
| vcov | `np.ndarray | None` | Covariance of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| se | `dict[str, float] | None` | Standard errors of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| nfev | `int` | Objective evaluations |
| nit | `int | None` | Iterations |
| optimizer_config | `dict[str, Any]` | Optimizer method, bounds, and options the run was called with |
| cov_status | `int` | `0` when the covariance was produced, otherwise why it was not |

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
| `ModelConfig` | Copy of `config` carrying `theta`. Parameters that were not estimated keep their calibration values. |

### MAPResult

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x | `np.ndarray` | Optimized unconstrained vector |
| theta | `dict[str, float]` | The estimated parameters at the optimum, in constrained space, keyed by `estimated_params`. Parameters that were not estimated are not included: they stay at the model's calibration. |
| success | `bool` | Optimizer convergence flag |
| message | `str` | Optimizer status message |
| fun | `float` | Objective value at optimum |
| logprior | `float` | Log-prior at optimum |
| logpost | `float` | Log-posterior at optimum |
| vcov | `np.ndarray | None` | Covariance of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| se | `dict[str, float] | None` | Standard errors of the estimated parameters at the optimum, or `None` if `cov=False` was passed to the estimator. |
| nfev | `int` | Objective evaluations |
| nit | `int | None` | Iterations |
| optimizer_config | `dict[str, Any]` | Optimizer method, bounds, and options the run was called with |
| cov_status | `int` | `0` when the covariance was produced, otherwise why it was not |

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
| param_names | `tuple[str, ...]` | Parameter order for samples. |
| samples | `np.ndarray` | Retained posterior samples as `(n_chains, n_draws, n_params)`. |
| logpost_trace | `np.ndarray` | Posterior trace for retained samples, as a density over the parameters. Shaped `(n_chains, n_draws)`. |
| logjac_trace | `np.ndarray` | Log-jacobian of the prior transforms at each retained draw; add it to `logpost_trace` for the unconstrained density the sampler walked. Shaped `(n_chains, n_draws)`. |
| accepted_burn_in | `np.ndarray` | Proposals accepted during burn-in per chain. |
| accepted_in_sample | `np.ndarray` | Proposals accepted after burn-in per chain. |
| wall_seconds | `float` | Wall-clock of the sampler. Excludes initialization. |
| init_seconds | `float` | Time spent on initialization, including the MAP and inverse Hessian computations if applicable. |
| burn_in_seconds | `np.ndarray` | Seconds spent on burn-in per chain. |
| in_sample_seconds | `np.ndarray` | Seconds spent on the retained phase per chain. |
| n_chains | `int` | Number of chains. |
| n_draws | `int` | Retained draw count per chain. |
| burn_in | `int` | Burn-in iterations per chain. |
| thin | `int` | Thinning interval. |
| sampler_config | `dict[str, Any]` | The sampler configuration used to produce the result. |

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| accept_rate | `list[float]` | Accepted proposals per chain over every step the sampler took, burn-in included. |
| accept_rate_burn_in | `list[float]` | Acceptance rate per chain over burn-in alone, or NaN when `burn_in` is `0`. |
| accept_rate_in_sample | `list[float]` | Acceptance rate per chain over the retained phase alone. |
| split_rhat | `dict[str, float]` | Split R-hat convergence diagnostic for each parameter, keyed by parameter name. |
| rhat | `dict[str, float]` | Maximum of bulk and folded rank-normalized R-hat convergence diagnostic for each parameter, keyed by parameter name. NaN if either is NaN. |
| ess | `dict[str, float]` | Effective sample size of each parameter's posterior mean, keyed by parameter name. |
| ess_per_second | `dict[str, float]` | `ess` divided by `wall_seconds`. |
| ess_tail | `dict[str, float]` | Tail effective sample size, the minimum of `ess_quantile(0.05)` and `ess_quantile(0.95)`, keyed by parameter name. NaN if either is NaN. |
| ess_bulk | `dict[str, float]` | Bulk effective sample size of each parameter, keyed by parameter name. |
| mcse | `dict[str, float]` | Monte Carlo standard error of each parameter's posterior mean, $\mathrm{sd}/\sqrt{\mathrm{ESS}}$. |

???+ note "Effective Sample Size"
    ESS is computed with the integrated autocorrelation time under Geyer's initial monotone sequence. Reported values are capped at `S * log10(S)` where `S` is `n_chains * n_draws`. A parameter whose chain never moved, or whose autocorrelation has not decayed within the retained draws, reports NaN rather than raising, and that NaN carries through `ess_per_second`, `mcse`, and `mcse_quantile`.

__Methods:__

```python
MCMCResult.summary(
    alpha: float = 0.05, # (1)!
    digits: int | None = None,
) -> pd.DataFrame
```

1. Significance level for the highest-posterior-density (HPD) intervals.

Return a summary table reporting:

- posterior means and standard deviations
- R-hat (maximum of bulk and folded rank-normalized R-hat)
- bulk/tail ESS and ESS/s
- MCSE for the mean
- HPD intervals for each parameter.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| alpha | Significance level used to determine the empirical HPD coverage. |
| digits | Number of decimal places to round the summary table to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `pd.DataFrame` | Summary table with one row per parameter and columns per sample diagnostic/descriptor. |

```python
MCMCResult.hpd_intervals(
    alpha: float = 0.05, # (1)!
    digits: int | None = None,
) -> dict[str, tuple[float, float]]
```

1. Significance level. Must satisfy `0 <= alpha < 1`; each interval covers approximately `1 - alpha` of the retained marginal draws.

Compute marginal highest-posterior-density (HPD) intervals for each parameter column.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| alpha | Significance level used to determine the empirical HPD coverage. |
| digits | Number of decimal places to round the HPD intervals to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `dict[str, tuple[float, float]]` | Shortest empirical marginal interval containing approximately `1 - alpha` of the retained posterior draws per parameter. |

&nbsp;

```python
MCMCResult.ess_quantile(
    q: float, # (1)!
    digits: int | None = None,
) -> dict[str, float]
```

1. Quantile whose effective sample size is reported. Must satisfy `0 < q < 1`.

Compute the effective sample size of the `q` quantile for each parameter column, taken as the ESS of the indicator chain $\mathbf{1}\{x_t \le \hat{x}_q\}$.
Errors are reported as NaN where a split half of the indicator chain is constant.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| q | Quantile whose effective sample size is reported. |
| digits | Number of decimal places to round the ESS to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `dict[str, float]` | Effective sample size of each parameter's `q` quantile, keyed by parameter name. |

&nbsp;

```python
MCMCResult.mcse_quantile(
    q: float, # (1)!
    digits: int | None = None,
) -> dict[str, float]
```

1. Quantile whose error is reported. Must satisfy `0 < q < 1`.

Compute the Monte Carlo standard error of the `q` quantile for each parameter column.
Errors arising at ESS computations report as NaN per parameter.

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
| `dict[str, float]` | Monte Carlo standard error of each parameter's `q` quantile. |

&nbsp;

```python
MCMCResult.posterior_point(
    point: Literal["mean", "map"] = "mean", # (1)!
    digits: int | None = None,
) -> dict[str, float]
```

1. `"mean"` averages the retained draws, and `"map"` takes the draw maximizing `logpost_trace`.

Reduce the retained posterior draws to a single parameter vector.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| point | Which summary of the retained draws to return. |
| digits | Number of decimal places to round the returned parameter vector to. `None` does not apply rounding. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `dict[str, float]` | Parameter values at the selected point, keyed by parameter name. |

&nbsp;

```python
MCMCResult.apply_to(
    config: ModelConfig,
    point: Literal["mean", "map"] = "mean",
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
| `ModelConfig` | Copy of `config` carrying the selected point. Parameters that were not estimated keep their calibration values. |

&nbsp;

```python
MCMCResult.posterior_kde_plot(grid_points: int = 1000) -> None
```

Plot marginal posterior kernel-density estimates for each retained parameter column. Warns and skips parameters with zero or undefined variance.

This is a quick visual diagnostic for posterior shape. It is useful for checking skewness, heavy tails, and obvious multimodality in the retained draws. A separate subplot is produced for each parameter and displayed immediately with `matplotlib.pyplot.show()`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| grid_point | Number of even-spaced points spanning each parameter's range, at least 2. The resulting grid is equivalent to a `np.linspace` call. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `None` | Displays a Matplotlib figure of marginal KDE curves and returns nothing. |

&nbsp;

```python
MCMCResult.logpost_trace_plot() -> None
```

Plot the retained log-posterior sequence across MCMC iterations.

This diagnostic helps identify abrupt changes in posterior fit, long stretches of poor exploration, or chains that remain unstable even after burn-in and thinning have been applied. The plot is generated from `MCMCResult.logpost_trace`, which stores one log-posterior value per retained draw, and is displayed immediately with `matplotlib.pyplot.show()`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| None | This method uses the retained log-posterior trace already stored on the result object. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `None` | Displays a Matplotlib figure of the retained log-posterior trace and returns nothing. |
