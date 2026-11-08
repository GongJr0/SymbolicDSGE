---
tags:
    - doc
---
# Filters

```python
from SymbolicDSGE import LinearKalman, ExtendedKalman, UnscentedKalman
```

Filters are a collection of algorithms used for state-path estimation in DSGE (and other state-space) models. They are used to recover the model's latent state variables conditional on the provided data and the specific method of filtration. Alongside their use as standalone estimators, filters are also integral to parameter estimation routines where they are used as the likelihood engine for a model's parameters.

In SymbolicDSGE, filters feed parameter estimation via [`Estimator`](../Estimator.md) and are available for use in [`Monte Carlo`](../monte_carlo/index.md) pipelines via [`filter_step`](../monte_carlo/operations/core/filter.md). Each `Filter` class also acts as a standalone filter working on `SolvedModel` instances:

```python
result = LinearKalman(joseph_cov=True).filter(solved, y, observables=["Infl", "Rate"])
```

## Available Filters

| __Class__ | __Required Solution__ | __Result__ |
|:----------|:---------------------:|-----------:|
| [`LinearKalman`](./LinearKalman.md) | First order or higher | [`FilterResult`](./FilterResult.md) |
| [`ExtendedKalman`](./ExtendedKalman.md) | First order or higher | [`FilterResult`](./FilterResult.md) |
| [`UnscentedKalman`](./UnscentedKalman.md) | Second order | [`UnscentedFilterResult`](./FilterResult.md#unscentedfilterresult) |

__Hierarchy and Available Options:__

```text
Filter
└── GaussianFilter          jitter, symmetrize
    ├── LinearKalman        + joseph_cov, return_shocks
    ├── ExtendedKalman      + joseph_cov, return_shocks
    └── UnscentedKalman     + alpha, beta, kappa
```

## Base Arguments

Every `Filter` implements the `filter` method below and can extend the arguments as needed for its own computations.
The arguments below are always available and are always consumed in the same way, regardless of the filter used.

```python
filter(
    model: SolvedModel,
    y: ndarray | DataFrame,
    observables: list[str] | tuple[str, ...] | None = None,
    R: ndarray | None = None,
    x0: Mapping[str, float] | None = None,
) -> FilterResult
```

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| model | The solved model to be filtered. |
| y | Observed data. An `ndarray` is read in the order `observables` names, a `DataFrame` is aligned by column label. |
| observables | Model observables the data corresponds to. Defaults to all of them in declaration order. |
| R | Measurement noise covariance. Shape `(n_obs, n_obs)`. Defaults to the model's `calibration.measurement_noise`. |
| x0 | Prior state mean, in levels, naming only the variables it sets. Defaults to the steady state. |

???+ note "Results Are in Levels"
    Most filter recursions run in deviations, but the outputs for all filters are reported in levels both for state estimates and for observables.
