---
tags:
    - doc
---
# SolvedModel

```python
FirstOrderSolvedModel(SolvedModel[FirstOrderSolution])
SecondOrderSolvedModel(SolvedModel[SecondOrderSolution])
PiecewiseSolvedModel(SolvedModel[PiecewiseSolution])
```

`SolvedModel` instances contain a compiled model, its typed policy solution, and the methods that operate on that policy.

???+ info "Type Parameterization"
    `SolvedModel` is a generic class parameterized by the solution type.
    Methods implemented for specific solution types are documented as `SolvedModel[SolutionType].method`. Methods that are solution-type agnostic are documented as `SolvedModel.method`.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| compiled | `#!python CompiledModel` | The compiled model object that resulted in the current solution. |
| policy | `#!python FirstOrderSolution | SecondOrderSolution | PiecewiseSolution` | Solver output containing the policy data for this model. |

&nbsp;

__Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| config | `#!python ModelConfig` | Parsed model configuration object. |
| kalman_config | `#!python KalmanConfig | None` | Parsed Kalman Filter configuration object. |


__Methods:__

```python
SolvedModel.sim(
    T: int,
    shocks: Mapping[str | Sequence[str], Shock | np.ndarray] | Sequence[Shock | ShockPath] | None = None, # (1)!
    shock_scale: float = 1.0, # (2)!
    x0: dict[str, float] | list[float] | np.ndarray | None = None,
    observables: bool = False
) -> SimResult
```

 1. Two shapes are accepted. The preferred one is a sequence of entries that name their own targets:
   - A [`#!python Shock`](./Shock.md) bound by `#!python .joint(...)` or `#!python .independent(...)`. `sim` materializes it into a `T` period draw at call time.
   - A [`#!python ShockPath`](./Shock.md) carrying a materialized array, `(T,)` or `(T, 1)` for one shock and `(T, k)` for a group.

   A mapping is also accepted, keyed from the outside: an innovation symbol, or a tuple of them for a joint draw, against an __unbound__ `#!python Shock` or a bare array. See the deprecation note below.

   When omitted (or `#!python None`), all shocks are zero.
2. Shocks are drawn from the specified distribution and all elements in the arrays are scaled by this parameter.

 Returns the simulated path defined by the given inputs.

???+ info "Univariate Shock Syntax"
    A univariate entry names one innovation symbol. For a model with a variable `x` and a shock symbol `e_x`:

    ```python
    sol.sim(T=10, shocks=[Shock(dist="norm").joint("e_x")])
    sol.sim(T=10, shocks=[ShockPath(path, "e_x")])            # path shaped (10,) or (10, 1)
    ```

    `#!python .independent("e_x", "e_y")` returns one such entry per symbol, each drawn from its own calibrated standard deviation. Splice the list into the spec.

???+ info "Correlated Shock Syntax"
    To draw innovations with nonzero covariance, one entry names the whole group:

    ```python
    sol.sim(T=10, shocks=[Shock(dist="norm").joint("e_x", "e_y")])
    sol.sim(T=10, shocks=[ShockPath(path, "e_x", "e_y")])      # path shaped (10, 2)
    ```

    Details regarding grouped entries:

    - A group may name any number of symbols. Each shock may appear in at most one entry across the whole spec.
    - For a drawn entry, the ordering of symbols does __not__ affect simulation results.
    - For a `ShockPath`, the ordering __is__ meaningful: it says which column of `path` drives which shock. `#!python ShockPath(arr, "e_y", "e_x")` means `#!python arr[:, 0]` is `e_y`, whichever order the model declares the two in.
    - Shock realizations are always aligned with the innovation ordering defined at model configuration or compilation (`B` matrix order).

??? info "Multivariate Shock Canonicalization and Reproducibility"
    When a grouped `Shock` entry is drawn, variables are __internally reordered to a canonical model-defined order before sampling__.
    This ensures that simulations are __reproducible under a fixed random seed__, regardless of the order in which variables are specified
    in a grouped entry (e.g. `#!python .joint("e_g", "e_z")` vs `#!python .joint("e_z", "e_g")`).

    This behavior is required because multivariate sampling methods (e.g. Cholesky-based Gaussian draws) are __order-dependent at the
    realization level__, even when the underlying covariance structure is permutation-invariant.

    Concretely:

    - Correlation and covariance matrices are constructed **after** canonicalizing variable order.
    - Sampling is performed in this canonical order.
    - Shock realizations are then mapped to the correct innovation indices used by the model.

    As a result, variable ordering in a grouped entry does **not** affect either the statistical properties
    *or the realized sample paths* of the simulation when the random seed is fixed.

???+ warning "Mapping Specs Are Deprecated"
    The mapping shape predates entries that name themselves. It is kept for compatibility and gives identical results; prefer the sequence.
    `Shock`s must be unbound in the deprecated spec and `ShockPath`s are not accepted.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of periods to simulate. The result has `T` rows. |
| shocks | A sequence of bound `Shock` or `ShockPath` entries, or a mapping keyed by innovation symbol or tuple of symbols. |
| shock_scale | Scaling factor for the shocks. |
| x0 | Initial level state at `t - 1`. A dense sequence covers all compiled variables in declaration order, including generated lags. A mapping may specify only selected variables; omitted variables start at their steady state. |
| observables | Include observable paths in the result if `#!python True`. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python SimResult` | Simulated level paths. `states` maps variable names to columns of `X`; `observables` and `y` are available when requested. |

&nbsp;

## `SolvedModel[PiecewiseSolution].sim_reference`

```python
SolvedModel[PiecewiseSolution].sim_reference(
    T: int,
    shocks: Mapping[str | Sequence[str], Shock | np.ndarray] | Sequence[Shock | ShockPath] | None = None,
    shock_scale: float = 1.0,
    x0: dict[str, float] | list[float] | np.ndarray | None = None,
    observables: bool = False,
) -> SimResult
```

Simulate the unconstrained first-order reference policy of a piecewise model. This method accepts the same inputs as `SolvedModel.sim(...)`, but ignores every constraint and does not run the OccBin regime search.

The result contains the reference policy's level path. It has no `regimes` or `diagnostics`, because it is an ordinary first-order simulation.

&nbsp;

```python
SolvedModel.irf(
    shocks: list[str],
    T: int,
    scale: float = 1.0,
    observables: bool = False
    ) -> dict[str, np.ndarray[float]]
```

Returns the IRF paths with shocks to specified variable(s).

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| shocks | List of variables to receive shocks.|
| T | Time period of the IRF. |
| scale | Shock scaling factor. |
| observables | Include observables in the output if `#!python True`.

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, np.ndarray[float]]` | The paths simulated for the IRF. Mirrors the return of `#!python SolvedModel.sim` with a specific shock configuration. |

&nbsp;

```python
SolvedModel.transition_plot(
    T: int,
    shocks: list[str],
    scale: float = 1.0,
    observables: bool = False
    ) -> None
```
Display the plot of transition paths generated by the specified shocks.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Time index to simulate. |
| shocks | List of variables to shock.|
| scale | Shock scaling factor. |
| observables | Include observables in the plot if `#!python True`. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python None` | Displays a plot of the paths created by a given config. |

&nbsp;

```python
SolvedModel.kalman(
    y: ndarray | DataFrame,
    filter_mode: Literal['linear', 'extended', 'unscented'] = 'linear',
    *,
    observables: list[str] | None = None, # (1)!
    x0: dict[str, float | float64] | list[float | float64] | NDF | None = None, # (2)!
    jitter: float | None = None, # (3)!
    symmetrize: bool = False,
    joseph_cov: bool = False, 
    return_shocks: bool = False,
    P0: ndarray | None = None,
    R: ndarray | None = None,
) -> FilterResult | UnscentedFilterResult
```

1. `None`: Use all compiled observables in model order.
2. `None`: Use a zero vector.
3. `None`: Use `0.0`.

Run a Kalman Filter application on the observables specified.

???+ info "`y` Array Alignment"
    When a DataFrame is used as `y`, column names will be used to align and order observables' names and position. However, for `ndarray` inputs, the method assumes names in `observables` and columns of `y` are position-aligned.

???+ info "State Units and Timing"
    This public path returns state histories in levels. 

    For linear and extended filters, `x0` and `P0` are the prior mean and covariance of the first observed state. For the unscented filter, they describe the state and covariance before the first observation.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| y | observations to filter. |
| filter_mode | `"linear"` for affine measurements, `"extended"` (EKF) for nonlinear measurements, or `"unscented"` (UKF), which runs against the model's second-order solution. `"unscented"` does not support `return_shocks`. Returns an `UnscentedFilterResult` instead of a `FilterResult`. |
| observables | Name of corresponding model measurements. |
| x0 | Initial state vector in levels. It is the prior for the first observation in linear and extended modes, and the state before the first observation in unscented mode. |
| jitter | Jitter term added to matrices when Cholesky fails. |
| symmetrize | Symmetrize covariances at each filter pass if `True`. |
| joseph_cov | Use Joseph form for covariance update if `True`. `filter_mode == "unscented"` has it's own update mechanism and will raise when this parameter is `True`. |
| return_shocks | Include the estimated shocks in the return object if `True`. |
| P0 | Initial state covariance override. `None` uses the stationary state-space covariance. Supply a full `(n_var, n_var)` matrix in compiled variable order; for unscented mode its state block is embedded automatically. |
| R | Constant measurement-error covariance override. If omitted, `R` is taken from the `KalmanConfig` (a fixed calibrated matrix, or rebuilt from named `R` parameters). |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python FilterResult` | `dataclass` containing information on filter state, measurements, and diagnostics. |

&nbsp;

```python
SolvedModel.to_dict() -> dict[str, Any]
```
Dictionary representation of the class instance.

__Inputs:__

`#!python None`

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, Any]` | Dictionary representation of the `#!python SolvedModel` object. |

&nbsp;

## `SolvedModel.save_sdsge`

```python
SolvedModel.save_sdsge(
    path: str | Path,
    *,
    yaml_text: str | None = None, # (1)!
    role: str = "reference",
    compile_kwargs: Mapping[str, Any] | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
) -> Path
```

1. Override the YAML embedded in the bundle. Defaults to `compiled.config.source_yaml` (populated automatically by `#!python ModelParser`).

Write a model-only `.sdsge` bundle around this `#!python SolvedModel`. For bundles that also carry estimation, Monte Carlo, or simulation members, use [`SolvedModel.to_bundle_builder`](#solvedmodelto_bundle_builder) and chain the additions before calling `.write()`.

???+ warning "Source YAML required"
    Raises `#!python ValueError` when `compiled.config.source_yaml` is `#!python None` and no `yaml_text=` override is supplied. Loading via `#!python ModelParser(path)` or `#!python ModelParser.from_string(text)` populates `source_yaml` automatically; programmatic `#!python ModelConfig` construction does not.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| path | Output `.sdsge` path. |
| yaml_text | Explicit YAML override. |
| role | `"reference"` or `"dgp"`. |
| compile_kwargs | Kwargs the loader will use to rebuild the `#!python SolvedModel`. |
| solve_kwargs | Kwargs the loader will use at the solve step. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python Path` | The path the bundle was written to. |

&nbsp;

## `SolvedModel.to_bundle_builder`

```python
SolvedModel.to_bundle_builder(
    *,
    yaml_text: str | None = None,
    role: str = "reference",
    compile_kwargs: Mapping[str, Any] | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
    created_by: str | None = None, # (1)!
) -> BundleBuilder
```

1. Defaults to `#!python "SymbolicDSGE <version>"` when omitted.

Return a [`#!python BundleBuilder`](./bundle/BundleBuilder.md) pre-seeded with this model's YAML. Chain estimation, Monte Carlo, or simulation members and then call `.write()` to materialize the archive.

```python
sol.to_bundle_builder() \
    .add_estimation(spec) \
    .add_mc(pipeline) \
    .write("experiment-1.sdsge")
```

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| yaml_text | Explicit YAML override; defaults to `compiled.config.source_yaml`. |
| role | `"reference"` or `"dgp"`. |
| compile_kwargs | Kwargs the loader will use to rebuild the `#!python SolvedModel`. |
| solve_kwargs | Kwargs the loader will use at the solve step. |
| created_by | Manifest `created_by` string. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python BundleBuilder` | A fluent builder with the model member already attached. |

See the [`bundle` module documentation](./bundle/index.md) and the [Bundle Authoring Guide](../guides/bundle_authoring_guide.md) for the complete picture.
