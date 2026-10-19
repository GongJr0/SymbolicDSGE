---
tags:
    - doc
---
# Shock

```python
class Shock(
    dist: Literal["norm", "t", "uni", "exp", "gamma", "beta"] | DistributionTable, 
    seed: int | None = 0,
    **dist_kwargs,
)
```

`Shock` is a horizon independent shock specification. Pass it directly inside `SolvedModel.sim(..., shocks=[...])`; the simulation supplies the horizon and the calibration.

__Inputs:__

| __Name__    |                                                                                                     __Description__ |
|:------------|--------------------------------------------------------------------------------------------------------------------:|
| dist        |                                      Distribution family (`"norm"`, `"t"`, `"uni"`, `"exp"`, `"gamma"`, `"beta"`) or a pre-built distribution table. |
| seed        |                                                                        Random seed. Pass `None` for unseeded draws. |
| dist_kwargs | Keyword arguments passed to the distribution draw method. Do not pass `scale`; simulation supplies the model scale. |

???+ warning "Scale is model supplied"
    A `Shock` stores distribution shape and location parameters only. The standard deviation of a single shock, and the covariance of a grouped one, come from the `SolvedModel` calibration at simulation time.

???+ note "Supported kwargs for native distributions"
    The following keyword arguments are supported for the native distribution families:

    | __Distribution__ |     __Supported kwargs__    |
    |:-----------------|:----------------------------|
    | `"norm"`         | `loc | mean`                |
    | `"t"`            | `loc | mean, df`            |
    | `"uni"`          | `loc | mean`                |
    | `"exp"`          | `loc | mean`                |
    | `"gamma"`        | `loc | mean, a`             |
    | `"beta"`         | `loc | mean, a, b`          |
    
    `loc` and `mean` are interchangeable and multivariate distributions must specify it as a dense vector.
    A mean of zero is only assumed when `loc` is omitted as a keyword argument. 

???+ info "Distribution tables"
    Distribution tables are exported from `SymbolicDSGE.core` and can be constructed directly.
    However this requires a pre-built inverse CDF table. This argument type
    is intended for building a `Shock` from another's `Shock.dist` attribute.
    To create a `Shock` from a quantile function or distribution object, use `Shock.from_ppf(...)` instead.

&nbsp;

```python
Shock.from_ppf(
    dist: QuantileFn | QuantileDist, 
    n_knots: int = 1024, 
    discrete: bool = False, 
    seed: int | None = 0, 
    **dist_kwargs
) -> Shock
```

`from_ppf` constructs a distribution table from a quantile function or a distribution object implementing a `ppf` method.

__Inputs:__

| __Name__ |                                           __Description__ |
|:---------|----------------------------------------------------------:|
| dist     | A ppf function or a distribution object implementing one. |
| n_knots  | Number of knots to use for the distribution table. |
| discrete | Whether to treat the distribution as discrete. |
| seed     | Random seed. Pass `None` for unseeded draws. |
| dist_kwargs | Keyword arguments passed to the distribution draw method. Do not pass `scale`; simulation supplies the model scale. |

???+ note "`QuantileFn` and `QuantileDist`"
    A `QuantileFn` is a callable that accepts an array of quantiles and returns the corresponding values. It is `Callable[[ndarray], ArrayLike]` so functions that only take a single float are not suitable. (A scipy distribution's ppf satisfies this signature.) A `QuantileFn` input ignores all `dist_kwargs` except `loc`/`mean`.
    For scipy distributions, shape parameters should be passed when constructing the distribution object.

    A `QuantileDist` is any object that implements a `ppf` method. The library constructs a `QuantileFn` from the `ppf` as `lambda q: dist.ppf(q, **dist_kwargs)`. Since `kwargs` are unpacked into the `ppf` call, `QuantileDist` can be unfrozen scipy distributions, e.g. `scipy.stats.laplace` can be passed directly.

&nbsp;

```python
Shock.joint(*keys : str) -> Shock
```

Bind one or more shock names to the `Shock` instance for a joint draw. A non-zero correlation among the group is exercised when present.
`joint` returns a new `Shock` instance and the original instance the call was made from remains usable for binding a separate shock(s).

__Inputs:__

| __Name__ |                                          __Description__ |
|:---------|---------------------------------------------------------:|
| keys     | One or more shock names to bind to the `Shock` instance. |

&nbsp;

```python
Shock.independent(*keys : str, offset_seeds: bool = True) -> list[Shock]
```

Return separate shock instances with a shared distribution specification. Binds multiple shock names at the same time but disregards any correlation the model configuration may imply. Returns a list of __new__ `Shock` instances, keeping the caller's instance usable for binding a separate shock(s). `offset_seeds` increments the seed for each instance so specifications don't return identical draws.

__Inputs:__

| __Name__     |                                          __Description__ |
|:-------------|---------------------------------------------------------:|
| keys         | One or more shock names to bind to the `Shock` instance. |
| offset_seeds |         Whether to increment the seed for each instance. |

&nbsp;

```python
class ShockPath(path: ndarray, *keys: str)
```

A pre-materialized `ndarray` path bound to `*keys` specified at the constructor. `path` column count must match the number of `*keys` and keys should be specified in the order the appear in `path`.

## Shock specs

A simulation takes a sequence of entries that name their own targets. Two forms are accepted, and both leave the horizon to the simulation:

| __Form__              |                                                                         __Meaning__ |
|:----------------------|------------------------------------------------------------------------------------:|
| `#!python Shock(...)` | Draw this family, reseeded per Monte Carlo replication, using the calibrated scale. |
| `#!python ShockPath(...)`    |  Use  `ShockPath.path` verbatim, shaped `(T,)`/`(T, 1)` for one shock or `(T, k)` for a grouped key. |

```python
sol.sim(T=10, shocks=[Shock(dist="norm", seed=1).joint("e_z")])
sol.sim(T=10, shocks=[Shock(dist="norm", seed=1).joint("e_g", "e_z")])

# IRF replicated with a simulation using a pre-materialized path
path = np.zeros(10)
path[0] = sol.config.calibration.parameters["sig_g"]
sol.sim(T=10, shocks=[ShockPath(path, "e_g")])
```

???+ warning "Mapping Specs Are Deprecated"
    A simulation still accepts the older shape, a mapping keyed from the outside:

    ```python
    sol.sim(T=10, shocks={"e_g": Shock(dist="norm", seed=1)})            # key binds the Shock
    sol.sim(T=10, shocks={("e_g", "e_z"): Shock(dist="norm", seed=1)})   # tuple key binds a group
    sol.sim(T=10, shocks={"e_g": path})                                  # key names a bare array
    ```

    `Shock`s must be unbound in the deprecated spec and `ShockPath`s are not accepted.

