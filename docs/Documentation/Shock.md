---
tags:
    - doc
---
# Shock

```python
class Shock(
    T: int,
    dist: Literal['norm', 't', 'uni'] | rv_generic | multi_rv_generic | None = None,
    multivar: bool = False,
    seed: int | None = 0, # (1)!
    dist_args: tuple = (),
    dist_kwargs: dict | None = None,
    shock_arr: ndarray | None = None,
    )
```

1. Notice the default behavior will produce seeded results. Pass `#!python None` explicitly to disable this behavior.

`#! Shock` provides the infrastructure necessary to produce simulation shocks. All distributions that are subclasses of `#!python SciPy`'s `#!python rv_generic` and `#!python multi_rv_generic` are supported by the interface. Moreover, Normal, Student-t, and Uniform shocks are supported natively inside the class object.

???+ warning "Multivariate Uniform Distribution Support"
    Multivariate uniforms are not identifiable even with known bounds and covariances. A specific support function must be supplied to determine an exact shape; with the exceptions of rectangular (no covariance) and ellipsoid (assumed shape) cases.

    Support functions, and derivation techniques for multivariate uniform distributions will not be implemented unless explicitly requested.

??? info "Custom Distributions"
    The underlying generators are written to accept any class implementing a `#!python .rvs` method (abstract method from `#!python SciPy`). Writing a subclass of either `#!python SciPy` abstraction and implementing said method allows the creation of random values in any desired way.

??? info "Generator Factories"
    Generator factories being utilized in this class are currently not part of the public API of `#!python SymbolicDSGE`. Robust and defensive alternatives of generators will be written specifically for the public API in future releases.

__Attributes:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Period length to generate shocks for. |
| dist | Distribution to use when drawing random values. |
| multivar | Generate shocks for multiple correlated components if `#! True`. |
| seed | Random state seed. |
| dist_args | Positional arguments passed-through to the distribution's `#!python .rvs` method. |
| dist_kwargs | Keyword arguments passed-through to the distribution's `#!python .rvs` method. |
| shock_arr | Array of shock values to configure. |

&nbsp;

__Methods:__

```python
Shock.shock_generator() -> Callable[[...], ndarray]
```

Creates a callable that returns an entire shock array according to the class attributes when called.

__Inputs:__

`#!python None`

__Returns:__

???+ warning "Single Return"
    Only one `Callable` specification from the table below is returned.

| __Type(s)__ | __Description__ |
|:------------|----------------:|
| `#!python Callable[[float], ndarray[float]]` | Callable taking a single shock standard deviation (sigma) parameter. (This signature is returned for univariate shocks)
| `#!python Callable[[ndarray[float]], ndarray[float]]` | Callable taking a shock covariance matrix. (This signature is returned for multivariate shocks.) |

&nbsp;

```python
@overload
Shock.place_shocks(
    shock_spec: dict[int, float],
) -> np.ndarray[float]

@overload
Shock.place_shocks(
    shock_spec: dict[tuple[int, int], float]
) -> np.ndarray[float]
```

Modifies the specified indices of a given shock array to the corresponding values. (Returns a modified zero vector if `shock_arr` isn't specified in the class instance)

`#!python place_shocks` supports univariate and multivariate array manipulations via overloading.

???+ warning "Array Shape Inference"
    In univariate cases, the array shape can be inferred accurately using `#!python Shock.T`.

    In multivariate cases, the number of columns cannot be inferred uniquely from T alone. If `shock_arr` is not provided, `SymbolicDSGE` infers K from `shock_spec` as `#!python K = max_col_idx + 1` where `#!python max_col_idx = max(col_idx for (t, col_idx) in shock_spec)`.

???+ note "Index Bound Enforcement"
    Out-of-bounds or negative indices raise `#!python IndexError`.

__Overloaded `shock_spec` Scheme:__

| __Mode__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| Univariate | `#!python dict[int, float]` | Keys indicate the time index to place the values at. |
| Multivariate | `#!python dict[tuple[int, int], float]` | Keys contain a two-dimensional indexer `(time_idx, arr_idx)` to access the 2D array using the conventional `(row, col)` specification. Values are placed in the element described by the indexer. |

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| shock_spec | `#!python dict` indicating the position and value required for a modification of `shock_arr`. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python np.ndarray[float]` | Array with specified indices modified as per the specification. |

__Examples:__

```python
# Univariate
Shock(T=10).place_shocks(
    {
        0: 1.0,
        3: -0.5 # (1)!
    }
)  # (2)!

# Multivariate (K inferred as 2)
Shock(T=10, multivar=True).place_shocks(
    {
        (0,0): 1.0,
        (0,1): 2.0, # (3)!
    }
)  # (4)!
```

1. Sets `output[0] = 1.0` and `output[3] = -0.5`.
2. Returns shape `(10,)`
3. Sets `output[0, 0] = 1.0` and `output[0, 1] = 2.0`.
4. Returns shape `(10, 2)`
