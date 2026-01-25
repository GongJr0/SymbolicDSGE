---
tags:
    - doc
---
# SolvedModel

```python
@dataclass(frozen=True)
class SolvedModel()
```

`SolvedModel` contains the policy/transition matrices and relevant methods.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| compiled | `#!python CompiledModel` | The compiled model object that resulted in the current solution. |
| policy | `#!python linearsolve.model` | Solver backend output (e.g., stability diagnostics, eigenvalues, raw solver objects). |
| A | `#!python np.ndarray` | The discovered state-transition matrix. |
| B | `#!python np.ndarray` | The discovered innovation impact matrix. |

&nbsp;

__Methods:__

```python
SolvedModel.sim(
    T: int,
    shocks: dict[str, Callable | np.ndarray], # (1)!
    shock_scale: float = 1.0, # (2)!
    x0: np.ndarray | None = None,
    observables: bool = False
) -> dict[str, np.ndarray[float]]
```

 1. The dictionary keys can be populated by:
   - An array of shocks `(T, 1) | (T,)` when shocks are for a single variable and `(T, K)` when drawing correlated shocks simultaneously.
   - A callable accepting either a shock standard deviation (sigma) or a covariance matrix depending on univariate or multivariate requirements. The callable should return arrays shaped as described above. Per-step generators are not supported.
1. Shocks are drawn from the specified distribution and all elements in the arrays are scaled by this parameter.

 Returns the simulated path defined by the given inputs.

???+ info "Univariate Shock Syntax"
    A univariate shock is defined as a dictionary entry for the given variable. For example, if a model specifies a variable `x`
    and a shock symbol `e_x`, the dictionary would expect `#!python {"x": ...}` where `...` is populated by a `ndarray` of shape
    `(T,)` or `(T,1)`, or by a univariate generator callable.

???+ info "Correlated Shock Syntax"
    To define a set of variables with nonzero shock covariance, a shared dictionary entry should be used. For example, a multivariate
    shock to `x` and `y` should be defined as `#!python {"x,y": ...}` where `...` is populated by a `(T, 2)` array or a multivariate
    generator callable.

    Details regarding the dictionary key scheme:

    - Variable names are parsed by splitting on commas; surrounding whitespace is stripped.
    - Multiple variables can be chained as required. There is no variable count limitation.
    - The ordering of variables in the key does __not__ affect simulation results.
    - Shock realizations are always aligned with the innovation ordering defined at model configuration or compilation (`B` matrix order).

??? info "Multivariate Shock Canonicalization and Reproducibility"
    When multivariate shock generators are used, variables are __internally reordered to a canonical model-defined order before sampling__.
    This ensures that simulations are __reproducible under a fixed random seed__, regardless of the order in which variables are specified
    in the shock dictionary key (e.g. `"g,z"` vs `"z,g"`).

    This behavior is required because multivariate sampling methods (e.g. Cholesky-based Gaussian draws) are __order-dependent at the
    realization level__, even when the underlying covariance structure is permutation-invariant.

    Concretely:

    - Correlation and covariance matrices are constructed **after** canonicalizing variable order.
    - Sampling is performed in this canonical order.
    - Shock realizations are then mapped to the correct innovation indices used by the model.

    As a result, variable ordering in multivariate shock keys does **not** affect either the statistical properties
    *or the realized sample paths* of the simulation when the random seed is fixed.

??? warning "Custom Shock Generators"
    While it is technically possible to replicate the internal generator factory’s behavior, this is __strongly discouraged__.
    Custom shock distributions and array manipulations are supported via the `#!python Shock` class (see the [docs](./Shock.md)).
    Bypassing the shock interface may lead to unexpected or unstable behavior.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Amount of steps to simulate the paths; excluding `x0`. |
| shocks | Array or callable generator of shocks keyed by their corresponding variable. |
| shock_scale | Scaling factor for the shocks. |
| x0 | Initial state of model variables shaped `(n,)`. (`None` defaults to zeroes) |
| observables | Include observable paths in the output `#!python dict` if `#!python True`. |

???+ info "Period Specification"
    Index 0 of the output will always be an initial state. (default or specified) The input `T` will result on `T+1` array indices from index 0 to `T`.

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, np.ndarray[float]]` | Arrays of paths paired with their corresponding variables. The key `#!python "_X"` contains the full state matrix. (Shape `(T+1, n)`) |

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
SolvedModel.to_dict() -> dict[str, Any]
```
Dictionary representation of the class instance.

__Inputs:__

`#!python None`

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python dict[str, Any]` | Dictionary representation of the `#!python SolvedModel` object. |
