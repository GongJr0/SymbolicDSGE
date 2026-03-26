---
tags:
    - doc
---
# DenHaanMarcet

```python
@dataclass(frozen=True)
class DenHaanMarcetResult()
```

`#!python dataclass` storing the output of a single Den Haan-Marcet (DHM) test application.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| statistic | `#!python float` | Realized DHM test statistic. |
| df | `#!python int` | Degrees of freedom of the chi-square reference distribution. |
| p_value | `#!python float` | Chi-square tail probability associated with `statistic`. |
| critical_value | `#!python float` | Chi-square critical value implied by the requested `alpha`. |
| rejects_null | `#!python bool` | Whether the test rejects the null at the requested significance level. |
| mean_moments | `#!python np.ndarray` | Sample mean of the stacked moment vector. |
| covariance | `#!python np.ndarray` | Estimated covariance matrix of the stacked moment vector. |
| moments | `#!python np.ndarray` | Realized stacked moment matrix. Shape `(n_obs, n_eq * n_inst)`. |
| residuals | `#!python np.ndarray` | Equation residuals used in the DHM test. Shape `(n_obs, n_eq)`. |
| raw_residuals | `#!python np.ndarray` | Explicit copy of the raw residual matrix for downstream diagnostics. Shape `(n_obs, n_eq)`. |
| instruments | `#!python np.ndarray` | Realized instrument matrix. Shape `(n_obs, n_inst)`. |
| states | `#!python np.ndarray` | State matrix used to evaluate the test. Shape `(T+1, n)` for simulation-driven runs. |
| shock_matrix | `#!python np.ndarray \| None` | Realized shock matrix used by `#!python one_sample`. `#!python None` for `#!python from_state_path`. |
| variables | `#!python list[str]` | Ordering of state columns in `states`. Matches compiled variable order. |
| equation_idx | `#!python np.ndarray` | Equation indices used in fallback compiled-equation mode, or a sequential index for custom FOCs. |
| instrument_idx | `#!python np.ndarray` | Instrument indices resolved against compiled variable order. |
| include_constant | `#!python bool` | Whether a constant instrument was included. |
| burn_in | `#!python int` | Number of leading transitions excluded from the test. |
| foc_expressions | `#!python tuple[str, ...] \| None` | Normalized custom FOC expressions used for the run. `#!python None` in compiled-equation mode. |

&nbsp;

```python
@dataclass(frozen=True)
class DenHaanMarcetMonteCarloResult()
```

`#!python dataclass` storing the output of repeated DHM applications under Monte Carlo shock draws.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| rejection_rate | `#!python float` | Fraction of replications rejecting the null. |
| alpha | `#!python float` | Significance level used for rejection decisions. |
| df | `#!python int` | Degrees of freedom shared across all replications. |
| critical_value | `#!python float` | Chi-square critical value implied by `alpha`. |
| statistics | `#!python np.ndarray` | DHM statistic from each replication. Shape `(n_rep,)`. |
| p_values | `#!python np.ndarray` | P-values from each replication. Shape `(n_rep,)`. |
| rejections | `#!python np.ndarray` | Boolean rejection flags from each replication. Shape `(n_rep,)`. |
| raw_residuals | `#!python np.ndarray` | Residual matrices stacked by replication. Shape `(n_rep, n_obs, n_eq)`. |
| variables | `#!python list[str]` | Ordering of state columns shared by every replication. |
| equation_idx | `#!python np.ndarray` | Equation indices used in the repeated test application. |
| foc_expressions | `#!python tuple[str, ...] \| None` | Normalized custom FOC expressions used for the Monte Carlo run. |

&nbsp;

```python
class DenHaanMarcet(
    solved: SolvedModel,
    focs: Sequence[str] | None = None,
    foc_locals: Mapping[str, str] | None = None
)
```

`#!python DenHaanMarcet` provides one-sample and Monte Carlo Den Haan-Marcet tests using a `#!python SolvedModel` object.

The class supports two equation sources:

- Compiled model equations selected by `#!python equation_idx`
- Custom FOC strings supplied either at construction or per method call

Custom FOC strings are parsed against the model configuration, validated against calibrated parameters and time-indexed variables, and normalized to a `#!python t / t-1` representation before compilation.

???+ note "Forward Object Convention"
    By default, forward terms use the conditional expectations (next state if shocks were `0.0`). If `#!python use_conditional_expectation=False`, the forward object becomes the realized simulation draw of the next time index.

???+ note "Equation and Instrument Indexing"
    In compiled-equation mode, `#!python equation_idx` indexes the model equations in the compiled `#!python config.equations.model` order. `#!python instrument_idx` indexes the compiled variable order and can be supplied either as integer positions or variable names.

???+ info "Custom FOC Syntax"
    Time-dependent variables must be written with explicit timing such as `#!python x(t)` or `#!python Pi(t+1)`. Bare variable names like `#!python x` are treated as symbols and rejected unless they are calibrated parameters or locally defined aliases.

???+ info "FOC Locals"
    `#!python foc_locals` allows pre-validation substitutions. Both scalar and time-indexed aliases are supported, for example:

    - `#!python {"gap_term": "sigma*(r(t) - Pi(t+1))"}`
    - `#!python {"C(t)": "exp(x(t))"}`

__Methods:__

```python
DenHaanMarcet.one_sample(
    T: int,
    shocks: dict[str, Callable | np.ndarray] | None = None, # (1)!
    *,
    focs: Sequence[str] | None = None, # (2)!
    foc_locals: Mapping[str, str] | None = None, # (3)!
    shock_scale: float = 1.0,
    x0: np.ndarray | None = None,
    equation_idx: Sequence[int] | None = None, # (4)!
    instrument_idx: Sequence[int | str] | None = None, # (5)!
    include_constant: bool = True,
    burn_in: int = 0,
    alpha: float = 0.05,
    use_conditional_expectation: bool = True, # (6)!
) -> DenHaanMarcetResult
```

 1. Shock specification follows the same conventions as `#!python SolvedModel.sim`, including comma-separated keys for correlated multivariate shocks.
 2. Per-call custom FOCs override constructor defaults.
 3. Per-call locals are merged on top of constructor-level locals.
 4. Only available when no custom FOCs are supplied.
 5. Accepts either integer indices or variable names resolved against compiled variable order.
 6. If `#!python True`, forward terms are evaluated using the model-implied conditional expectation `#!python states[:-1] @ A.T` rather than the realized next row `#!python states[1:]`.

Simulate a state path from the solved model and run a single DHM test.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of transitions to simulate. The resulting state matrix has shape `(T+1, n)`. |
| shocks | Shock specification used to generate the realized state path. |
| focs | Optional custom FOC strings used instead of compiled model equations. |
| foc_locals | Optional alias dictionary applied before FOC validation and compilation. |
| shock_scale | Multiplicative scaling applied to realized shocks. |
| x0 | Initial state vector. `#!python None` defaults to zeros with controls backfilled from the policy rule. |
| equation_idx | Optional subset of compiled-equation indices to test. |
| instrument_idx | Optional subset of instruments taken from compiled variable order. |
| include_constant | Include a constant term in the instrument vector if `#!python True`. |
| burn_in | Number of leading transitions excluded from the DHM test. |
| alpha | Significance level used for `critical_value` and `rejects_null`. |
| use_conditional_expectation | Use `#!python E_t[X_{t+1}] = A X_t` instead of realized next states when evaluating forward terms. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python DenHaanMarcetResult` | Single-run DHM output containing the evaluated state path, instruments, moments, residuals, and test statistics. |

&nbsp;

```python
DenHaanMarcet.from_state_path(
    states: np.ndarray,
    *,
    focs: Sequence[str] | None = None,
    foc_locals: Mapping[str, str] | None = None,
    equation_idx: Sequence[int] | None = None,
    instrument_idx: Sequence[int | str] | None = None,
    include_constant: bool = True,
    burn_in: int = 0,
    alpha: float = 0.05,
    use_conditional_expectation: bool = True
) -> DenHaanMarcetResult
```

Run a DHM test on a user-supplied state path without simulating shocks internally.

???+ note "State Path Convention"
    `states` must be a 2D matrix in compiled variable order. Consecutive rows are interpreted as consecutive periods. The method evaluates `#!python len(states) - 1 - burn_in` transitions by pairing `#!python states[:-1]` with either `#!python states[1:]` or `#!python states[:-1] @ A.T`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| states | Consecutive state vectors in compiled variable order. |
| focs | Optional custom FOC strings used instead of compiled model equations. |
| foc_locals | Optional alias dictionary applied before FOC validation and compilation. |
| equation_idx | Optional subset of compiled-equation indices to test. |
| instrument_idx | Optional subset of instruments taken from compiled variable order. |
| include_constant | Include a constant term in the instrument vector if `#!python True`. |
| burn_in | Number of leading transitions excluded from the DHM test. |
| alpha | Significance level used for `critical_value` and `rejects_null`. |
| use_conditional_expectation | Use `#!python E_t[X_{t+1}] = A X_t` instead of the realized next row. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python DenHaanMarcetResult` | Single-run DHM output built from the provided state path. `#!python shock_matrix` is `#!python None` in this route. |

&nbsp;

```python
DenHaanMarcet.monte_carlo(
    T: int,
    shocks: dict[str, Shock], # (1)!
    *,
    focs: Sequence[str] | None = None,
    foc_locals: Mapping[str, str] | None = None,
    n_rep: int,
    shock_scale: float = 1.0,
    x0: np.ndarray | None = None,
    equation_idx: Sequence[int] | None = None,
    instrument_idx: Sequence[int | str] | None = None,
    include_constant: bool = True,
    burn_in: int = 0,
    alpha: float = 0.05,
    use_conditional_expectation: bool = True
) -> DenHaanMarcetMonteCarloResult
```

 1. Monte Carlo mode requires `#!python Shock` instances, not arbitrary callables or pre-realized arrays. Each replication clones the provided shocks and deterministically increments their seeds when a base seed is present.

Run repeated DHM tests under independently regenerated shock draws.

???+ warning "Monte Carlo Shock Requirements"
    The Monte Carlo interface is intentionally stricter than `#!python one_sample`. Repeated draws require generator-style `#!python Shock` objects so the class can safely create fresh realizations without mutating user-owned shock specifications.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of transitions to simulate in each replication. |
| shocks | Dictionary of `#!python Shock` objects keyed by their corresponding variable or comma-separated multivariate group. |
| focs | Optional custom FOC strings used instead of compiled model equations. |
| foc_locals | Optional alias dictionary applied before FOC validation and compilation. |
| n_rep | Number of Monte Carlo replications. |
| shock_scale | Multiplicative scaling applied to realized shocks. |
| x0 | Initial state vector. `#!python None` defaults to zeros with controls backfilled from the policy rule. |
| equation_idx | Optional subset of compiled-equation indices to test. |
| instrument_idx | Optional subset of instruments taken from compiled variable order. |
| include_constant | Include a constant term in the instrument vector if `#!python True`. |
| burn_in | Number of leading transitions excluded from each DHM test. |
| alpha | Significance level used for rejection decisions. |
| use_conditional_expectation | Use `#!python E_t[X_{t+1}] = A X_t` instead of realized next states when evaluating forward terms. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python DenHaanMarcetMonteCarloResult` | Monte Carlo summary containing per-replication statistics, rejection flags, and stacked raw residual matrices. |
