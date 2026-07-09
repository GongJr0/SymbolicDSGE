---
tags:
    - doc
---
# Wald Tests

```python
wald_test_step(
    name: str,
    *,
    source: str,
    field: str,
    target: ndarray,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> MCStep
```

`wald_test_step` wraps `run_wald_test(...)`. It selects a 1D or 2D array from a producer step and runs the requested HAC Wald diagnostic.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| source | required | Producer step name. |
| field | required | Producer field. Use `states` or `observables` for data steps, a filter output field for filter steps, or `payload` for transform steps. |
| target | required | Hypothesized moment value. |
| kind | `"mean"` | Moment tested against `target`. |
| columns | `None` | Optional column subset. |
| burn_in | `0` | Rows dropped before the test. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| kernel | `"bartlett"` | HAC kernel. |
| bandwidth | `"auto"` | HAC bandwidth selector or explicit integer. |
| alpha | `0.05` | Test size. |

__Kinds:__

| __Kind__ | __Description__ |
|:---------|----------------:|
| `mean` | Tests `E[g_t] = target`. |
| `covariance` | Tests the vech representation of the covariance matrix against `target`. |
| `second_moment` | Tests the vech representation of the raw second moment against `target`. |
