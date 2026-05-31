---
tags:
    - doc
---
# Wald Tests

```python
wald_test_step(
    name: str,
    *,
    source: Literal[
        "states",
        "observables",
        "x_pred",
        "x_filt",
        "y_pred",
        "y_filt",
        "innov",
        "std_innov",
        "payload",
    ],
    target: ndarray,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> MCStep
```

`wald_test_step` wraps `run_wald_test(...)`. It selects a 1D or 2D array from generated data, a `FilterResult`, or a named payload, then runs the requested HAC Wald diagnostic.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |

__Kinds:__

| __Kind__ | __Description__ |
|:---------|----------------:|
| `mean` | Tests `E[g_t] = target`. |
| `covariance` | Tests the vech representation of the covariance matrix against `target`. |
| `second_moment` | Tests the vech representation of the raw second moment against `target`. |
