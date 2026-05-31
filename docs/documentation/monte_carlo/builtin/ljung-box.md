---
tags:
    - doc
---

# Ljung-Box Tests

```python
ljung_box_test_step(
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
    filter_key: str = "filter",
    payload_key: str | None = None,
    payload_key: str | None = None,
    column: Sequence[int] | int | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 10,
    alpha: float = 0.05,
) -> MCStep
```

`ljung_box_test_step` wraps `run_ljung_box_test(...)`. It selects a 1D array from generated data, a `FilterResult`, or a named payload, then runs the Ljung-Box test for autocorrelation up to the specified number of lags.

???+ warning "Lag Handling"
    - If `lags` is greater than the length of the input array minus `burn_in`, the test will run up to the maximum possible lag.
    - There is no automatic lag selection heuristic, if `lags` isn't specified, the default value of `10` is used regardless of the input array length.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |