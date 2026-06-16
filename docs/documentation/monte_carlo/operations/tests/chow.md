---
tags:
    - doc
---

# Chow Tests

```python
chow_test_step(
    name: str,
    *,
    y_source: InpSources,
    x_source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    t_break: int = 10,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep
```

`chow_test_step` wraps `run_chow_test(...)`. It resolves a single response column from `y_source` and a regressor matrix from `x_source`, then tests for a structural break in the regression coefficients at the known break point `t_break` by comparing the pooled residual sum of squares against the two sub-sample fits.

???+ info "Reference Distribution"
    The statistic is compared against an $F(p,\ T - 2p)$ distribution, where $p$ is the number of regressor columns and $T$ is the number of observations.

???+ warning "Break point and sample size"
    `t_break` must satisfy `0 < t_break < T`, and the sample must provide enough observations for two separate fits (`T > 2p`); otherwise the step reports a non-`OK` status. `y_source` must resolve to exactly one column, and the response and regressor arrays must share the same number of rows.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
