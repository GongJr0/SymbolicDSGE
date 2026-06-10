---
tags:
    - doc
---

# Breusch-Pagan Tests

```python
breusch_pagan_test_step(
    name: str,
    *,
    residual_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    residual_payload_key: str | None = None,
    x_payload_key: str | None = None,
    residual_col: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
    robust: bool = False,
) -> MCStep
```

`breusch_pagan_test_step` wraps `run_breusch_pagan_test(...)`. It resolves a single residual column from `residual_source` and a regressor matrix from `X_source`, then tests the residuals for heteroskedasticity by regressing their squares on the regressors.

???+ info "Reference Distribution"
    The Lagrange-multiplier statistic is compared against a $\chi^2(p)$ distribution, where $p$ is the number of regressor columns. Set `robust=True` for Koenker's studentized variant, which is robust to non-normal residuals.

???+ warning "Residuals must be 1D"
    `residual_source` must resolve to exactly one column, and the residual and regressor arrays must share the same number of rows.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
