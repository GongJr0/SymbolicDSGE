---
tags:
    - doc
---
# Reference Filtering

```python
reference_filter_step(
    name: str = "filter",
    *,
    filter_mode: Literal["linear", "extended"] = "linear",
    observables: list[str] | None = None,
    x0: ndarray | None = None,
    p0_mode: Literal["diag", "eye"] | None = None,
    p0_scale: float | None = None,
    jitter: float | None = None,
    symmetrize: bool | None = None,
    return_shocks: bool = False,
    R: ndarray | None = None,
    estimate_R_diag: bool = False,
    R_scale: float = 1.0,
) -> MCStep
```

`reference_filter_step` wraps `run_reference_filter(...)`. It reads `context.data.observables` and calls `reference.kalman(...)`.

When `observables=None`, generated `MCData.observable_names` are used if available. If names are not available, `reference.kalman(...)` falls back to its normal observable resolution.
