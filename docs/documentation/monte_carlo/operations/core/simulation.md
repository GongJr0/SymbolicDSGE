---
tags:
    - doc
---
# DGP Simulation

```python
simulation_step(
    name: str = "datagen",
    *,
    T: int,
    shocks: Mapping[str, Shock | Callable | ndarray] | None = None,
    seed_increment: int | Literal["auto"] = "auto",
    shock_scale: float = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCStep
```

`simulation_step` wraps `simulate_dgp(...)`. It requires `MCPipeline.run(..., dgp=...)` and calls `dgp.sim(...)` in each replication.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of simulated periods, excluding the initial state. |
| shocks | Shock specification passed into DGP simulation after per-replication seed handling. |
| seed_increment | Integer seed offset per replication, or `"auto"` to increment by the number of seeded `Shock` objects. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are included in `MCData.observables`. |

???+ info "Seed Convention"
    For generator-style `Shock` objects with integer seeds, replication `rep_idx` receives `shock.seed + rep_idx * seed_increment`. With `seed_increment="auto"`, the increment is the number of seeded `Shock` entries. Array shocks and callable shocks are passed through unchanged.
