---
tags:
    - doc
---
# Simulation

```python
simulation_step(
    name: str = "datagen",
    *,
    target: Literal["reference", "dgp"] = "dgp",
    T: int,
    shocks: Mapping[str, Shock | Callable | ndarray] | None = None,
    seed_increment: int | Literal["auto"] = "auto",
    shock_scale: float = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCStep
```

`simulation_step` wraps `simulate(...)`. Each replication it drives one solved model's simulation: the **DGP** by default, or the **reference** model when `target="reference"`. The chosen model must be supplied to `MCPipeline.run(...)` — `dgp=...` for `target="dgp"`, and the always-required `reference=...` for `target="reference"`. Simulating the reference (the "null" / correctly-specified case) is a size study; simulating a distinct DGP is a power/misspecification study.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| target | Which solved model to simulate: `"dgp"` (default) or `"reference"`. |
| T | Number of simulated periods, excluding the initial state. |
| shocks | Shock specification passed into the simulation after per-replication seed handling. When omitted, shocks are generated from the target model's shock configuration. |
| seed_increment | Integer seed offset per replication, or `"auto"` to increment by the number of seeded `Shock` objects. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are included in `MCData.observables`. |

???+ info "Seed Convention"
    For generator-style `Shock` objects with integer seeds, replication `rep_idx` receives `shock.seed + rep_idx * seed_increment`. With `seed_increment="auto"`, the increment is the number of seeded `Shock` entries. Array shocks and callable shocks are passed through unchanged.
