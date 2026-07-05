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

`simulation_step` wraps `simulate(...)`. Each replication drives one solved model's simulation: the **DGP** by default, or the **reference** model when `target="reference"`. The selected model must be supplied to `MCPipeline.run(...)`: pass `dgp=...` for `target="dgp"` and pass `reference=...` for all runs. Simulating the reference is a size study; simulating a distinct DGP is a power or misspecification study.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| target | Which solved model to simulate: `"dgp"` (default) or `"reference"`. |
| T | Number of simulated periods, excluding the initial state. |
| shocks | Explicit shock mapping passed into `SolvedModel.sim(...)` after per-replication seed handling. Use keys matching simulated variables or grouped keys such as `"g,z"`. With `None`, the simulation is deterministic. |
| seed_increment | Integer seed offset per replication, or `"auto"` to increment by the number of seeded `Shock` objects. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are included in `MCData.observables`. |

???+ warning "Stochastic runs need explicit shocks"
    The Monte Carlo API does not infer a shock specification from model metadata or `shock_map`. Provide a `shocks` mapping directly, or let the UI shock registry compile into one. A `Shock` object carries the distribution, distribution parameters, and optional seed for that run.

???+ info "Seed Convention"
    For generator style `Shock` objects with integer seeds, replication `rep_idx` receives `shock.seed + rep_idx * seed_increment`. With `seed_increment="auto"`, the increment is the number of seeded `Shock` entries. Array shocks and callable shocks are passed through unchanged.
