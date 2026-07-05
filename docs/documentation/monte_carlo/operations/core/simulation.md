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

`simulation_step` wraps `simulate(...)`. Each replication drives one solved model's simulation: the DGP by default, or the reference model when `target="reference"`. The selected model must be supplied to `MCPipeline.run(...)`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| target | Which solved model to simulate: `"dgp"` (default) or `"reference"`. |
| T | Number of simulated periods, excluding the initial state. |
| shocks | Shock mapping passed into `SolvedModel.sim(...)` after per replication seed handling. Use the same key convention as `SolvedModel.sim(...)`, including grouped keys such as `"g,z"`. With `None`, the simulation is deterministic. |
| seed_increment | Integer seed offset per replication, or `"auto"` to increment by the number of seeded `Shock` objects. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are included in `MCData.observables`. |

???+ info "Seed Convention"
    For generator style `Shock` objects with integer seeds, replication `rep_idx` receives `shock.seed + rep_idx * seed_increment`. With `seed_increment="auto"`, the increment is the number of seeded `Shock` entries. Array shocks and callable shocks are passed through unchanged.
