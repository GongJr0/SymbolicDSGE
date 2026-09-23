---
tags:
    - doc
---
# Simulation

```python
simulation_step(
    name: str = "datagen",
    n_retain: int = -1,
    *,
    target: str,
    T: int,
    shocks: Mapping[str | Sequence[str], Shock | ndarray] | Sequence[Shock | ShockPath] | None = None,
    shock_scale: float = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCStep
```

`simulation_step` generates one replication's data from a solved model. The selected model must be supplied to `MCPipeline.run(...)`. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| target | Name of the model to simulate. Must be supplied to `MCPipeline.run(...)`. | 
| n_retain | Number of replications to retain in the output. If `-1`, all replications are retained. |
| T | Number of simulated periods, excluding the initial state. |
| shocks | Shock spec resolved once and redrawn per replication. Mirrors `SolvedModel.sim(...)`: a sequence of bound `Shock` or `ShockPath` entries, or a mapping keyed by innovation symbol or tuple of symbols. With `None`, the simulation is deterministic. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are produced alongside states, and downstream steps may read `field="observables"`. |

???+ info "Seed Convention"
    Normal and uniform `Shock` specifications are drawn inside the native loop from a counter based engine keyed on `(shock.seed, entry index)` and addressed by `rep_idx`. Every replication therefore reads a distinct, non-overlapping stream, and a seeded specification replays bit for bit regardless of `n_rep` or `n_jobs`. A specification with `seed=None` takes a fresh key each run.

    Other specifications (Student-t, scipy distribution objects, `ShockPath` entries) are drawn in Python before the run. There, replication `rep_idx` receives `shock.seed + rep_idx * k`, where `k` is the number of seeded `Shock` entries; a `ShockPath` is passed through unchanged and is therefore identical across replications.

## Reproducing One Replication

```python
replication_shocks(
    model: SolvedModel,
    step: MCStep,
    rep_idx: int,
) -> dict[tuple[str, ...], ndarray]
```

Because each replication addresses its own stream rather than replaying a shared one, rerunning a pipeline with a smaller `n_rep` does not reproduce a given replication. `replication_shocks` returns the shock paths replication `rep_idx` saw, one per spec entry, keyed by that entry's `target` tuple: a `(T, 1)` block for a univariate entry and a `(T, width)` block for a grouped one. It lives in `SymbolicDSGE.monte_carlo`.

```python
from SymbolicDSGE import ShockPath
from SymbolicDSGE.monte_carlo import replication_shocks

shocks = replication_shocks(dgp, datagen_step, rep_idx=417)
sample = dgp.sim(T=200, shocks=[ShockPath(p, *k) for k, p in shocks.items()], shock_scale=1.0)
```

Scaling is applied to the returned paths; `sim` is therefore called with `shock_scale=1.0`. `model` must be the role the step targeted, and `step` the step the run used. Only seeded specifications are reproducible: a `seed=None` entry was drawn from a key the run discarded, and comes back as a fresh path.
