---
tags:
    - doc
---
# Simulation

```python
simulation_step(
    name: str = "datagen",
    target: Literal["reference", "dgp"] = "dgp",
    n_retain: int = -1,
    *,
    T: int,
    shocks: Mapping[str, Shock | Callable | ndarray] | None = None,
    shock_scale: float = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCStep
```

`simulation_step` generates one replication's data from a solved model: the DGP by default, or the reference model when `target="reference"`. The selected model must be supplied to `MCPipeline.run(...)`. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| target | Which solved model to simulate: `"dgp"` (default) or `"reference"`. |
| n_retain | Number of replications to retain in the output. If `-1`, all replications are retained. |
| T | Number of simulated periods, excluding the initial state. |
| shocks | Shock mapping resolved once and redrawn per replication. Use innovation symbol keys, including grouped keys such as `"e_g,e_z"`. With `None`, the simulation is deterministic. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are produced alongside states, and downstream steps may read `field="observables"`. |

???+ info "Seed Convention"
    Normal and uniform `Shock` specifications are drawn inside the native loop from a counter based engine keyed on `(shock.seed, entry index)` and addressed by `rep_idx`. Every replication therefore reads a distinct, non-overlapping stream, and a seeded specification replays bit for bit regardless of `n_rep` or `n_jobs`. A specification with `seed=None` takes a fresh key each run.

    Other specifications (Student-t, scipy distribution objects, callables, literal arrays) are drawn in Python before the run. There, replication `rep_idx` receives `shock.seed + rep_idx * k`, where `k` is the number of seeded `Shock` entries; array and callable shocks are passed through unchanged.

## Reproducing One Replication

```python
replication_shocks(
    model: SolvedModel,
    step: MCStep,
    rep_idx: int,
) -> dict[str, ndarray]
```

Because each replication addresses its own stream rather than replaying a shared one, rerunning a pipeline with a smaller `n_rep` does not reproduce a given replication. `replication_shocks` returns the shock paths replication `rep_idx` saw, keyed exactly as the specification is: a `(T,)` column per univariate entry and a `(T, width)` block per grouped entry. It lives in `SymbolicDSGE.monte_carlo`.

```python
from SymbolicDSGE.monte_carlo import replication_shocks

shocks = replication_shocks(dgp, datagen_step, rep_idx=417)
sample = dgp.sim(T=200, shocks=shocks, shock_scale=1.0)
```

Scaling is applied to the returned paths, so `sim` is called with `shock_scale=1.0`. `model` must be the role the step targeted, and `step` the step the run used. Only seeded specifications are reproducible: a `seed=None` entry was drawn from a key the run discarded, so it comes back as a fresh path.
