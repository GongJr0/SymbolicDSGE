---
tags:
    - guide
---

# Bundle Loading Guide

??? tip "__TL;DR__"
    Open a `.sdsge` bundle with `load_bundle(...)` and reach every component through the typed `LoadedBundle` fields:

    - `reference` and `dgp`: Solved models.
    - `estimation.estimator` and `estimation.result`: Estimator and `*Result` (MLE, MAP, or MCMC).
    - `mc.pipeline` and `mc.traces`: Monte Carlo pipeline and result traces.
    - `simulation`: Dictionary of Simulation prefills (`SimSpec`) for each model slot

    You can find a demonstration notebook [here](../assets/bundle_loading.ipynb).

This guide walks through opening a `.sdsge` bundle and reaching each library object it carries.
We use `experiment-1.sdsge` as produced by the [Bundle Authoring Guide](bundle_authoring_guide.md). Substitute any other bundle path.

???+ tip "What `load_bundle` actually does"
    `load_bundle` parses every embedded YAML, runs `DSGESolver.compile(**compile_kwargs).solve(**solve_kwargs)` with the kwargs recorded at write time, decodes every tabular member by `Member.format` (CSV or Parquet), and rebuilds live estimation results and Monte Carlo pipelines when present. Loading is deterministic: the resulting policy matrices match those the author had in hand.

## Open the bundle

```python
import numpy as np

from SymbolicDSGE import load_bundle
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation.results import MCMCResult

from typing import cast

loaded = load_bundle("experiment-1.sdsge")  # (1)!
```

1. `load_bundle` is available from `SymbolicDSGE`; it calls `SymbolicDSGE.bundle.build_from`. Both names are interchangeable.

`loaded` is a [`LoadedBundle`](../documentation/bundle/LoadedBundle.md). Every component is reachable through a typed field.

```python
print("Created by:", loaded.manifest.created_by)
print("Created at:", loaded.manifest.created_at)
print("Format version:", loaded.manifest.sdsge_version)
```

???+ info "Manifest provenance"
    `Manifest.checksums` carries SHA-256 hex digests over each member's bytes. This is useful for integrity checks before trusting the contents downstream.

## Reach the solved models

`reference` and `dgp` are full `SolvedModel` instances. They behave exactly like models you would have solved in process, including IRFs, simulation, and Kalman filtering.

```python
# Cast so type checkers do not treat these values as optional.
reference = cast(SolvedModel, loaded.reference)
dgp = cast(SolvedModel, loaded.dgp)

# The `if` check narrows the type if casting is not preferred.
if reference is not None:
    print("Stable:", reference.policy.stab == 0)
    print("Eigenvalues:", reference.policy.eig.round(2), "\n")

if dgp is not None:
    print("Stable:", dgp.policy.stab == 0)
    print("Eigenvalues:", dgp.policy.eig.round(2))
```

???+ note "DGP slot may be absent"
    `loaded.dgp` is `None` whenever the bundle did not carry a `dgp.yaml`. Test before use.

## Reach the estimation tab

`LoadedEstimation` carries every text and bulk component of the estimation run.

```python
estimation = loaded.estimation

# Type checkers cannot infer that `estimation` is not `None`.
# Asserting (alternative to casting) here silences the errors 
# by forcing the type to narrow.
assert estimation, "No estimation tab found in the bundle."

est = estimation.estimator
res = cast(MCMCResult, estimation.result)
```

`estimation.estimator` is a live `Estimator` instance for the `reference` model.
`estimation.spec` is an `EstimatorSpec` and can be used to construct `Estimator`s for
any model.

```python
# This also works and builds the estimator for the DGP model.
# Observables and parameters stay the same.
Estimator.from_spec(estimation.spec, compiled=dgp.compiled)
```

`estimation.result` (when present) is the result a bundled run produced: a `MLEResult` for MLE, a `MAPResult` for MAP, or an `MCMCResult` for MCMC.

### Run an estimation from a loaded bundle

All result classes carry their options as a dictionary. `optimizer_config` in `MLEResult`/`MAPResult` and `sampler_config` in `MCMCResult` can be unpacked diectly to re-run the routine that produced a given result. Note that `random_state` must be set for a result to be reproducible in case of `MCMCResult`.

```python
repro = est.mcmc(**res.sampler_config)
np.array_equal(res.samples, repro.samples)
```

```bash
>>> True # if random_state was set in the original run
```

See the [Estimation Guide](estimation_guide.md) for the run methods in detail.

## Reach the Monte Carlo tab

`LoadedMC.pipeline` is the first class [`MCPipeline`](../documentation/monte_carlo/pipeline.md) rebuilt at load time. `LoadedMC.spec` remains available for archive inspection and UI rendering, and `LoadedMC.resources` holds the side channel arrays or custom callables that were reattached while rebuilding the pipeline. When the bundle carries a completed run, `document` holds the trace free summary and `traces` holds the bulk columns.

```python
mc = loaded.mc
assert mc, "No Monte Carlo tab found in the bundle."

print("Runtime Steps:", [step.name for step in mc.pipeline.per_rep_steps])
print("Post-Processing:", [step.name for step in mc.pipeline.postproc_steps])
```

### Run a Monte Carlo pipeline from a loaded bundle

The loaded pipeline runs against the loaded models without the `[ui]` extra.

```python
# The pipeline's simulation datagen needs a DGP. The authoring notebook
# bundles a model under role "dgp", so `loaded.dgp` resolves. If a bundle
# omits `reference` or `dgp`, provide the missing model when running again.
assert mc, "No Monte Carlo tab found in the bundle."
assert mc.document, "No Monte Carlo document found in the bundle."

n_rep = mc.document["n_rep"]
mc_result = mc.pipeline.run(
    reference=reference,
    dgp=dgp,
    n_rep=n_rep,
    n_jobs=-1,
    fail_fast=True,
    verbosity=1,
)
print("Successful reps:", mc_result.n_successful, "/", mc_result.n_rep)

jb = mc_result.test_summaries["jb_test"]
stat_ci = jb.statistic_confidence_interval(0.95)
pval_ci = jb.pval_confidence_interval(0.95)

print(stat_ci, pval_ci, sep="\n")
```

???+ info "Validating without running"
    `LoadedMC.pipeline` has already been rebuilt from the stored spec. If you still want to inspect the serialized graph directly, `validate_pipeline_spec(loaded.mc.spec, has_reference=bool(loaded.reference), has_dgp=bool(loaded.dgp))` returns `(steps, postprocs)` when the graph is well formed and raises with a specific message otherwise.

See the [Monte Carlo Guide](monte_carlo_guide.md) for the pipeline grammar and the [`monte_carlo` API reference](../documentation/monte_carlo/index.md) for the core runner exports.

## Reach the simulation prefill

Simulation prefills ride inline in the manifest, so they are reachable from `loaded.simulation` directly (no separate member). It is a `{role: SimSpec}` map, and each `SimSpec` unpacks straight into `SolvedModel.sim`.

```python
prefills = loaded.simulation  # dict[str, SimSpec] | None
assert prefills, "No simulation tab found in the bundle."


for role, spec in prefills.items():
    print(role, "\n", spec)

reference.sim(**prefills["reference"]).states["r"][:5]
```

???+ note "Determinism"
    Replaying a prefill against its model reproduces the author's intended simulation exactly. The bundle stores no simulation outputs and no live `Shock` objects. It stores only each shock's `Shock.to_dict()` parameters.

## Round-trip safety

Two properties to rely on after `load_bundle`:

1. __Manifest integrity:__ every member declared in the manifest is present in the archive (and vice versa). `BundleArchive.open` validates this on read and raises if the bundle is malformed.
2. __Format version compatibility:__ `load_bundle` will raise if a bundle format is incompatible with the library version reading it.

???+ warning "Reproducing simulations across machines"
    SymbolicDSGE bundles deterministic simulation instructions, not a complete numerical runtime. Bit-exact reproduction may depend on the execution environment, including Python, NumPy, SciPy, native BLAS/LAPACK dependencies, platform, thread settings, and CPU feature path.

    Across different machines or builds, floating-point results should be treated as numerically reproducible within appropriate tolerances rather than guaranteed bit-identical.

## Further steps

- [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md): extract the same components to disk for inspection or editing.
- [`LoadedBundle` API reference](../documentation/bundle/LoadedBundle.md).
- [Bundle Authoring Guide](bundle_authoring_guide.md): the other half of the round trip.
