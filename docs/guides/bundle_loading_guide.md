---
tags:
    - guide
---

# Bundle Loading Guide

??? tip "__TL;DR__"
    Open a `.sdsge` bundle with `load_bundle(...)` and reach every component through the typed `LoadedBundle` fields: the solved `SolvedModel`s, the estimation spec, first class estimation result, observed data, posterior arrays, Monte Carlo pipeline, run output, traces, and the simulation prefill. Loading is deterministic: the policy matrices match the author's.

    You can find a demonstration notebook [here](../assets/bundle_loading.ipynb).

This guide walks through opening a `.sdsge` bundle and reaching each library object it carries: the `SolvedModel`s, the estimation spec, estimation result, observed data, posterior arrays, Monte Carlo pipeline, run output, traces, and the simulation prefill.

We use `experiment-1.sdsge` as produced by the [Bundle Authoring Guide](bundle_authoring_guide.md). Substitute any other bundle path.

???+ tip "What `load_bundle` actually does"
    `load_bundle` parses every embedded YAML, runs `DSGESolver.compile(**compile_kwargs).solve(**solve_kwargs)` with the kwargs recorded at write time, decodes every tabular member by `Member.format` (CSV or Parquet), and rebuilds live estimation results and Monte Carlo pipelines when present. Loading is deterministic: the resulting policy matrices match those the author had in hand.

## Open the bundle

```python
import numpy as np

from SymbolicDSGE import load_bundle
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.estimation.results import MCMCResult, MAPResult, MLEResult

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

A quick deterministic simulation against generated shocks confirms the policy round trip:

```python
T = 20
rng = np.random.default_rng(42)
shocks = {
    "e_g,e_z": rng.standard_normal((T, 2)), # (1)!
}
sim = reference.sim(
    T=T,
    shocks=shocks,
    observables=True,
)
print(sim.observables["Infl"][:5])
```

1. See [`SolvedModel.sim`](../documentation/SolvedModel.md) for the shock specification grammar.

???+ note "DGP slot may be absent"
    `loaded.dgp` is `None` whenever the bundle did not carry a `dgp.yaml`. Test before use.

## Reach the estimation tab

`LoadedEstimation` carries every text and bulk component of the estimation run.

```python
estimation = loaded.estimation

if estimation is not None:
    print("Method:", estimation.spec.method) # (1)!
    print("Observables:", estimation.spec.observables)
    print("Parameters:", [p.name for p in estimation.spec.parameters])
```

1. `estimation.spec` is an [`EstimationSpec`](../documentation/bundle/index.md#estimation-spec-and-result-types) instance. It round trips to and from JSON via `to_dict()` / `from_dict()`.

`estimation.result` is the first class result the run produced: a `MLEResult` for MLE, a `MAPResult` for MAP, or an `MCMCResult` for MCMC. The loader rebuilds it from the stored metadata and, for MCMC, the `posterior` traces, so no manual reconstruction is needed.

```python
result = estimation.result
if isinstance(result, MCMCResult):
    print("Acceptance:", round(result.accept_rate, 2))
    print("Draws:", result.n_draws, "burn-in:", result.burn_in)
elif isinstance(result, MAPResult):
    print("Point estimate:", result.theta)
    print("Log-posterior:", result.logpost)
elif isinstance(result, MLEResult):
    print("Point estimate:", result.theta)
    print("Log-likelihood:", result.loglik)
```

Observed data and (when present) MCMC posterior are numpy arrays decoded from the embedded CSV or Parquet member.

```python
if estimation.observed is not None:
    print("Observed shape:", estimation.observed.shape) # (1)!
if estimation.posterior is not None:
    samples = estimation.posterior["samples"] # (2)!
    print("Posterior mean:", samples.mean(axis=0))
```

1. Shape is `(n_periods, n_observables)` with column order matching `estimation.spec.observables`.
2. The same arrays already power `result.samples` / `result.logpost_trace`; `estimation.posterior` exposes them raw for callers who want the columns directly. The `logpost` key holds the one dimensional log posterior trace.

???+ tip "MCMC diagnostics are ready to use"
    A loaded MCMC `result` is a live `MCMCResult`. The loader already paired the metadata with the `posterior` traces. Call diagnostics on it directly (`result.hpd_intervals(...)`, `result.posterior_traces()`, `result.joint_hpd_set(...)`); there is no rebuild step.

### Run an estimation from a loaded bundle

`EstimationSpec.to_estimator_inputs()` lowers the loaded spec to concrete arguments: `estimated_params`, `theta0`, `bounds`, and `priors` as built `Prior` objects. Pass these to `DSGESolver.estimate(...)` when you want to reproduce the run or when a bundle stored the spec without a result. The lowering lives in the core library, so no `[ui]` extra is required.

```python
inputs = estimation.spec.to_estimator_inputs() # (1)!
inputs
```

1. Selects `estimate=True` parameters, materializes their initials/bounds, and, for MAP/MCMC, builds a `Prior` object from each `PriorSpec`. Raises if MAP/MCMC parameters lack a prior.
2. `solver.estimate` forwards `**method_kwargs` to the underlying `mle`/`map`/`mcmc` call. `bounds` is accepted by MLE/MAP but not by MCMC, so we gate it on the method.
3. The `CompiledModel` reuses the layout `load_bundle` already produced when solving the embedded YAML. No recompile is needed.
4. The observed matrix is the data the original run was fit against. It is already reconstructed by `load_bundle` and stored on `LoadedEstimation.observed`.

???+ info "Why `to_estimator_inputs` exists"
    The spec is authored by users or the GUI, so it carries `PriorSpec` for declarative reasons. The estimator needs built `Prior` objects. `to_estimator_inputs` is where the materialization happens, and where MAP/MCMC's prior invariant is enforced.

See the [Estimation Guide](estimation_guide.md) for the run methods in detail.

## Reach the Monte Carlo tab

`LoadedMC.pipeline` is the first class [`MCPipeline`](../documentation/monte_carlo/pipeline.md) rebuilt at load time. `LoadedMC.spec` remains available for archive inspection and UI rendering, and `LoadedMC.resources` holds the side channel arrays or custom callables that were reattached while rebuilding the pipeline. When the bundle carries a completed run, `document` holds the trace free summary and `traces` holds the bulk columns.

```python
mc = loaded.mc

if mc is not None:
    print("Runtime Steps:", [step.name for step in mc.pipeline.per_rep_steps])
    print("Post-Processing:", [step.name for step in mc.pipeline.postproc_steps])
```

### Run a Monte Carlo pipeline from a loaded bundle

The loaded pipeline runs against the loaded models without the `[ui]` extra.

```python
# The pipeline's simulation datagen needs a DGP. The authoring notebook
# bundles a model under role "dgp", so `loaded.dgp` resolves. If a bundle
# omits `reference` or `dgp`, provide the missing model when running again.
n_rep = mc.document["n_rep"]
mc_result = mc.pipeline.run(
    reference=reference,
    dgp=dgp,
    n_rep=n_rep,
    n_jobs=-1,  # (1)!
    fail_fast=True,
    verbosity=1,
)
print("Successful reps:", mc_result.n_successful, "/", mc_result.n_rep)

jb = mc_result.test_summaries["jb_test"]
stat_ci = jb.statistic_confidence_interval(0.95)
pval_ci = jb.pval_confidence_interval(0.95)

list(zip(stat_ci, pval_ci))
```

1. The rerun executes the same native loop the author ran. `n_jobs=-1` fans the replications across all available cores; `None` keeps it single-threaded.

???+ info "Validating without running"
    `LoadedMC.pipeline` has already been rebuilt from the stored spec. If you still want to inspect the serialized graph directly, `validate_pipeline_spec(loaded.mc.spec, has_reference=loaded.reference is not None, has_dgp=loaded.dgp is not None)` returns `(ordered, postprocs)` when the graph is well formed and raises with a specific message otherwise.

See the [Monte Carlo Guide](monte_carlo_guide.md) for the pipeline grammar and the [`monte_carlo` API reference](../documentation/monte_carlo/index.md) for the core runner exports.

## Reach the simulation prefill

Simulation prefills ride inline in the manifest, so they are reachable from `loaded.simulation` directly (no separate member). It is a `{role: SimSpec}` map, and each `SimSpec` unpacks straight into `SolvedModel.sim`.

```python
prefills = loaded.simulation  # dict[str, SimSpec] | None

if prefills is not None:
    for role, spec in prefills.items():
        print(role, "\n", spec)

reference.sim(**prefills["reference"]).states["r"][:5]
```

???+ note "Determinism"
    Replaying a `SimSpec` against its model reproduces the author's intended simulation exactly. The bundle stores no simulation outputs and no live `Shock` objects. It stores only each shock's `Shock.to_dict()` parameters. `sim` rebuilds the `Shock` and materializes a `T` horizon draw, so a fixed seed yields identical paths for author and receiver.

## Round-trip safety

Two properties to rely on after `load_bundle`:

1. **Manifest integrity**: every member declared in the manifest is present in the archive (and vice versa). `BundleArchive.open` validates this on read and raises if the bundle is malformed.
2. **Format version compatibility**: a bundle newer than the installed reader supports raises immediately with a clear message. Older bundles read forward without intervention.

???+ warning "Reproducing simulations across machines"
    SymbolicDSGE bundles deterministic simulation instructions, not a complete numerical runtime. Bit-exact reproduction requires the same execution environment, including Python, NumPy, SciPy, native BLAS/LAPACK dependencies, platform, thread settings, and CPU feature path.

    Across different machines or builds, floating-point results should be treated as numerically reproducible within appropriate tolerances rather than guaranteed bit-identical.

## Further steps

- [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md): extract the same components to disk for inspection or editing.
- [`LoadedBundle` API reference](../documentation/bundle/LoadedBundle.md).
- [Bundle Authoring Guide](bundle_authoring_guide.md): the other half of the round trip.
