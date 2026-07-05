---
tags:
    - guide
---

# Bundle Loading Guide

??? tip "__TL;DR__"
    Open a `.sdsge` bundle with `load_bundle(...)` and reach every component through the typed `LoadedBundle` fields: the re-solved `SolvedModel`s, the estimation spec, first class estimation result, observed data, posterior arrays, Monte Carlo pipeline, run output, traces, and the simulation prefill. Loading is deterministic: the policy matrices match the author's.

    You can find a demonstration notebook [here](../assets/bundle_loading.ipynb).

This guide walks through opening a `.sdsge` bundle and reaching each library object it carries: the re-solved `SolvedModel`s, the estimation spec, first class estimation result, observed data, posterior arrays, Monte Carlo pipeline, run output, traces, and the simulation prefill.

We use `experiment-1.sdsge` as produced by the [Bundle Authoring Guide](bundle_authoring_guide.md). Substitute any other bundle path.

???+ tip "What `load_bundle` actually does"
    `load_bundle` re-parses every embedded YAML, re-runs `DSGESolver.compile(**compile_kwargs).solve(**solve_kwargs)` with the kwargs recorded at write time, decodes every tabular member by `Member.format` (CSV or Parquet), and rebuilds live estimation results and Monte Carlo pipelines when present. Loading is deterministic: the resulting policy matrices match those the author had in hand.

## Open the bundle

```python
from SymbolicDSGE import load_bundle

loaded = load_bundle("experiment-1.sdsge") # (1)!
```

1. `load_bundle` is a top-level re-export of `SymbolicDSGE.bundle.build_from`. Both names are interchangeable.

`loaded` is a [`LoadedBundle`](../documentation/bundle/LoadedBundle.md). Every component is reachable through a typed field.

```python
print("Created by:", loaded.manifest.created_by)
print("Created at:", loaded.manifest.created_at)
print("Format version:", loaded.manifest.sdsge_version)
```

???+ info "Manifest provenance"
    `Manifest.checksums` carries SHA-256 hex digests over each member's bytes. This is useful for integrity checks before trusting the contents downstream.

## Reach the re-solved models

`reference` and `dgp` are full `SolvedModel` instances. They behave exactly like models you would have solved in process, including IRFs, simulation, and Kalman filtering.

```python
reference = loaded.reference
if reference is not None:
    print("Stable:", reference.policy.stab == 0)
    print("Eigenvalues:", reference.policy.eig)
    print("A shape:", reference.A.shape)
```

A quick deterministic simulation against pre-generated shocks confirms the policy round-tripped:

```python
import numpy as np

T = 20
rng = np.random.default_rng(42)
shocks = {
    "g,z": rng.standard_normal((T, 2)), # (1)!
}
sim = reference.sim(
    T=T,
    shocks=shocks,
    observables=True,
)
print(sim["Infl"][:5])
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

`estimation.result` is the first class result the run produced: an `OptimizationResult` for MLE / MAP, or an `MCMCResult` for MCMC. The loader rebuilds it from the stored metadata and, for MCMC, the `posterior` traces, so no manual reconstruction is needed.

```python
from SymbolicDSGE.estimation.results import (
    MCMCResult,
    OptimizationResult,
)

result = estimation.result
if isinstance(result, OptimizationResult):
    print("Point estimate:", result.theta)
    print("Log-posterior:", result.logpost)
elif isinstance(result, MCMCResult):
    print("Acceptance:", result.accept_rate)
    print("Draws:", result.n_draws, "burn-in:", result.burn_in)
    print("HPD intervals:", result.hpd_intervals(alpha=0.05))
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
2. The same arrays already power `result.samples` / `result.logpost_trace`; `estimation.posterior` exposes them raw for callers who want the columns directly. The `logpost` key holds the 1-D log-posterior trace.

???+ tip "MCMC diagnostics are ready to use"
    A loaded MCMC `result` is a live `MCMCResult`. The loader already paired the metadata with the `posterior` traces. Call diagnostics on it directly (`result.hpd_intervals(...)`, `result.posterior_traces()`, `result.joint_hpd_set(...)`); there is no rebuild step.

### Re-run an estimation from a loaded bundle

`EstimationSpec.to_estimator_inputs()` lowers the loaded spec to concrete arguments: `estimated_params`, `theta0`, `bounds`, and `priors` as built `Prior` objects. Pass these to `DSGESolver.estimate(...)` when you want to reproduce the run or when a bundle stored the spec without a result. The lowering lives in the core library, so no `[ui]` extra is required.

```python
from SymbolicDSGE import DSGESolver

inputs = estimation.spec.to_estimator_inputs() # (1)!

solver = DSGESolver(loaded.reference.config, loaded.reference.kalman_config)
extras: dict = {} # (2)!
if inputs.bounds is not None and estimation.spec.method != "mcmc":
    extras["bounds"] = inputs.bounds

fresh_result = solver.estimate(
    compiled=loaded.reference.compiled, # (3)!
    y=estimation.observed, # (4)!
    method=estimation.spec.method,
    estimated_params=inputs.estimated_params,
    theta0=inputs.theta0,
    priors=inputs.priors,
    observables=estimation.spec.observables,
    **extras,
)
```

1. Selects `estimate=True` parameters, materializes their initials/bounds, and (for MAP/MCMC) builds a `Prior` object from each `PriorSpec`. Raises if MAP/MCMC parameters lack a prior.
2. `solver.estimate` forwards `**method_kwargs` to the underlying `mle`/`map`/`mcmc` call. `bounds` is accepted by MLE/MAP but not by MCMC, so we gate it on the method.
3. The `CompiledModel` reuses the layout `load_bundle` already produced when re-solving the embedded YAML. No recompile is needed.
4. The observed matrix is the data the original run was fit against. It is already reconstructed by `load_bundle` and stored on `LoadedEstimation.observed`.

???+ info "Why `to_estimator_inputs` exists"
    The spec is human-authored (or GUI-authored): it carries `PriorSpec` for declarative reasons. The estimator needs built `Prior` objects. `to_estimator_inputs` is the seam where the materialization happens, and where MAP/MCMC's prior-required invariant is enforced.

See the [Estimation Guide](estimation_guide.md) for the run methods in detail.

## Reach the Monte Carlo tab

`LoadedMC.pipeline` is the first class [`MCPipeline`](../documentation/monte_carlo/pipeline.md) rebuilt at load time. `LoadedMC.spec` remains available for archive inspection and UI rendering, and `LoadedMC.resources` holds the side-channel arrays or custom callables that were reattached while rebuilding the pipeline. When the bundle carries a completed run, `document` holds the trace-free summary and `traces` holds the bulk columns. The convenience method `wire()` re-merges those pieces into the canonical UI shape.

```python
mc = loaded.mc

if mc is not None:
    print("Runtime steps:", [step.name for step in mc.pipeline.per_rep_steps])
    print("Stored graph nodes:", [n.id for n in mc.spec.nodes])

    if mc.document is not None:
        wire = mc.wire() # (1)!
        print("Run kind:", wire["kind"])
```

1. `mc.wire()` returns `None` when either `document` or `traces` is missing. A bundle authored with only the pipeline carries neither.

### Re-run a Monte Carlo pipeline from a loaded bundle

The loaded pipeline runs against the loaded models without the `[ui]` extra.

```python
mc_result = loaded.mc.pipeline.run(
    reference=loaded.reference,
    dgp=loaded.dgp,
    n_rep=500,
    retain_payloads=False, # (1)!
    retain_test_results=False,
    retain_contexts=False,
    fail_fast=True,
    verbosity=0,
)
print("Successful reps:", mc_result.n_successful, "/", mc_result.n_rep)
```

1. These retention flags keep the rerun result compact. Summaries and traces are still produced.

???+ info "Validating without running"
    `LoadedMC.pipeline` has already been rebuilt from the stored spec. If you still want to inspect the serialized graph directly, `validate_pipeline_spec(loaded.mc.spec, has_reference=loaded.reference is not None, has_dgp=loaded.dgp is not None)` returns `(ordered, postprocs)` when the graph is well formed and raises with a specific message otherwise.

See the [Monte Carlo Guide](monte_carlo_guide.md) for the pipeline grammar and the [`monte_carlo` API reference](../documentation/monte_carlo/index.md) for the core runner exports.

## Reach the simulation prefill

Simulation prefills ride inline in the manifest, so they are reachable from `loaded.simulation` directly (no separate member). It is a `{role: SimSpec}` map, and each `SimSpec` unpacks straight into `SolvedModel.sim`.

```python
prefills = loaded.simulation  # dict[str, SimSpec] | None

if prefills is not None:
    for role, spec in prefills.items():
        print(role, "T:", spec.T, "| shocks:", list((spec.shocks or {}).keys()))

    # A SimSpec is a Mapping over sim's keyword arguments, so replay is a splat.
    reference_spec = prefills.get("reference")
    if reference_spec is not None and loaded.reference is not None:
        result = loaded.reference.sim(**reference_spec)
```

???+ note "Determinism"
    Replaying a `SimSpec` against its model reproduces the author's intended simulation exactly. The bundle stores no simulation outputs and no live `Shock` objects — only each shock's `Shock.to_dict()` parameters. `sim` rebuilds the `Shock` and materializes a `T`-horizon draw, so a fixed seed yields identical paths for author and receiver.

## Round-trip safety

Two properties to rely on after `load_bundle`:

1. **Manifest integrity**: every member declared in the manifest is present in the archive (and vice versa). `BundleArchive.open` validates this on read and raises if the bundle is malformed.
2. **Format-version compatibility**: a newer-than-supported bundle raises immediately with a clear message. Older bundles read forward without intervention.

???+ warning "Reproducing simulations across machines"
    Deterministic reproduction requires the receiver run the same numpy / SciPy versions on the same platform. The bundle does not pin those, so record them externally if exact reproducibility across heterogeneous environments matters.

## Further steps

- [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md): extract the same components to disk for inspection or editing.
- [`LoadedBundle` API reference](../documentation/bundle/LoadedBundle.md).
- [Bundle Authoring Guide](bundle_authoring_guide.md): the other half of the round trip.
