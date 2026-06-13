---
tags:
    - guide
---

# Bundle Loading Guide

??? tip "__TL;DR__"
    Open a `.sdsge` bundle with `load_bundle(...)` and reach every component through the typed `LoadedBundle` fields — the re-solved `SolvedModel`s, the estimation spec / result / observed data / posterior, the Monte Carlo pipeline / result / traces, and the simulation prefill. Loading is deterministic: the policy matrices match the author's.

    You can find a demonstration notebook [here](../assets/bundle_loading.ipynb).

This guide walks through opening a `.sdsge` bundle and reaching each library object it carries — the re-solved `SolvedModel`s, the estimation spec / result / observed data / posterior, the Monte Carlo pipeline / result / traces, and the simulation prefill.

We use `experiment-1.sdsge` as produced by the [Bundle Authoring Guide](bundle_authoring_guide.md). Substitute any other bundle path.

???+ tip "What `load_bundle` actually does"
    `load_bundle` re-parses every embedded YAML, re-runs `DSGESolver.compile(**compile_kwargs).solve(**solve_kwargs)` with the kwargs recorded at write time, and decodes every tabular member by `Member.format` (CSV or Parquet). Loading is deterministic — the resulting policy matrices match those the author had in hand.

## Open the bundle

```python
from SymbolicDSGE import load_bundle

loaded = load_bundle("experiment-1.sdsge") # (1)!
```

1. `load_bundle` is a top-level re-export of `SymbolicDSGE.bundle.build_from`. Both names are interchangeable.

`loaded` is a [`LoadedBundle`](../documentation/bundle/LoadedBundle.md) — every component is reachable through a typed field.

```python
print("Created by:", loaded.manifest.created_by)
print("Created at:", loaded.manifest.created_at)
print("Format version:", loaded.manifest.sdsge_version)
```

???+ info "Manifest provenance"
    `Manifest.checksums` carries SHA-256 hex digests over each member's bytes — useful for integrity checks before trusting the contents downstream.

## Reach the re-solved models

`reference` and `dgp` are full `SolvedModel` instances. They behave exactly like models you would have solved in-process — including IRFs, simulation, and Kalman filtering.

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

1. `estimation.spec` is an [`EstimationSpec`](../documentation/bundle/index.md#estimation-result-metadata) instance — round-trippable to / from JSON via `to_dict()` / `from_dict()`.

The result metadata discriminates by type. For MLE / MAP runs the result is an `OptimizationResultMeta`; for MCMC it is an `MCMCResultMeta` paired with bulk traces in `estimation.posterior`.

```python
from SymbolicDSGE.estimation.spec import (
    MCMCResultMeta,
    OptimizationResultMeta,
)

result = estimation.result
if isinstance(result, OptimizationResultMeta):
    print("Point estimate:", result.theta)
    print("Log-posterior:", result.logpost)
elif isinstance(result, MCMCResultMeta):
    print("Acceptance:", result.accept_rate)
    print("Draws:", result.n_draws, "burn-in:", result.burn_in)
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
2. Pair with `result.param_names` for column-to-parameter mapping. The `logpost` key holds the 1-D log-posterior trace.

???+ warning "MCMCResult reconstruction"
    `MCMCResultMeta` carries the scalar slice only. To rebuild a live `MCMCResult` for sampling diagnostics, construct it explicitly:

    ```python
    from SymbolicDSGE.estimation.results import MCMCResult

    mcmc = MCMCResult(
        param_names=result.param_names,
        samples=estimation.posterior["samples"],
        logpost_trace=estimation.posterior["logpost"],
        accept_rate=result.accept_rate,
        n_draws=result.n_draws,
        burn_in=result.burn_in,
        thin=result.thin,
    )
    mcmc.hpd_intervals(alpha=0.05)
    ```

???+ info "Re-running a loaded estimation"
    `loaded.estimation.spec.to_estimator_inputs()` lowers the spec to concrete `Estimator` arguments — `estimated_params`, `theta0`, `bounds`, and `priors` rebuilt from their specs — so the run can be replayed (paired with `loaded.estimation.observed`) without the `[ui]` extra. See the [Estimation Guide](estimation_guide.md).

## Reach the Monte Carlo tab

`LoadedMC.spec` is the [`PipelineSpec`](../documentation/monte_carlo/pipeline.md) describing the graph. When the bundle carries a completed run, `document` holds the trace-free summary and `traces` holds the bulk columns. The convenience method `wire()` re-merges them into the canonical UI shape.

```python
mc = loaded.mc

if mc is not None:
    print("Pipeline nodes:", [n.id for n in mc.spec.nodes])
    print("Pipeline edges:", [(e.source, e.target) for e in mc.spec.edges])

    if mc.document is not None:
        wire = mc.wire() # (1)!
        print("Run kind:", wire["kind"])
```

1. `mc.wire()` returns `None` when either `document` or `traces` is missing — a bundle authored with only the pipeline spec carries neither.

???+ info "Re-running a loaded pipeline"
    The compile path lives in the core `monte_carlo` module, so a loaded pipeline can be re-run against the loaded models **without the `[ui]` extra**:

    ```python
    from SymbolicDSGE.monte_carlo import run_pipeline

    result = run_pipeline(
        loaded.mc.spec, # (1)!
        reference=loaded.reference,
        dgp=loaded.dgp,
        n_rep=500,
        fail_fast=True,
    )
    ```

    1. `run_pipeline` validates the graph, compiles each step through the step catalogue, and runs it — `validate_pipeline_spec` / `build_pipeline` are also exported if you want the stages separately.

    See the [Monte Carlo Guide](monte_carlo_guide.md) for the pipeline grammar.

## Reach the simulation prefill

The `SimSpec` rides inline in the manifest, so it is reachable from `loaded.simulation` directly (no separate member).

```python
sim_spec = loaded.simulation

if sim_spec is not None:
    print("T:", sim_spec.T)
    print("Observables flag:", sim_spec.observables)
    if sim_spec.shock_generation is not None:
        print("Seed:", sim_spec.shock_generation.seed)
        print("Distribution:", sim_spec.shock_generation.dist)
```

???+ note "Determinism"
    Replaying `sim_spec` against `loaded.reference` reproduces the author's intended simulation exactly. The bundle stores no simulation outputs — they are reconstructed on demand from `(SolvedModel, seed, shock spec)`.

## Round-trip safety

Two properties to rely on after `load_bundle`:

1. **Manifest integrity**: every member declared in the manifest is present in the archive (and vice versa). `BundleArchive.open` validates this on read and raises if the bundle is malformed.
2. **Format-version compatibility**: a newer-than-supported bundle raises immediately with a clear message. Older bundles read forward without intervention.

???+ warning "Reproducing simulations across machines"
    Deterministic reproduction requires the receiver run the same numpy / SciPy versions on the same platform. The bundle does not pin those — record them externally if exact reproducibility across heterogeneous environments matters.

## Further steps

- [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md) — extract the same components to disk for inspection or editing.
- [`LoadedBundle` API reference](../documentation/bundle/LoadedBundle.md).
- [Bundle Authoring Guide](bundle_authoring_guide.md) — the other half of the round-trip.
