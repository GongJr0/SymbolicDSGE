---
tags:
    - doc
---
# Bundle

The `bundle` module produces and consumes `.sdsge` archives: versioned zip files containing a model and Kalman configuration, optional estimation spec/result/data, optional Monte Carlo pipeline/result, and an optional simulation prefill. Authored bundles are portable: a receiver only needs `pip install SymbolicDSGE` to reach every component the bundle carries, and `pip install "SymbolicDSGE[ui]"` to hydrate the GUI with one.

For task-oriented walkthroughs see the [Bundle Authoring Guide](../../guides/bundle_authoring_guide.md) and [Bundle Loading Guide](../../guides/bundle_loading_guide.md). For the CLI counterparts see the [Portable Experiments](../../portable_experiments/index.md) section.

???+ note "Top-level imports"
    `BundleBuilder`, `LoadedBundle`, and `load_bundle` are re-exported at `SymbolicDSGE` root. Everything else in this section lives under `SymbolicDSGE.bundle`.

## Module layout

| Class / function | Description |
| --- | --- |
| [`BundleBuilder`](BundleBuilder.md) | Fluent assembler for the in-code authoring path. |
| [`load_bundle`](load_bundle.md) | Open a `.sdsge` and reconstruct its components into Python objects. |
| [`LoadedBundle`](LoadedBundle.md) | Container holding every component returned by `load_bundle`. |
| [`Manifest`](Manifest.md) | Versioned archive index for `manifest.json`. |

## Estimation spec and result types

Estimation specs and archive metadata live in `SymbolicDSGE.estimation.spec`. Live result objects live in `SymbolicDSGE.estimation.results`. The writer accepts live `OptimizationResult` and `MCMCResult` objects, stores their metadata representation, and `load_bundle` reconstructs live result objects on read.

| Class / function | Purpose |
| --- | --- |
| `EstimationSpec` | Serializable spec for an estimation run (method, parameters, observables, kwargs, posterior point). |
| `EstimationSpec.from_targets(...)` | Build a spec from just the parameter names, initials, priors, and bounds. Mirrors `DSGESolver.estimate` inputs and sets `estimate=True` for you. |
| `EstimationSpec.to_estimator_inputs()` | Lower a spec to concrete `EstimatorInputs` for a run. Builds `Prior` objects from each `PriorSpec`. |
| `EstimatorInputs` | Concrete arguments lowered from `EstimationSpec`: `estimated_params`, `theta0`, `priors`, and `bounds`. Directly feedable to `DSGESolver.estimate(...)`. |
| `OptimizationResultMeta` | Archive metadata for `OptimizationResult` (`kind`, `theta`, `success`, `message`, `fun`, `loglik`, `logprior`, `logpost`, `nfev`, `nit`). |
| `MCMCResultMeta` | Archive metadata for `MCMCResult` (`param_names`, `accept_rate`, `n_draws`, `burn_in`, `thin`). Bulk `samples` and `logpost_trace` ride a Parquet member alongside the metadata and pair with it at load time. |

Live `OptimizationResult` and `MCMCResult` carry `.to_meta()` projections; `MCMCResult.posterior_arrays()` returns the bulk `{"samples", "logpost"}` dict the bundle expects. See [`Estimator.to_spec`](../Estimator.md) for the estimator-side spec projection.

???+ tip "Authoring fast paths"
    - `Estimator.to_spec(method="map", priors={...})` snapshots an `Estimator`'s configuration into an `EstimationSpec` for bundling.
    - `BundleBuilder.add_estimation(estimator, result=result)` accepts the live result directly. No manual `*Meta` construction is needed.

## Convention summary

| Topic | Behavior |
| --- | --- |
| Format version | `Manifest.sdsge_version`; readers reject bundles with a newer-than-supported version. |
| Compression | Parquet members are stored uncompressed (`ZIP_STORED`); text members deflate. |
| Determinism | Bundle stores simulation specs + seed, not simulation results. Reproducibility relies on numpy `PCG64`. |
| Round trip | Model YAML re-parsed and re-solved with stored `compile_kwargs` and `solve_kwargs`. |

???+ warning "Bundle format scope"
    `.sdsge` is designed for sharing model setups and result snapshots, not as a long term data archive. The format is versioned; older versions read forward, but bundles from newer unsupported versions are rejected. Keep the producing library version recorded alongside the bundle for reproducibility.
