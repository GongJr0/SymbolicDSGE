---
tags:
    - guide
---

# Bundle Authoring Guide

This guide walks through assembling a complete `.sdsge` bundle from code — model + Kalman config, estimation spec and result, observed data, Monte Carlo pipeline and traces, simulation prefill. Every member type the bundle supports is covered.

We start from `MODELS/POST82.yaml` (also used in the [Quick Start](quickstart.md)). The full assembly fits in a single script you can copy and run.

???+ tip "Where the bundle ends up"
    The example writes `experiment-1.sdsge` in the current directory. Open it later via [`load_bundle`](../documentation/bundle/load_bundle.md), inspect components with [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md), or open it directly in the GUI with `sdsge-ui experiment-1.sdsge`.

## Solve a reference model

```python
from SymbolicDSGE import DSGESolver, ModelParser
from numpy import array, float64

parser = ModelParser("MODELS/POST82.yaml") # (1)!
model, kalman = parser.get_all()

solver = DSGESolver(model, kalman)
compiled = solver.compile(
    linearize=False, # (2)!
)
sol = solver.solve(
    compiled,
    steady_state=array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float64),
)
```

1. `ModelParser` populates `ModelConfig.source_yaml` automatically; the bundle re-uses it without re-reading the file.
2. Set `True` for models authored in nonlinear levels — see the [Quick Start](quickstart.md#compilation) for details.

???+ note "Why we solve first"
    The bundle preserves the YAML and the recorded compile/solve kwargs, then re-runs them at load time. Solving here is only required if we want to attach an estimation, an MC pipeline, or a `SolvedModel`-derived result. Author-only bundles do not need a live `SolvedModel`.

## Specify the estimation tab

We define a small MAP run estimating `psi_pi` and `psi_x` against synthetic observed data. The spec captures the run shape and parameter priors; the result captures the optimization outcome. Observed data is fed in as a numpy matrix and stored as a CSV column-pair with semantic headers.

```python
import numpy as np

from SymbolicDSGE.estimation.spec import (
    EstimationParameterSpec,
    EstimationSpec,
    OptimizationResultMeta,
    PriorSpec,
)

estimation_spec = EstimationSpec(
    method="map", # (1)!
    parameters=[
        EstimationParameterSpec(
            name="psi_pi",
            initial=1.5,
            estimate=True,
            lower=1.0,
            upper=3.0,
            prior=PriorSpec(
                distribution="normal",
                parameters={"loc": 1.5, "scale": 0.25},
            ),
        ),
        EstimationParameterSpec(
            name="psi_x",
            initial=0.5,
            estimate=True,
            lower=0.0,
            upper=1.0,
            prior=PriorSpec(
                distribution="normal",
                parameters={"loc": 0.5, "scale": 0.2},
            ),
        ),
    ],
    observables=["Infl", "Rate"], # (2)!
    method_kwargs={"options": {"maxiter": 50}},
)

rng = np.random.default_rng(0)
observed = rng.standard_normal((40, 2)) # (3)!
estimation_result_meta = OptimizationResultMeta(
    kind="map",
    theta={"psi_pi": 1.48, "psi_x": 0.55}, # (4)!
    success=True,
    message="Optimization converged.",
    fun=-87.3,
    loglik=-85.1,
    logprior=-2.2,
    logpost=-87.3,
    nfev=124,
    nit=14,
)
```

1. `"mle"`, `"map"`, or `"mcmc"`. MAP/MCMC require priors; MLE does not.
2. Order matters — the loader cross-checks these against the model's declared observables at compile time.
3. Synthetic data shaped `(n_periods, n_observables)`. Substitute the real `y` matrix you fitted the model to.
4. The bundle stores only the result metadata, not the raw `scipy.optimize.OptimizeResult` — `theta` carries the same information by parameter name.

???+ info "Source of the result metadata"
    A live `OptimizationResult` from `Estimator.run(...)` projects to `OptimizationResultMeta` field-for-field. The same applies to `MCMCResult` ↔ `MCMCResultMeta` — except MCMC also requires the `samples` and `logpost` bulk arrays as a separate `posterior` dict.

## Build a Monte Carlo pipeline

`PipelineSpec` is the pydantic-free representation of an MC graph. Here we attach the pipeline spec without a completed run; the loader returns the spec for the GUI to render and the user to fill in `n_rep` and click **Run**.

```python
from SymbolicDSGE.monte_carlo.spec import EdgeSpec, NodeSpec, PipelineSpec

mc_pipeline = PipelineSpec(
    nodes=[
        NodeSpec(
            id="sim",
            step_type="simulation",
            name="DGP Simulation",
            params={"T": 100},
        ),
        NodeSpec(
            id="jb",
            step_type="jarque_bera",
            name="Normality Check",
            params={"source": "observables"},
        ),
    ],
    edges=[EdgeSpec(source="sim", target="jb")], # (1)!
)
```

1. Edges describe the directed graph of step dependencies. See the [Monte Carlo Guide](monte_carlo_guide.md) for the full pipeline grammar.

???+ note "Attaching a completed MC run"
    If you already have an `MCPipelineResult` from a live run, pass it as `result=` to `add_mc(...)` below. `BundleBuilder` splits the trace-free document from the bulk traces internally — no manual unpacking required.

## Specify a simulation prefill

`SimSpec` rides inline in the manifest. It controls what the GUI's Outputs tab pre-fills when the receiver opens the bundle.

```python
from SymbolicDSGE.bundle import ShockGeneration, SimSpec

simulation = SimSpec(
    role="reference",
    T=25,
    observables=True,
    shock_scale=1.0,
    shock_generation=ShockGeneration(
        dist="norm",
        seed=42, # (1)!
        loc=0.0,
    ),
)
```

1. The seed makes the replayed simulation deterministic — both the bundle author and the receiver produce identical paths when clicking **Run**.

## Assemble and write

`BundleBuilder` chains every component into one archive. Each `add_*` call returns `self`; the final `.write(path)` materializes the bundle and returns the path written.

```python
from SymbolicDSGE import BundleBuilder

bundle_path = (
    BundleBuilder(created_by="experiment-1") # (1)!
    .add_model(
        "reference",
        model.source_yaml, # (2)!
        compile_kwargs={"linearize": False},
    )
    .add_estimation(
        estimation_spec,
        result=estimation_result_meta,
        observed=observed,
        observable_names=["Infl", "Rate"], # (3)!
    )
    .add_mc(mc_pipeline)
    .set_simulation(simulation)
    .write("experiment-1.sdsge")
)

print("Bundle written to:", bundle_path)
```

1. `created_by` is purely metadata recorded in the manifest. Defaults to `"SymbolicDSGE <version>"` when omitted.
2. `model.source_yaml` is the YAML text retained on the `ModelConfig` by `ModelParser`. For programmatically constructed configs you would supply the YAML as a string here directly.
3. The names must match the model's `observables` in count and order — the loader validates this at compile time and refuses to write a mismatched bundle.

???+ warning "Observable name validation"
    Order is significant. `["Rate", "Infl"]` is rejected even if `["Infl", "Rate"]` would have been accepted — the bundle's downstream consumers index observed data by column position, so reordering would silently produce wrong inference. See the [`sdsge-compile` validation reference](../portable_experiments/sdsge-compile.md#validation) for the message format.

## Add raw data alongside the model

`add_raw_data` covers any extra CSV files you want to ship in `data/`. They are not interpreted by the loader — they are passthrough storage for context the receiver may want.

```python
import io
import pandas as pd

aux = pd.DataFrame({
    "date": pd.date_range("2000-01-01", periods=40, freq="Q"),
    "gdp_growth": rng.standard_normal(40),
})
csv_buffer = io.StringIO()
aux.to_csv(csv_buffer, index=False)

BundleBuilder() \
    .add_model("reference", model.source_yaml) \
    .add_raw_data("auxiliary_series", csv_buffer.getvalue()) \
    .write("with-raw-data.sdsge")
```

???+ info "CSV vs Parquet for raw data"
    `add_raw_data` re-encodes CSV input as Parquet by default. Pass `as_parquet=False` to embed the CSV verbatim — useful for hand-zip-friendly bundles.

## Inspect the result

The bundle is a zip — every member is human-inspectable.

```bash
unzip -l experiment-1.sdsge
```

```text
Archive:  experiment-1.sdsge
  Length      Date    Time    Name
---------  ---------- -----   ----
     ...    ...        ...    manifest.json
     ...    ...        ...    model/reference.yaml
     ...    ...        ...    estimation/spec.json
     ...    ...        ...    estimation/result.json
     ...    ...        ...    estimation/observed.parquet
     ...    ...        ...    montecarlo/pipeline.json
```

For a structured view, decompile it:

```bash
sdsge-decompile experiment-1.sdsge -o experiment-1/
```

Or open it directly in Python — see the [Bundle Loading Guide](bundle_loading_guide.md).

## Further steps

- [`sdsge-compile`](../portable_experiments/sdsge-compile.md) — the directory-driven CLI equivalent.
- [`SolvedModel.save_sdsge`](../documentation/SolvedModel.md) — one-shot bundle write for the model-only case.
- [`SolvedModel.to_bundle_builder`](../documentation/SolvedModel.md) — pre-seeded builder factory, equivalent to the chain in this guide.
- [`BundleBuilder` API reference](../documentation/bundle/BundleBuilder.md).
