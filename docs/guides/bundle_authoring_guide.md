---
tags:
    - guide
---

# Bundle Authoring Guide

??? tip "__TL;DR__"
    Assemble a complete `.sdsge` bundle from code with `BundleBuilder` — chain `add_model`, `add_estimation`, `add_mc`, `set_simulation`, and `add_raw_data`, then `.write(...)` the archive. Each member is optional, so the same builder covers a model-only bundle and a full experiment alike.

    You can find a demonstration notebook [here](../assets/bundle_authoring.ipynb).

This guide walks through assembling a complete `.sdsge` bundle from code — model + Kalman config, estimation spec and result, observed data, Monte Carlo pipeline and traces, simulation prefill. Every member type the bundle supports is covered.

We start from `MODELS/POST82.yaml` (also used in the [Quick Start](quickstart.md)). The full assembly fits in a single script you can copy and run.

???+ tip "Where the bundle ends up"
    The example writes `experiment-1.sdsge` in the current directory. Open it later via [`load_bundle`](../documentation/bundle/load_bundle.md), inspect components with [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md), or open it directly in the GUI with `sdsge-ui experiment-1.sdsge`.

## Solve a reference model

We initialize a `BundleBuilder` to store all upcoming components. A model is then defined and solved to be attached to the bundle. You can refer to the [Quick Start](quickstart.md) for details on basic model authoring and solving.

```python
from SymbolicDSGE import DSGESolver, ModelParser, BundleBuilder
from numpy import array, float64

bundle = BundleBuilder(created_by="Central Bank") # (1)!

parser = ModelParser("MODELS/POST82.yaml") # (2)!
model, kalman = parser.get_all()

solver = DSGESolver(model, kalman)
compiled = solver.compile(
    linearize=False, # (3)!
)
sol = solver.solve(
    compiled,
    steady_state=[0.0, 0.0, 0.0, 0.0, 0.0],
)

bundle.add_model(
    "reference",
    model.source_yaml, # (4)!
    compile_kwargs={"linearize": False},
    solve_kwargs={"steady_state": [0.0, 0.0, 0.0, 0.0, 0.0]},
)

bundle.add_model(
    "dgp",
    model.source_yaml,
    compile_kwargs={"linearize": False},
    solve_kwargs={"steady_state": [0.0, 0.0, 0.0, 0.0, 0.0]},
)
```

1. We can set `created_by` to any string; it is recorded in the bundle manifest for provenance. Defaults to `"SymbolicDSGE <version>"` when omitted.
2. `ModelParser` populates `ModelConfig.source_yaml` automatically; the bundle re-uses it without re-reading the file.
3. Set `True` for models authored in nonlinear levels — see the [Quick Start](quickstart.md#compilation) for details.
4. The bundle keeps the YAML text originating our model. The loader will respect the solve/compile kwargs and re-solve the model to obtain the exact same `SolvedModel` deterministically.

???+ note "Why we solve first"
    The bundle preserves the YAML and the recorded compile/solve kwargs, then re-runs them at load time. Solving here is only required if we want to attach an estimation, an MC pipeline, or a `SolvedModel`-derived result. Author-only bundles do not need a live `SolvedModel`.

???+ note "Model Roles"
    The model framework inside bundles can work with two model roles: `reference` and `dgp`. At least one of them must be present. `reference` is later used as the subject of Monte Carlo experiments and simulation prefill. `dgp` is used as the data-generating process for Monte Carlo experiments. We attach the same model twice here for demonstration, but in practice you may want to attach two different models.

## Specify the estimation tab

We define a small MCMC run estimating `psi_pi` and `psi_x` against synthetic observed data. The estimation is carried out as it would be in a live run.
You can refer to the [Estimation Guide](estimation_guide.md) and [API Reference](../documentation/Estimator.md) for details on the `Estimator` API usage.

```python
import numpy as np
from SymbolicDSGE import Estimator
from SymbolicDSGE.bayesian import make_prior

priors = {
    "psi_pi": make_prior(
        distribution="normal",
        parameters={"mean": 1.5, "std": 0.25},
        transform="identity",
    ),
    "psi_x": make_prior(
        distribution="normal",
        parameters={"mean": 0.5, "std": 0.2},
        transform="identity",
    ),
}

rng = np.random.default_rng(0)
observed = rng.standard_normal((40, 3))
estim = Estimator(
    solver=solver,
    compiled=compiled,
    observables=["OutGap", "Infl", "Rate"],
    y=observed,
    priors=priors,
)
res = estim.mcmc(  # (1)!
    n_draws=1000,
    burn_in=200,
    thin=2,
)

bundle.add_estimation(  # (2)!
    source=estim,
    result=res,
)
```

1. We can bundle results from an executed estimation, or we can bundle an estimation spec without results.
2. `add_estimation` can bundle live results and initialized `Estimator` instances. On the backend, these are converted to human-readable specifications. Bundling live objects does not make the final bundle depend on unreadable binary objects.

???+ note "Estimation Methods"
    MCMC returns a special result object `MCMCResult` while MLE and MAP both return `OptimizationResult`. The bundler handles both cases.

## Build a Monte Carlo pipeline

We create a Monte Carlo pipeline and run it as we would in a live session. Similar to estimation, we can bundle a pipeline spec without running it, or we can bundle a live `MCPipelineResult` from a run. The bundler converts a live `MCPipeline` to a portable [`PipelineSpec`](../documentation/monte_carlo/spec.md) and splits the live result into document and trace members. You can refer to the [Monte Carlo Guide](monte_carlo_guide.md) and [API Reference](../documentation/monte_carlo/pipeline.md) for details on the `MCPipeline` API usage.

```python
from SymbolicDSGE import Shock
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.operations import (
    core as c,  # (1)!
    tests as t,  # (2)!
)

gz_shock = Shock(T=200, seed=42, multivar=True, dist="norm")  # (3)!
r_shock = Shock(T=200, seed=42, multivar=False, dist="norm")

mc_pipeline = MCPipeline([
        c.simulation_step(T=200, shocks={"g,z": gz_shock, "r": r_shock}),
        t.jarque_bera_test_step("jb_test", source="observables", column=0),
])
mc_res = mc_pipeline.run(
    reference=sol,
    dgp=sol,
    n_rep=1000,
    retain_payloads=False,
    retain_contexts=True,
    verbosity=2,
)

bundle.add_mc(pipeline=mc_pipeline, result=mc_res)
```

1. `core` contains data generation, raw data consumption, and Kalman filtering.
2. `tests` contains multiple built-in statistical tests.
3. Notice we don't call `Shock.shock_generator` here. This is because the MC pipeline needs to manage the seed per-replication to avoid repeating the same shock path across replications.

## Specify a simulation prefill

`SimSpec` rides inline in the manifest. It controls what the GUI's Outputs tab pre-fills when the receiver opens the bundle on `sdsge-ui`.

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

bundle.set_simulation(simulation)
```

1. The seed makes the replayed simulation deterministic — both the bundle author and the receiver produce identical paths when clicking **Run**.

## Add raw data alongside the model

`add_raw_data` covers any extra CSV files you want to ship in `data/`. They are not interpreted by the loader — they are passthrough storage for context the receiver may want.

```python
import io
import pandas as pd

aux = pd.DataFrame({
    "date": pd.date_range("2000-01-01", periods=40, freq="QS"),
    "gdp_growth": rng.standard_normal(40),
})
csv_buf = io.StringIO()
aux.to_csv(csv_buf, index=False)

bundle.add_raw_data(
    name="auxiliary_series",
    data=csv_buf.getvalue(),
)
```

???+ info "CSV vs Parquet for raw data"
    `add_raw_data` re-encodes CSV input as Parquet by default. Pass `as_parquet=False` to embed the CSV verbatim — useful for hand-zip-friendly bundles.

## Write the bundle

`BundleBuilder` chains every component into one archive. Each `add_*` call returns `self`; the final `.write(path)` materializes the bundle and returns the path written.

The bundle is written to disk as a zip file that's aliased as a `.sdsge` file.

```python
bundle_path = bundle.write("experiment-1.sdsge")
print(f"Bundle written to {bundle_path}")
```

## Inspect the result

Since `.sdsge` is just an alias, a bundle acts exactly like a zip file. You can use any `zip` utility to inspect its contents or use the `sdsge-decompile` CLI to extract it into a directory structure.

```bash
unzip -l experiment-1.sdsge
```

```text
Archive:  experiment-1.sdsge
  Length      Date    Time    Name
---------  ---------- -----   ----
     3122  16-06-2026 14:26   manifest.json
     2599  16-06-2026 14:26   model/reference.yaml
     2599  16-06-2026 14:26   model/dgp.yaml
      973  16-06-2026 14:26   estimation/spec.json
      420  16-06-2026 14:26   estimation/result.json
     1939  16-06-2026 14:26   estimation/observed.parquet
    19943  16-06-2026 14:26   estimation/posterior.parquet
      832  16-06-2026 14:26   montecarlo/pipeline.json
     4475  16-06-2026 14:26   montecarlo/result.json
    15645  16-06-2026 14:26   montecarlo/traces.parquet
     1282  16-06-2026 14:26   data/auxiliary_series.parquet
---------                     -------
    53829                     11 files
```

For a structured view, decompile it:

```bash
sdsge-decompile experiment-1.sdsge -o experiment-1/
```

Or open it directly in Python: see the [Bundle Loading Guide](bundle_loading_guide.md).

## Further steps

- [`sdsge-compile`](../portable_experiments/sdsge-compile.md) — the directory-driven CLI equivalent.
- [`SolvedModel.save_sdsge`](../documentation/SolvedModel.md) — one-shot bundle write for the model-only case.
- [`SolvedModel.to_bundle_builder`](../documentation/SolvedModel.md) — pre-seeded builder factory, equivalent to the chain in this guide.
- [`BundleBuilder` API reference](../documentation/bundle/BundleBuilder.md).

[Download Guide Notebook](../assets/bundle_authoring.ipynb){ .md-button download="" }
