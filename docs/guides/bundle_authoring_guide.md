---
tags:
    - guide
---

# Bundle Authoring Guide

??? tip "__TL;DR__"
    Assemble a complete `.sdsge` bundle from code with `BundleBuilder`. Chain `add_model`, `add_estimation`, `add_mc`, `set_simulation`, and `add_raw_data`, then `.write(...)` the archive. Each member is optional, so the same builder covers a model only bundle and a full experiment alike.

    You can find a demonstration notebook [here](../assets/bundle_authoring.ipynb).

This guide walks through assembling a complete `.sdsge` bundle from code: model + Kalman config, estimation spec and result, observed data, Monte Carlo pipeline and traces, simulation prefill. Every member type the bundle supports is covered.

We start from `MODELS/POST82.yaml` (also used in the [Quick Start](quickstart.md)). The full assembly fits in a single script you can copy and run.

???+ tip "Where the bundle ends up"
    The example writes `experiment-1.sdsge` in the current directory. Open it later via [`load_bundle`](../documentation/bundle/load_bundle.md), inspect components with [`sdsge-decompile`](../portable_experiments/sdsge-decompile.md), or open it directly in the GUI with `sdsge-ui experiment-1.sdsge`.

## Solve a reference model

We initialize a `BundleBuilder` to store all upcoming components. A model is then defined and solved to be attached to the bundle. You can refer to the [Quick Start](quickstart.md) for details on basic model authoring and solving.

```python
from SymbolicDSGE import DSGESolver, ModelParser, BundleBuilder
from numpy import array, float64

bundle = BundleBuilder(created_by="experiment-1") # (1)!

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
2. `ModelParser` populates `ModelConfig.source_yaml` automatically; the bundle uses it without reading the file again.
3. Set `True` for models authored in nonlinear levels. See the [Quick Start](quickstart.md#compilation) for details.
4. The bundle keeps the YAML text originating our model. The loader will respect the solve/compile kwargs and solve the model again to obtain the exact same `SolvedModel` deterministically.

???+ note "Why we solve first"
    The bundle preserves the YAML and the recorded compile/solve kwargs, then runs them again at load time. Solving here is only required if we want to attach an estimation, an MC pipeline, or a result derived from a `SolvedModel`. Bundles that only carry authored model text do not need a live `SolvedModel`.

???+ note "Model Roles"
    The model framework inside bundles can work with two model roles: `reference` and `dgp`. At least one of them must be present. `reference` is later used as the subject of Monte Carlo experiments and simulation prefill. `dgp` is used as the data generating process for Monte Carlo experiments. We attach the same model twice here for demonstration, but in practice you may want to attach two different models.

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
2. `add_estimation` can bundle live results and initialized `Estimator` instances. These are converted to readable specifications for storage. Bundling live objects does not make the final bundle depend on unreadable binary objects.

???+ note "Estimation Methods"
    MCMC returns a special result object `MCMCResult` while MLE and MAP both return `OptimizationResult`. The bundler handles both cases.

## Build a Monte Carlo pipeline

We create a Monte Carlo pipeline and run it as we would in a live session. Similar to estimation, we can bundle a live `MCPipeline`, and optionally include a live `MCPipelineResult` from a run. The bundle stores the pipeline and splits the live result into document and trace members. You can refer to the [Monte Carlo Guide](monte_carlo_guide.md) and [API Reference](../documentation/monte_carlo/pipeline.md) for details on the `MCPipeline` API usage.

```python
from SymbolicDSGE import Shock
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.operations import (
    core as c,  # (1)!
    tests as t,  # (2)!
)

gz_shock = Shock(seed=0, multivar=True, dist="norm")  # (3)!
r_shock = Shock(seed=1, multivar=False, dist="t", dist_kwargs={"df": 3})

mc_pipeline = MCPipeline([
        c.simulation_step(
            target="dgp",
            T=200,
            shocks={"g,z": gz_shock, "r": r_shock},
        ),
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
3. Notice we don't call `Shock.shock_generator` here. The MC pipeline needs to manage the seed per replication to avoid repeating the same shock path across replications.

## Specify a simulation prefill

Simulation prefills ride inline in the manifest, keyed by role. They control what the GUI's Outputs tab prefills when the receiver opens the bundle on `sdsge-ui`. A `SimSpec`'s fields are exactly the keyword arguments of `SolvedModel.sim`, and each shock is stored as its `Shock.to_dict()` parameters. No live `Shock` is serialized.

```python
from SymbolicDSGE.bundle import SimSpec
from SymbolicDSGE.core.shock_generators import Shock

simulation = SimSpec(
    T=200,
    observables=True,
    shock_scale=1.0,
    shocks={
        "r": Shock(seed=42, dist="norm", dist_kwargs={"loc": 0.0}).to_dict(), # (1)!
    },
)

bundle.set_simulation("reference", simulation)
```

1. The seed makes the replayed simulation deterministic. Both the bundle author and the receiver produce identical paths when clicking **Run**. Because a `Shock` is horizon independent, `to_dict()` carries no `T`; the period count comes from the `SimSpec`.

## Add raw data alongside the model

`add_raw_data` covers any extra CSV files you want to ship in `data/`. They are not interpreted by the loader. They are passthrough storage for context the receiver may want.

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
    `add_raw_data` encodes CSV input as Parquet by default. Pass `as_parquet=False` to embed the CSV verbatim, which is useful for hand zipped bundles.

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

- [`sdsge-compile`](../portable_experiments/sdsge-compile.md): the directory driven CLI equivalent.
- [`SolvedModel.save_sdsge`](../documentation/SolvedModel.md): one shot bundle write for the model only case.
- [`SolvedModel.to_bundle_builder`](../documentation/SolvedModel.md): pre seeded builder factory, equivalent to the chain in this guide.
- [`BundleBuilder` API reference](../documentation/bundle/BundleBuilder.md).

[Download Guide Notebook](../assets/bundle_authoring.ipynb){ .md-button download="" }
