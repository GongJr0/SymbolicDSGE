---
tags:
    - guide
---

# Monte Carlo Pipeline Guide

??? tip "__TL;DR__"
    You can find a demonstration notebook [here](../assets/monte_carlo.ipynb).

???+ warning "Read the Quickstart and Kalman Guides"
    This guide assumes familiarity with `SolvedModel.sim(...)`, `Shock`, and `SolvedModel.kalman(...)`.

This guides demonstrates the setup of an example Monte Carlo experiment.
The `monte_carlo` module is written for two cases:

1. Comparing two models: a reference and data-generating model.
2. Comparing a reference model to raw data.

This demonstration focuses on the first case where two models are present.

## Model Instantiation

```python
import numpy as np
import pandas as pd

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.monte_carlo.step_factories import (
    reference_filter_step,
    simulation_step,
    wald_test_step,
)

model, kalman = ModelParser("../../MODELS/POST82.yaml").get_all() # (1)!
ss_seed = np.zeros(5, dtype=np.float64)  # (2)!

# Solve the reference model
solver = DSGESolver(model, kalman)
compiled = solver.compile()
reference = solver.solve(compiled, ss_seed=ss_seed)

# Change parameters and re-compile to get the DGP model
dgp_params = {str(k): v for k, v in model.calibration.parameters.items()} # (3)!

dgp_params["rho_g"] = 0.90 # AR persistence param
dgp_params["rho_z"] = 0.75 # AR persistence param

dgp = solver.solve(
    compiled,
    parameters=dgp_params,
    ss_seed=ss_seed,
)

```

1. This is the core configuration file for both models.
2. The configuration is based on a NK3 gap model.
3. We extract the parameters from the original config and modify them slightly.

Now, we have two models to compare in a Monte Carlo experiment.
We will determine whether the reference model is misspecified relative to the DGP using MC repeated Wald tests.

???+ note "Data Retention"
    Each step produces and records their output independently. Retaining full output resolution for every step can easily exceed available memory in consumer hardware.
    `n_retain` controls how many replications persist outside the hot loop to give more granular memory management. All steps contain an `n_retain` argument, which defaults to `-1` (retain all). Setting it to `0` retains nothing, and any positive integer retains that many evenly spaced replications.

    For reference, the exact run described in this guide requires a modest 1.42 GiB at 10,000 replications. At a more realistic 100,000 replications, the same run requires 14.2 GiB.


## Pipeline Setup

```python
from SymbolicDSGE.monte_carlo import MCPipeline

T = 200  # (1)!
n_obs = len(reference.compiled.observable_names)  # (2)!

pipeline = MCPipeline(
    per_rep_steps=[...],  # (3)!
    postproc_steps=[...],  # (4)!
    )
```

1. Length of each simulated sample.
2. Number of observables the model(s) have.
3. Steps here are executed per replication.
4. This field is reserved for `POSTPROC` steps; these execute once after the replication loop concludes.

`MCPipeline` is used to compile the steps that need to be executed for each repetition.
Every step of `MCPipeline` must be an `MCStep` object.
The pipeline will be built using the step-generating functions under `SymbolicDSGE.monte_carlo.step_factories`.

### Data Generation

???+ warning "Step Ordering"
    Data generation is done exactly once, in the first step of the pipeline.

Using the `simulation_step` function, we generate an `MCStep` object that samples the DGP model with a given simulation specification.

```python
from SymbolicDSGE import Shock

datagen_step = simulation_step(
    T=T,
    target="dgp",  # (1)!
    n_retain=-1,  # (2)!
    shocks={
        "g,z": Shock(dist="norm", multivar=True, seed=0),
        "r": Shock(dist="norm", seed=1),
    },
    observables=True,
)
```

1. `target` can be either `"reference"` or `"dgp"` and defaults to `"dgp"`. It selects which model role is simulated.

???+ note "Replication Streams"
    Normal and uniform shocks are drawn inside the native loop. Each replication addresses its own stream of a counter based engine keyed on the specification's seed and its position in the mapping, so replications never overlap and two specifications sharing a seed stay independent.

    A seeded specification replays bit for bit across runs, and the result does not depend on `n_rep` or `n_jobs`. A specification with `seed=None` draws from a fresh key each run.

`simulation_step` forwards `T`, `shocks`, `shock_scale`, `x0`, and `observables` to `SolvedModel.sim(...)`. It adds `target`, which selects the model role.
The `shocks` argument follows the same dictionary convention as `SolvedModel.sim(...)`. Each MC iteration runs the selected model with this specification and passes the output data downstream.

To inspect one replication on its own, `replication_shocks` hands back the exact shock paths that replication saw, keyed the same way the specification is:

```python
from SymbolicDSGE.monte_carlo import replication_shocks

shocks = replication_shocks(dgp, datagen_step, rep_idx=417)
sample = dgp.sim(T=T, shocks=shocks, shock_scale=1.0)
```

Scaling is already applied to what comes back, which is why `shock_scale` is `1.0` above. Only seeded specifications are reproducible this way; one with `seed=None` was drawn from a key the run discarded.

### Filtering

The first step after datagen is filtering the reference model using a Kalman filter against the DGP simulated observables.
`reference_filter_step` is a pre-built function configuring the reference model's Kalman filter to be run per iteration for this purpose.

```python
kf_step = reference_filter_step()
```

`reference_filter_step` accepts `filter_mode`, `observables`, `x0`, `P0`, `R`, `jitter`, `symmetrize`, and `return_shocks`, mirroring the `SolvedModel.kalman` configuration.

### Testing

With filtered outputs, we run a test step using the `wald_test_step` function.
`kind = "mean"` and `target = np.zeros(n_obs)` tests the first moment of the standardized innovations against a zero vector.

```python

mean_test_step = wald_test_step(
    "std_innov_mean",  # (1)!
    source="filter",  # (2)!
    field="std_innov",  # (3)!
    target=np.zeros(n_obs),  # (4)!
    kind="mean",  # (5)!
    burn_in=20,  # (6)!
)
```

1. Name of the step. (This will be used as the key to access the results)
2. Producer step to read from. In this case, it is the Kalman filter (named `"filter"` by default).
3. Field in the producer's output. `std_innov` is the standardized innovations of the Kalman filter.
4. Target to test against. In this case we're testing if the mean of each observable is zero.
5. Kind of the wald test. Available inputs are: `Literal["mean", "covariance", "second_moment"]`.
6. Number of periods to discard before running the tests.

Every test step writes a statistic, a p-value, and a status per replication. These are aggregated into an MC summary once the loop concludes.

### Built-in and Custom Transforms

Custom transforms run inside the native loop, so they are wrapped with `custom_transform` and compiled by Numba. The wrapper snapshots the function source and the globals it references so the operation can travel inside a `.sdsge` archive, and it enforces the native transform contract: exactly two positional arguments and an integer status return.

```python
from SymbolicDSGE.monte_carlo import custom_transform

@custom_transform
def custom_standardize(sample, output) -> int:  # (1)!
    n, p = sample.shape
    for j in range(p):
        mean = 0.0
        for i in range(n):
            mean += sample[i, j]
        mean /= n

        var = 0.0
        for i in range(n):
            var += (sample[i, j] - mean) ** 2
        std = (var / n) ** 0.5

        for i in range(n):
            output[i, j] = (sample[i, j] - mean) / std

    return 0  # (2)!
```

1. `sample` is the C-contiguous 2D `float64` slice selected for the current replication, `output` is the buffer the function writes into. Both are supplied by the runner; the function never allocates.
2. `0` marks success. Any other value marks the replication as failed for this step.

???+ note "Built-in Transforms"
    There are multiple built-in transforms available in `SymbolicDSGE` and [standardization](../documentation/monte_carlo/operations/transforms/standardize.md) is one of them. All built-in transforms are documented and `standardize_step` is used as an example in this guide.

With a custom function defined, the step can be created using the generic `transform_step` function.

```python
from SymbolicDSGE.monte_carlo.step_factories import standardize_step, transform_step

custom_std = transform_step(
    "custom_std",  # (1)!
    custom_standardize,  # (2)!
    source="filter",  # (3)!
    field="innov",
    output_shape=(T, n_obs),  # (4)!
)

builtin_std = standardize_step(
    "builtin_std",
    source="filter",
    field="innov",  # (5)!
)

```

1. Name of the step. It becomes the trace key `payload.custom_std`.
2. The function to be executed. A bare function is wrapped in `NumbaCustomFunc` automatically.
3. `source` and `field` select the upstream array. `columns`, `burn_in`, and `drop_initial` narrow it further.
4. Required. The output buffer is planned before the run starts, so the transform must declare the exact `(rows, columns)` it writes.
5. In this case, the `innov` attribute (raw innovations) of the filter output is used.

### Post-Processing

Post-processing is executed separately from the replication loop. The `kde_step` function is the only built-in. Custom post-processing steps stay in plain Python and are encapsulated by a `pandas_operation` decorator, which extends the numeric namespace with allowed `pandas` functionality. Post-processing functions do not see individual replications. Instead, they receive a flattened `traces` dictionary containing transform payloads, test results, and regression results.

Access to a given array follows a `"."` separated path, for example, the custom standardization step (which is a payload) is accessed as `traces["payload.custom_std"]`. Payload traces are stacked across retained replications, so a transform writing `(T, p)` per replication appears as `(n_retained, T, p)`. Test and regression results are structured:

__Test Traces:__

- `"test.{name}.pval"`: Array of p-values for each replication.
- `"test.{name}.statistic"`: Array of test statistics for each replication.
- `"test.{name}.status"`: Array of test statuses for each replication.

__Regressions Traces:__

- `"regression.{name}.coef"`: 2D array of regression coefficients for each replication.
- `"regression.{name}.r2"`: Array of R-squared values for each replication.
- `"regression.{name}.status"`: Array of regression statuses for each replication.

A custom post-processing function is defined as follows:

```python
from SymbolicDSGE.monte_carlo import pandas_operation

@pandas_operation
def get_std_obs_mean(*, traces: dict[str, Any]) -> pd.DataFrame:
    stacked = traces["payload.custom_std"]  # (1)!
    return pd.DataFrame({"mean": stacked.mean(axis=(0, 1))})
```

1. Shape `(n_retained, T, n_obs)`. Averaging over the first two axes gives the per-observable mean across the experiment.

To create a step out of this function, we use `postproc_step`:

```python
from SymbolicDSGE.monte_carlo.step_factories import kde_step, postproc_step

custom_postproc = postproc_step(
    "custom_postproc",  # (1)!
    get_std_obs_mean,  # (2)!
)

builtin_kde = kde_step(
    "builtin_kde",
    trace="payload.builtin_std",  # (3)!
    grid_points=100,  # (4)!
)

```

1. Name of the step. It is the key the artifact is stored under in `MCPipelineResult.postproc`.
2. The function to be executed. Any callable with the signature of a custom post-processing function can be used here.
3. The trace to be used for the KDE. This is a payload in this case, but it can also be a test or regression result.
4. The number of grid points to use for the KDE. This is only applicable to the built-in KDE step.

???+ tip "Render Hints"
    Wrapping a returned artifact in `Summary(value, title=...)` or `Raw(value)` tells downstream consumers such as the GUI how to display it. Both are exported from `SymbolicDSGE.monte_carlo` and are optional; returning a bare object is fine.

### Complete Pipeline

```python
pipeline = MCPipeline(
    per_rep_steps=[
    datagen_step,
    kf_step,
    custom_std,
    builtin_std,
    mean_test_step,
],
postproc_steps=[
    custom_postproc,
    builtin_kde,
])
```

## Running the Pipeline

The `MCPipeline` object explains the procedure that will run per iteration.
`MCPipeline.run` lowers that procedure to native kernels and executes it repeatedly.

```python
mc = pipeline.run(
    reference=reference,
    dgp=dgp,
    n_rep=10000,
    fail_fast=True,  # (1)!
    n_jobs=-1,  # (2)!
    verbosity=2,  # (3)!
    check_memory_availability=True,  # (4)!
)
```

1. Whether to raise on the first failing replication. With `False`, failures are recorded in `MCPipelineResult.failures` and the run continues.
2. Number of worker threads. `None` runs single-threaded, a positive value is taken literally, and a negative value resolves to `max(1, cpu_count + 1 + n_jobs)` following the joblib convention.
3. Verbosity level for logging output `{0, 1, 2}`. `0` prints nothing, `1` prints the run total and the post-processing total, `2` additionally prints per-step throughput.
4. Whether to size the run's retained arenas before allocating them. A run that spills past physical memory warns and proceeds; one that does not fit even with swap counted raises `MemoryError` with a per-step breakdown. `False` allocates unconditionally.

???+ warning "Per-step Timings Require `verbosity=2`"
    The native loop only collects per-step profiling when it is asked to. At any lower verbosity `meta.step_elapsed_s`, `meta.step_counts`, and `meta.step_failures` are empty dictionaries.

```bash
>>> MC run concluded successfully in 0.25s with 40529.48 it/s.
Per-step Report:

    datagen: 0 failures, 47156.51 worker it/s (0.21 worker-s), 40529.48 wall it/s.
    filter: 0 failures, 2908.28 worker it/s (3.44 worker-s), 40529.48 wall it/s.
    custom_std: 0 failures, 219709.70 worker it/s (0.05 worker-s), 40529.48 wall it/s.
    builtin_std: 0 failures, 172673.13 worker it/s (0.06 worker-s), 40529.48 wall it/s.
    std_innov_mean: 0 failures, 87742.76 worker it/s (0.11 worker-s), 40529.48 wall it/s.

Post-processing Report:

    custom_postproc: Succeeded in 0.0184s.
    builtin_kde: Succeeded in 9.1390s.
```

???+ note "Worker Time vs Wall Time"
    Worker rates divide by the time summed across threads, so they describe the cost of a step. Wall rates divide by the elapsed run time, so they describe what the step contributed to the observed duration. The two coincide when `n_jobs` resolves to `1`.

This returns a `MCPipelineResult` object containing test summaries for each test step executed in the pipeline.
To extract the test results, their p-values, and other relevant statistics, we can access the test summaries by key (step names).

```python

summary = pd.DataFrame(
    {
        name: {
            "mean_statistic": res.mean_statistic,
            "mean_pval": res.mean_pval,
            "rejection_rate": res.rejection_rate,
            "ci_low": res.pval_confidence_interval()[0],
            "ci_high": res.pval_confidence_interval()[1],
        }
        for name, res in mc.test_summaries.items()
    }
).T
print(summary.round(4))

```

```bash
>>>                mean_statistic  mean_pval  rejection_rate  ci_low  ci_high
std_innov_mean           2.758      0.546           0.051   0.047    0.056

```

Regression results are accessed similarly in `MCPipelineResult.regression_summaries`, and post-processing artifacts are accessed by key in `MCPipelineResult.postproc: dict[str, Any]`.

???+ tip "Retaining Fewer Replications"
    Large runs can routinely exceed hardware memory in size requirements. It is strongly recommended to set `n_retain=0` for intermediate steps that are not needed for later
    analysis or post-processing. This will allow you to retain much more replications for steps you care about before running out of memory. `n_retain=0` will not make the data unavailable to downstream steps, all data makes it through the pipeline, but non-retained replications are discarded after used. `n_retain=-1` is the default and retains all replications.

???+ tip "Sizing a Run Before It Allocates"
    `MCPipeline.validate_memory_requirements` takes the arguments `run` takes and reports what they would allocate, broken down by step. `run` performs the same check unless `check_memory_availability=False`. If a run is provably too large (meaning your RAM and swap combined are smaller than the required allocation), a `MemoryError` will block the run instead of seeing it crash hours later. `check_memory_availability=False` also disables this behavior.

    ```python
    print(pipeline.validate_memory_requirements(reference=reference, dgp=dgp, n_rep=400_000, n_jobs=-1))
    ```

    ```bash
    >>> Memory Availability Error:
        step               per rep   retained  n_retain
        datagen          12.50 KiB   4.77 GiB        -1
        filter          126.57 KiB  48.28 GiB        -1
        custom_std        4.69 KiB   1.79 GiB        -1
        builtin_std       4.69 KiB   1.79 GiB        -1
        std_innov_mean        16 B   6.10 MiB        -1
        -----------------------------------------------
        worker lanes (x20)           3.57 MiB
        run metadata                36.62 MiB
        allocated                   56.67 GiB
        reserve (1.00 GiB + 2.5%)    2.42 GiB
        total                       59.09 GiB
        available                    9.55 GiB
        ceiling (+ swap free)       24.03 GiB

    MemoryError: This run requires 59.09 GiB, which does not fit in 9.55 GiB free RAM 
    + 14.48 GiB free swap. Lower n_retain or n_rep, or pass check_memory_availability=False to run anyway.

    ```

    `allocated` is what the run commits, and it is exact. The `reserve` on top of it is held for the process that hosts the run rather than for the arenas: the interpreter growing as results are read back, a notebook kernel holding its own copy, and the allocator's transient peaks. None of that scales with the size of the run, which is why the reserve is a flat floor plus a small fraction instead of a multiple. A shock specification the native draw cannot reproduce adds a `prematerialized shocks` row, which is an `(n_rep, T, n_exog)` array built in Python before the loop starts.

    Passing `available` means the run no longer fits in physical memory and will page, which costs throughput. Passing `ceiling` means it does not fit even with swap counted. At that point, a mid-run crash is significantly more likely, and this is the point where `MCPipeline.run` raises `MemoryError`.

## Conclusion

This guide demonstrates the usage of basic MC functionality through the pre-configured steps available in the library.
Custom transforms are available through `transform_step(...)` and bundle-safe custom operations can be wrapped with `custom_transform` or `pandas_operation`. See the [Monte Carlo custom operation API reference](../documentation/monte_carlo/custom_ops.md) for the current contract.

For future reference or a ready-made boilerplate, you can visit [this](../assets/monte_carlo.ipynb) link to access a notebook containing the process outlined in this guide.

[Download MC Guide Notebook](../assets/monte_carlo.ipynb){ .md-button download="" }
