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
from SymbolicDSGE.monte_carlo.operations.core import (
    reference_filter_step,
    simulation_step,
)
from SymbolicDSGE.monte_carlo.operations.tests import wald_test_step

model, kalman = ModelParser("../../MODELS/POST82.yaml").get_all() # (1)!
steady_state = np.zeros(5, dtype=np.float64)  # (2)!

# Solve the reference model
solver = DSGESolver(model, kalman)
compiled = solver.compile()
reference = solver.solve(compiled, steady_state=steady_state)

# Change parameters and re-compile to get the DGP model
dgp_params = {str(k): v for k, v in model.calibration.parameters.items()} # (3)!

dgp_params["rho_g"] = 0.90 # AR persistence param
dgp_params["rho_z"] = 0.75 # AR persistence param

dgp = solver.solve(
    compiled,
    parameters=dgp_params,
    steady_state=steady_state,
)

```

1. This is the core configuration file for both models.
2. The configuration is based on a NK3 gap model.
3. We extract the parameters from the original config and modify them slightly.

Now, we have two models to compare in a Monte Carlo experiment.
We will determine whether the reference model is misspecified relative to the DGP using MC repeated Wald tests.

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
4. This field is reserved for `POSTPROC` steps; these execture once after the replication loop concludes.

`MCPipeline` is used to compile the steps that need to be executed for each repetition.
Every step of `MCPipeline` must be an `MCStep` object.
The pipeline will be built using the step-generating functions under `SymbolicDSGE.monte_carlo.operations`.

### Data Generation

???+ warning "Step Ordering"
    Data generation is done exactly once, in the first step of the pipeline.

Using the `simulation_step` function, we generate an `MCStep` object that samples the DGP model with a given simulation specification.

```python
from SymbolicDSGE import Shock

datagen_step = simulation_step(
    T=T,
    shocks={
        "g,z": Shock(T=T, dist="norm", multivar=True, seed=0),
        "r": Shock(T=T, dist="norm", seed=1),
    },
    observables=True,
)
```

`simulation_step` takes all `kwarg`s that `SolvedModel.sim` accepts.
Each MC iteration will trigger a simulation with this specification and passthrough the output data.

### Filtering

The first step after datagen is filtering the reference model using a Kalman filter against the DGP simulated observables.
`reference_filter_step` is a pre-built function configuring the `SolvedModel.kalman` filter to be called per iteration for this purpose.

```python
kf_step = reference_filter_step(estimate_R_diag=False)
```

`reference_filter_step` accepts all `SolvedModel.kalman` kwargs similar to the simulation step function.

### Testing

With filtered outputs, we run a test step using the `wald_test_step` function.
`kind = "mean"` and `target = np.zeros(n_obs)` tests the first moment of the standardized innovations against a zero vector.

```python

mean_test_step = wald_test_step(
    "std_innov_mean",  # (1)!
    source="std_innov",  # (2)!
    target=np.zeros(n_obs),  # (3)!
    kind="mean",  # (4)!
    burn_in=20,  # (5)!
)
```

1. Name of the step. (This will be used as the key to access the results)
2. Data to use when conducting the test. In this case, it is `std_innov` from `FilterResult` objects returned by the kalman filter.
3. Target to test against. In this case we're testing if the mean of each observable is zero.
4. Kind of the wald test. Available inputs are: `Literal["mean", "covariance", "second_moment"]`.
5. Number of periods to discard before running the tests.

Each test returns a `TestResult` object and the results are aggregated to produce an MC summary.

### Built-in and Custom Transforms

A special wrapper `numpy_operation` is used when defining custom transforms to restrict the namespace availabe. This eliminates some obvious security like `exec` and `eval` on top of restricting what portion of `numpy` is allowed. All custom transforms are parsed as `NumpyCustomFunc` regardless of the decorator. Decorating allows to show intent on the author's side. A custom transform function is defined as follows:

```python
from SymbolicDSGE.monte_carlo import numpy_operation
from SymbolicDSGE.monte_carlo.operations.transforms import transform_step

@numpy_operation
def custom_standardize(
    context: MCContext,  # (1)! 
    reference: SolvedModel, # (2)!
    dgp: SolvedModel | None, # (3)!
    rep_idx: int  # (4)!
) -> np.ndarray | None:
    del reference, dgp, rep_idx  # (5)!
    data: MCData = context.require_data()  # (6)!
    obs = data.observables

    if obs is not None:
        return (obs - obs.mean(axis=0)) / obs.std(axis=0)
    
    return None
```

1. `context` is a required argument for all custom transforms. It is used to access the current `MCContext` object at a given replication. `require_data()` and `require_payload()` are the two main methods to access data and payloads.
2. `reference` is the reference model used in the pipeline. It is a required argument.
3. `dgp` is the data-generating model used in the pipeline. It is a required argument.
4. `rep_idx` is the current replication index. It is a required argument.
5. Unused arguments are deleted both for clarity and to avoid linter warnings.
6. Refer to [MC Core Containers](../documentation/monte_carlo/core_containers.md) for more information on the `MCData` object and its attributes.

???+ note "Built-in Transforms"
    There are multiple built-in transforms available in `SymbolicDSGE` and [standardization](../documentation/monte_carlo/operations/transforms/standardize.md) is one of them. All built-in transforms are documented and `standardize_step` is used as an example in this guide.

With a custom function defined, the step can be created using the generic `transform_step` function.

```python
from SymbolicDSGE.monte_carlo.operations.transforms import transform_step, standardize_step

custom_std = transform_step(
    "custom_std",  # (1)!
    func=custom_standardize,  # (2)!
    store_key=None, # (3)!
)

builtin_std = standardize_step(
    "builtin_std",
    source="innov",  # (4)!
)

```

1. Name of the step. Used as the key in the payload dictionary when `store_key` is `None`.
2. The function to be executed. Any callable with the signature of a custom transform can be used here.
3. The key to store the output of the transform in the payload dictionary. If `None`, the step name is used as the key.
4. The source of the data to be transformed. In this case, it is the `innov` attribute of the `FilterResult` object returned by the kalman filter.

### Post-Processing

Post-processing is executed separately from the replication loop. The `kde_step` function is the only built-in. However, custom post-processing steps are more permissive than their transform counterparts. These steps are encapsulated by a `pandas_operation` decorator which extends the `numpy_operation` namespace with allowed `pandas` functionality. Post-processing functions do not have access to the per-replication context and data objects. Instead, they receive a flattened `traces` dictionary containing payloads, test results, and regression results.

Access to a given given array follows a `"."` separated path, for example, the custom standardization step (which is a payload) is accessed as `traces["payload.custom_std"]`. Payloads contain whatever the step returned, but test and regression results are structured:

__Test Traces:__

- `"test.{name}.pval"`: Array of p-values for each replication.
- `"test.{name}.statistic"`: Array of test statistics for each replication.
- `"test.{name}.status"`: Array of test statuses for each replication.

__Regressions Traces:_

- `"regression.{name}.coef"`: 2D array of regression coefficients for each replication.
- `"regression.{name}.r2"`: Array of R-squared values for each replication.
- `"regression.{name}.status"`: Array of regression statuses for each replication.

A custom post-processing function is defined as follows:

```python
from SymbolicDSGE.monte_carlo import pandas_operation

@pandas_operation
def get_std_obs_mean(*, traces: dict[str, Any]) -> pd.Series:
    return traces["payload.custom_std"].mean(axis=0)
```

To create a step out of this function, we use `postproc_step`:

```python
from SymbolicDSGE.monte_carlo.operations.postproc import postproc_step, kde_step

custom_postproc = postproc_step(
    "custom_postproc",  # (1)!
    func=get_std_obs_mean,  # (2)!
    store_key=None,  # (3)!
)

builtin_kde = kde_step(
    "builtin_kde",
    trace="payload.builtin_std",  # (4)!
    grid_points=100,  # (5)!
)

```

1. Name of the step. Used as the key in the traces dictionary when `store_key` is `None`.
2. The function to be executed. Any callable with the signature of a custom post-processing function can be used here.
3. The key to store the output of the post-processing function in the traces dictionary. If `None`, the step name is used as the key.
4. The trace to be used for the KDE. This is a payload in this case, but it can also be a test or regression result.
5. The number of grid points to use for the KDE. This is only applicable to the built-in KDE step.


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
`MCPipeline.run` is then used to run the procedure repeatedly.

```python
mc = pipeline.run(
    reference=reference,
    dgp=dgp,
    n_rep=1000,
    retain_payloads=False,  # (1)!
    retain_test_results=False,  # (2)!
    retain_contexts=False,  # (3)!
    verbosity=2,  # (4)!
)
```

1. Whether to keep payloads in the result or discard after the replication loop. Depending on the type of the payload, this can be a large amount of data.
2. Whether to keep test result objects in the result or discard after the replication loop. Results are automatically aggregated into traces; this shouldn't be enabled unless you need access to the full `TestResult` featureset per replication.
3. Whether to keep contexts in the result or discard after the replication loop. Contexts are memory-heavy and can cause you to run out of memory for large runs. Proceed with caution if you need to enable this option.
4. Verbosity level for logging output `{0, 1, 2}`. `0` prints nothing, `1` prints throughout for the loop and time elapsed for the post-processing, `2` prints per-step throughput for the loop and time elapsed for the post-processing.

```bash
>>> MC run concluded successfully with 1434.06 it/s.
Per-step Report:

    datagen: 0 faliures, 3724.37 it/s.
    filter: 0 faliures, 3529.53 it/s.
    custom_std: 0 faliures, 20365.73 it/s.
    builtin_std: 0 faliures, 20673.25 it/s.
    std_innov_mean: 0 faliures, 25025.34 it/s.

Post-processing Report:

    custom_postproc: Succeeded in 0.0003s.
    builtin_kde: Succeeded in 1.3634s.
```

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
>>>             mean_statistic  mean_pval  rejection_rate  ci_low  ci_high
std_innov_mean           2.698      0.555           0.053   0.041    0.069

```

Regression results are accessed similarly in `MCPipelineResult.regression_summaries`; and post-processing are accessed by key in `MCPipelineResult.postproc: dict[str, Any]`.

## Conclusion

This guide demonstrates the usage of basic MC functionality through the pre-configured steps available in the library.
Custom transforms are available through `transform_step(...)` and bundle-safe custom operations can be wrapped with `custom_operation`. See the [Monte Carlo custom operation API reference](../documentation/monte_carlo/custom_ops.md) for the current contract.

For future reference or a ready-made boilerplate, you can visit [this](../assets/monte_carlo.ipynb) link to access a notebook containing the process outlined in this guide.

[Download MC Guide Notebook](../assets/monte_carlo.ipynb){ .md-button download="" }
