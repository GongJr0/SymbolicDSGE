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

from SymbolicDSGE import DSGESolver, ModelParser, Shock
from SymbolicDSGE.monte_carlo import (
    MCPipeline,
    reference_filter_step,
    simulation_step,
    wald_test_step,
)

model, kalman = ModelParser("../../MODELS/POST82.yaml").get_all() # (1)!
steady_state = np.zeros(5, dtype=np.float64)  # (2)!

# Solve the reference model
solver = DSGESolver(model, kalman)
compiled = solver.compile(
    n_state=3,
    n_exog=3,
)
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

T = 200  # (1)!
n_obs = len(reference.compiled.observable_names)  # (2)!

pipeline = MCPipeline([...])
```

1. Length of each simulated sample.
2. Number of observables the model(s) have.

`MCPipeline` is used to compile the steps that need to be executed for each repetition.
Every step of `MCPipeline` must be an `MCStep` object.
The pipeline will be built using the step-generating functions built-in to `SymbolicDSGE`.

### Data Generation

???+ warning "Step Ordering"
    Data generation is done exactly once, in the first step of the pipeline.

Using the `simulation_step` function, we generate an `MCStep` object that samples the DGP model with a given simulation specification.

```python
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

With filtered outputs, we run two test steps using the `wald_test_step` function.
One test checks mean equivalence of observables, and the other checks the second moment.

```python

mean_test_step = wald_test_step(
            "std_innov_mean",  # (1)!
            source="std_innov",  # (2)!
            target=np.zeros(n_obs),  # (3)!
            kind="mean",  # (4)!
            burn_in=20,  # (5)!
        )
second_moment_test_step = wald_test_step(
    "std_innov_second_moment",
    source="std_innov",
    target=np.eye(n_obs),
    kind="second_moment",
    burn_in=20,
)
```

1. Name of the step. (This will be used as the key to access the results)
2. Data to use when conducting the test. In this case, it is `std_innov` from `FilterResult` objects returned by the kalman filter.
3. Target to test against. In this case we're testing if the mean of each observable is zero.
4. Kind of the wald test. Available inputs are: `Literal["mean", "covariance", "second_moment"]`.
5. Number of periods to discard before running the tests.

Each test returns a `TestResult` object and the results are aggregated to produce an MC summary.

### Complete Pipeline

```python
pipeline = MCPipeline([
    datagen_step,
    kf_step,
    mean_test_step,
    second_moment_test_step,
])
```

## Running the Pipeline

The `MCPipeline` object explains the procedure that will run per iteration.
`MCPipeline.run` is then used to run the procedure repeatedly.

```python
mc = pipeline.run(
    reference=reference,
    dgp=dgp,
    n_rep=500,
    retain_payloads=False,
    retain_test_results=False,
)

mc.succeeded, mc.n_successful
```

```bash
>>> (True, 500)
```

This returns a `MCPipelineResult` object containing summaries for each test step executed in the pipeline.
To extract the test results, their p-values, and other relevant statistics, we can access the summaries by key (step names).

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
        for name, res in mc.summaries.items()
    }
).T
print(summary.round(4))

```

```bash
>>>                          mean_statistic  mean_pval  rejection_rate  ci_low  \
std_innov_mean                    5.652      0.363           0.224   0.190   
std_innov_second_moment        1400.913      0.000           1.000   0.992   

                         ci_high  
std_innov_mean             0.263  
std_innov_second_moment    1.000  
```

## Conclusion

This guide demonstrates the usage of basic MC functionality through the pre-configured steps available in the library.
Guides on constructing custom `MCStep` objects will be provided once the user-facing API of custom OPs are locked.

For future reference or a ready-made boilerplate, you can visit [this](../assets/monte_carlo.ipynb) link to access a notebook containing the process outlined in this guide.

[Download MC Guide Notebook](../assets/monte_carlo.ipynb){ .md-button download="" }