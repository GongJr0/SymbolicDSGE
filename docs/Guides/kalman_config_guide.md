---
tags:
    - guide
---

# Kalman Filter Configuration Guide

??? tip "__TL;DR__"
    You can see an example config [here](../assets/test.yaml).

???+ warning "Read Model Configuration Guide"
    This guide refers to fields used in model configuration and some parameters relevant to Kalman Filters are part of the model parameter family. Make sure you've read the [model configuration guide](./model_config_guide.md) before reading this one.

`SymbolicDSGE` uses a single configuration file and appends the Kalman Filter (KF) configuration to the same YAML that carries model information. All KF related configuration entries live under the parent field `kalman:` and are parsed into a `KalmanConfig` object at parse time. The config block accepts `R` and `P0`.

???+ note "Config Overrides"
    Runtime filter options and observable subset selection are passed to `SolvedModel.kalman(...)`.

## Observables

The model configuration's `observables` list defines the measurement equations available to the Kalman filter. `kalman.R` is built in that order. A filter call can select a subset through `SolvedModel.kalman(..., observables=...)`; the selected covariance block is sliced from the configured full `R`.

???+ info "Array Alignment"
    `DataFrame` inputs are aligned by column name. `ndarray` inputs are interpreted positionally against the selected observables.

## Measurement Covariance

Measurement Covariance ($R$) is constructed through parameters defined in the model configuration. We use the parent `R:` and populate it as such:

```yaml
kalman:
    R:
        std:...
        corr:...
```

Then we populate each section with respective parameter names. The `std` map must include every observable declared in the model configuration when `R` is supplied. The `corr` map is optional for each observable pair; omitted pairs are treated as zero correlation.

```yaml
kalman:
    R:
        std:
            Infl: meas_infl # (1)!
            Rate: meas_rate # (2)!

        corr:
            Infl, Rate: meas_rho_ir # (3)!
```

1. Defined as a parameter and given calibration value in model config.
2. Defined as a parameter and given calibration value in model config.
3. Defined as a parameter and given calibration value in model config.

???+ info "No Standard Deviation Defaults"
    `SymbolicDSGE` does not infer measurement standard deviations when constructing $R$. Each configured observable needs an explicit standard deviation parameter.


## State Covariance

State Covariance ($P$) defines the inter-state variation through a covariance matrix. $P$ is an inferred parameter in Kalman Filters, but an initial guess $P_0$ is provided through the configuration. `P0` supports diagonal and scaled identity initialization. Initial guesses are not relevant beyond an often short burn in period, but a well specified $P_0$ guess can help convergence speeds. In the config we define a parent `P0:` and populate the following fields:

```yaml
kalman:
    P0:
        mode: diag # (1)!
        scale: 10.0 # (2)!
        diag: # (3)!
            g: 1.0
            z: 1.0
            r: 1.0
            Pi: 1.0
            x: 1.0
```

1. `P0` construction mode. `#!python "eye"` uses $I_n \times \operatorname{scale}$ while `#!python "diag"` constructs the matrix from below defined diagonal values before scaling by `scale`.
2. Scaling factor of P0.
3. Diagonal entries of the covariance matrix.

???+ note "Diagonal Values"
    The `diag` field is directly used as the matrix values before scaling. Therefore, entries in the `diag` field correspond to variances instead of standard deviations.

## Filter Options

`jitter` and `symmetrize` are runtime options on `SolvedModel.kalman(...)`. `jitter` is added to covariance matrices if their Cholesky decomposition fails. `symmetrize=True` applies $(M + M^\top)/2$ to covariance matrices during filtering.

## Conclusion

The configuration fields above provide all KF necessary information that can't (or shouldn't) be inferred from the model state. This config field is only relevant to `#!python SolvedModel.kalman`. If you've read to this point and want to check a complete configuration file including `kalman` and model configurations, you can visit [this](https://github.com/GongJr0/SymbolicDSGE/blob/main/MODELS/POST82.yaml) link.

[Download Test Config](../assets/test.yaml){ .md-button download="" }
