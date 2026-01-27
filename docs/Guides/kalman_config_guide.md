---
tags:
    - guide
---

# Kalman Filter Configuration Guide

??? tip "__TL;DR__"
    You can see an example config [here](../assets/test.yaml).

???+ warning "Read Model Configuration Guide"
    This guide refers to fields used in model configuration and some parameters relevant to Kalman Filters are part of the model parameter family. Please make sure you've read the [model configuration guide](./model_config_guide.md) before reading this one.

`SymbolicDSGE` uses a single configuration file and appends the Kalman Filter (KF) configuration to the same YAML that carries model information. Although Kalman Filtering can be done in a completely model-detached fashion, there's integration infrastructure provided in the model objects and configurations.

All KF related configuration entries live under the parent field `kalman:` and are parsed into a `KalmanConfig` object at parse time.

???+ note "Config Overrides"
    All configuration fields (except `R:`) can be overridden through function parameters when calling the filter.

## Observables

The config uses a tag `y` to define the list of model measurement equations to include in the Kalman Filter.

```yaml
y: ["Infl", "Rate"] # (1)!
```

1. The listed names are observable names defined in the model configuration

To ensure matrix alignment, names given here will be reordered to match their ordering in the model configuration. You can check the order via `ModelConfig.observables` or `CompiledModel.observable_names` to confirm the positions to expect the output arrays in.

???+ example "Alignment Demonstration"
    Assume we have three observables and use two of them. In the model config, we defined `#!yaml observables: [obs1, obs2, obs3]` and for the KF configuration, we used `#!yaml y: [obs3, obs1]`.

    The alignment occurs on the positional index of observables as given in the model config. So we re-order to `y: [obs1, obs3]` internally.

    This will reflect in all `ndarray`s outputted from a KF run and it is generally advisable to check alignment or confirm the updated positions of observable arrays to avoid future confusion.

## Measurement Covariance

Measurement Covariance ($R$) is constructed through parameters defined in the model configuration. We use the parent `R:` and populate it as such:

```yaml
kalman:
    R:
        std:...
        corr:...
```

Then we populate each sub-field with respective parameter names:

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

???+ info "No Defaults"
    `SymbolicDSGE` does not fall back to defaults when constructing $R$. As of now, heuristics to infer $R$ are not implemented and any "default" is practically guaranteed to be inaccurate. There are future plans to implement robust $R$ inference pipelines; but the no-default behavior will not change until said implementations are in place.


## State Covariance

State Covariance ($P$) defines the inter-state variation through a covariance matrix. $P$ is an inferred parameter in Kalman Filters; but an initial guess $P_0$ is provided through the configuration. As of now, the configuration only allows the creation of diagonally-defined matrices (assumes no correlation) but a full $P_0$ construction will be implemented. Initial guesses are not relevant beyond an (often short) burn-in period; but a well-specified $P_0$ guess can help convergence speeds. In the config we define a parent `P0:` and populate the following fields:

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
    The `diag` field is directly used as the pre-scaling matrix values. Therefore, entries in the `diag` field correspond to variances instead of standard deviations.

## Filter Options

`SymbolicDSGE` defines two KF options that can be adjusted through the configuration file. These options do not belong to a parent tag and are defined as follows:

```yaml
kalman:
    jitter: 1e-10 # (1)!
    symmetrize: true # (2)!
```

1. The number specified is added to covariance matrices if their Cholesky decomposition fails.
2. Covariance matrices are symmetrized if this field is `#!yaml true`. Symmetrization is applied as $(M + M^\top)/2$.

# Conclusion

The configuration fields above provide all KF necessary information that can't (or shouldn't) be inferred from the model state. This config field is only relevant to `#!python SolvedModel.kalman`. If you've read to this point and want to check a complete configuration file including `kalman` and model configurations, you can visit [this](https://github.com/GongJr0/SymbolicDSGE/blob/main/MODELS/POST82.yaml) link.

[Download Test Config](../assets/test.yaml){ .md-button download="" }
