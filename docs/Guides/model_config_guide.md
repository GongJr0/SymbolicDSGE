---
tags:
    - guide
---

# Model Configuration Guide

??? tip "__TL;DR__"
    You can see an example config [here](../assets/test.yaml).

`SymbolicDSGE` models are configured through a YAML file. Similar to many familiar DSGE engines, the configuration contains:

- Parameter declarations
- Constraint definitions
- Model equations
- Measurement equations (For post-solution observables)
- Parameter calibration
- Shock symbol declarations

This guide contains detailed information about config sections, how they are parsed, and the conventions users are expected to follow for correct parsing.
The ordering of fields does not matter for the parser, however ordering of the components can change the model behavior. We will start with an empty config and build components to create a valid model in this guide.

To start with, the configuration accepts a `name` field to specify the model's alias. This name is accessible in the parsed model but never used; it remains in the model object as a reference for users.

```yaml
name: "Test Model"
```

## Variables

The `variables` field contains the names for all primary model variables. (no time indices or parameters) It is declared as a list and the ordering of variables will be respected in the solver unless explicitly given a separate ordering. In addition to variable names, each variable requires an explicit boolean entry in the `constrained` field. Having this toggle allows the constraint equations to be predefined in the config but only used when explicitly enabled.

Variables are declared as follows:
```yaml
variables: [g, z, r, Pi, x]
constrained:
    g: false
    z: false
    r: false
    x: false
    Pi: false
```
## Parameters

Parameters are "constants" that appear in the model equations in some capacity.
Common examples of parameters are:

- Shock persistence terms
- Shock (co)variances
- Steady state values
- Model parameters such as the discount factor (often $\beta$)

Ordering of the parameters does not matter in the configuration file.
The `parameters` field is again declared as a list.
```yaml
parameters: [beta, kappa, tau_inv,
             psi_pi, psi_x, rho_r,
             rho_g, rho_z,
             pi_star, r_star,
             sig_r, sig_g, sig_z,
             rho_gz]
```

???+ note "Calibration Values"
    `SymbolicDSGE` currently expects each parameter to have known values.
    estimation/inference will be implemented but are not accessible as of now.

## Shocks

???+ note "Wording Convention"
    This guide uses the term "(co)variance" to refer to a shock term's variance/covariance parameters for brevity. It's important to note that `SymbolicDSGE` expects __standard deviations__ and __correlation coefficients__ in the configuration.

Shocks are the symbols that represent the stochastic components of the model.
A shock symbol is separate from its (co)variance and is used to indicate where a respective innovation should be applied in the model equations.

```yaml
shock_map:
    e_r: r
    e_g: g
    e_z: z
```

Shock realizations are only injected when the user selects them at simulation time. Therefore, declaring extra variables here and including them in the model equations can be used to test multiple shock configurations from a single model config.

## Observables

Observables map model units to real life variables via equations. For the `observables` field we only declare the names we desire to use as observable variables.
```yaml
observables: [Infl, Rate]
```

## Equations

Equations contain the bulk of model dynamics. In `SymbolicDSGE` the field is used as a parent to model equations, constraints, and observable equations. We declare the necessary fields:
```yaml
equations:
    model: ...
    constraint: ...
    observables: ...
```
The equations field treats all variables as a function of time; to refer to past, current, and future observations we use `#!python x(t-1)`, `#!python x(t)`, and `#!python x(t+1)` respectively.

### Model Equations

This field contains the state-space definition. Multiple equations are supplied to form all necessary interactions.
```yaml
equations:
    model:
        - Pi(t) = beta*Pi(t+1) + kappa*x(t) + z(t) # (1)!

        - x(t) = x(t+1) - tau_inv*(r(t) - Pi(t+1)) + g(t) # (2)!

        - r(t) = rho_r*r(t-1) + (1 - rho_r)*(psi_pi*Pi(t) + psi_x*x(t)) + e_r # (3)!

        - g(t) = rho_g*g(t-1) + e_g # (4)!

        - z(t) = rho_z*z(t-1) + e_z # (5)!
    constraint: ...
    observables: ...
```

1. New Keynesian Phillips Curve (NKPC)
2. IS/Euler Equation
3. Taylor Rule
4. Demand Shock
5. Cost-Push Shock

Here, we use these variables and parameters that we defined to create the namespace.

### Constraints
The constraints field is available and parsed in `SymbolicDSGE`. However, the constraints are currently not supported in the solver and therefore are not enforced. The `constraint` field takes a `{variable: equation, ...}` style dictionary. The behavior is similar for the `constrained` toggle; it will be parsed and cross-checked with the given equations. However, the solver will not act on the given equations. For correctness, we will leave the field empty in the example config.

```yaml
equations:
    model:
        - Pi(t) = beta*Pi(t+1) + kappa*x(t) + z(t)

        - x(t) = x(t+1) - tau_inv*(r(t) - Pi(t+1)) + g(t)

        - r(t) = rho_r*r(t-1) + (1 - rho_r)*(psi_pi*Pi(t) + psi_x*x(t)) + e_r

        - g(t) = rho_g*g(t-1) + e_g

        - z(t) = rho_z*z(t-1) + e_z
    constraint: {...}
    observables: ...
```

### Observables
This field contains the mappings of model variables to real-life observed variables. In our example, we defined two observables in the namespace; and we will define the equations to construct them here. Observable equations can be constructed from any parameter/variable combinations. If a constant is required as a scaling factor or an offset, it should be declared as a parameter (to ensure `#! SymPy` parses correctly). As a note, observable equations are expected to correspond to current time. Observable equations must be functions of current state variables. (no `t+1` terms)

```yaml
equations:
    model:
        - Pi(t) = beta*Pi(t+1) + kappa*x(t) + z(t)

        - x(t) = x(t+1) - tau_inv*(r(t) - Pi(t+1)) + g(t)

        - r(t) = rho_r*r(t-1) + (1 - rho_r)*(psi_pi*Pi(t) + psi_x*x(t)) + e_r

        - g(t) = rho_g*g(t-1) + e_g

        - z(t) = rho_z*z(t-1) + e_z
    constraint: {...}
    observables:
        Infl: 4*Pi(t) + pi_star # (1)!

        Rate: 4*r(t) + (r_star + pi_star) # (2)!
```

1. Annualized inflation from quarterly gap
2. Annualized nominal rate from quarterly gap.


## Calibration
The `calibration` field stores values and shock variance specifications to annotate the corresponding values of all model components except the variables.
The field is a parent containing two sub-fields:
```yaml
calibration:
    parameters: ...
    shocks: ...
```

### Parameters
This section is used to define the known values of model parameters.
All parameters defined in the namespace must (for now) have a value entry here.
```yaml
calibration:
    parameters:
        beta: 0.99

        psi_pi: 2.19
        psi_x: 0.30
        rho_r: 0.84

        pi_star: 3.43
        r_star: 3.01

        kappa: 0.58
        tau_inv: 1.86

        rho_g: 0.83
        rho_z: 0.85
        rho_gz: 0.36

        sig_r: 0.18
        sig_g: 0.18
        sig_z: 0.64
    shocks: ...
```

### Shocks
The shocks section maps shock (co)variances to the corresponding terms in model equations. Shock terms that are defined but not included in this field will use default values.

- Innovations without a specified standard deviation will assume `1.0`
- Correlation between excluded pairs will assume `0.0`

???+ warning "Shock Parameter Convention"
    To align with `SciPy` distributions' signatures, the standard deviations of stochastic terms are used instead of the variance.

???+ info "Shock Selection at Simulations"
    At simulation time, shocks are specified by the exogenous state variable names (e.g., `g`, `z`) or by grouped keys (`#!python "g,z"`), not by the innovation symbols (`e_g`)

```yaml
calibration:
    parameters:
        beta: 0.99

        psi_pi: 2.19
        psi_x: 0.30
        rho_r: 0.84

        pi_star: 3.43
        r_star: 3.01

        kappa: 0.58
        tau_inv: 1.86

        rho_g: 0.83
        rho_z: 0.85
        rho_gz: 0.36

        sig_r: 0.18
        sig_g: 0.18
        sig_z: 0.64
    shocks:
        std:
            e_r: sig_r
            e_g: sig_g
            e_z: sig_z
        corr:
            e_g, e_z: rho_gz
```

Innovation terms are paired with the relevant (co)variance parameters through the `std` and `corr` fields of the configuration.

## Conclusion
With all components defined, the configuration file now fully specifies a solvable symbolic DSGE model. The parser will construct the symbolic state-space representation, apply calibration, and prepare the model for solution and simulation.

For future reference or a ready-made boilerplate, you can visit [this](https://github.com/GongJr0/SymbolicDSGE/blob/main/MODELS/POST82.yaml) link to see a test configuration in the `SymbolicDSGE` repository.

[Download Test Config](../assets/test.yaml){ .md-button download="" }
