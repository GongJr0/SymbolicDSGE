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
The ordering of top level fields does not matter for the parser. Component order is preserved as a stable fallback in places where the model does not imply a unique order, but variables do not need to be manually arranged for the solver backend. We will start with an empty config and build components to create a valid model in this guide.

To start with, the configuration accepts a `name` field to specify the model's alias. This name is accessible in the parsed model but never used; it remains in the model object as a reference for users.

```yaml
name: Test Model
```

## Variables

The `variables` field contains the names and some optional configuration for all primary model variables. (no time indices or parameters) It is declared as a list or mapping.

When using a mapping instead of a list, each variable can specify a preferred linearization method and the parameter corresponding to its steady state level. This additional information is only used when linearizing a nonlinear model.

Variables are declared as follows:

```yaml
variables: [g, z, r_shadow, r, Pi, x]  # as list
variables: # as mapping
    g: # (1)!
    z:
    r_shadow:
    r:
        linearization: log  # (2)!
        ss_seed: r_star # (3)!
    ...
```

1. Exogenous processes are already linear and have no steady states. When fields are not specified we infer `linearization: none` automatically.
2. Can be one of `log`, `taylor`, or `none`.
3. Newton seed for the steady state. Omitted seeds at zero. Doubles as the expansion point under `log` or `taylor`.

???+ info "Ordering Convention"
    At compile time, the solver infers its canonical layout from the model config: after auxiliaries are generated (where necessary), a variable occurring at `t-1` is a state and every other variable is a control. During model compilation the canonical order is derived as `[states; controls]`.

    You need not know or check the internal ordering to interact with methods requiring parameters supplied as a vector. The primary parameters needing ordered input are `x0` (initial state) and `ss_seed` (steady state solver seed). Both of which accept a dictionary mapping names to values. For dense vector inputs:

    - `x0` requires all variables, including auxiliaries in the order `[declared_variables; generated_auxiliaries]`.
    - `ss_seed` requires all declared variables (the `variables` field) in declaration order.

## Parameters

Parameters are "constants" that appear in the model equations in some capacity.
Common examples of parameters are:

- Shock persistence terms
- Shock (co)variances
- Steady state values
- Model parameters such as the discount factor (often $\beta$)

Parameters are declared by their entries under [`calibration.parameters`](#parameters_1).
There is no separate `parameters` list; a name is a parameter if and only if it is calibrated.

```yaml
calibration:
    parameters:
        beta: 0.99
        kappa: 0.58
        tau_inv: 1.86
```

???+ note "Calibration Values"
    `SymbolicDSGE` currently expects each parameter to have known values.
    These calibration values are used as defaults and as initial guesses for estimation workflows (`mle`, `map`, `mcmc`) when no explicit `theta0` is supplied.

## Shocks

???+ note "Wording Convention"
    This guide uses the term "(co)variance" to refer to shock variance and covariance parameters for brevity. It's important to note that `SymbolicDSGE` expects __standard deviations__ and __correlation coefficients__ in the configuration.

Shocks are the symbols that represent the stochastic components of the model.
A shock symbol is separate from its (co)variance and is used to indicate where a respective innovation should be applied in the model equations.

```yaml
shocks:
    - e_r
    - e_g
    - e_z
```

Shock realizations are only injected when the user selects them at simulation time. Therefore, declaring extra variables here and including them in the model equations can be used to test multiple shock configurations from a single model config.

## Observables

Observables map model units to real life variables via equations. For the `observables` field we only declare the names we desire to use as observable variables.

```yaml
observables: [OutGap, Infl, Rate]
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

This field contains the elementary state-space definition. Equations are supplied as a mapping from equation name to equation, forming all necessary interactions.

```yaml
equations:
    model:
        nkpc: Pi(t) = beta*Pi(t+1) + kappa*x(t) + z(t) # (1)!

        euler: x(t) = x(t+1) - tau_inv*(r(t) - Pi(t+1)) + g(t) # (2)!

        taylor_shadow: r_shadow(t) = rho_r*r_shadow(t-1) + (1 - rho_r)*(psi_pi*Pi(t) + psi_x*x(t)) + e_r # (3)!

        rate_link: r(t) = r_shadow(t) # (4)!

        g_process: g(t) = rho_g*g(t-1) + e_g # (5)!

        z_process: z(t) = rho_z*z(t-1) + e_z # (6)!
    constraint: ...
    observables: ...
```

1. New Keynesian Phillips Curve (NKPC)
2. IS/Euler Equation
3. Shadow Taylor rule
4. Actual policy rate outside the binding regime
5. Demand shock
6. Cost-push shock

Here, we use these variables and parameters that we defined to create the namespace.

### Constraints

The `constraint` field stores named piecewise OBC conditions. Each condition defines `bind` and, optionally, `relax` expressions. Conditions are parsed as `SymPy` relational expressions and must be contemporaneous. The corresponding `regime` entry replaces model equations when that condition binds.

```yaml
equations:
    model:
        nkpc: Pi(t) = beta*Pi(t+1) + kappa*x(t) + z(t)

        euler: x(t) = x(t+1) - tau_inv*(r(t) - Pi(t+1)) + g(t)

        taylor_shadow: r_shadow(t) = rho_r*r_shadow(t-1) + (1 - rho_r)*(psi_pi*Pi(t) + psi_x*x(t)) + e_r

        rate_link: r(t) = r_shadow(t)

        g_process: g(t) = rho_g*g(t-1) + e_g

        z_process: z(t) = rho_z*z(t-1) + e_z
    constraint:
        ZLB:
            bind: r_shadow(t) < 0
            relax: r_shadow(t) >= 0 # (1)!
    regime:
        ZLB:
            rate_link: r(t) = 0
    observables: ...
```

1. The binding regime pins the actual policy rate while the shadow rate determines whether the constraint binds. If `relax` is omitted, the parser derives it as `Not(bind)`.

### Observables

This field contains the mappings of model variables to real-life observed variables. In our example, we defined two observables in the namespace; and we will define the equations to construct them here. Observable equations can be constructed from any parameter/variable combinations. If a constant is required as a scaling factor or an offset, it should be declared as a parameter (to ensure `#! SymPy` parses correctly). As a note, observable equations are expected to correspond to current time. Observable equations must be functions of current state variables. (no `t+1` terms)

```yaml
equations:
    model:
        nkpc: Pi(t) = beta*Pi(t+1) + kappa*x(t) + z(t)

        euler: x(t) = x(t+1) - tau_inv*(r(t) - Pi(t+1)) + g(t)

        taylor_shadow: r_shadow(t) = rho_r*r_shadow(t-1) + (1 - rho_r)*(psi_pi*Pi(t) + psi_x*x(t)) + e_r

        rate_link: r(t) = r_shadow(t)

        g_process: g(t) = rho_g*g(t-1) + e_g

        z_process: z(t) = rho_z*z(t-1) + e_z
    constraint:
        ZLB:
            bind: r_shadow(t) < 0
            relax: r_shadow(t) >= 0
    regime:
        ZLB:
            rate_link: r(t) = 0
    observables:
        OutGap: x(t) # (1)!

        Infl: 4*Pi(t) + pi_star # (2)!

        Rate: 4*r(t) + (r_star + pi_star) # (3)!
```

1. Output gap is a direct mapping of the model variable to the observable.
2. Annualized inflation from quarterly gap
3. Annualized nominal rate from quarterly gap.

## Calibration

The `calibration` field stores values and shock variance specifications to annotate the corresponding values of all model components except the variables.
The field is a parent containing two sections:

```yaml
calibration:
    parameters: ...
    shocks: ...
```

### Parameters

This section declares the model parameters and their known values.
Any name referenced by the equations, shock calibration, or Kalman block must appear here.

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
    At simulation time, shocks are specified by the name of the shock itself (e.g., `e_g`, `e_z` instead of `g`, `z`) or by grouped keys (`#!python "e_g,e_z"`). The latter allows a multivariate distribution to be specified for all, or a subset of, the shocks in the model.

```yaml
calibration:
    parameters:
        beta: 0.99

        psi_pi: 2.19
        psi_x: 0.30
        rho_r: 0.84

        pi_star: 3.43
        r_star: 3.01 # (1)!

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

1. This parameter will be used for linearization as declared, but it is not reserved for that purpose.

Innovation terms are paired with the relevant (co)variance parameters through the `std` and `corr` fields of the configuration.

## Conclusion

With all components defined, the configuration file now fully specifies a solvable symbolic DSGE model. The parser will construct the symbolic state-space representation, apply calibration, and prepare the model for solution and simulation.

For future reference or a ready to use boilerplate, you can visit [this](https://github.com/GongJr0/SymbolicDSGE/blob/main/MODELS/POST82.yaml) link to see a test configuration in the `SymbolicDSGE` repository.

[Download Test Config](../assets/test.yaml){ .md-button download="" }
