---
tags:
    - guide
---

# Spec Tab

The Spec tab exposes the active model structure, shock configuration, and user-defined output functions.

## Model Overview

The model panel reports:

- Variables and observables.
- Inferred state and exogenous-state counts.
- The active model name and role.

## Shock Configuration

Select one of two shock sources:

| Source | Description |
| --- | --- |
| Shock | Generate innovations from a normal, Student-t, or uniform distribution. |
| Raw | Enter complete shock paths directly. |

Generated shocks expose the distribution seed and parameters together with the model's configured shock standard deviations and correlations. Changes are applied when a simulation is run.

Raw paths accept values separated by spaces, commas, or semicolons. Each supplied path must contain exactly the number of periods requested in the Outputs tab.

## Transforms and Plots

The Python panels register functions that run against simulation output:

- **Transform** functions return a one-dimensional array and add it as an output series.
- **Plot** functions return a Matplotlib figure and add it to the Figures panel.

Function arguments are matched to available state and observable names. Submit a function to register it for the active model role, or remove it from the list beneath the editor.

???+ note "Execution Scope"
    Submitted functions run in the local backend process. They should only contain code you trust.
