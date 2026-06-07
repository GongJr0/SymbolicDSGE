---
tags:
    - guide
---

# Builder Tab

The Builder tab provides a YAML editor for creating or modifying a model configuration.

The editor supplies SymbolicDSGE configuration completions and diagnostics for supported fields and input types. The currently loaded configuration is placed in the editor when a load event occurs.

## Loading a Configuration

1. Enter or edit the YAML configuration.
2. Select the intended `reference` or `dgp` role in the sidebar.
3. Select **Load** in the Builder panel to parse the editor contents.
4. Select **Solve** in the sidebar when the configuration is ready.

The sidebar **Load** button reads from the configured YAML path instead of the editor contents.

???+ note "Model Roles"
    The `reference` role corresponds to the main model of interest, while the `dgp` role is only required for Monte Carlo comparisons of the reference model to a data generating process. The `dgp` role is optional and can be left empty if not needed.

???+ warning "Loading Replaces the Active Slot"
    Loading a configuration replaces the parsed model in the selected role and clears its existing solved model. Solve the newly loaded configuration before using simulation or Monte Carlo features.

## Syncing

Select **Sync** to refresh model status from the backend session. Syncing does not overwrite non-empty editor contents.

Use the **Linearize** switch before solving configurations written in nonlinear levels. The configuration must provide the required linearization specification.
