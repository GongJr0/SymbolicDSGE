---
tags:
    - guide
---

# Outputs Tab

The Outputs tab simulates the active solved model and displays the resulting series.

## Running a Simulation

1. Select the model role in the sidebar.
2. Load and solve the model.
3. Configure shocks in the Spec tab.
4. Set the number of periods, `T`.
5. Choose whether to include observables.
6. Select **Simulate**.

The simulation uses the current generated-shock parameters or raw paths from the Spec tab.

## Inspecting Results

Simulation results are divided into three panels:

| Panel | Description |
| --- | --- |
| Graph | Select and compare generated state, observable, and transform series. |
| Table | Inspect the same series in tabular form and filter or sort columns. |
| Figures | Display figures produced by submitted Plot functions. |

Series keep consistent graph colors as selections change. The internal combined state array `_X` is omitted from the graph and table views.

The table rounds displayed values to three decimal places. CSV export retains the full stored precision.
