---
tags:
    - guide
---

# GUI Usage

The SymbolicDSGE GUI is a localhost workspace for loading, inspecting, solving, and experimenting with model configurations.

Install the optional UI dependencies and launch the application with:

```bash
pip install "SymbolicDSGE[ui]"
sdsge-ui
```

A solved model can also open the GUI with itself preloaded as the reference model:

```python
solved.serve()
```

## Shared Controls

The sidebar controls the active model role and is available from every tab.

| Control | Description |
| --- | --- |
| Role | Select the `reference` or `dgp` model slot. |
| YAML Path | Path to a model configuration file. |
| Load | Parse the selected YAML file into the active model slot. |
| Sync | Refresh the displayed backend session state. |
| Linearize | Pass `linearize=True` when compiling the model. |
| Solve | Compile and solve the loaded model in the active slot. |

???+ note "Reference and DGP Roles"
    Most single-model workflows use the reference slot. The Monte Carlo builder requires solved reference and DGP models.

## Tabs

- [Builder](builder.md): edit and load YAML model configurations.
- [Spec](spec.md): inspect the model, configure shocks, and submit transforms or plots.
- [Outputs](outputs.md): simulate the active model and inspect its generated series.
- [MC Pipeline](monte_carlo.md): visually construct and run repeated simulation experiments.

Panels throughout the GUI can be resized, folded, and rearranged where drag controls are available.
