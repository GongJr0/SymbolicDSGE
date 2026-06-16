---
tags:
    - guide
---

# MC Pipeline Tab

The MC Pipeline tab visually constructs repeated experiments comparing a solved reference model against a solved DGP model.

## Building a Pipeline

Add steps from the palette by selecting or dragging them onto the canvas. Select a node to edit its name and parameters in the Step Inspector.

The graph represents data dependencies:

- The data source node is the single root. It may be a DGP simulation or raw data node.
- Filters connect directly from the data source and may feed multiple dependent steps.
- Built-in transforms and custom transforms may consume data, filter outputs, or upstream payloads. They can feed other transforms, tests, or regressions.
- Tests and regressions are terminal and cannot feed downstream nodes.

Step order does not affect the current built-in test and regression results. The backend infers a valid execution order from the dependency graph.

Target vectors accept comma- or space-separated values. Target matrices use one row per line, with values separated by commas or spaces.

Custom transform nodes use a restricted numpy-oriented function contract so they can be validated and, when bundled, restored from the `.sdsge` archive.

## Running and Inspecting Results

Set the replication count and fail-fast behavior, then select **Validate** or **Run pipeline**.

The Run Summary contains three tabs:

| Tab | Description |
| --- | --- |
| Performance | Overall and per-step timings, generated-data summaries, and failures. |
| Tests | Test statistics, p-values, confidence intervals, rejection rates, and trace distributions. |
| Regressions | Coefficient distributions, status counts, fit metrics, and OLS diagnostics where available. |

## Persistence and Clearing

The pipeline graph, node positions, run settings, and latest completed result persist across browser refreshes. Graph edits replace the saved graph without removing the latest result. A successful run replaces the saved result.

The trash button in the Pipeline panel resets the graph to its initial state. **Clear workspace** clears both the graph and latest persisted result.

???+ note "Local Persistence"
    MC workspace state is stored in the browser used to access the localhost GUI. It is not written into the model configuration.
