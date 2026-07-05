---
tags:
    - guide
---

# Portable Experiments

A `.sdsge` file packages everything needed to reopen a model in the GUI or recover its components in code — model and Kalman configuration, estimation spec + results, Monte Carlo pipeline + traces, observed data, and a simulation prefill — into a single shareable archive. The file is a zip with a versioned manifest (`manifest.json`) and an aliased extension; nothing about reading or writing it requires a custom tool beyond the bundled CLIs.

The container exists so a non-coding collaborator can run an experiment by opening one file, and so a coding collaborator can reach every component the bundle carries without re-deriving them.

## What lives in a bundle

| Component | Format | Source |
| --- | --- | --- |
| Model config | YAML | `ModelParser` source (path or string) |
| Estimation spec | JSON | `EstimationSpec.to_json()` |
| Estimation result metadata | JSON | `OptimizationResultMeta` / `MCMCResultMeta` |
| Observed data | CSV or Parquet | Author-supplied or `Estimator` inputs |
| MCMC posterior | CSV or Parquet | `MCMCResult.samples` + `logpost_trace` |
| Monte Carlo pipeline | JSON | `PipelineSpec.to_json()` |
| Monte Carlo result + traces | JSON + CSV/Parquet | `MCPipelineResult` |
| Simulation prefill | Inline (manifest) | `{role: SimSpec}` |

???+ note "Authoring formats"
    Bulk numeric members are written as Parquet by default for size. CSV authoring is also supported — `sdsge-compile --csv-only` keeps everything as CSV, and `sdsge-decompile --csv` re-encodes Parquet members back to CSV. The reader is format-agnostic: a hand-zipped CSV-only bundle and a CLI-built Parquet bundle both validate.

## Entry points

There are three ways to produce or consume a `.sdsge`:

- **Command line** — [`sdsge-compile`](sdsge-compile.md) walks a directory layout and assembles a bundle; [`sdsge-decompile`](sdsge-decompile.md) extracts a bundle into a re-compilable directory.
- **In-code** — [`SolvedModel.save_sdsge()`](../documentation/SolvedModel.md) and [`SymbolicDSGE.load_bundle()`](../documentation/bundle/index.md) handle the round-trip from Python; the [Authoring](../guides/bundle_authoring_guide.md) and [Loading](../guides/bundle_loading_guide.md) guides walk through both.
- **GUI** — passing a `.sdsge` to [`sdsge-ui`](../gui_usage/index.md) launches the playground with every tab pre-populated.

???+ info "When to author from code vs from a directory"
    Author from code when the bundle is the output of a Python workflow (you already have a `SolvedModel`, an `MCPipelineResult`, etc.). Author from a directory when the bundle is composed by hand or by a non-Python tool — drop CSVs / YAMLs / JSONs into the conventional layout and call `sdsge-compile`.

## Determinism

The bundle stores simulation specs and a seed, not simulation results — `Run` on the receiver's side reproduces the author's intended simulation deterministically (numpy `PCG64` + fixed seed). Re-solving the embedded YAML at load time is also deterministic, so the receiver's policy matrices match the author's.

???+ warning "Source YAML embedding"
    `SolvedModel.save_sdsge()` requires a source YAML — populated automatically when the model was loaded via `ModelParser(path)` or `ModelParser.from_string(text)`. Programmatically constructed `ModelConfig` instances need `yaml_text=` passed explicitly.

## Where to next

- [`sdsge-compile`](sdsge-compile.md): CLI reference for directory → `.sdsge`.
- [`sdsge-decompile`](sdsge-decompile.md): CLI reference for `.sdsge` → directory.
- [Bundle Authoring Guide](../guides/bundle_authoring_guide.md): assemble a full bundle in code.
- [Bundle Loading Guide](../guides/bundle_loading_guide.md): open a bundle and reach each library object.
- [`bundle` API reference](../documentation/bundle/index.md): class- and function-level documentation.
