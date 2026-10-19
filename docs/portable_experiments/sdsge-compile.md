---
tags:
    - guide
---

# `sdsge-compile`

`sdsge-compile` assembles a `.sdsge` bundle from a conventional directory layout.

```bash
sdsge-compile <source-directory> [-o OUTPUT] [--created-by NAME]
```

## Directory layout

```text
my-experiment/
    ├── manifest.json                       # the index
    ├── {model_name}.yaml                   # model config, one per model
    ├── simulation.json                     # simulation prefills
    ├── data/*.parquet|.csv                 # raw observable files
    ├── shock/
    │   ├── path/{id}.parquet|.csv          # supplied paths, pooled by content id
    │   └── table/{id}.parquet|.csv         # tabulated distributions, same
    ├── estimation/
    │   ├── spec.json
    │   ├── result.json                     # optional
    │   ├── observed.parquet|.csv           # optional
    │   └── posterior.parquet|.csv          # optional, MCMC only
    └── montecarlo/
        ├── pipeline.json
        ├── custom/{step}.pkl               # custom ops, cloudpickled
        ├── data/{step}.parquet|.csv        # raw_model_data arrays
        └── result/                         # optional; present with a run
            ├── meta.json                   # the run's own metadata + run_config
            ├── tests/{test_steps.json, test_traces.parquet|.csv}
            ├── regressions/{regression_steps.json, regression_traces.parquet|.csv}
            ├── transforms/{transform_steps.json, {step}_{field}.parquet|.csv}
            └── postproc/{postproc_steps.json, {step}_{field}.parquet|.csv}

```

???+ note "Model files"
    Each model YAML is named after the model it carries. An estimation member that names no model resolves to `reference`.

## Arguments

| Argument         | Description                                                                       |
|------------------|-----------------------------------------------------------------------------------|
| `source`         | Directory containing the bundle members.                                          |
| `-o`, `--output` | Output `.sdsge` path. Defaults to `<source>.sdsge` next to the source directory.  |
| `--created-by`   | Override the manifest `created_by` field. Defaults to `"SymbolicDSGE <version>"`. |

## See also

- [`sdsge-decompile`](sdsge-decompile.md): the inverse operation.
- [Bundle Authoring Guide](../guides/bundle_authoring_guide.md): assemble a bundle from code.
- [`BundleBuilder`](../documentation/bundle/BundleBuilder.md): the in-code equivalent of this CLI.
