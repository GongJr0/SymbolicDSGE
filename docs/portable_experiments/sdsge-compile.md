---
tags:
    - guide
---

# `sdsge-compile`

`sdsge-compile` assembles a `.sdsge` bundle from a conventional directory layout. The layout is auto-detected — no separate manifest file is required to drive the compile.

```bash
sdsge-compile <source-directory> [-o OUTPUT] [--csv-only] [--created-by NAME]
```

## Directory layout

A bundle source directory has at least one model YAML at its root. Every other entry is optional.

```text
my-experiment/
    ├── manifest.json                       # the index, and the simulation prefill
    ├── reference.yaml                      # model_config, at the root here
    ├── dgp.yaml                            # optional second role
    ├── data/*.parquet|.csv                 # raw observable files
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

???+ note "Model role files"
    `reference.yaml` is the main model. `dgp.yaml` is only required when the bundle ships a Monte Carlo pipeline that needs a data-generating process. At least one of the two must be present.

???+ info "Options sidecars"
    `<role>.options.json` carries the `compile_kwargs` and `solve_kwargs` the loader will use to rebuild the `SolvedModel`. The file is a JSON object:

    ```json
    {
        "compile_kwargs": {"n_state": 3, "n_exog": 2},
        "solve_kwargs": {}
    }
    ```

    Both keys are optional; an absent file is equivalent to empty kwargs.

## Arguments

| Argument         | Description                                                                                                                           |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `source`         | Directory containing the bundle members.                                                                                              |
| `-o`, `--output` | Output `.sdsge` path. Defaults to `<source>.sdsge` next to the source directory.                                                      |
| `--csv-only`     | Keep CSV bulk members as CSV instead of converting to Parquet. The reader is format-agnostic, so the resulting bundle is still valid. |
| `--created-by`   | Override the manifest `created_by` field. Defaults to `"SymbolicDSGE <version>"`.                                                     |

## Examples

Compile with all defaults — Parquet conversion on, output alongside the directory.

```bash
sdsge-compile my-experiment/
```

CSV-only bundle for a hand-zip-friendly workflow:

```bash
sdsge-compile my-experiment/ --csv-only -o my-experiment.sdsge
```

## Validation

`sdsge-compile` cross-checks observed-data columns against the model's declared observables. The check is **strict on order** because downstream consumers operate by column index. Three failure modes have dedicated messages:

| Failure                 | Message excerpt                                                                                | Remedy                                         |
|-------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------|
| Numeric-looking headers | `inferred observable names look numeric; file may be missing a header row`                     | Add observable names as the first CSV row.     |
| Name mismatch           | `columns [...] do not match model observables [...]. Rename columns to match (order matters).` | Rename CSV/Parquet columns to match the model. |
| Count mismatch          | `has K columns but model declares N observables`                                               | Add or remove columns to match.                |

???+ warning "Order matters"
    The observable-name validation enforces same names **and** same order — even sets that match cause failure if reordered. The bundle's consumers index observed data by position, so reordering would silently produce wrong results.

## Member format inference

The reader dispatches each member by file extension:

| Extension       | Format  |
|-----------------|---------|
| `.yaml`, `.yml` | YAML    |
| `.json`         | JSON    |
| `.csv`          | CSV     |
| `.parquet`      | Parquet |

???+ note "Both formats present"
    Authoring both `observed.csv` and `observed.parquet` (or any other CSV/Parquet pair) is rejected at compile time with `both ... and ... exist in <dir>; choose one`.

## Embedding pre-computed result members

`estimation/result.json` and `montecarlo/result.json` are embedded verbatim — they are code-generated artifacts. Their wire shape is:

```json
{
    "type": "mcmc",
    "data": {
        "param_names": ["beta", "sigma"],
        "accept_rate": 0.31,
        "n_draws": 1000,
        "burn_in": 100,
        "thin": 2,
        "random_state": 0
    }
}
```

`type` is inferred from the result type (`"mle"`, `"map"`, `"mcmc"`); `data` is the result's textual data (complete for MLE/MAP, without traces for MCMC).

## See also

- [`sdsge-decompile`](sdsge-decompile.md) — the inverse operation.
- [Bundle Authoring Guide](../guides/bundle_authoring_guide.md) — assemble a bundle from code.
- [`BundleBuilder`](../documentation/bundle/BundleBuilder.md) — the in-code equivalent of this CLI.
