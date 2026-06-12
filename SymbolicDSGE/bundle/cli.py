"""CLI entry points for ``sdsge-compile`` and ``sdsge-decompile``.

``sdsge-compile <dir>`` walks a conventional directory layout and assembles a
``.sdsge`` bundle::

    my-bundle/
    ├── reference.yaml              # required (or dgp.yaml, or both)
    ├── reference.options.json      # optional {"compile_kwargs":..., "solve_kwargs":...}
    ├── dgp.yaml / dgp.options.json # optional
    ├── estimation/
    │   ├── spec.json               # required if estimation/ is present
    │   ├── result.json             # optional ({"type":"mcmc"|"optimization","data":{...}})
    │   ├── observed.csv|.parquet   # optional
    │   └── posterior.csv|.parquet  # optional
    ├── montecarlo/
    │   ├── pipeline.json           # required if montecarlo/ is present
    │   ├── result.json             # optional
    │   └── traces.csv|.parquet     # optional
    ├── simulation.json             # optional (SimSpec)
    └── data/*.csv|*.parquet        # optional raw data members

``sdsge-decompile <file>`` extracts the bundle's members back into a directory
re-compilable into an equivalent bundle. Pass ``--csv`` to re-encode Parquet
members as CSV (for editing).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..core.model_parser import ModelParser
from ..estimation.spec import (
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
)
from ..monte_carlo.spec import PipelineSpec
from .builder import BundleBuilder
from .container import BundleArchive
from .manifest import Manifest, Member, SimSpec
from .parquet import (
    collapse_columns,
    csv_to_columns,
    from_parquet_columns,
    trace_to_csv,
)


class CompileError(ValueError):
    """Raised when the compile source directory cannot be assembled into a bundle."""


# -- entry points -----------------------------------------------------------


def main_compile(argv: Sequence[str] | None = None) -> int:
    """``sdsge-compile`` entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="sdsge-compile",
        description="Assemble a .sdsge bundle from a directory layout.",
    )
    parser.add_argument(
        "source", type=Path, help="Directory containing the bundle members."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .sdsge path (default: <source>.sdsge alongside the directory).",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Skip CSV->Parquet conversion (paired with format-agnostic reader).",
    )
    parser.add_argument(
        "--created-by",
        default=None,
        help="Override the manifest 'created_by' field.",
    )
    args = parser.parse_args(argv)
    try:
        out = compile_directory(
            args.source,
            args.output,
            csv_only=args.csv_only,
            created_by=args.created_by,
        )
    except (CompileError, FileNotFoundError, ValueError) as exc:
        print(f"sdsge-compile: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def main_decompile(argv: Sequence[str] | None = None) -> int:
    """``sdsge-decompile`` entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="sdsge-decompile",
        description="Extract a .sdsge bundle into a directory.",
    )
    parser.add_argument("source", type=Path, help=".sdsge file to extract.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <source stem>/ alongside the file).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Re-encode Parquet members as CSV in the output (for editing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    args = parser.parse_args(argv)
    try:
        out = decompile_bundle(
            args.source,
            args.output,
            also_csv=args.csv,
            force=args.force,
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"sdsge-decompile: {exc}", file=sys.stderr)
        return 1
    print(f"extracted to {out}")
    return 0


# -- compile ----------------------------------------------------------------


def compile_directory(
    source: Path,
    output: Path | None = None,
    *,
    csv_only: bool = False,
    created_by: str | None = None,
) -> Path:
    """Assemble a ``.sdsge`` bundle from ``source`` per the layout convention."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"compile source must be a directory: {source}")
    out_path = (
        Path(output) if output is not None else source.parent / f"{source.name}.sdsge"
    )

    builder = BundleBuilder(created_by=created_by)

    model_observables: dict[str, list[str]] = {}
    for role in ("reference", "dgp"):
        yaml_file = source / f"{role}.yaml"
        if not yaml_file.exists():
            continue
        yaml_text = yaml_file.read_text(encoding="utf-8")
        opts = _load_model_options(source / f"{role}.options.json")
        builder.add_model(
            role,
            yaml_text,
            compile_kwargs=opts.get("compile_kwargs"),
            solve_kwargs=opts.get("solve_kwargs"),
        )
        parsed = ModelParser.from_string(yaml_text).parsed.model
        model_observables[role] = [str(o) for o in parsed.observables]

    if not model_observables:
        raise CompileError(
            f"compile source {source} must contain at least one of "
            "'reference.yaml' or 'dgp.yaml'."
        )

    _compile_raw_data(builder, source / "data", csv_only=csv_only)
    _compile_estimation(
        builder,
        source / "estimation",
        model_observables=model_observables,
        csv_only=csv_only,
    )
    _compile_mc(builder, source / "montecarlo", csv_only=csv_only)

    sim_path = source / "simulation.json"
    if sim_path.exists():
        sim_spec = SimSpec.from_dict(json.loads(sim_path.read_text(encoding="utf-8")))
        builder.set_simulation(sim_spec)

    return builder.write(out_path)


def _load_model_options(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _compile_raw_data(
    builder: BundleBuilder, data_dir: Path, *, csv_only: bool
) -> None:
    if not data_dir.is_dir():
        return
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file():
            continue
        stem = entry.stem
        if entry.suffix == ".csv":
            builder.add_raw_data(
                stem, entry.read_text(encoding="utf-8"), as_parquet=not csv_only
            )
        elif entry.suffix == ".parquet":
            # Embed verbatim — re-encoding would lose any pre-existing layout.
            builder.add_member(
                Member(path=f"data/{stem}.parquet", kind="raw_data"),
                entry.read_bytes(),
            )
        else:
            raise CompileError(
                f"unrecognized raw data file '{entry.name}' "
                f"(expected .csv or .parquet)"
            )


def _compile_estimation(
    builder: BundleBuilder,
    est_dir: Path,
    *,
    model_observables: dict[str, list[str]],
    csv_only: bool,
) -> None:
    if not est_dir.is_dir():
        return

    spec_path = est_dir / "spec.json"
    if not spec_path.exists():
        raise CompileError(f"{est_dir}/ is present but spec.json is missing.")
    spec = EstimationSpec.from_json(spec_path.read_text(encoding="utf-8"))

    result: OptimizationResultMeta | MCMCResultMeta | None = None
    result_path = est_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        data = payload["data"]
        if payload.get("type") == "mcmc":
            result = MCMCResultMeta.from_dict(data)
        else:
            result = OptimizationResultMeta.from_dict(data)

    expected_observables = model_observables.get("reference") or model_observables.get(
        "dgp"
    )

    observed: NDArray[np.float64] | None = None
    observable_names: list[str] | None = None
    obs_path = _pick_one_of(est_dir, "observed")
    if obs_path is not None:
        observed, observable_names = _load_observed(obs_path, expected_observables)

    posterior: dict[str, NDArray[Any]] | None = None
    post_path = _pick_one_of(est_dir, "posterior")
    if post_path is not None:
        posterior = _load_posterior(post_path)

    builder.add_estimation(
        spec,
        result=result,
        observed=observed,
        observable_names=observable_names,
        posterior=posterior,
        as_parquet=not csv_only,
    )


def _compile_mc(builder: BundleBuilder, mc_dir: Path, *, csv_only: bool) -> None:
    if not mc_dir.is_dir():
        return

    pipeline_path = mc_dir / "pipeline.json"
    if not pipeline_path.exists():
        raise CompileError(f"{mc_dir}/ is present but pipeline.json is missing.")
    pipeline = PipelineSpec.from_json(pipeline_path.read_text(encoding="utf-8"))
    builder.add_mc(pipeline)

    # Pre-split mc_result + mc_trace authoring: embed verbatim via add_member
    # (the in-code add_mc takes a live MCPipelineResult and splits it).
    result_path = mc_dir / "result.json"
    if result_path.exists():
        builder.add_member(
            Member(path="montecarlo/result.json", kind="mc_result"),
            result_path.read_bytes(),
        )

    trace_path = _pick_one_of(mc_dir, "traces")
    if trace_path is not None:
        if trace_path.suffix == ".csv":
            target = "montecarlo/traces.csv"
        else:
            target = "montecarlo/traces.parquet"
        builder.add_member(
            Member(path=target, kind="mc_trace"), trace_path.read_bytes()
        )


def _pick_one_of(directory: Path, stem: str) -> Path | None:
    csv_path = directory / f"{stem}.csv"
    parquet_path = directory / f"{stem}.parquet"
    if csv_path.exists() and parquet_path.exists():
        raise CompileError(
            f"both {csv_path.name} and {parquet_path.name} exist in {directory}; "
            "choose one."
        )
    if csv_path.exists():
        return csv_path
    if parquet_path.exists():
        return parquet_path
    return None


def _load_observed(
    path: Path, expected_observables: list[str] | None
) -> tuple[NDArray[np.float64], list[str] | None]:
    columns = _read_columns(path)
    if not columns:
        raise CompileError(f"{path}: file is empty.")

    # Mechanical layout: collapse_columns recovers a single (n, k) 'y' matrix.
    collapsed = collapse_columns(columns)
    y = collapsed.get("y")
    if isinstance(y, np.ndarray) and y.ndim == 2:
        if expected_observables is not None and y.shape[1] != len(expected_observables):
            raise CompileError(
                f"{path}: has {y.shape[1]} columns but model declares "
                f"{len(expected_observables)} observables "
                f"{expected_observables}."
            )
        return y.astype(np.float64, copy=False), None

    # Semantic-header layout: column names ARE observable names.
    inferred = list(columns.keys())
    if any(_looks_numeric(name) for name in inferred):
        raise CompileError(
            f"{path}: inferred observable names {inferred!r} look numeric; "
            "file may be missing a header row. Add observable names as the "
            "first row and re-run."
        )
    if expected_observables is not None and inferred != expected_observables:
        raise CompileError(
            f"{path}: columns {inferred!r} do not match model observables "
            f"{expected_observables!r}. Rename columns to match (order matters)."
        )
    matrix = np.column_stack(
        [
            np.asarray(
                [np.nan if v is None else v for v in columns[name]], dtype=np.float64
            )
            for name in inferred
        ]
    )
    return matrix, inferred


def _load_posterior(path: Path) -> dict[str, NDArray[Any]]:
    return collapse_columns(_read_columns(path))


def _read_columns(path: Path) -> dict[str, list[Any]]:
    if path.suffix == ".csv":
        return csv_to_columns(path.read_text(encoding="utf-8"))
    return from_parquet_columns(path.read_bytes())


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# -- decompile --------------------------------------------------------------


def decompile_bundle(
    source: Path,
    output: Path | None = None,
    *,
    also_csv: bool = False,
    force: bool = False,
) -> Path:
    """Extract ``source`` into a directory that recompiles to an equivalent bundle."""
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"decompile source must be a file: {source}")
    out_dir = (
        Path(output).resolve() if output is not None else source.parent / source.stem
    )

    if out_dir.exists():
        if not force:
            raise FileExistsError(
                f"output directory exists: {out_dir}; pass --force to overwrite."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    archive = BundleArchive.open(source)
    manifest = archive.manifest

    rewritten: list[Member] = []
    for member in manifest.members:
        raw = archive.read(member.path)
        write_path = _authoring_path_for(member)
        new_path: str
        new_format: str

        if also_csv and member.format == "parquet":
            new_path, data = _parquet_member_to_csv(member, raw)
            # If we already remapped (model), keep the authoring path; only the
            # format-rewrite branch changes the extension.
            if write_path != member.path:
                new_path = write_path[: -len(".parquet")] + ".csv"
            new_format = "csv"
        else:
            new_path = write_path
            data = raw
            new_format = member.format

        target = out_dir / new_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rewritten.append(
            Member(
                path=new_path,
                kind=member.kind,
                format=new_format,
                role=member.role,
                columns=member.columns,
                options=dict(member.options),
            )
        )

        # Model options sidecar at the authoring location so recompile picks it up.
        if member.kind == "model_config" and member.options and member.role:
            (out_dir / f"{member.role}.options.json").write_text(
                json.dumps(dict(member.options), indent=2), encoding="utf-8"
            )

    # Simulation prefill rides inline in the manifest; on decompile extract it
    # to simulation.json at root so a recompile picks it up via the standard
    # directory layout.
    if manifest.simulation is not None:
        (out_dir / "simulation.json").write_text(
            json.dumps(manifest.simulation.to_dict(), indent=2), encoding="utf-8"
        )

    out_manifest = Manifest(
        created_by=manifest.created_by,
        created_at=manifest.created_at,
        sdsge_version=manifest.sdsge_version,
        members=rewritten,
        simulation=manifest.simulation,
        # Checksums are sha256(bytes); skip on decompile — recompile recomputes.
        checksums={},
    )
    (out_dir / "manifest.json").write_text(out_manifest.to_json(), encoding="utf-8")

    return out_dir


def _authoring_path_for(member: Member) -> str:
    """Map a bundle member's archive path to its compile-input authoring path.

    Most kinds share the same path in both directions; ``model_config`` is the
    exception because the bundle stores it under ``model/{role}.yaml`` while the
    compile convention is ``{role}.yaml`` at the directory root.
    """
    if member.kind == "model_config" and member.role:
        return f"{member.role}.yaml"
    return member.path


def _parquet_member_to_csv(member: Member, raw: bytes) -> tuple[str, bytes]:
    """Re-emit a Parquet member as CSV, preserving observable-name metadata."""
    columns = from_parquet_columns(raw)
    new_path = member.path[: -len(".parquet")] + ".csv"

    if member.kind == "estimation_data" and member.columns:
        collapsed = collapse_columns(columns)
        y = collapsed.get("y")
        if isinstance(y, np.ndarray) and y.ndim == 2:
            return new_path, _matrix_to_csv(y, list(member.columns))

    # Generic path: emit columns with their current names (1-D each, since they
    # came from a flat parquet file — no 2-D expansion).
    return new_path, trace_to_csv({k: np.asarray(v) for k, v in columns.items()})


def _matrix_to_csv(matrix: NDArray[Any], headers: list[str]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(headers)
    for i in range(matrix.shape[0]):
        writer.writerow(
            [
                (
                    ""
                    if not math.isfinite(float(matrix[i, j]))
                    else repr(float(matrix[i, j]))
                )
                for j in range(matrix.shape[1])
            ]
        )
    return out.getvalue().encode("utf-8")
