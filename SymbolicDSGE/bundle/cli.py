"""CLI entry points for ``sdsge-compile`` and ``sdsge-decompile``.

``sdsge-decompile <file>`` extracts a bundle's members into a directory, keeping
``manifest.json`` beside them so the layout stays self-describing::

    my-bundle/
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

Each step kind's metas ride one JSON keyed by step name. Tests and regressions
pack every step's columns into one block, qualified ``{step}.{field}`` and
extended to ``{step}.{field}.{idx}`` where a column is 2-D. Transform payloads and
postproc ``Raw`` arrays take a member each, since they share no shape with
anything. Pass ``--csv`` to re-encode Parquet members as CSV, which is the only
way to read one in a text editor; the loader takes either format, so a CSV bundle
is as valid as a Parquet one.

``sdsge-compile <dir>`` packs such a directory back into a ``.sdsge``. It is the
inverse of decompile and nothing more: member bytes are copied through unchanged,
the manifest is rebuilt from the one in the directory, and checksums are
recomputed because an edited member is a different member. Nothing stops a
directory being written by hand, but ``manifest.json`` is what compile reads, so
a hand-authored one has to declare its members there. Building a bundle from live
objects is :class:`~SymbolicDSGE.bundle.builder.BundleBuilder`'s job, not this
one.
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

from .builder import BundleBuilder
from .container import MANIFEST_NAME, BundleArchive
from .manifest import Manifest, Member, SimSpec
from .parquet import collapse_columns, from_parquet_columns, trace_to_csv


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
        "--created-by",
        default=None,
        help="Override the manifest 'created_by' field.",
    )
    args = parser.parse_args(argv)
    try:
        out = compile_directory(
            args.source,
            args.output,
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
    created_by: str | None = None,
) -> Path:
    """Pack ``source`` back into a bundle, the inverse of :func:`decompile_bundle`.

    The directory's own ``manifest.json`` is the index: every member it lists is
    read from disk and written into the archive unchanged, so a decompile that
    re-encoded nothing round-trips byte for byte. Checksums are recomputed rather
    than carried over, since this tool cannot know whether a member was edited.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"compile source must be a directory: {source}")
    manifest_path = source / MANIFEST_NAME
    if not manifest_path.exists():
        raise CompileError(
            f"{source} has no {MANIFEST_NAME}; compile packs a directory that "
            "sdsge-decompile produced."
        )
    out_path = (
        Path(output) if output is not None else source.parent / f"{source.name}.sdsge"
    )

    manifest = Manifest.from_json(manifest_path.read_text(encoding="utf-8"))
    builder = BundleBuilder(created_by=created_by or manifest.created_by)
    for member in manifest.members:
        member_path = source / member.path
        if not member_path.is_file():
            raise CompileError(
                f"{MANIFEST_NAME} lists {member.path!r} but no such file is in "
                f"{source}."
            )
        builder.add_member(
            Member(
                path=_archive_path_for(member),
                kind=member.kind,
                format=member.format,
                role=member.role,
                columns=member.columns,
                options=dict(member.options),
            ),
            member_path.read_bytes(),
        )

    if manifest.simulation is not None:
        for role, spec in manifest.simulation.items():
            builder.set_simulation(role, spec)

    return builder.write(out_path)


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

    out_manifest = Manifest(
        created_by=manifest.created_by,
        created_at=manifest.created_at,
        sdsge_version=manifest.sdsge_version,
        last_breaking_version=manifest.last_breaking_version,
        members=rewritten,
        simulation=manifest.simulation,
        # Checksums are sha256(bytes); skip on decompile — recompile recomputes.
        checksums={},
    )
    (out_dir / MANIFEST_NAME).write_text(out_manifest.to_json(), encoding="utf-8")

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


def _archive_path_for(member: Member) -> str:
    """Map an authoring path back to its place in the archive.

    The mirror of :func:`_authoring_path_for`, and the only asymmetry between the
    two layouts: everything else sits at the same path in both.
    """
    if member.kind == "model_config" and member.role:
        return f"model/{member.role}.yaml"
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
