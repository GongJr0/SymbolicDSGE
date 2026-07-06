"""Tests for the ``sdsge-compile`` / ``sdsge-decompile`` CLI commands.

Drives ``main_compile`` / ``main_decompile`` through their argv lists so the
production entry points exercise the same code paths the installed scripts will.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from SymbolicDSGE.bundle.cli import (
    CompileError,
    compile_directory,
    decompile_bundle,
    main_compile,
    main_decompile,
)
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.bundle.manifest import SimSpec
from SymbolicDSGE.core.model_parser import ModelParser

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")
_OBSERVABLES = [
    str(o) for o in ModelParser.from_string(_MODEL_YAML).parsed.model.observables
]


# -- helpers ----------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _observed_csv(matrix: np.ndarray, headers: list[str]) -> str:
    rows = "\n".join(",".join(repr(float(v)) for v in row) for row in matrix)
    return ",".join(headers) + "\n" + rows + "\n"


def _baseline_dir(tmp_path: Path, *, with_estimation: bool = True) -> Path:
    src = tmp_path / "bundle"
    src.mkdir()
    _write_text(src / "reference.yaml", _MODEL_YAML)
    _write_text(
        src / "reference.options.json",
        json.dumps({"compile_kwargs": {"n_state": 3, "n_exog": 2}}),
    )
    if with_estimation:
        spec = {
            "method": "mle",
            "parameters": [{"name": "beta", "initial": 0.99, "estimate": True}],
            "method_kwargs": {},
            "compile_kwargs": {},
            "posterior_point": "mean",
        }
        _write_text(src / "estimation" / "spec.json", json.dumps(spec))
        matrix = np.arange(12, dtype=np.float64).reshape(6, len(_OBSERVABLES))
        _write_text(
            src / "estimation" / "observed.csv",
            _observed_csv(matrix, _OBSERVABLES),
        )
    return src


# -- compile entry point ----------------------------------------------------


def test_main_compile_emits_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _baseline_dir(tmp_path, with_estimation=False)
    out = tmp_path / "out.sdsge"
    rc = main_compile([str(src), "-o", str(out), "--created-by", "cli-test"])
    assert rc == 0
    assert out.exists()
    assert f"wrote {out}" in capsys.readouterr().out

    loaded = build_from(out)
    assert loaded.manifest.created_by == "cli-test"
    assert loaded.reference is not None


def test_main_compile_default_output_path(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path, with_estimation=False)
    rc = main_compile([str(src)])
    assert rc == 0
    default_out = src.parent / f"{src.name}.sdsge"
    assert default_out.exists()


def test_compile_requires_at_least_one_model(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CompileError, match="reference.yaml"):
        compile_directory(empty)


def test_compile_observed_round_trips_through_loader(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    out = compile_directory(src, tmp_path / "obs.sdsge")
    loaded = build_from(out)
    assert loaded.estimation is not None
    assert loaded.estimation.observed is not None
    expected = np.arange(12, dtype=np.float64).reshape(6, len(_OBSERVABLES))
    np.testing.assert_allclose(loaded.estimation.observed, expected)


def test_compile_csv_only_keeps_csv_member(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    out = compile_directory(src, tmp_path / "csv.sdsge", csv_only=True)
    loaded = build_from(out)
    data_member = next(
        m for m in loaded.manifest.members if m.kind == "estimation_data"
    )
    assert data_member.format == "csv"
    assert data_member.path.endswith(".csv")


def test_compile_rejects_observable_name_mismatch(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    matrix = np.zeros((3, len(_OBSERVABLES)))
    # Wrong header names (model expects 'Infl,Rate').
    _write_text(
        src / "estimation" / "observed.csv",
        _observed_csv(matrix, ["foo", "bar"]),
    )
    with pytest.raises(CompileError, match="do not match model observables"):
        compile_directory(src, tmp_path / "bad.sdsge")


def test_compile_rejects_headerless_numeric_csv(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    # Numeric "headers" — file is effectively missing its header row.
    _write_text(
        src / "estimation" / "observed.csv",
        "0.1,0.2\n0.3,0.4\n0.5,0.6\n",
    )
    with pytest.raises(CompileError, match="missing a header row"):
        compile_directory(src, tmp_path / "nohdr.sdsge")


def test_compile_rejects_column_count_mismatch_on_mechanical_layout(
    tmp_path: Path,
) -> None:
    src = _baseline_dir(tmp_path)
    # Mechanical y.0..y.{k-1} layout with the wrong k.
    _write_text(
        src / "estimation" / "observed.csv",
        "y.0,y.1,y.2\n1,2,3\n4,5,6\n",
    )
    with pytest.raises(CompileError, match="model declares"):
        compile_directory(src, tmp_path / "wrongk.sdsge")


def test_compile_rejects_both_csv_and_parquet_for_same_member(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    _write_text(src / "estimation" / "observed.parquet", "junk")
    with pytest.raises(CompileError, match="choose one"):
        compile_directory(src, tmp_path / "ambig.sdsge")


# -- decompile entry point --------------------------------------------------


def test_main_decompile_extracts_members(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out_bundle = compile_directory(_baseline_dir(tmp_path), tmp_path / "in.sdsge")
    out_dir = tmp_path / "extracted"
    rc = main_decompile([str(out_bundle), "-o", str(out_dir)])
    assert rc == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "reference.yaml").exists()
    assert (out_dir / "reference.options.json").exists()
    assert (out_dir / "estimation" / "spec.json").exists()
    assert f"extracted to {out_dir.resolve()}" in capsys.readouterr().out


def test_decompile_then_recompile_yields_equivalent_bundle(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    # Add a sim prefill so the inline-in-manifest path is also exercised.
    src_sim = SimSpec(
        T=8,
        shocks={
            "u": {
                "dist": "norm",
                "multivar": False,
                "seed": 42,
                "dist_args": [],
                "dist_kwargs": {"loc": 0.0},
            }
        },
    )
    # Write simulation.json (a {role: SimSpec} map) into the source dir and compile.
    _write_text(src / "simulation.json", json.dumps({"reference": src_sim.to_dict()}))
    pass1 = compile_directory(src, tmp_path / "pass1.sdsge")
    loaded1 = build_from(pass1)
    assert loaded1.simulation is not None and loaded1.simulation["reference"].T == 8

    out_dir = decompile_bundle(pass1, tmp_path / "extracted", also_csv=True)
    assert (out_dir / "simulation.json").exists()  # inline SimSpec extracted
    pass2 = compile_directory(out_dir, tmp_path / "pass2.sdsge")
    loaded2 = build_from(pass2)

    assert loaded2.simulation is not None and loaded2.simulation["reference"].T == 8
    np.testing.assert_allclose(loaded1.estimation.observed, loaded2.estimation.observed)
    assert loaded1.estimation.spec.to_dict() == loaded2.estimation.spec.to_dict()


def test_decompile_csv_mode_rewrites_parquet_member_to_csv(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    bundle = compile_directory(src, tmp_path / "p.sdsge")
    # Confirm the input bundle uses parquet.
    data_member = next(
        m for m in build_from(bundle).manifest.members if m.kind == "estimation_data"
    )
    assert data_member.format == "parquet"

    out_dir = decompile_bundle(bundle, tmp_path / "as_csv", also_csv=True)
    assert (out_dir / "estimation" / "observed.csv").exists()
    assert not (out_dir / "estimation" / "observed.parquet").exists()

    # Header should be the semantic observable names (not mechanical y.0/y.1).
    header = (out_dir / "estimation" / "observed.csv").read_text().splitlines()[0]
    assert header == ",".join(_OBSERVABLES)


def test_decompile_rejects_existing_dir_without_force(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    bundle = compile_directory(src, tmp_path / "b.sdsge")
    out_dir = tmp_path / "exists"
    out_dir.mkdir()
    with pytest.raises(FileExistsError, match="--force"):
        decompile_bundle(bundle, out_dir)


def test_decompile_force_overwrites(tmp_path: Path) -> None:
    src = _baseline_dir(tmp_path)
    bundle = compile_directory(src, tmp_path / "b.sdsge")
    out_dir = tmp_path / "occupied"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("old")
    decompile_bundle(bundle, out_dir, force=True)
    assert not (out_dir / "stale.txt").exists()
    assert (out_dir / "manifest.json").exists()


def test_main_compile_returns_nonzero_on_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main_compile([str(empty)])
    assert rc == 1
    assert "sdsge-compile:" in capsys.readouterr().err
