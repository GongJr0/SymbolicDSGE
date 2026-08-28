"""Tests for the ``sdsge-compile`` / ``sdsge-decompile`` CLI commands.

Drives ``main_compile`` / ``main_decompile`` through their argv lists so the
production entry points exercise the same code paths the installed scripts will.

The pair is an extract/pack round trip over an already-built bundle, so every
fixture here starts from :class:`BundleBuilder` rather than a hand-written
directory: authoring a bundle from loose files is not something either command
does.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.cli import (
    CompileError,
    compile_directory,
    decompile_bundle,
    main_compile,
    main_decompile,
)
from SymbolicDSGE.bundle.loader import build_from
from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
)

_MODEL_YAML = Path("MODELS/test.yaml").read_text(encoding="utf-8")


# -- helpers ----------------------------------------------------------------


def _pipeline() -> MCPipeline:
    observables = np.random.default_rng(0).normal(size=(4, 20, 2))
    return MCPipeline(
        [
            raw_model_data_step(
                "dat", observables=observables, observable_names=("y", "x")
            ),
            jarque_bera_test_step("jb", source="dat", field="observables", column=0),
        ]
    )


def _bundle(tmp_path: Path, *, with_result: bool = True) -> Path:
    """A bundle carrying every member kind the CLI has to move."""
    pipe = _pipeline()
    result = (
        pipe.run(reference=cast(SolvedModel, object()), n_rep=4, verbosity=0)
        if with_result
        else None
    )
    return (
        BundleBuilder(created_by="cli-test")
        .add_model("reference", _MODEL_YAML, compile_kwargs={"linearize": False})
        .add_mc(pipe, result=result)
        .add_raw_data("series", "a,b\n1,2.5\n3,4.5\n")
        .set_simulation(
            "reference",
            T=8,
            shocks={"u": Shock(dist="norm", seed=42, dist_kwargs={"loc": 0.0})},
        )
        .write(tmp_path / "in.sdsge")
    )


def _member_digests(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name != "manifest.json"
        }


# -- round trip -------------------------------------------------------------


def test_decompile_then_compile_returns_the_same_bytes(tmp_path: Path) -> None:
    # Compile packs rather than rebuilds, so a round trip that re-encodes nothing
    # is byte-for-byte identical, member paths included.
    bundle = _bundle(tmp_path)
    extracted = decompile_bundle(bundle, tmp_path / "flat")

    packed = compile_directory(extracted, tmp_path / "out.sdsge")

    assert _member_digests(packed) == _member_digests(bundle)


def test_round_tripped_bundle_still_loads(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    original = build_from(bundle)

    packed = compile_directory(
        decompile_bundle(bundle, tmp_path / "flat"), tmp_path / "out.sdsge"
    )
    loaded = build_from(packed)

    assert loaded.reference is not None
    assert loaded.simulation is not None and loaded.simulation["reference"]["T"] == 8
    assert loaded.mc is not None and loaded.mc.result is not None
    np.testing.assert_array_equal(
        loaded.mc.result.test_summaries["jb"].statistic_trace,
        original.mc.result.test_summaries["jb"].statistic_trace,
    )


def test_csv_mode_round_trips_into_a_readable_bundle(tmp_path: Path) -> None:
    # ``--csv`` is the only way to read a bulk member in an editor, and what it
    # produces is still a bundle: the loader takes either format.
    bundle = _bundle(tmp_path)
    extracted = decompile_bundle(bundle, tmp_path / "flat", also_csv=True)
    assert not any(extracted.rglob("*.parquet"))

    packed = compile_directory(extracted, tmp_path / "out.sdsge")
    loaded = build_from(packed)

    assert [
        m.format for m in loaded.manifest.members if m.kind == "mc_test_traces"
    ] == ["csv"]
    np.testing.assert_array_equal(
        loaded.mc.result.test_summaries["jb"].statistic_trace,
        build_from(bundle).mc.result.test_summaries["jb"].statistic_trace,
    )


def test_model_config_moves_to_the_root_and_back(tmp_path: Path) -> None:
    # The one path that differs between the two layouts, in both directions.
    bundle = _bundle(tmp_path, with_result=False)
    extracted = decompile_bundle(bundle, tmp_path / "flat")
    assert (extracted / "reference.yaml").exists()

    packed = compile_directory(extracted, tmp_path / "out.sdsge")
    member = build_from(packed).manifest.model_member("reference")
    assert member is not None and member.path == "model/reference.yaml"
    # The options the loader rebuilds the model with survive the round trip.
    assert member.options["compile_kwargs"] == {"linearize": False}


# -- compile refuses what it cannot pack -------------------------------------


def test_compile_requires_a_manifest(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "reference.yaml").write_text(_MODEL_YAML, encoding="utf-8")

    with pytest.raises(CompileError, match="manifest.json"):
        compile_directory(bare, tmp_path / "out.sdsge")


def test_compile_reports_a_member_the_manifest_lists_but_the_directory_lacks(
    tmp_path: Path,
) -> None:
    extracted = decompile_bundle(_bundle(tmp_path), tmp_path / "flat")
    (extracted / "reference.yaml").unlink()

    with pytest.raises(CompileError, match="reference.yaml"):
        compile_directory(extracted, tmp_path / "out.sdsge")


# -- decompile output directory ---------------------------------------------


def test_decompile_rejects_existing_dir_without_force(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, with_result=False)
    out_dir = tmp_path / "exists"
    out_dir.mkdir()

    with pytest.raises(FileExistsError, match="--force"):
        decompile_bundle(bundle, out_dir)


def test_decompile_force_overwrites(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, with_result=False)
    out_dir = tmp_path / "occupied"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("old")

    decompile_bundle(bundle, out_dir, force=True)

    assert not (out_dir / "stale.txt").exists()
    assert (out_dir / "manifest.json").exists()


# -- entry points ------------------------------------------------------------


def test_main_decompile_extracts_members(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bundle = _bundle(tmp_path)
    out_dir = tmp_path / "extracted"

    assert main_decompile([str(bundle), "-o", str(out_dir)]) == 0

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "reference.yaml").exists()
    assert f"extracted to {out_dir.resolve()}" in capsys.readouterr().out


def test_main_compile_emits_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    extracted = decompile_bundle(_bundle(tmp_path), tmp_path / "flat")
    target = tmp_path / "out.sdsge"

    assert main_compile([str(extracted), "-o", str(target)]) == 0

    assert target.is_file()
    assert f"wrote {target}" in capsys.readouterr().out


def test_main_compile_default_output_path(tmp_path: Path) -> None:
    extracted = decompile_bundle(_bundle(tmp_path), tmp_path / "flat")

    assert main_compile([str(extracted)]) == 0

    assert (extracted.parent / f"{extracted.name}.sdsge").is_file()


def test_main_compile_carries_created_by_from_the_manifest(tmp_path: Path) -> None:
    extracted = decompile_bundle(_bundle(tmp_path), tmp_path / "flat")

    packed = compile_directory(extracted, tmp_path / "out.sdsge")

    assert build_from(packed).manifest.created_by == "cli-test"


def test_main_compile_returns_nonzero_on_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main_compile([str(empty)]) == 1

    assert "sdsge-compile:" in capsys.readouterr().err


def test_simulation_prefill_survives_the_round_trip(tmp_path: Path) -> None:
    # The prefill is not a member: it rides inline in the manifest, so it is the
    # one thing compile picks up from the index rather than from a file.
    extracted = decompile_bundle(_bundle(tmp_path), tmp_path / "flat")
    written = json.loads((extracted / "manifest.json").read_text(encoding="utf-8"))
    assert written["simulation"]["reference"]["T"] == 8

    packed = compile_directory(extracted, tmp_path / "out.sdsge")
    loaded = build_from(packed)

    assert loaded.simulation is not None
    assert loaded.simulation["reference"]["shocks"]["u"].seed == 42
