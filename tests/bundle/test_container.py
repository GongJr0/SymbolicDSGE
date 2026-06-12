from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from SymbolicDSGE.bundle.container import BundleArchive, write_bundle
from SymbolicDSGE.bundle.manifest import Manifest, Member


def _manifest(members: list[Member]) -> Manifest:
    return Manifest(created_by="test", members=members)


def test_write_and_open_round_trip(tmp_path: Path) -> None:
    members = [
        Member(path="model/reference.yaml", kind="model_config", role="reference"),
        Member(path="data/x.parquet", kind="raw_data"),
    ]
    files = {
        "model/reference.yaml": b"name: TEST\n",
        "data/x.parquet": b"PARQUET-BYTES",
    }
    target = tmp_path / "bundle.sdsge"
    write_bundle(target, _manifest(members), files)

    archive = BundleArchive.open(target)
    assert archive.manifest.created_by == "test"
    assert archive.read("model/reference.yaml") == files["model/reference.yaml"]
    assert archive.read("data/x.parquet") == files["data/x.parquet"]
    assert archive.read_text("model/reference.yaml") == "name: TEST\n"


def test_compression_policy(tmp_path: Path) -> None:
    members = [
        Member(path="spec.json", kind="estimation_spec"),
        Member(path="data/x.parquet", kind="raw_data"),
    ]
    files = {"spec.json": b'{"a": 1}', "data/x.parquet": b"PARQUET"}
    target = tmp_path / "bundle.sdsge"
    write_bundle(target, _manifest(members), files)

    with zipfile.ZipFile(target) as zf:
        info = {z.filename: z.compress_type for z in zf.infolist()}
    assert info["data/x.parquet"] == zipfile.ZIP_STORED
    assert info["spec.json"] == zipfile.ZIP_DEFLATED
    assert info["manifest.json"] == zipfile.ZIP_DEFLATED


def test_member_file_mismatch_raises(tmp_path: Path) -> None:
    members = [Member(path="a.json", kind="estimation_spec")]
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "b.sdsge", _manifest(members), {"b.json": b"{}"})


def test_open_rejects_non_bundle(tmp_path: Path) -> None:
    target = tmp_path / "plain.zip"
    with zipfile.ZipFile(target, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(ValueError):
        BundleArchive.open(target)
