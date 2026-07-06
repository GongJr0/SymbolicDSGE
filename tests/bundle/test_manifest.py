from __future__ import annotations

import pytest

from SymbolicDSGE.bundle.manifest import (
    SDSGE_FORMAT_VERSION,
    Manifest,
    Member,
    SimSpec,
    format_for_path,
)

_SHOCK = {
    "dist": "norm",
    "multivar": False,
    "seed": 42,
    "dist_args": [],
    "dist_kwargs": {"loc": 0.0},
}


def test_member_format_inference() -> None:
    assert format_for_path("model/reference.yaml") == "yaml"
    assert format_for_path("a/b.JSON") == "json"
    assert format_for_path("data/x.parquet") == "parquet"
    assert format_for_path("data/x.csv") == "csv"
    with pytest.raises(ValueError):
        format_for_path("data/x.bin")


def test_member_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError):
        Member(path="x.json", kind="not_a_kind")


def test_member_format_filled_from_path() -> None:
    member = Member(path="model/reference.yaml", kind="model_config", role="reference")
    assert member.format == "yaml"


def test_manifest_round_trip() -> None:
    manifest = Manifest(
        created_by="SymbolicDSGE 9.9.9",
        created_at="2026-06-12T00:00:00+00:00",
        members=[
            Member(
                path="model/reference.yaml",
                kind="model_config",
                role="reference",
                options={"compile_kwargs": {"n_state": 3, "n_exog": 2}},
            ),
            Member(
                path="estimation/observed.parquet",
                kind="estimation_data",
                columns=["Infl", "Rate"],
            ),
        ],
        simulation={"reference": SimSpec(T=10, shocks={"u": _SHOCK})},
        checksums={"model/reference.yaml": "abc"},
    )
    restored = Manifest.from_json(manifest.to_json())
    assert restored.to_dict() == manifest.to_dict()
    assert restored.model_member("reference") is not None
    assert restored.model_member("dgp") is None
    assert restored.members_by_kind("estimation_data")[0].columns == ["Infl", "Rate"]
    assert restored.simulation is not None
    assert restored.simulation["reference"].shocks["u"]["seed"] == 42


def test_manifest_rejects_newer_version() -> None:
    payload = Manifest(created_by="x").to_dict()
    payload["sdsge_version"] = SDSGE_FORMAT_VERSION + 1
    with pytest.raises(ValueError):
        Manifest.from_dict(payload)
