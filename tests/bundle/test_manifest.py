from __future__ import annotations

import pytest

from SymbolicDSGE.bundle.manifest import (
    SDSGE_FORMAT_VERSION,
    SDSGE_LAST_BREAKING_VERSION,
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


def test_manifest_reads_a_newer_bundle_that_broke_nothing() -> None:
    # The bump that produced this bundle left last_breaking_version alone, so
    # nothing in it postdates a break this reader is missing.
    payload = Manifest(created_by="x").to_dict()
    payload["sdsge_version"] = SDSGE_FORMAT_VERSION + 3

    restored = Manifest.from_dict(payload)
    assert restored.sdsge_version == SDSGE_FORMAT_VERSION + 3
    assert restored.last_breaking_version == SDSGE_LAST_BREAKING_VERSION


def test_manifest_rejects_a_bundle_written_after_a_later_break() -> None:
    payload = Manifest(created_by="x").to_dict()
    payload["sdsge_version"] = SDSGE_FORMAT_VERSION + 3
    payload["last_breaking_version"] = SDSGE_FORMAT_VERSION + 1

    with pytest.raises(ValueError, match="upgrade SymbolicDSGE"):
        Manifest.from_dict(payload)


def test_manifest_rejects_a_bundle_predating_the_last_break() -> None:
    # A version 1 bundle keys its shock specs by the driven variable, so it
    # rebuilds into specs naming shocks the model does not have.
    payload = Manifest(created_by="x").to_dict()
    payload["sdsge_version"] = SDSGE_LAST_BREAKING_VERSION - 1

    with pytest.raises(ValueError, match="predates"):
        Manifest.from_dict(payload)


def test_manifest_without_the_field_assumes_its_own_version_broke() -> None:
    # Written before last_breaking_version existed, so nothing says otherwise.
    payload = Manifest(created_by="x").to_dict()
    payload["sdsge_version"] = SDSGE_FORMAT_VERSION + 1
    del payload["last_breaking_version"]

    with pytest.raises(ValueError, match="upgrade SymbolicDSGE"):
        Manifest.from_dict(payload)
