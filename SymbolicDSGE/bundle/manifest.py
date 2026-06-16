"""``.sdsge`` manifest schema (the bundle index).

Stdlib dataclasses only — the bundle layer lives in the core library (no ``[ui]``
extra), so it stays pydantic-free. The manifest is stored as ``manifest.json`` at
the archive root and enumerates every member with its ``kind`` and ``format`` so a
reader can dispatch each one (format-agnostic: a hand-zipped CSV bundle and a
CLI-built Parquet bundle both validate). The simulation prefill (#141) rides inline
here rather than as its own member.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

#: Bundle format version. Bump on breaking manifest changes; readers reject a
#: ``sdsge_version`` they do not recognise.
SDSGE_FORMAT_VERSION = 1

MemberKind = Literal[
    "model_config",
    "raw_data",
    "estimation_spec",
    "estimation_result",
    "estimation_data",
    "estimation_trace",
    "mc_pipeline",
    "mc_result",
    "mc_trace",
    "mc_raw_data",
    "mc_custom_op",
]
MEMBER_KINDS: frozenset[str] = frozenset(get_args(MemberKind))

MemberFormat = Literal["yaml", "json", "csv", "parquet", "pickle"]
_FORMAT_BY_EXT: dict[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
    ".pkl": "pickle",
}


def format_for_path(path: str) -> str:
    """Infer a member ``format`` from its file extension."""
    ext = posixpath.splitext(path)[1].lower()
    try:
        return _FORMAT_BY_EXT[ext]
    except KeyError as exc:
        raise ValueError(
            f"Cannot infer bundle member format from path {path!r}; "
            f"expected one of {sorted(_FORMAT_BY_EXT)}."
        ) from exc


@dataclass
class ShockGeneration:
    """RNG settings for replayed shock generation (determinism via seed)."""

    dist: str = "norm"
    seed: int | None = 0
    loc: float = 0.0
    df: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dist": self.dist,
            "seed": None if self.seed is None else int(self.seed),
            "loc": float(self.loc),
            "df": float(self.df),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShockGeneration:
        seed = data.get("seed", 0)
        return cls(
            dist=str(data.get("dist", "norm")),
            seed=None if seed is None else int(seed),
            loc=float(data.get("loc", 0.0)),
            df=float(data.get("df", 5.0)),
        )


@dataclass
class SimSpec:
    """Simulation/output-tab prefill (#141).

    No simulation results are stored — replaying these specs against the
    preloaded model reproduces the intended run (numpy PCG64 + fixed seed).
    Raw shock paths, when present, are carried inline (they are small).
    """

    role: str = "reference"
    T: int = 0
    observables: bool = True
    shock_scale: float = 1.0
    shock_generation: ShockGeneration | None = None
    shock_std: dict[str, float] = field(default_factory=dict)
    shock_corr: dict[str, float] = field(default_factory=dict)
    shocks: dict[str, list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "role": self.role,
            "T": int(self.T),
            "observables": bool(self.observables),
            "shock_scale": float(self.shock_scale),
            "shock_std": {str(k): float(v) for k, v in self.shock_std.items()},
            "shock_corr": {str(k): float(v) for k, v in self.shock_corr.items()},
        }
        if self.shock_generation is not None:
            out["shock_generation"] = self.shock_generation.to_dict()
        if self.shocks is not None:
            out["shocks"] = {
                str(k): [float(x) for x in v] for k, v in self.shocks.items()
            }
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimSpec:
        gen = data.get("shock_generation")
        shocks = data.get("shocks")
        return cls(
            role=str(data.get("role", "reference")),
            T=int(data.get("T", 0)),
            observables=bool(data.get("observables", True)),
            shock_scale=float(data.get("shock_scale", 1.0)),
            shock_generation=(
                ShockGeneration.from_dict(gen) if gen is not None else None
            ),
            shock_std={
                str(k): float(v) for k, v in dict(data.get("shock_std", {})).items()
            },
            shock_corr={
                str(k): float(v) for k, v in dict(data.get("shock_corr", {})).items()
            },
            shocks=(
                {str(k): [float(x) for x in v] for k, v in dict(shocks).items()}
                if shocks is not None
                else None
            ),
        )


@dataclass
class Member:
    """One archive entry described in the manifest.

    ``options`` carries kind-specific metadata — for ``model_config`` it holds the
    ``compile_kwargs``/``solve_kwargs`` needed to rebuild the ``SolvedModel``.
    """

    path: str
    kind: str
    format: str = ""
    role: str | None = None
    columns: list[str] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in MEMBER_KINDS:
            raise ValueError(
                f"Unknown bundle member kind {self.kind!r}; "
                f"expected one of {sorted(MEMBER_KINDS)}."
            )
        if not self.format:
            self.format = format_for_path(self.path)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "format": self.format,
        }
        if self.role is not None:
            out["role"] = self.role
        if self.columns is not None:
            out["columns"] = list(self.columns)
        if self.options:
            out["options"] = dict(self.options)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Member:
        return cls(
            path=str(data["path"]),
            kind=str(data["kind"]),
            format=str(data.get("format", "")),
            role=None if data.get("role") is None else str(data["role"]),
            columns=(
                list(data["columns"]) if data.get("columns") is not None else None
            ),
            options=dict(data.get("options", {})),
        )


@dataclass
class Manifest:
    """The ``manifest.json`` index of a ``.sdsge`` bundle."""

    created_by: str = ""
    created_at: str | None = None
    sdsge_version: int = SDSGE_FORMAT_VERSION
    members: list[Member] = field(default_factory=list)
    simulation: SimSpec | None = None
    checksums: dict[str, str] = field(default_factory=dict)

    def members_by_kind(self, kind: str) -> list[Member]:
        return [m for m in self.members if m.kind == kind]

    def model_member(self, role: str) -> Member | None:
        for member in self.members:
            if member.kind == "model_config" and member.role == role:
                return member
        return None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sdsge_version": int(self.sdsge_version),
            "created_by": self.created_by,
            "members": [m.to_dict() for m in self.members],
        }
        if self.created_at is not None:
            out["created_at"] = self.created_at
        if self.simulation is not None:
            out["simulation"] = self.simulation.to_dict()
        if self.checksums:
            out["checksums"] = dict(self.checksums)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        version = int(data.get("sdsge_version", SDSGE_FORMAT_VERSION))
        if version > SDSGE_FORMAT_VERSION:
            raise ValueError(
                f"Bundle sdsge_version {version} is newer than this library "
                f"supports ({SDSGE_FORMAT_VERSION}); upgrade SymbolicDSGE."
            )
        sim = data.get("simulation")
        return cls(
            created_by=str(data.get("created_by", "")),
            created_at=(
                None if data.get("created_at") is None else str(data["created_at"])
            ),
            sdsge_version=version,
            members=[Member.from_dict(m) for m in data.get("members", [])],
            simulation=SimSpec.from_dict(sim) if sim is not None else None,
            checksums={
                str(k): str(v) for k, v in dict(data.get("checksums", {})).items()
            },
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls.from_dict(json.loads(text))
