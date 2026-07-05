"""``.sdsge`` manifest schema (the bundle index).

Stdlib dataclasses only — the bundle layer lives in the core library (no ``[ui]``
extra), so it stays pydantic-free. The manifest is stored as ``manifest.json`` at
the archive root and enumerates every member with its ``kind`` and ``format`` so a
reader can dispatch each one (format-agnostic: a hand-zipped CSV bundle and a
CLI-built Parquet bundle both validate). The simulation prefill (#141) rides inline
here rather than as its own member.
"""

from __future__ import annotations

from ..core.shock_generators import Shock

import json
import posixpath
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, TypedDict
from numpy import ndarray

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
    "mc_postproc",
    "mc_postproc_table",
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


class ShockParameters(TypedDict):
    dist: str  # Cannot serialize custom distributions.
    multivar: bool
    seed: int | None
    dist_args: tuple
    dist_kwargs: dict[str, Any]
    shock_arr: ndarray | None


@dataclass
class SimSpec(Mapping):
    """Simulation/output-tab prefill (#141).

    No simulation results are stored — replaying these specs against the
    preloaded model reproduces the intended run (numpy PCG64 + fixed seed).
    Raw shock paths, when present, are carried inline (they are small).
    """

    T: int = 0
    x0: list[float] | ndarray | None = None
    observables: bool = True
    shock_scale: float = 1.0
    shocks: dict[str, ShockParameters] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-serializable form: shocks stay as their ``Shock.to_dict``."""
        return {
            "T": int(self.T),
            "x0": None if self.x0 is None else [float(x) for x in self.x0],
            "observables": bool(self.observables),
            "shock_scale": float(self.shock_scale),
            "shocks": self.shocks,
        }

    def to_sim_kwargs(self) -> dict[str, Any]:
        """The ``SolvedModel.sim`` keyword form: shocks as live ``Shock`` objects.

        ``sim`` materializes each ``Shock`` into its horizon-bound draw at the
        simulation boundary, so this is exactly ``model.sim(**spec)``.
        """
        out = self.to_dict()
        out["shocks"] = (
            {k: Shock.from_dict(v) for k, v in self.shocks.items()}
            if self.shocks
            else None
        )
        return out

    # Mapping protocol: a SimSpec unpacks straight into ``model.sim(**spec)``.
    # The view is the materialized sim kwargs, distinct from ``to_dict``'s
    # JSON-serializable form.
    def __getitem__(self, key: str) -> Any:
        return self.to_sim_kwargs()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_sim_kwargs())

    def __len__(self) -> int:
        return len(self.to_sim_kwargs())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimSpec:
        return cls(
            T=int(data.get("T", 0)),
            x0=data.get("x0", None),
            observables=bool(data.get("observables", True)),
            shock_scale=float(data.get("shock_scale", 1.0)),
            shocks=data.get("shocks", None),
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
    simulation: dict[str, SimSpec] | None = None
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
            out["simulation"] = {k: v.to_dict() for k, v in self.simulation.items()}
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
            simulation=(
                {k: SimSpec.from_dict(v) for k, v in sim.items()}
                if sim is not None
                else None
            ),
            checksums={
                str(k): str(v) for k, v in dict(data.get("checksums", {})).items()
            },
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls.from_dict(json.loads(text))
