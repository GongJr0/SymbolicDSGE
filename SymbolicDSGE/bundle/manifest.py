"""``.sdsge`` manifest schema (the bundle index).

Stdlib dataclasses only. The bundle layer lives in the core library (no ``[ui]``
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

import numpy as np
from numpy import float64, ndarray

from ..core.shock_generators import Shock, ShockParameters

#: Bundle format version. Bump on every manifest change.
SDSGE_FORMAT_VERSION = 3

#: The version at which the format last broke. A reader rejects bundles older
#: than this, and each bundle records its own so a reader can tell a version it
#: predates from a version that postdates it: a bump that breaks nothing
#: leaves this alone and stays readable by older versions.
SDSGE_LAST_BREAKING_VERSION = 3

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
    "mc_raw_model_data",
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


@dataclass
class SimSpec:
    """Simulation/output-tab prefill (#141).

    The internal carrier between :meth:`BundleBuilder.set_simulation`, the
    manifest, and the loader. No public signature names it: the builder takes
    the ``sim`` keywords and lowers them here, and the loader materializes it
    back through :meth:`to_sim_kwargs` before handing anything to a caller.

    No simulation results are stored. Replaying these specs against the
    preloaded model reproduces the intended run (numpy PCG64 + fixed seed).
    Raw shock paths, when present, are carried inline (they are small).
    """

    T: int = 0
    x0: Mapping[str, float] | list[float] | ndarray | None = None
    observables: bool = False
    shock_scale: float = 1.0
    #: Per key, either a ``Shock.to_dict()`` mapping or a raw path as a nested
    #: list. Both are what ``SolvedModel.sim`` accepts, in JSON-safe form.
    shocks: dict[str, ShockParameters | list[Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-serializable form: shocks stay as parameters or raw paths."""
        return {
            "T": int(self.T),
            "x0": _x0_to_json(self.x0),
            "observables": bool(self.observables),
            "shock_scale": float(self.shock_scale),
            "shocks": self.shocks,
        }

    def to_sim_kwargs(self) -> dict[str, Any]:
        """The ``SolvedModel.sim`` keyword form: ``model.sim(**spec.to_sim_kwargs())``.

        Each shock parameter mapping becomes a live :class:`Shock`, which ``sim``
        materializes into its horizon-bound draw; each raw path becomes the array
        ``sim`` passes through unchanged.
        """
        out = self.to_dict()
        out["shocks"] = (
            {key: _shock_from_json(value) for key, value in self.shocks.items()}
            if self.shocks
            else None
        )
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimSpec:
        return cls(
            T=int(data.get("T", 0)),
            x0=data.get("x0", None),
            observables=bool(data.get("observables", False)),
            shock_scale=float(data.get("shock_scale", 1.0)),
            shocks=data.get("shocks", None),
        )


def _x0_to_json(x0: Mapping[str, float] | list[float] | ndarray | None) -> Any:
    """``x0`` in JSON form, keeping the name-keyed and positional shapes apart."""
    if x0 is None:
        return None
    if isinstance(x0, Mapping):
        return {str(name): float(value) for name, value in x0.items()}
    return [float(value) for value in x0]


def _shock_from_json(value: ShockParameters | list[Any]) -> Any:
    """One stored shock as ``sim`` takes it: a :class:`Shock` or a raw path."""
    if isinstance(value, Mapping):
        return Shock.from_dict(value)
    return np.asarray(value, dtype=float64)


@dataclass
class Member:
    """One archive entry described in the manifest.

    ``options`` carries kind-specific metadata. For ``model_config`` it holds the
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
    last_breaking_version: int = SDSGE_LAST_BREAKING_VERSION
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
            "last_breaking_version": int(self.last_breaking_version),
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
        if version < SDSGE_LAST_BREAKING_VERSION:
            raise ValueError(
                f"Bundle sdsge_version {version} predates the format's last "
                f"breaking change ({SDSGE_LAST_BREAKING_VERSION}); rebuild the "
                f"bundle from its sources."
            )
        # Absent on bundles written before the field existed, and those are
        # already rejected above, so assuming the worst costs nothing.
        last_break = int(data.get("last_breaking_version", version))
        if last_break > SDSGE_FORMAT_VERSION:
            raise ValueError(
                f"Bundle sdsge_version {version} was written after a breaking "
                f"change at {last_break}, which this library ({SDSGE_FORMAT_VERSION}) "
                f"predates; upgrade SymbolicDSGE."
            )
        sim = data.get("simulation")
        return cls(
            created_by=str(data.get("created_by", "")),
            created_at=(
                None if data.get("created_at") is None else str(data["created_at"])
            ),
            sdsge_version=version,
            last_breaking_version=last_break,
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
