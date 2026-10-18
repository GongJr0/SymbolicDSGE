"""``.sdsge`` manifest schema (the bundle index).

Stdlib dataclasses only; the bundle layer lives in the core library, with no
``[ui]`` extra. The manifest is stored as ``manifest.json`` at
the archive root and enumerates every member with its ``kind`` and ``format`` so a
reader can dispatch each one (format-agnostic: a hand-zipped CSV bundle and a
CLI-built Parquet bundle both validate). The simulation prefill (#141) rides inline
here rather than as its own member.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypedDict, get_args

from numpy import ndarray

from ..core.shock.generators import ShockParameters, ShockPathParameters

# --- LAST BREAK ---
# V12: Breakage from adding custom shock serialization (#571).
# Custom shocks used to remain python objects and unserializable.
# The scipy handling refactor introduces quantile tabulation to approximate
# any distribution through a ppf function. Tables are serializable and are encoded
# as array members. Therefore the old shock format (inlined in manifest) is no longer
# available.
#
# The change breaks forward and backward compatibility, both through the on-disk format
# and the refactored shock handling code.
# ------

#: Bundle format version. Bump on every manifest or member change.
SDSGE_FORMAT_VERSION = 12

#: Oldest bundle format this reader can reconstruct.
SDSGE_MIN_READABLE_VERSION = 12

#: Most recent format change requiring a newer reader. Written into the
#: manifest so older readers reject bundles they cannot interpret.
SDSGE_LAST_BREAKING_VERSION = 12

MemberKind = Literal[
    "model_config",
    "raw_data",
    # Shock arrays, pooled across every scope that carries one
    "shock_path",
    "shock_table",
    # Simulation prefill
    "simulation",
    # Estimation
    "estimation_spec",
    "estimation_result",
    "estimation_data",
    "estimation_trace",
    # Monte Carlo: the pipeline and the side-channels it references by key
    "mc_pipeline",
    "mc_data",
    "mc_func",
    # Monte Carlo: a recorded run, one meta member per step kind beside its bulk
    "mc_result_meta",
    "mc_datagen_steps",
    "mc_datagen_trace",
    "mc_filter_steps",
    "mc_filter_trace",
    "mc_test_steps",
    "mc_test_traces",
    "mc_regression_steps",
    "mc_regression_traces",
    "mc_transform_steps",
    "mc_transform_trace",
    "mc_postproc_steps",
    "mc_postproc_raw",
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


class SimSpec(TypedDict):
    """Simulation/output-tab prefill (#141).

    The internal carrier between :meth:`BundleBuilder.set_simulation`, the
    manifest, and the loader. No public signature names it: the builder takes
    the ``sim`` keywords and lowers them here, and the loader materializes it
    into a :class:`SimulationPrefill`.

    No simulation results are stored. Replaying these specs against the
    preloaded model reproduces the intended run (numpy PCG64 + fixed seed).
    The arrays a shock spec carries are not inline: they pool as their own
    members, named by the id the entry references them by.
    """

    T: int
    x0: Mapping[str, float] | list[float] | ndarray | None
    observables: bool
    shock_scale: float
    shocks: list[ShockParameters | ShockPathParameters] | None


def default_sim_spec() -> SimSpec:
    """Return a SimSpec with defaults written in."""
    return SimSpec(
        T=0,
        x0=None,
        observables=False,
        shock_scale=1.0,
        shocks=None,
    )


def _x0_to_json(x0: Mapping[str, float] | list[float] | ndarray | None) -> Any:
    """``x0`` in JSON form, keeping the name-keyed and positional shapes apart."""
    if x0 is None:
        return None
    if isinstance(x0, Mapping):
        return {str(name): float(value) for name, value in x0.items()}
    return [float(value) for value in x0]


@dataclass
class Member:
    """One archive entry described in the manifest.

    ``options`` carries kind-specific metadata. For ``model_config`` it holds the
    ``compile_kwargs``/``solve_kwargs`` needed to rebuild the ``SolvedModel``.
    """

    path: str
    kind: str
    format: str = ""
    model_name: str | None = None
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
        """Convert to a JSON-serializable mapping for the manifest.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``path``, ``kind``, ``format``, and optionally
            ``model_name``, ``columns``, and ``options``.

        """
        out: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "format": self.format,
        }
        if self.model_name is not None:
            out["model_name"] = self.model_name
        if self.columns is not None:
            out["columns"] = list(self.columns)
        if self.options:
            out["options"] = dict(self.options)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Member:
        """Construct a :class:`Member` from a manifest entry.

        Parameters
        ----------
        data : Mapping[str, Any]
            JSON-format dictionary representing a bundle member, typically extracted from the manifest
            of a loaded ``.sdsge`` bundle.

        Returns
        -------
        Member
            Member instance populated with the data from the manifest entry, including path, kind, format,

        """
        return cls(
            path=str(data["path"]),
            kind=str(data["kind"]),
            format=str(data.get("format", "")),
            model_name=(
                None
                if (model_name := data.get("model_name", data.get("role"))) is None
                else str(model_name)
            ),
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
    checksums: dict[str, str] = field(default_factory=dict)

    def members_by_kind(self, kind: str) -> list[Member]:
        """Select all members of a given kind from the manifest.

        Parameters
        ----------
        kind : str
            A string-value :type:`MemberKind`

        Returns
        -------
        list[Member]
            List of :class:`Member` instances from the manifest that match the specified kind.

        """
        return [m for m in self.members if m.kind == kind]

    def model_member(self, name: str) -> Member | None:
        """Get the model config member with a given name if it exists.

        Parameters
        ----------
        name : str
            The name of the model.

        Returns
        -------
        Member | None
            The :class:`Member` instance representing the model configuration with the specified name,
            or None if no such member exists in the manifest.

        """
        for member in self.members:
            if member.kind == "model_config" and member.model_name == name:
                return member
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert a bundle manifest to a JSON-serializable mapping.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary representing the manifest, including versioning information,
            creation metadata, member list, simulation specifications, and checksums.

        """
        out: dict[str, Any] = {
            "sdsge_version": int(self.sdsge_version),
            "last_breaking_version": int(self.last_breaking_version),
            "created_by": self.created_by,
            "members": [m.to_dict() for m in self.members],
        }
        if self.created_at is not None:
            out["created_at"] = self.created_at
        if self.checksums:
            out["checksums"] = dict(self.checksums)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Construct a :class:`Manifest` from a JSON-serializable mapping.

        Parameters
        ----------
        data : Mapping[str, Any]
            JSON-format dictionary representing a bundle manifest, typically extracted from the ``manifest.json`` file of a loaded ``.sdsge`` bundle.

        Returns
        -------
        Manifest
            Manifest instance populated with the data from the JSON mapping, including versioning information, creation metadata, member list, simulation specifications, and checksums.

        """
        version = int(data.get("sdsge_version", SDSGE_FORMAT_VERSION))
        if version < SDSGE_MIN_READABLE_VERSION:
            raise ValueError(
                f"Bundle sdsge_version {version} predates the oldest readable "
                f"format ({SDSGE_MIN_READABLE_VERSION}); rebuild the "
                f"bundle from its sources."
            )
        # Without an explicit compatibility boundary, require a reader at
        # least as recent as the bundle format.
        last_break = int(data.get("last_breaking_version", version))
        if last_break > SDSGE_FORMAT_VERSION:
            raise ValueError(
                f"Bundle sdsge_version {version} was written after a breaking "
                f"change at {last_break}, which this library ({SDSGE_FORMAT_VERSION}) "
                f"predates; upgrade SymbolicDSGE."
            )
        return cls(
            created_by=str(data.get("created_by", "")),
            created_at=(
                None if data.get("created_at") is None else str(data["created_at"])
            ),
            sdsge_version=version,
            last_breaking_version=last_break,
            members=[Member.from_dict(m) for m in data.get("members", [])],
            checksums={
                str(k): str(v) for k, v in dict(data.get("checksums", {})).items()
            },
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Convert the manifest to a JSON string.

        Parameters
        ----------
        indent : int | None
            Level of indentation for pretty-printing the JSON output. If None, the most compact representation is used.

        Returns
        -------
        str
            JSON-formatted string representation of the manifest, suitable for writing to a file or transmitting over a network.

        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Construct a :class:`Manifest` from a JSON string.

        Parameters
        ----------
        text : str
            JSON-formatted string representing a bundle manifest, typically read from the ``manifest.json`` file of a loaded ``.sdsge`` bundle.

        Returns
        -------
        Manifest
            Manifest instance populated with the data from the JSON string, including versioning information, creation metadata, member list, simulation specifications, and checksums.

        """
        return cls.from_dict(json.loads(text))
