"""Low-level ``.sdsge`` zip read/write.

A ``.sdsge`` file is a plain zip with a ``manifest.json`` at the root plus the
members it enumerates. Text members (``yaml``/``json``/``csv``) are deflated;
Parquet members are already compressed, so they are stored uncompressed
(``ZIP_STORED``). This layer is format-blind: it moves bytes and the manifest,
nothing more. :mod:`SymbolicDSGE.bundle.builder` / :mod:`.loader` interpret them.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path

from .manifest import Manifest

MANIFEST_NAME = "manifest.json"


def _compression_for(path: str) -> int:
    """Parquet rides ``ZIP_STORED`` (already compressed); everything else deflates."""
    return zipfile.ZIP_STORED if path.endswith(".parquet") else zipfile.ZIP_DEFLATED


def write_bundle(
    path: str | Path,
    manifest: Manifest,
    files: Mapping[str, bytes],
) -> None:
    """Write a ``.sdsge`` archive: ``manifest.json`` plus each member.

    Every ``manifest.members`` path must be present in ``files`` (and vice
    versa), so a written bundle is always self-consistent.
    """
    declared = {member.path for member in manifest.members}
    provided = set(files)
    if declared != provided:
        missing = sorted(declared - provided)
        extra = sorted(provided - declared)
        raise ValueError(
            "Bundle members and provided files disagree; "
            f"missing files for {missing}, unreferenced files {extra}."
        )

    target = Path(path)
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(
            MANIFEST_NAME, manifest.to_json(), compress_type=zipfile.ZIP_DEFLATED
        )
        for member_path, data in files.items():
            archive.writestr(
                member_path, data, compress_type=_compression_for(member_path)
            )


class BundleArchive:
    """An opened ``.sdsge`` bundle: its manifest plus member bytes.

    Members are read eagerly into memory on :meth:`open` (bundles are small), so
    no file handle outlives the call.
    """

    def __init__(self, manifest: Manifest, files: Mapping[str, bytes]) -> None:
        self.manifest = manifest
        self._files = dict(files)

    @classmethod
    def open(cls, path: str | Path) -> BundleArchive:
        with zipfile.ZipFile(Path(path), "r") as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                raise ValueError(
                    f"{path!r} is not a valid .sdsge bundle: no {MANIFEST_NAME}."
                )
            manifest = Manifest.from_json(archive.read(MANIFEST_NAME).decode("utf-8"))
            files = {
                member.path: archive.read(member.path)
                for member in manifest.members
                if member.path in names
            }
        missing = [m.path for m in manifest.members if m.path not in files]
        if missing:
            raise ValueError(
                f"Bundle {path!r} manifest references missing members: {missing}."
            )
        return cls(manifest, files)

    def read(self, member_path: str) -> bytes:
        try:
            return self._files[member_path]
        except KeyError as exc:
            raise KeyError(f"No bundle member at {member_path!r}.") from exc

    def read_text(self, member_path: str) -> str:
        return self.read(member_path).decode("utf-8")
