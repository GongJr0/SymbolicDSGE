"""Step construction for tests that author a pipeline the way the client posts one.

The GUI resolves a step's op kind, lifts its source legs out of the flat form
fields, and separates a custom op's source from its kwargs before posting. A
test that hand-writes a step stands in for that client, so it does the same.
"""

from __future__ import annotations

from typing import Any

from SymbolicDSGE.monte_carlo.spec import OP_TYPES, POSTPROC_KINDS, SourceSpec, StepMeta


def source(
    arg: str,
    source_step: str,
    field: str,
    *,
    columns: list[int] | None = None,
    burn_in: int = 0,
) -> SourceSpec:
    return SourceSpec(
        arg=arg,
        source_step=source_step,
        field=field,
        columns=columns,
        burn_in=burn_in,
    )


def step(
    *,
    step_type: str,
    name: str,
    kwargs: dict[str, Any] | None = None,
    source_args: list[SourceSpec] | None = None,
    n_retain: int = -1,
) -> StepMeta:
    return StepMeta(
        name=name,
        op_type=OP_TYPES.get(step_type, ""),
        step_type=step_type,
        kwargs=dict(kwargs or {}),
        source_args=list(source_args or []),
        n_retain=n_retain,
    )


_LEG_SUFFIXES = ("_source", "_field", "_columns", "_column")
_BARE_LEG_KEYS = {
    "source": "source",
    "field": "field",
    "columns": "columns",
    "column": "columns",
}


def _leg_key(key: str) -> tuple[str, str] | None:
    """Split a flat form key into ``(arg, role)``, or ``None`` if it is not one.

    The spelling is the whole rule: a step with one leg writes the roles bare,
    several legs prefix each by its arg, and a leg taking exactly one column
    says ``column``. Nothing else is consulted.
    """
    if key in _BARE_LEG_KEYS:
        return "sample", _BARE_LEG_KEYS[key]
    for suffix in _LEG_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            role = "columns" if suffix in ("_columns", "_column") else suffix[1:]
            return key[: -len(suffix)], role
    return None


def as_posted(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Resolve a flatly-authored pipeline the way the GUI resolves one.

    The client names each step's op kind, lifts its source legs out of the form
    fields, and splits a custom op's source off its kwargs before posting. A test
    that writes the flat form shape runs it through here so it exercises the same
    payload the browser sends.

    Steps are authored under ``nodes``/``postprocs``; ``edges`` are ignored, since
    the posted pipeline carries no graph of its own.
    """
    replication = [_posted_step(raw) for raw in pipeline.get("nodes") or []]
    postproc = [_posted_step(raw) for raw in pipeline.get("postprocs") or []]
    return {"replication_steps": replication, "postproc_steps": postproc}


def _posted_step(raw: dict[str, Any]) -> dict[str, Any]:
    step_type = raw["step_type"]
    params = dict(raw.get("params") or {})
    burn_in = int(params.pop("burn_in", 0) or 0)
    code = params.pop("code", None)
    n_retain = int(params.pop("n_retain", -1))

    legs: dict[str, dict[str, Any]] = {}
    if step_type not in POSTPROC_KINDS:
        for key in list(params):
            split = _leg_key(key)
            if split is None:
                continue
            arg, role = split
            legs.setdefault(arg, {})[role] = params.pop(key)

    sources = [
        source(
            arg,
            str(parts["source"]),
            str(parts["field"]),
            columns=_columns(parts.get("columns")),
            burn_in=burn_in,
        )
        for arg, parts in legs.items()
        if "source" in parts and "field" in parts
    ]
    posted: dict[str, Any] = dict(
        step(
            step_type=step_type,
            name=raw["name"],
            kwargs=params,
            source_args=sources,
            n_retain=n_retain,
        )
    )
    if code is not None:
        posted["code"] = code
    return posted


def _columns(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    values = value if isinstance(value, (list, tuple)) else [value]
    columns = [int(item) for item in values]
    return columns or None
