"""Node construction for tests that author a spec the way the client posts one.

The GUI resolves a node's op kind and lifts its source legs out of the flat form
fields before posting. A test that hand-writes a node stands in for that client,
so it does the same: the op kind comes from the kind taxonomy, and source legs
are given as their own objects.
"""

from __future__ import annotations

from typing import Any

from SymbolicDSGE.monte_carlo.spec import OP_TYPES, NodeSpec, SourceSpec


def source(
    arg: str,
    source_step: str,
    field: str,
    *,
    columns: list[int] | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> SourceSpec:
    return SourceSpec(
        arg=arg,
        source_step=source_step,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )


def node(
    *,
    id: str,
    step_type: str,
    name: str,
    params: dict[str, Any] | None = None,
    sources: list[SourceSpec] | None = None,
) -> NodeSpec:
    return NodeSpec(
        id=id,
        op_type=OP_TYPES.get(step_type, ""),
        step_type=step_type,
        name=name,
        params=dict(params or {}),
        sources=list(sources or []),
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

    The client lifts source legs out of the form fields and names each node's
    op kind before posting. A test that writes the flat form shape runs it
    through here so it exercises the same payload the browser sends.
    """
    posted = dict(pipeline)
    posted["nodes"] = [_posted_node(node) for node in pipeline["nodes"]]
    return posted


def _posted_node(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw.get("params") or {})
    burn_in = int(params.pop("burn_in", 0) or 0)
    drop_initial = bool(params.pop("drop_initial", False))

    legs: dict[str, dict[str, Any]] = {}
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
            drop_initial=drop_initial,
        )
        for arg, parts in legs.items()
        if "source" in parts and "field" in parts
    ]
    node_spec = dict(raw)
    node_spec["op_type"] = OP_TYPES.get(raw["step_type"], "")
    node_spec["params"] = params
    node_spec["sources"] = sources
    return node_spec


def _columns(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    values = value if isinstance(value, (list, tuple)) else [value]
    columns = [int(item) for item in values]
    return columns or None
