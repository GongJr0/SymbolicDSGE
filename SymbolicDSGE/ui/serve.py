"""Unified launcher for the SymbolicDSGE web UI.

:func:`serve_from` is the single entry point all three call sites share:

- the ``sdsge-ui`` CLI (``sdsge-ui [BUNDLE.sdsge]``);
- :meth:`SymbolicDSGE.core.solved_model.SolvedModel.serve` (in-process);
- programmatic callers (``from SymbolicDSGE.ui import serve_from``).

``source`` is polymorphic:

- ``None`` -> empty session (the Builder tab is the entry point);
- a :class:`~SymbolicDSGE.core.solved_model.SolvedModel` -> preload as the
  ``reference`` slot;
- a path / string -> open the ``.sdsge`` bundle, hydrate ``reference``/``dgp``
  and the estimation/MC/sim prefill into the session's :class:`Workspace`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from .estimation import emit_estimation_wire
from .session import Workspace

if TYPE_CHECKING:
    from SymbolicDSGE.bundle.loader import LoadedBundle
    from SymbolicDSGE.core.solved_model import SolvedModel


def serve_from(
    source: "str | Path | SolvedModel | None" = None,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Launch the SymbolicDSGE web UI, optionally hydrated from ``source``."""
    from SymbolicDSGE.core.solved_model import SolvedModel

    from .cli import run_server

    if source is None:
        run_server(host=host, port=port, open_browser=open_browser)
        return

    if isinstance(source, SolvedModel):
        run_server(
            reference=source,
            host=host,
            port=port,
            open_browser=open_browser,
        )
        return

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f"serve_from: bundle path does not exist or is not a file: {path}"
        )

    from SymbolicDSGE.bundle.loader import build_from

    loaded = build_from(path)
    workspace = build_workspace(loaded)
    run_server(
        reference=loaded.reference,
        dgp=loaded.dgp,
        workspace=workspace,
        host=host,
        port=port,
        open_browser=open_browser,
    )


def build_workspace(loaded: "LoadedBundle") -> Workspace:
    """Project a :class:`LoadedBundle` into a :class:`Workspace` preload payload.

    The estimation/MC tabs are seeded with their canonical wire dicts (same
    shape an in-process run would produce), so the frontend can repaint without
    re-running anything. The simulation prefill rides as the SimSpec dict so
    the Outputs tab pre-fills the seed/T/shock controls.
    """
    estimation_wire: dict[str, Any] | None = None
    estimation_spec_dict: dict[str, Any] | None = None
    if loaded.estimation is not None:
        estimation_spec_dict = loaded.estimation.spec.to_dict()
        if loaded.estimation.result is not None:
            estimation_wire = emit_estimation_wire(
                loaded.estimation.result,
                traces=loaded.estimation.posterior,
            )

    mc_wire: dict[str, Any] | None = None
    mc_pipeline_dict: dict[str, Any] | None = None
    if loaded.mc is not None:
        mc_pipeline_dict = loaded.mc.spec.to_dict()
        mc_wire = loaded.mc.wire()

    simulation_dict: dict[str, dict[str, Any]] | None = (
        {role: spec.to_dict() for role, spec in loaded.simulation.items()}
        if loaded.simulation is not None
        else None
    )

    return Workspace(
        estimation=estimation_wire,
        estimation_spec=estimation_spec_dict,
        mc=mc_wire,
        mc_pipeline=mc_pipeline_dict,
        simulation=simulation_dict,
    )
