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

from .estimation import (
    build_estimation_prefill,
    emit_estimation_wire,
    estimator_spec_wire,
)
from .session import TabState, Workspace

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
        source=str(path),
        host=host,
        port=port,
        open_browser=open_browser,
    )


def build_workspace(loaded: "LoadedBundle") -> Workspace:
    """Project a :class:`LoadedBundle` into a :class:`Workspace` preload payload.

    The estimation/MC tabs land as ``{spec, result, view}``: the bundle's own
    two members carried over as they stand, plus the view the GUI repaints
    from, which for estimation is the spec and result projected into the shape
    its form posts back. The simulation prefill rides as the SimSpec dict so
    the Outputs tab pre-fills the seed/T/shock controls.
    """
    estimation = TabState()
    if loaded.estimation is not None:
        estimation.spec = estimator_spec_wire(loaded.estimation.spec)
        if loaded.estimation.result is not None:
            estimation.result = emit_estimation_wire(loaded.estimation.result)
        if loaded.reference is not None:
            # The form is per-role, so the view is keyed by it. A bundle holds
            # one estimation, tied to the reference model it was run against.
            estimation.view = {
                "reference": build_estimation_prefill(
                    loaded.estimation.spec,
                    loaded.estimation.result,
                    loaded.reference.compiled,
                )
            }

    mc = TabState()
    if loaded.mc is not None:
        mc.spec = loaded.mc.spec.to_dict()
        mc.result = loaded.mc.wire()

    simulation_dict: dict[str, dict[str, Any]] | None = (
        {role: spec.to_dict() for role, spec in loaded.simulation.items()}
        if loaded.simulation is not None
        else None
    )

    return Workspace(
        estimation=estimation,
        mc=mc,
        simulation=simulation_dict,
    )
