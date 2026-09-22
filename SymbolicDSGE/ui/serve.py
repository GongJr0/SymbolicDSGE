"""Unified launcher for the SymbolicDSGE web UI.

:func:`serve_from` is the single entry point all three call sites share:

- the ``sdsge-ui`` CLI (``sdsge-ui [BUNDLE.sdsge]``);
- :meth:`SymbolicDSGE.core.solved_model.SolvedModel.serve` (in-process);
- programmatic callers (``from SymbolicDSGE.ui import serve_from``).

``source`` is polymorphic:

- ``None`` -> empty session (the Builder tab is the entry point);
- a :class:`~SymbolicDSGE.core.solved_model.SolvedModel` -> preload as the
  ``reference`` slot;
- a path / string -> open the ``.sdsge`` bundle, hydrate named models
  and the estimation/MC/sim prefill into the session's :class:`Workspace`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..monte_carlo.serialize import serialize_pipeline_result
from ..monte_carlo.spec import pipeline_meta
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
            models={"reference": source},
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

    from SymbolicDSGE.bundle.loader import load_bundle

    loaded = load_bundle(path)
    workspace = build_workspace(loaded)
    run_server(
        models=loaded.models,
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
    eout: dict[str, TabState] = {}
    estimation = TabState()

    if loaded.estimation is not None:
        spec = loaded.estimation.estimator.to_spec()
        estimation.spec = estimator_spec_wire(spec)
        if loaded.estimation.result is not None:
            estimation.result = emit_estimation_wire(loaded.estimation.result)
        member = loaded.manifest.members_by_kind("estimation_spec")[0]
        model_name = "reference" if member.model_name is None else member.model_name
        model = loaded.models[model_name] if loaded.models is not None else None
        if model is not None:
            estimation.view = build_estimation_prefill(
                spec,
                loaded.estimation.result,
                model.compiled,
            )
        eout = {model_name: estimation}

    mc = TabState()
    if loaded.mc is not None:
        mc.spec = dict(pipeline_meta(loaded.mc.pipeline.to_spec()))
        if loaded.mc.result is not None:
            mc.result = serialize_pipeline_result(loaded.mc.result)

    # Spec only: the session replays it against the model once both are
    # installed, which is what fills the result.
    simulation = {
        name: TabState(spec=spec.to_dict())
        for name, spec in (loaded.manifest.simulation or {}).items()
    }

    return Workspace(
        estimation=eout,
        mc=mc,
        simulation=simulation,
    )
