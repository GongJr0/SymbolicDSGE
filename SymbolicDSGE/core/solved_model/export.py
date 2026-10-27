"""Handing a solved model to something outside the library: the UI, a bundle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..config import ModelConfig

if TYPE_CHECKING:
    from .base import SolvedModel


def serve(
    model: SolvedModel[Any],
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Launch the web playground with ``model`` preloaded as ``reference``."""
    try:
        from ...ui.serve import serve_from
    except ImportError as exc:  # pragma: no cover - exercised without [ui]
        raise ImportError(
            "The SymbolicDSGE UI extra is required for .serve(). "
            "Install it with: pip install 'SymbolicDSGE[ui]'"
        ) from exc

    serve_from(
        source=model,
        host=host,
        port=port,
        open_browser=open_browser,
    )


def to_bundle_builder(
    model: SolvedModel | ModelConfig,
    *,
    name: str = "reference",
    compile_kwargs: Mapping[str, Any] | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> Any:
    """A :class:`BundleBuilder` pre-seeded with the model's YAML."""
    from ...bundle.builder import BundleBuilder

    return BundleBuilder(created_by=created_by).add_model(
        name,
        model,
        compile_kwargs=compile_kwargs,
        solve_kwargs=solve_kwargs,
    )
