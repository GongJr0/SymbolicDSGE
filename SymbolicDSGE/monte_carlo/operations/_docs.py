"""Docstring composition for the user-facing step factories.

Every ``builtins`` module states its shared factory contract once in a
module-level base string; each factory then carries only its own summary,
unique keyword arguments, and an example as an ordinary literal docstring.
:func:`with_base_doc` stitches the two at import so ``help()`` / ``pydoc`` /
``inspect.getdoc`` show the full composed text, while the source keeps a
readable per-factory literal.

Why assignment rather than a literal: a triple-quoted string is only harvested
into ``__doc__`` because the *compiler* recognizes it as the first statement of
the def. ``help`` reads ``__doc__`` at call time, so assigning a composed value
(what this decorator does) is just as reliable for introspection — it merely
can't be written as a bare literal. The factory keeps its own literal docstring
as the summary; the shared contract is appended beneath it (base-first would
make every factory's ``pydoc`` one-liner identical boilerplate).
"""

from __future__ import annotations

import inspect
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def with_base_doc(base: str) -> Callable[[F], F]:
    """Append the shared ``base`` contract beneath a factory's own docstring.

    The decorated function's literal docstring stays the summary (its first
    line remains the ``help`` / ``pydoc`` one-liner); ``base`` is appended as a
    shared footer. A factory with no docstring of its own gets ``base`` alone.

    Both parts are run through :func:`inspect.cleandoc`, which handles the usual
    docstring shape (first line flush against ``\"\"\"``, the rest indented to
    the code block) that :func:`textwrap.dedent` mangles.
    """
    footer = inspect.cleandoc(base)

    def decorate(func: F) -> F:
        specific = inspect.cleandoc(func.__doc__) if func.__doc__ else ""
        func.__doc__ = f"{specific}\n\n{footer}" if specific else footer
        return func

    return decorate
