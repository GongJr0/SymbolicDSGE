"""Native (Cython + C) numeric kernels for SymbolicDSGE.

This package is a *leaf*: it must not import anything from the rest of
``SymbolicDSGE``. Each subsystem (``core``, ``kalman``, ...) builds one private
compiled extension (``_<name>``) from pure-C numeric kernels plus a thin Cython
shim that maps NumPy buffers to ``double*``.

The numba/Python fallback lives in the *consumer* module, never here. A consumer
selects the native path like so::

    try:
        from .._ckernels.core import simulate_linear_states_into
    except ImportError:  # extension not built (sdist without a compiler, etc.)
        # ... numba/Python reference implementation ...

Importing an unbuilt subsystem therefore raises ``ImportError``, which is exactly
what the consumer's ``try`` expects. Keeping ``_ckernels`` free of any
``SymbolicDSGE`` imports avoids an import cycle once consumers prefer native.

See ``README.md`` for the layout and the recipe for adding a subsystem.
"""
