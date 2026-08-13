"""Native (Cython + C) numeric kernels for SymbolicDSGE.

This package is a *leaf*: it must not import anything from the rest of
``SymbolicDSGE``. Each subsystem (``core``, ``kalman``, ...) builds one private
compiled extension (``_<name>``) from pure-C numeric kernels plus a Cython
shim.

See ``README.md`` for the layout and the recipe for adding a subsystem.
"""
