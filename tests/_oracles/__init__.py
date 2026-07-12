"""Reference (numba / numpy) implementations kept as parity oracles for the
native ``_ckernels`` kernels. Not part of the shipped library; imported only by
the parity tests via ``from _oracles.<subsystem> import ...`` (``tests`` is on
the path through ``pythonpath`` in ``[tool.pytest.ini_options]``)."""
