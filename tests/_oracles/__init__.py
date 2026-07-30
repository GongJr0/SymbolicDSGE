"""Reference implementations kept as parity oracles for native kernels.

The ``monte_carlo`` package is a complete snapshot of the former Python Monte
Carlo stack. It is test-only and intentionally independent from the live native
lowering path. Tests import oracle modules through ``_oracles.<subsystem>``
because ``tests`` is on ``pythonpath`` in ``[tool.pytest.ini_options]``.
"""
