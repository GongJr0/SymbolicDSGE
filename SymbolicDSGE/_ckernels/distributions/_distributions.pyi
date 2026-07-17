"""Type stubs for the compiled ``_distributions`` extension (placeholder).

The distribution kernels are not implemented yet; this stub exists so the
landsite is ready and the package's ``py.typed`` coverage stays consistent (a
compiled module carries no inspectable types, so without a stub its symbols
degrade to ``Any`` for consumers). As each kernel lands in ``_distributions.pyx``
/ ``distributions.c``, add the matching ``def`` signature here -- typed NumPy
arrays in, status code or output array out -- kept in sync with the numba
reference. The parity tests guard the runtime behavior, not this stub.
"""
