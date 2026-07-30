from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.monte_carlo.allocation import _FieldSpec, _compile_field_layout


def test_compile_field_layout_uses_separate_dtype_local_offsets() -> None:
    size, fields = _compile_field_layout(
        {
            "payload": _FieldSpec((4, 3), np.float64),
            "status": _FieldSpec((), np.int64),
            "loglik": _FieldSpec((), np.float64),
            "failure": _FieldSpec((2,), np.int64),
        }
    )

    assert size == (13, 3)
    assert fields["payload"].offset == 0
    assert fields["loglik"].offset == 12
    assert fields["status"].offset == 0
    assert fields["failure"].offset == 1


def test_compile_field_layout_rejects_non_native_dtype() -> None:
    with pytest.raises(TypeError, match="unsupported dtype"):
        _compile_field_layout({"bad": _FieldSpec((1,), np.int32)})
