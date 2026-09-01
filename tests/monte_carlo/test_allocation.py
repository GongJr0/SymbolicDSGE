from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.monte_carlo._offsets import ArenaOffset
from SymbolicDSGE.monte_carlo.allocation import _FieldSpec, _compile_field_layout

_EMPTY = ArenaOffset(foffset=(), fwidth=(), ioffset=(), iwidth=())


def test_compile_field_layout_reads_each_lane_in_its_own_order() -> None:
    """A field takes the next buffer in the lane its dtype selects."""
    size, fields = _compile_field_layout(
        {
            "payload": _FieldSpec((4, 3), np.float64),
            "status": _FieldSpec((), np.int64),
            "loglik": _FieldSpec((), np.float64),
            "failure": _FieldSpec((2,), np.int64),
        },
        ArenaOffset(foffset=(0, 12), fwidth=(12, 1), ioffset=(0, 1), iwidth=(1, 2)),
    )

    assert size == (13, 3)
    assert fields["payload"].offset == 0
    assert fields["loglik"].offset == 12
    assert fields["status"].offset == 0
    assert fields["failure"].offset == 1


def test_compile_field_layout_rejects_non_native_dtype() -> None:
    with pytest.raises(TypeError, match="unsupported dtype"):
        _compile_field_layout({"bad": _FieldSpec((1,), np.int32)}, _EMPTY)


def test_a_field_with_no_native_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="no buffer in the native layout"):
        _compile_field_layout({"payload": _FieldSpec((2,), np.float64)}, _EMPTY)


def test_a_native_buffer_no_field_names_is_rejected() -> None:
    with pytest.raises(ValueError, match="fields name them"):
        _compile_field_layout(
            {"payload": _FieldSpec((2,), np.float64)},
            ArenaOffset(foffset=(0, 2), fwidth=(2, 2), ioffset=(), iwidth=()),
        )


def test_a_field_the_layout_left_out_reports_absence() -> None:
    """The width is the layout's, and the shape says the buffer was dropped."""
    _, fields = _compile_field_layout(
        {"se": _FieldSpec((3,), np.float64)},
        ArenaOffset(foffset=(0,), fwidth=(0,), ioffset=(), iwidth=()),
    )

    assert fields["se"].shape == (0,)
    assert fields["se"].flat_count == 0


def test_a_field_that_is_merely_empty_keeps_its_own_shape() -> None:
    """A transform wider than its source resolves to zero rows, not to absence."""
    _, fields = _compile_field_layout(
        {"payload": _FieldSpec((0, 2), np.float64)},
        ArenaOffset(foffset=(0,), fwidth=(0,), ioffset=(), iwidth=()),
    )

    assert fields["payload"].shape == (0, 2)
