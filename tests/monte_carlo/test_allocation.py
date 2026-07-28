from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.monte_carlo.allocation import BufferSpec, allocate_buffers


def test_allocate_buffers_creates_nested_pipeline_owned_arrays() -> None:
    buffers = allocate_buffers(
        {
            "ols": {
                "coef_trace": BufferSpec((4, 3), np.float64),
                "status_trace": BufferSpec((4,), np.int64),
            }
        }
    )

    assert buffers["ols"]["coef_trace"].shape == (4, 3)
    assert buffers["ols"]["coef_trace"].dtype == np.float64
    assert buffers["ols"]["status_trace"].dtype == np.int64


def test_allocate_buffers_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="negative dimension"):
        allocate_buffers({"step": {"bad": BufferSpec((-1,), np.float64)}})
