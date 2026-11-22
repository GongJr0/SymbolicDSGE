from __future__ import annotations

import pytest

from SymbolicDSGE.monte_carlo.mc_constructs import SourceArgs
from SymbolicDSGE.monte_carlo.step_factories import standardize_step


def test_source_args_normalizes_a_single_column_to_a_tuple() -> None:
    selector = SourceArgs(
        arg="sample", source_step="obs", field="payload", columns=0, burn_in=2
    )
    assert selector.columns == (0,)
    assert selector.burn_in == 2


def test_source_args_rejects_negative_burn_in() -> None:
    with pytest.raises(ValueError, match="burn_in"):
        SourceArgs(arg="sample", source_step="obs", field="payload", burn_in=-1)


def test_step_rejects_an_empty_source() -> None:
    with pytest.raises(ValueError, match="`source` must be non-empty"):
        standardize_step("bad", source="", field="payload")


def test_step_splits_source_arguments_from_op_kwargs() -> None:
    step = standardize_step(
        "std", source="obs", field="payload", columns=0, burn_in=2, ddof=1
    )
    assert dict(step.kwargs) == {"ddof": 1}
    (selector,) = step.source_args
    assert (selector.arg, selector.source_step, selector.field) == (
        "sample",
        "obs",
        "payload",
    )
