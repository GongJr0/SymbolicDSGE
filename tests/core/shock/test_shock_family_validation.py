"""What each family requires of ``dist_kwargs`` at the width it drives.

``validate_shock_family`` is the only place a family's parameters are checked.
The kernel reads them off a union whose members overlap, so a missing shape is
indistinguishable from a zero one by the time a plan is built. Targets here need
not be model shocks; whether they are is ``validate_shock_targets``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as scipy_t

from SymbolicDSGE.core.shock.generators import Shock

#: The minimal well-formed kwargs for each family, so a test that means to fail
#: on width or on one parameter is not also failing on a missing one.
VALID_KWARGS = {
    "norm": {},
    "t": {"df": 5.0},
    "uni": {},
    "exp": {},
    "gamma": {"a": 2.5},
    "beta": {"a": 2.0, "b": 5.0},
}

OUT_OF_DOMAIN = [0.0, -1.0, np.nan, np.inf, -np.inf]


# --- what binds -------------------------------------------------------------


@pytest.mark.parametrize("dist", list(VALID_KWARGS))
def test_every_family_binds_at_width_one(dist) -> None:
    shock = Shock(dist, seed=0, dist_kwargs=VALID_KWARGS[dist]).joint("e_u")
    assert shock.target == ("e_u",)


@pytest.mark.parametrize("dist", ["norm", "t"])
def test_the_multivariate_capable_families_bind_to_a_group(dist) -> None:
    shock = Shock(dist, dist_kwargs=VALID_KWARGS[dist]).joint("e_u", "e_v")
    assert shock.target == ("e_u", "e_v")


def test_independent_binding_keeps_a_univariate_family_legal() -> None:
    # Width is per entry, not per spec: a family with no multivariate form still
    # drives several shocks, one width-1 entry each.
    copies = Shock("beta", dist_kwargs=VALID_KWARGS["beta"]).independent("e_u", "e_v")
    assert [shock.target for shock in copies] == [("e_u",), ("e_v",)]


def test_a_scipy_distribution_skips_the_family_checks() -> None:
    # Scipy owns its own parameterization, so neither df <= 2 nor the group is
    # refused here although the "t" family would refuse the first.
    shock = Shock(scipy_t, dist_kwargs={"df": 1.0}).joint("e_u", "e_v")
    assert shock.target == ("e_u", "e_v")


# --- width ------------------------------------------------------------------


@pytest.mark.parametrize("dist", ["uni", "exp", "gamma", "beta"])
def test_a_univariate_only_family_refuses_a_group(dist) -> None:
    with pytest.raises(
        NotImplementedError,
        match=rf"Multivariate '{dist}' shocks are not implemented",
    ):
        Shock(dist, dist_kwargs=VALID_KWARGS[dist]).joint("e_u", "e_v")


# --- parameters -------------------------------------------------------------


def test_family_parameters_are_checked_at_bind_not_construction() -> None:
    # A template carries no width, so nothing width-dependent can be settled
    # until it binds. This is what lets one template serve any model.
    template = Shock("gamma")
    with pytest.raises(ValueError, match=r"shape parameter \('a'\)"):
        template.joint("e_u")


@pytest.mark.parametrize(
    "dist, kwargs, message",
    [
        ("t", {}, "Student-t shocks require 'df'"),
        ("gamma", {}, r"shape parameter \('a'\)"),
        ("beta", {}, r"shape parameters \('a' and 'b'\)"),
        ("beta", {"a": 2.0}, r"shape parameters \('a' and 'b'\)"),
        ("beta", {"b": 5.0}, r"shape parameters \('a' and 'b'\)"),
    ],
)
def test_a_missing_shape_parameter_names_what_it_wants(dist, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Shock(dist, dist_kwargs=kwargs).joint("e_u")


@pytest.mark.parametrize(
    "dist, parameter", [("gamma", "a"), ("beta", "a"), ("beta", "b")]
)
@pytest.mark.parametrize("invalid", OUT_OF_DOMAIN)
def test_a_shape_parameter_must_be_finite_and_positive(
    dist, parameter, invalid
) -> None:
    kwargs = {**VALID_KWARGS[dist], parameter: invalid}
    with pytest.raises(ValueError, match="finite and positive"):
        Shock(dist, dist_kwargs=kwargs).joint("e_u")


@pytest.mark.parametrize("df", [2.0, 1.5, *OUT_OF_DOMAIN])
def test_student_t_needs_a_finite_df_above_two(df) -> None:
    # The standardization divides by sqrt(df - 2), so df <= 2 is not a wide
    # shock but an undefined one.
    with pytest.raises(ValueError, match="df > 2 for finite variance"):
        Shock("t", dist_kwargs={"df": df}).joint("e_u")


# --- the family name itself -------------------------------------------------


def test_an_unknown_family_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown shock distribution family"):
        Shock("cauchy")  # type: ignore[arg-type]


def test_something_that_is_not_a_distribution_is_rejected() -> None:
    with pytest.raises(TypeError, match="scipy.stats distribution object"):
        Shock(7)  # type: ignore[arg-type]
