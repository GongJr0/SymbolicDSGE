# type: ignore
import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.bayesian.support import OutOfSupportError
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    AffineProbitTransform,
    Identity,
    LogTransform,
    LogitTransform,
    LowerBoundedTransform,
    ProbitTransform,
    SoftplusTransform,
    UpperBoundedTransform,
    get_transform,
)
from SymbolicDSGE.bayesian.transforms.transform import TransformMethod


@pytest.fixture(
    params=[
        ("identity", Identity(), float64(0.5), np.array([-1.0, 0.5, 2.0]), np.inf),
        ("log", LogTransform(), float64(1.3), np.array([0.2, 0.7, 2.5]), float64(-0.1)),
        (
            "softplus",
            SoftplusTransform(),
            float64(1.3),
            np.array([0.2, 0.7, 2.5]),
            float64(-0.1),
        ),
        (
            "logit",
            LogitTransform(),
            float64(0.4),
            np.array([0.2, 0.4, 0.8]),
            float64(1.2),
        ),
        (
            "probit",
            ProbitTransform(),
            float64(0.4),
            np.array([0.2, 0.4, 0.8]),
            float64(1.2),
        ),
        (
            "affine_logit",
            AffineLogitTransform(float64(-2.0), float64(3.0)),
            float64(0.4),
            np.array([-1.0, 0.4, 2.2]),
            float64(3.0),
        ),
        (
            "affine_probit",
            AffineProbitTransform(float64(-2.0), float64(3.0)),
            float64(0.4),
            np.array([-1.0, 0.4, 2.2]),
            float64(3.0),
        ),
        (
            "lower_bounded",
            LowerBoundedTransform(float64(-1.0)),
            float64(0.4),
            np.array([-0.3, 0.4, 2.2]),
            float64(-1.0),
        ),
        (
            "upper_bounded",
            UpperBoundedTransform(float64(2.0)),
            float64(0.4),
            np.array([-0.3, 0.4, 1.2]),
            float64(2.0),
        ),
    ],
    ids=lambda case: case[0],
)
def transform_case(request):
    return request.param


def test_transform_roundtrip_scalar_and_vector(transform_case):
    _, transform, x_scalar, x_vec, _ = transform_case

    y_scalar = transform.forward(x_scalar)
    y_vec = transform.forward(x_vec)

    x_scalar_back = transform.inverse(y_scalar)
    x_vec_back = transform.inverse(y_vec)

    assert np.isfinite(y_scalar)
    assert np.all(np.isfinite(y_vec))
    assert np.allclose(x_scalar_back, x_scalar, rtol=1e-7, atol=1e-7)
    assert np.allclose(x_vec_back, x_vec, rtol=1e-7, atol=1e-7)


def test_transform_gradients_are_inverse_pairs(transform_case):
    _, transform, x_scalar, _, _ = transform_case

    y = transform.forward(x_scalar)
    grad_f = transform.grad_forward(x_scalar)
    grad_i = transform.grad_inverse(y)

    assert np.isfinite(grad_f)
    assert np.isfinite(grad_i)
    assert np.allclose(grad_f * grad_i, 1.0, rtol=1e-6, atol=1e-6)


def test_transform_logdet_forward_inverse_consistency(transform_case):
    _, transform, x_scalar, _, _ = transform_case

    y = transform.forward(x_scalar)
    logdet_f = transform.log_det_abs_jacobian_forward(x_scalar)
    logdet_i = transform.log_det_abs_jacobian_inverse(y)

    assert np.isfinite(logdet_f)
    assert np.isfinite(logdet_i)
    assert np.allclose(logdet_f + logdet_i, 0.0, rtol=1e-6, atol=1e-6)


def test_transform_support_mapping_membership(transform_case):
    _, transform, x_scalar, x_vec, _ = transform_case

    y_scalar = transform.forward(x_scalar)
    y_vec = transform.forward(x_vec)

    assert transform.support.contains(x_scalar)
    assert transform.support.contains(x_vec)
    assert transform.maps_to.contains(y_scalar)
    assert transform.maps_to.contains(y_vec)


def test_transform_forward_raises_out_of_support(transform_case):
    _, transform, _, _, x_invalid = transform_case

    with pytest.raises(OutOfSupportError):
        transform.forward(x_invalid)


@pytest.mark.parametrize(
    ("transform", "boundary", "expect_finite"),
    [
        (LogTransform(), float64(0.0), True),  # at_boundary(x, "low") -> x+eps
        (SoftplusTransform(), float64(0.0), True),
        (LogitTransform(), float64(0.0), True),
        (LogitTransform(), float64(1.0), True),
    ],
    ids=["log@low", "softplus@low", "logit@low", "logit@high"],
)
def test_transform_boundary_behavior(transform, boundary, expect_finite):
    y = transform.forward(boundary)
    g = transform.grad_forward(boundary)
    j = transform.log_det_abs_jacobian_forward(boundary)

    if expect_finite:
        assert np.isfinite(y)
        assert np.isfinite(g)
        assert np.isfinite(j)
    else:
        assert np.isneginf(y)
        assert np.isposinf(g)
        assert np.isposinf(j)


def test_transform_eps_default_is_small_positive():
    transform = Identity()
    assert transform.eps == float64(1e-8)


def test_transform_dispatch_for_all_registered_methods():
    expected = {
        TransformMethod.IDENTITY: Identity,
        TransformMethod.LOG: LogTransform,
        TransformMethod.SOFTPLUS: SoftplusTransform,
        TransformMethod.LOGIT: LogitTransform,
        TransformMethod.PROBIT: ProbitTransform,
        TransformMethod.AFFINE_LOGIT: AffineLogitTransform,
        TransformMethod.AFFINE_PROBIT: AffineProbitTransform,
        TransformMethod.LOWER_BOUNDED: LowerBoundedTransform,
        TransformMethod.UPPER_BOUNDED: UpperBoundedTransform,
    }
    for method, klass in expected.items():
        assert get_transform(method.value) is klass


def test_transform_dispatch_rejects_unknown_method():
    with pytest.raises(ValueError):
        get_transform("unknown")


def test_transform_repr_returns_class_name(transform_case):
    _, transform, _, _, _ = transform_case
    assert repr(transform) == transform.__class__.__name__
