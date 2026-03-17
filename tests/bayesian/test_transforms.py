# type: ignore
import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.bayesian.support import OutOfSupportError, Support
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    AffineProbitTransform,
    CholeskyCorrTransform,
    Identity,
    LogTransform,
    LogitTransform,
    LowerBoundedTransform,
    ProbitTransform,
    SoftplusTransform,
    UpperBoundedTransform,
    get_transform,
)
from SymbolicDSGE.bayesian.transforms.transform import Transform, TransformMethod


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
            float64(3.1),
        ),
        (
            "affine_probit",
            AffineProbitTransform(float64(-2.0), float64(3.0)),
            float64(0.4),
            np.array([-1.0, 0.4, 2.2]),
            float64(3.1),
        ),
        (
            "lower_bounded",
            LowerBoundedTransform(float64(-1.0)),
            float64(0.4),
            np.array([-0.3, 0.4, 2.2]),
            float64(-1.1),
        ),
        (
            "upper_bounded",
            UpperBoundedTransform(float64(2.0)),
            float64(0.4),
            np.array([-0.3, 0.4, 1.2]),
            float64(2.1),
        ),
    ],
    ids=lambda case: case[0],
)
def transform_case(request):
    return request.param


def test_transform_roundtrip_scalar_and_vector(transform_case):
    _, transform, x_scalar, x_vec, _ = transform_case

    y_scalar = transform.safe_forward(x_scalar)
    y_vec = transform.safe_forward(x_vec)

    x_scalar_back = transform.safe_inverse(y_scalar)
    x_vec_back = transform.safe_inverse(y_vec)

    assert np.isfinite(y_scalar)
    assert np.all(np.isfinite(y_vec))
    assert np.allclose(x_scalar_back, x_scalar, rtol=1e-7, atol=1e-7)
    assert np.allclose(x_vec_back, x_vec, rtol=1e-7, atol=1e-7)


def test_transform_gradients_are_inverse_pairs(transform_case):
    _, transform, x_scalar, _, _ = transform_case

    y = transform.safe_forward(x_scalar)
    grad_f = transform.safe_grad_forward(x_scalar)
    grad_i = transform.safe_grad_inverse(y)

    assert np.isfinite(grad_f)
    assert np.isfinite(grad_i)
    assert np.allclose(grad_f * grad_i, 1.0, rtol=1e-6, atol=1e-6)


def test_transform_logdet_forward_inverse_consistency(transform_case):
    _, transform, x_scalar, _, _ = transform_case

    y = transform.safe_forward(x_scalar)
    logdet_f = transform.safe_log_det_abs_jacobian_forward(x_scalar)
    logdet_i = transform.safe_log_det_abs_jacobian_inverse(y)

    assert np.isfinite(logdet_f)
    assert np.isfinite(logdet_i)
    assert np.allclose(logdet_f + logdet_i, 0.0, rtol=1e-6, atol=1e-6)


def test_transform_support_mapping_membership(transform_case):
    _, transform, x_scalar, x_vec, _ = transform_case

    y_scalar = transform.safe_forward(x_scalar)
    y_vec = transform.safe_forward(x_vec)

    assert transform.support.contains(x_scalar)
    assert transform.support.contains(x_vec)
    assert transform.maps_to.contains(y_scalar)
    assert transform.maps_to.contains(y_vec)


def test_transform_forward_raises_out_of_support(transform_case):
    _, transform, _, _, x_invalid = transform_case

    with pytest.raises(OutOfSupportError):
        transform.safe_forward(x_invalid)


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
    y = transform.safe_forward(boundary)
    g = transform.safe_grad_forward(boundary)
    j = transform.safe_log_det_abs_jacobian_forward(boundary)

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


def test_safe_forward_vector_adjusts_only_boundary_entries():
    transform = LogitTransform()
    x = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    y = transform.safe_forward(x)

    assert np.all(np.isfinite(y))
    assert np.allclose(y[1], 0.0, atol=1e-12)
    assert y[0] < 0.0
    assert y[2] > 0.0


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
        TransformMethod.CHOLESKY_CORR: CholeskyCorrTransform,
    }
    for method, klass in expected.items():
        assert get_transform(method.value) is klass


def test_transform_dispatch_rejects_unknown_method():
    with pytest.raises(ValueError):
        get_transform("unknown")


def test_transform_repr_returns_class_name(transform_case):
    _, transform, _, _, _ = transform_case
    assert repr(transform) == transform.__class__.__name__


def _valid_lkj_chol_3x3() -> np.ndarray:
    rho12 = 0.2
    rho13 = -0.15
    rho23 = 0.3
    corr = np.array(
        [
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0],
        ],
        dtype=np.float64,
    )
    return np.linalg.cholesky(corr).astype(np.float64)


def test_cholesky_corr_transform_roundtrip_and_support():
    transform = CholeskyCorrTransform(K=3)
    L = _valid_lkj_chol_3x3()

    z = transform.safe_forward(L)
    L_back = transform.safe_inverse(z)

    assert transform.support.contains(L)
    assert transform.maps_to.contains(z)
    assert z.shape == (3,)
    assert np.allclose(L_back, L, atol=1e-10, rtol=0.0)


def test_cholesky_corr_transform_logdet_forward_inverse_consistency():
    transform = CholeskyCorrTransform(K=3)
    L = _valid_lkj_chol_3x3()
    z = transform.safe_forward(L)

    logdet_f = transform.safe_log_det_abs_jacobian_forward(L)
    logdet_i = transform.safe_log_det_abs_jacobian_inverse(z)

    assert np.isfinite(logdet_f)
    assert np.isfinite(logdet_i)
    assert np.allclose(logdet_f + logdet_i, 0.0, atol=1e-10, rtol=0.0)


def test_cholesky_corr_transform_grad_logdet_matches_central_difference():
    transform = CholeskyCorrTransform(K=3)
    z = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    eps = 1e-6

    grad = transform.grad_log_det_abs_jacobian_inverse(z)
    approx = np.empty_like(z)
    for i in range(z.shape[0]):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        approx[i] = (
            transform.log_det_abs_jacobian_inverse(zp)
            - transform.log_det_abs_jacobian_inverse(zm)
        ) / (2.0 * eps)

    assert np.allclose(grad, approx, atol=1e-6, rtol=1e-6)


def test_cholesky_corr_transform_grad_methods_raise_not_implemented():
    transform = CholeskyCorrTransform(K=3)
    with pytest.raises(NotImplementedError):
        transform.grad_forward(_valid_lkj_chol_3x3())
    with pytest.raises(NotImplementedError):
        transform.grad_inverse(np.array([0.1, -0.2, 0.3], dtype=np.float64))


def test_cholesky_corr_transform_rejects_invalid_factor():
    transform = CholeskyCorrTransform(K=3)
    bad = np.eye(3, dtype=np.float64)
    bad[2, 0] = 0.8
    bad[2, 2] = 0.9

    with pytest.raises(ValueError, match="unit norm"):
        transform.safe_forward(bad)


class _FiniteDummyTransform(Transform):
    def __repr__(self) -> str:
        return "FiniteDummy"

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def grad_forward(self, x):
        return np.ones_like(x, dtype=np.float64)

    def grad_inverse(self, y):
        return np.ones_like(y, dtype=np.float64)

    def log_det_abs_jacobian_forward(self, x):
        return np.zeros_like(x, dtype=np.float64)

    def log_det_abs_jacobian_inverse(self, y):
        return np.zeros_like(y, dtype=np.float64)

    def grad_log_det_abs_jacobian_inverse(self, y):
        return np.zeros_like(y, dtype=np.float64)

    @property
    def support(self) -> Support:
        return Support(float64(0.0), float64(1.0))

    @property
    def maps_to(self) -> Support:
        return Support(float64(-1.0), float64(1.0))


def test_transform_base_adjusted_inverse_and_safe_inverse_helpers():
    transform = _FiniteDummyTransform()

    assert transform._get_adjusted_inverse(float64(-1.0)) == pytest.approx(
        -1.0 + transform.eps
    )
    assert transform._get_adjusted_inverse(float64(1.0)) == pytest.approx(
        1.0 - transform.eps
    )
    adjusted = transform._get_adjusted_inverse(
        np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    )
    assert np.allclose(
        adjusted,
        np.array(
            [-1.0 + transform.eps, 0.0, 1.0 - transform.eps],
            dtype=np.float64,
        ),
    )
    with pytest.raises(OutOfSupportError):
        transform._get_adjusted_inverse(float64(2.0))
    with pytest.raises(OutOfSupportError):
        transform._get_adjusted_inverse(np.array([-1.0, 2.0], dtype=np.float64))

    assert transform.safe_inverse(float64(-1.0)) == pytest.approx(-1.0 + transform.eps)
    assert transform.safe_grad_inverse(float64(0.0)) == pytest.approx(1.0)
    assert transform.safe_log_det_abs_jacobian_inverse(float64(0.0)) == pytest.approx(
        0.0
    )


def test_transform_base_abstract_method_bodies_are_reachable_for_coverage():
    dummy = _FiniteDummyTransform()

    assert Transform.forward(dummy, float64(0.0)) is None
    assert Transform.inverse(dummy, float64(0.0)) is None
    assert Transform.grad_forward(dummy, float64(0.0)) is None
    assert Transform.grad_inverse(dummy, float64(0.0)) is None
    assert Transform.log_det_abs_jacobian_forward(dummy, float64(0.0)) is None
    assert Transform.log_det_abs_jacobian_inverse(dummy, float64(0.0)) is None
    assert Transform.grad_log_det_abs_jacobian_inverse(dummy, float64(0.0)) is None
    assert Transform.support.fget(dummy) is None
    assert Transform.maps_to.fget(dummy) is None
    assert Transform.eps.fget(dummy) == pytest.approx(1e-8)


def test_cholesky_corr_transform_scalar_and_validation_error_branches():
    with pytest.raises(ValueError, match="at least 2"):
        CholeskyCorrTransform(K=1)

    transform = CholeskyCorrTransform(K=2)
    z_scalar = float64(0.3)
    L = transform.inverse(z_scalar)
    assert L.shape == (2, 2)
    assert transform.grad_log_det_abs_jacobian_inverse(z_scalar) == pytest.approx(
        -2.0 * np.tanh(0.3)
    )

    with pytest.raises(ValueError, match="scalar input"):
        CholeskyCorrTransform(K=3).inverse(float64(0.2))
    with pytest.raises(ValueError, match="1D array"):
        transform.inverse(np.zeros((1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="Expected 1 unconstrained CPC elements"):
        transform.inverse(np.array([0.1, 0.2], dtype=np.float64))
    with pytest.raises(ValueError, match="Expected a 2x2"):
        transform.forward(np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="lower triangular"):
        transform.forward(np.array([[1.0, 0.1], [0.0, 1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="must be positive"):
        transform.forward(np.array([[0.0, 0.0], [0.1, 1.0]], dtype=np.float64))


@pytest.mark.parametrize(
    ("transform", "x", "y_invalid", "should_raise"),
    [
        (LogTransform(), float64(2.0), float64(np.inf), True),
        (LowerBoundedTransform(float64(-1.0)), float64(0.5), float64(np.inf), True),
        (UpperBoundedTransform(float64(2.0)), float64(0.5), float64(np.inf), True),
        (
            AffineLogitTransform(float64(-2.0), float64(3.0)),
            float64(0.5),
            float64(np.inf),
            False,
        ),
        (
            AffineProbitTransform(float64(-2.0), float64(3.0)),
            float64(0.5),
            float64(np.inf),
            False,
        ),
    ],
)
def test_transform_direct_methods_and_error_branches(
    transform, x, y_invalid, should_raise
):
    y = transform.forward(x)
    assert np.isfinite(y)
    assert np.isfinite(transform.inverse(float64(y)))
    assert np.isfinite(transform.grad_forward(x))
    assert np.isfinite(transform.grad_inverse(float64(y)))
    assert np.isfinite(transform.log_det_abs_jacobian_forward(x))
    assert np.isfinite(transform.log_det_abs_jacobian_inverse(float64(y)))
    assert np.isfinite(transform.grad_log_det_abs_jacobian_inverse(float64(y)))

    if should_raise:
        with pytest.raises(OutOfSupportError):
            transform.inverse(y_invalid)
    else:
        assert np.isfinite(transform.inverse(y_invalid))
