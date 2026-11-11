# type: ignore
import pytest

from SymbolicDSGE.estimation import make_prior


def _normal_prior(mean, std):
    return make_prior(
        distribution="normal",
        parameters={"mean": mean, "std": std},
        transform="identity",
    )


@pytest.fixture
def mcmc_estimator(post82_estimator):
    """POST82 estimator with normal priors on ``psi_pi`` / ``rho_r``, ready for
    mcmc. The native mcmc path drives the compiled kernels, so these tests need a
    real model rather than a SimpleNamespace stub + fake loglik."""
    return post82_estimator(
        estimated_params={
            "psi_pi": _normal_prior(2.0, 0.5),
            "rho_r": _normal_prior(0.8, 0.1),
        }
    )


@pytest.fixture
def transformed_estimator(post82_estimator):
    """``rho_r`` under a logit, ``psi_pi`` under the identity.

    A non-identity transform is what makes the two prior densities differ: with
    everything on the identity the jacobian is zero and there is nothing for
    these tests to see.
    """
    return post82_estimator(
        estimated_params={
            "psi_pi": make_prior(
                distribution="normal",
                parameters={"mean": 2.19, "std": 0.5},
                transform="identity",
            ),
            "rho_r": make_prior(
                distribution="beta",
                parameters={"a": 8.0, "b": 2.0},
                transform="logit",
            ),
        },
    )
