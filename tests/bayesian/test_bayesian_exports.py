# type: ignore
from SymbolicDSGE.bayesian import Prior, make_prior


def test_bayesian_exports_make_prior_and_prior():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    assert isinstance(prior, Prior)
