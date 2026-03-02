# type: ignore
import pandas as pd
import sympy as sp

from SymbolicDSGE.regression.candidate_ranking import CandidateRanking


def _toy_exprs() -> pd.DataFrame:
    x = sp.Symbol("x")
    return pd.DataFrame(
        {
            "loss": [0.1, 0.3, 1.0],
            "complexity": [1, 2, 3],
            "sympy_format": [x, x + 1, x + sp.Dummy("C_1") + 2],
        }
    )


def test_compute_scores_adds_expected_columns():
    scores = CandidateRanking.compute_scores(_toy_exprs())
    assert set(scores.columns) == {"loss", "complexity", "constant_count", "total"}


def test_compute_scores_total_is_component_sum():
    scores = CandidateRanking.compute_scores(_toy_exprs())
    summed = scores["loss"] + scores["complexity"] + scores["constant_count"]
    assert (scores["total"] - summed).abs().max() < 1e-12


def test_loss_score_prefers_lower_loss():
    score = CandidateRanking._loss_score(_toy_exprs())
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]


def test_complexity_score_prefers_lower_complexity():
    score = CandidateRanking._complexity_score(_toy_exprs())
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]


def test_constant_count_score_penalizes_more_constants_and_dummy_terms():
    score = CandidateRanking._constant_count_score(_toy_exprs())
    assert score.iloc[0] >= score.iloc[1] >= score.iloc[2]
