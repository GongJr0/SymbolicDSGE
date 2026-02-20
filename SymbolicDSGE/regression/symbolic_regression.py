from .sr_backend import SymbolicRegressorBackend
from .model_parametrizer import ModelParametrizer
from .config import TemplateConfig, HessianMode
from .discovery_result import DiscoveryResult
from .hessian_validator import HessianValidator
from .candidate_ranking import CandidateRanking

import sympy as sp
import numpy as np
import pandas as pd


class SymbolicRegressor(SymbolicRegressorBackend):
    """
    Symbolic Regression class with expression candidate selection based on linearity constraints.
    The class respects the restrictions defined in the TemplateConfig and returns both qualified and disqualified expressions.
    """

    def __init__(self, parametrizer: ModelParametrizer) -> None:
        super().__init__(parametrizer)

    def classify_expressions(self, exprs: pd.DataFrame) -> DiscoveryResult:
        """
        Classify expressions into qualified and disqualified based on linearity constraints defined in the TemplateConfig.

        :param expressions: Expressions discovered during a fit.
        :type expressions: pd.DataFrame
        :return: DiscoveryResult containing qualified and disqualified expressions.
        :rtype: DiscoveryResult
        """
        qual_idx: list[int] = []
        disqual_idx: list[int] = []
        top_5_idx: list[int] = []

        scores = CandidateRanking.compute_scores(exprs)

        exprs = exprs.copy()
        exprs = pd.concat([exprs, scores], axis=1)

        hessian_validator = HessianValidator(self.parametrizer)

        qual_idx = [
            idx
            for idx in exprs.index
            if hessian_validator.hessian_compliant(exprs.loc[idx, "sympy_format"])
        ]
        disqual_idx = exprs.index.difference(qual_idx).tolist()

        # Top 5 by score
        top5 = scores.loc[qual_idx].sort_values(by="total", ascending=False).head(5)
        top_5_idx = top5.index.tolist()

        return DiscoveryResult(
            top_candidates=exprs.loc[top_5_idx, :],
            qualified_expressions=exprs.loc[qual_idx, :],
            disqualified_expressions=exprs.loc[disqual_idx, :],
        )
