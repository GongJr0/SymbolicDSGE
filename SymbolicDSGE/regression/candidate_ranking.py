import sympy as sp
import pandas as pd


class CandidateRanking:

    @staticmethod
    def compute_scores(exprs: pd.DataFrame) -> pd.DataFrame:
        scores = pd.DataFrame(index=exprs.index)
        scores["loss"] = CandidateRanking._loss_score(exprs)
        scores["complexity"] = CandidateRanking._complexity_score(exprs)
        scores["constant_count"] = CandidateRanking._constant_count_score(exprs)

        scores["total"] = scores.sum(axis=1)

        return scores

    @staticmethod
    def _loss_score(exprs: pd.DataFrame) -> pd.Series:
        loss = exprs["loss"]
        min_loss = loss.min()
        max_loss = loss.max()

        norm_loss = (loss - min_loss) / (max_loss - min_loss + 1e-8)
        out: pd.Series = 1 - norm_loss  # Higher loss -> worse score
        return out

    @staticmethod
    def _complexity_score(exprs: pd.DataFrame) -> pd.Series:
        compl = exprs["complexity"]
        min_compl = compl.min()
        max_compl = compl.max()

        norm_compl = (compl - min_compl) / (max_compl - min_compl + 1e-8)
        out: pd.Series = 1 - norm_compl  # Higher complexity -> worse score
        return out

    @staticmethod
    def _constant_count_score(exprs: pd.DataFrame) -> pd.Series:
        def _get_constant_count(expr: sp.Expr) -> int:
            terms = expr.as_ordered_terms()
            count = 0

            for term in terms:
                if len(term.free_symbols) == 0 and term.is_number:  # pyright: ignore
                    count += 1
                elif isinstance(term, sp.Dummy):
                    count += 1

            return count

        sp_exprs = exprs["sympy_format"]
        counts = sp_exprs.apply(_get_constant_count)

        min_count = counts.min()
        max_count = counts.max()

        norm_counts = (counts - min_count) / (max_count - min_count + 1e-8)
        out: pd.Series = 1 - norm_counts  # More constants -> worse score
        return out
