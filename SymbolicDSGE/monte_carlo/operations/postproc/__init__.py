"""Post-loop Monte Carlo summaries: run once over the assembled traces.

Shared contract for the factories in this group:

- Input: the across-replication ``traces`` registry (length-n_rep arrays keyed
  like "test.<name>.pval", "regression.<name>.coef", "payload.<name>").
- Output location: ``result.postproc[name]`` holds the op's return verbatim;
  a returned mapping fans out to "<name>.<key>" entries on serialize.
"""

from .builtins import kde_step, postproc_step

__all__ = ["kde_step", "postproc_step"]
