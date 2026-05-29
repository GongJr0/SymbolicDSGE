---
tags:
    - doc
---
# Monte Carlo

The `monte_carlo` module provides a bounded pipeline for repeated simulation, filtering, transformation, and diagnostic testing. The main use case is to treat one `SolvedModel` as the data-generating process (DGP), treat another `SolvedModel` as the reference model, and aggregate diagnostic test results over independent replications.

???+ info "Reference and DGP Roles"
    The built-in simulation step draws data from `dgp`. The built-in filtering step then runs `reference.kalman(...)` on the generated observables. The reference model is not simulated by the built-in DGP pipeline.
