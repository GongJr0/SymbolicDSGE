---
tags:
    - info
---

# Linear Kalman Filter Benchmark

???+ tip "__TL;DR__"
    If you're not interested in the details, you can skip to the [results](#results).
    However, it's recommended to give a quick skip since not all comparisons are measuring equivalent algorithms.

The linear Gaussian Kalman filter (KF) is the backbone of many DSGE applications. 
It provides state estimation, and is the main likelihood engine of choice for linear first-order DSGE models.
Therefore, the performance of a library's KF has direct and noticeable implications on estimation workflows.
This benchmark compares the SymbolicDSGE KF implementation against Dynare's on multiple published models.

## Model Shapes and Compilation Differences

Dynare handles observables differently than SymbolicDSGE. In Dynare, observables are defined in the model block, increasing the state-space dimension of the model.
SymbolicDSGE treats observables as *deterministic functions of the state*, hence observables do not enter the state-space. However, SymbolicDSGE only allows current-period states into observables, making auxiliary states necessary for some models.
Conversely, SymbolicDSGE routinely has more parameters for an equivalent model due to the KF's measurement covariance `R` being treated as a symbolic and estimable component. Each member of the `R` matrix, when declared, are separate parameters in SymbolicDSGE, while Dynare allows scalar initialization of the covariance matrix.
Differences in model dimensions are reported, but benchmarks do not try to equalize the state-space dimensions. Efficiency of the model representation is considered part of the library's performance.

## Dynare's Kalman Filter

Dynare implements the Kalman filter as a likelihood engine, and chooses to derive predicted, filtered, and smoothed states from a Kalman smoother.
SymbolicDSGE has a general-purpose Kalman filter that's capable of being likelihood-only, but also can return a complete filter history.
Since SymbolicDSGE does not implement a Kalman smoother as of version `2.0.0.dev8` (version of execution), the benchmark compares two separate cases:

1. SymbolicDSGE's likelihood-only KF against Dynare's likelihood-only KF.
2. SymbolicDSGE's full-history KF against Dynare's smoother.

The second-case is not a fair comparison by any means. A filter and smoother are not equal in per-step workload.
Additionally, for `T` observations, a filter will run `T` iterations while a smoother runs `2T` or `2T-1`.
Since Dynare exposes filter traces through their `calib_smoother` and `evaluate_smoother`, this measurement does compare the runtime of obtaining predicted and filtered states in either library.However, the comparison does not aim to imply a conclusion about the performance of Dynare's smoother implementation or claim that SymbolicDSGE has equivalent functionality.

???+ info "Matching Dynare's defaults"
    SymbolicDSGE's defaults largely overlap with Dynare's, with the main difference being the *Joseph form* covariance update.
    The Joseph update is numerically robust, but expensive.
    Dynare uses the simple covariance update, and SymbolicDSGE matches that with `joseph_cov=False`.

## Benchmark Setup

The benchmark times the filters only. The preparation, including solution and compilation, are not included in both libraries. Seeds are fixed and the absolute maximum difference between results are reported alongside the timing. There are 5 models, each published and fairly well-known; all models are tested for both likelihood-only and full-history filtering.
In increasing model complexity, the models are:

- [Lubik and Schorfheide (2004) - POST82 Calibration](https://github.com/GongJr0/SymbolicDSGE/blob/main/tests/fixtures/models/post82_first_order.mod)
- [Ireland (2004)](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Ireland_2004/Ireland_2004.mod)
- [Gali and Monacelli (2005)](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Gali_Monacelli_2005/Gali_Monacelli_2005.mod)
- [Gali (2015) - Chapter 3](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Gali_2015/Gali_2015_chapter_3.mod)
- [Smets and Wouters (2007)](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod)

> All models except Lubik-Schorfheide (2004) are taken from Johannes Pfeifer's `DSGE_mod` and are distributed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). The `.mod` files are originally written for reproduction; each of the used scripts remove the workflows and keep the model definitions. The Lubik-Schorfheide (2004) model is distributed under the [MIT License](https://opensource.org/license/mit/).


The benchmark script is available in [SymbolicDSGE's GitHub repository](https://github.com/GongJr0/SymbolicDSGE) and is runnable from the repo root:

```bash
python ./benchmarks/bench_linear_kf.py --help
```

### Run Configuration

Each bencmark runs on a simulated sample of 200 observations with 200 repetitions. Simulation outputs are produced by SymbolicDSGE once in preparation, and consequent repetitions do not redraw. Three warmup runs are performed for both libraries before timing.

The commands being timed are:

__SymbolicDSGE__

```python
KalmanFilter.run(
            case.A, case.B, case.C, case.d,
            case.Q, case.R, case.y,
            P0=None, # (1)!
            _store_history=False, # (2)!
            symmetrize=False,  # (3)!
            joseph_cov=False,
    )
```

1. Executes a Lyapunov solve to use stationary state covariance; Dynare does the same.
2. `True` when full-history is being timed
3. Default is `True`, disabled to match Dynare

__Dynare__

```matlab
% Loglik only filter
kalman_filter( ...
      Y, 1, periods, zeros(n_endo, 1), P0, ...
      1e-10, 0, false, 0, A, Q, R, H, Z, ...
      n_endo, n_obs, n_exo, 1, 0, {}, false, false);

% Full history
options_.lik_init = 1;  % (1)!
options_.kalman_algo = 1;  % (2)!
options_.first_obs = 1;
options_.nobs = periods;
options_.prefilter = 0;
options_.filtered_vars = 1;  % (3)!
options_.filter_step_ahead = 1;  % (4)!

[oo_, M_, options_, bayestopt_] = evaluate_smoother( ...
        'calibration', var_list_, M_, oo_, options_, bayestopt_, estim_params_);

```

1. `lik_init=1` is Dynare's filter initalization specifier for `P0`. The [Dynare Manual](https://www.dynare.org/manual/the-model-file.html#lik_init%20=%20INTEGER) recommendeds `1` for stationary models, which uses the same unconditional covariance calculation SymbolicDSGE executes.
2. `kalman_algo` chooses the underlying filter implementation the method resolves to, `1` is the linear Gaussian multivariate filter.
3. Enables the calculation of filtered variables. Corresponds to SymbolicDSGE's `x_filt`
4. Enables the calculation of predicted variables. Corresponds to SymbolicDSGE's `x_pred`

The benchmark runs on the following hardware and software configuration:

- CPU: Intel(R) Core(TM) i9-10900K CPU @ 4.90GHz
- MEM: 32GB DDR4 2933MHz
- OS: Windows 11 Pro 22H2
- Python: `cpython 3.11.14`
- SymbolicDSGE: `2.0.0.dev8`
- Dynare: `7.1`
- Dynare Runtimes: MATLAB R2026a, Octave 11.3.0
- MATLAB BLAS/LAPACK: Intel openMKL build 20241031 (LAPACK 3.11.0)
- Octave BLAS/LAPACK: OpenBLAS 0.3.31 (LAPACK 3.12.0)

The following metrics are reported for each benchmark:

- Median: the median runtime in microseconds.
- Runtime Ratio D/S (`SymbolicDSGE = 1`): The runtime ratio of Dynare over SymbolicDSGE. (> 1 means SymbolicDSGE is faster).
- `max |Δ ...|`: the maximum absolute difference in between the two libraries's outputs.

## Results

### Model Shapes

| Model                    | Declared variables S/D | Filter state dimension S/D | Predetermined variables S/D | Observables S/D | Shocks S/D | Parameters S/D |
|--------------------------|:----------------------:|:--------------------------:|:---------------------------:|:---------------:|:----------:|:--------------:|
| Lubik-Schorfheide (2004) |         5 / 8          |           5 / 8            |             3/3             |       3/3       |    3/3     |     20/15      |
| Ireland (2004)           |        11 / 13         |          11 / 13           |             6/6             |       2/2       |    4/4     |     16/10      |
| Gali-Monacelli (2005)    |        15 / 15         |          15 / 15           |             3/3             |       2/2       |    2/2     |      10/9      |
| Gali (2015), Ch. 3       |        20 / 22         |          20 / 22           |             5/5             |       2/2       |    3/3     |     19/12      |
| Smets-Wouters (2007)     |        37 / 40         |          37 / 40           |            20/20            |       7/7       |    7/7     |     57/34      |

### Likelihood Only

Each Dynare runtime cell reports `median μs (D/S)`. The final column is the maximum absolute log-likelihood difference over MATLAB and Octave.

| Model                    | Filter states S/D | SymbolicDSGE μs | Dynare MATLAB μs (D/S) | Dynare Octave μs (D/S) | max \|Δ loglik\| |
|--------------------------|:-----------------:|----------------:|-----------------------:|-----------------------:|-----------------:|
| Lubik-Schorfheide (2004) |       5 / 8       |          119.40 |      1,421.15 (11.90×) |    20,081.52 (168.19×) |        3.411e-13 |
| Ireland (2004)           |      11 / 13      |          351.10 |       1,301.30 (3.71×) |     18,696.43 (53.25×) |        1.592e-12 |
| Gali-Monacelli (2005)    |      15 / 15      |          643.00 |       1,703.85 (2.65×) |     18,992.07 (29.54×) |        9.877e-13 |
| Gali (2015), Ch. 3       |      20 / 22      |        1,158.10 |       2,248.85 (1.94×) |     20,075.08 (17.33×) |        3.411e-13 |
| Smets-Wouters (2007)     |      37 / 40      |        7,549.70 |       3,515.60 (0.47×) |      24,098.04 (3.21×) |        1.116e-10 |

### Retained History

SymbolicDSGE retains filter histories. Dynare's closest public alternative is `calib_smoother`, which also computes smoothed paths, so this is not an equivalent workload. Each Dynare runtime cell reports `median μs (D/S)`. The final columns are maxima over MATLAB and Octave.

| Model                    | Filter states S/D | SymbolicDSGE μs | Dynare MATLAB smoother μs (D/S) | Dynare Octave smoother μs (D/S) | max \|Δ updated\| | max \|Δ predicted\| |
|--------------------------|:-----------------:|----------------:|--------------------------------:|--------------------------------:|------------------:|--------------------:|
| Lubik-Schorfheide (2004) |       5 / 8       |          252.15 |              14,769.60 (58.57×) |             96,778.99 (383.82×) |         1.846e-15 |           1.721e-15 |
| Ireland (2004)           |      11 / 13      |          671.30 |              14,598.60 (21.75x) |             89,367.63 (133.13x) |         1.103e-14 |           9.989e-15 |
| Gali-Monacelli (2005)    |      15 / 15      |        1,201.90 |              15,163.00 (12.65×) |              91,416.48 (76.06×) |         5.884e-14 |           5.063e-14 |
| Gali (2015), Ch. 3       |      20 / 22      |        2,191.30 |               16,820.15 (7.68×) |              99,310.52 (45.32×) |         1.279e-13 |           1.048e-13 |
| Smets-Wouters (2007)     |      37 / 40      |       13,171.15 |               25,777.80 (1.90×) |             185,365.44 (13.39×) |         6.295e-11 |           6.237e-11 |

## Outlook

???+ note "Note on Octave"

    This section will briefly discuss the outcomes of the Octave comparison and leave the technical details to the MATLAB discussion.
    In all tested mdoel sizes SymbolicDSGE performs better than Octave + Dynare.
    Being an open-source alternative, SymbolicDSGE is therefore seen as a definite contender in scenarios where a MATLAB license is not available.
    However, the performance curve clearly shows that SymbolicDSGE will be slower than Octave given a sufficiently large model.


While targeting a small subset of models, the benchmark suggests a coherent performance conclusion.
SymbolicDSGE prefers hand-written kernels over BLAS where possible; as opposed to Dynare, which uses the BLAS implementation of the underlying MATLAB (or Octave) build.
At small sizes calling into BLAS can be a noticable portion of a routine's runtime since the actual algebra is very small.
SymbolicDSGE does not have this overhead, plausibly giving it an advantage for small-to-medium scale models.

Moreover, allocations and memory operations in MATLAB (a dynamically typed language) can be slower than their C counterparts.
For a smaller model, this also is a considerable portion of the runtime.
See SymbolicDSGE's Lubik-Schorfheide (2004) runs, the only difference between likelihood-only and full history runs are a set of `memcpy` operations recording the current state in a given iteration.
This alone doubles the runtime for a Lubik-Schorfheide (2004)-sized model in SymbolicDSGE.
Again, the memory operations will be less and less noticable as the model sizes increase and linear algebra becomes the dominating factor in the hot path.

Both of the above points are plausible advantages of SymbolicDSGE in smaller models.
MATLAB's advantages, on the other hand, show up in larger models where highly optimized and potentially mutli-threaded BLAS routines begin to amortize their overhead.
Smets-Wouters (2007) is the clear example; MATLAB + Dynare at those sizes run 2x faster than SymbolicDSGE.
While the exact point where BLAS overtakes hand-written C will depend on the system, it is clear that past-medium scale, Dynare should be the library of choice for numerical workflows.

For fast-iterating research work, scratch models, and extensive estimation routines for smaller core models (like a NK3 base), SymbolicDSGE is thought to be a contender.
This is not definitive proof of any performance outcome, however it shows that a quick benchmark of SymbolicDSGE against Dynare before a large workflow can be informative and beneficial to determine which library better suites the task.

???+ note "Preparation for Large Workflows"
    If you're planning to use SymbolicDSGE for a large estimation workflow or Monte Carlo pipeline, it is recommended to clone the repository and build the Cython extension targeting your architecture. (`-march=native` as a compiler flag.)
    All results here are derived from the `release` build of SymbolicDSGE and have no architecture specific optimizations.
