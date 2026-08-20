---
tags:
    - benchmark
---

# Point Estimation Benchmark (MLE/MAP)

???+ tip "TL;DR"
    If you're not interested in the details, you can skip to the [results](#results).
    However, it's recommended to give a quick skim.

This benchmark is designed as an extension of the [Linear Gaussian Kalman Filter benchmark](./linear_kf.md) which isolated the most expensive part of likelihood evaluation, i.e. the Kalman filter and tested SymbolicDSGE against Dynare.
The results established that SymbolicDSGE's choice of avoiding BLAS/LAPACK is and advantage for small to medium sized models; while the performance advantage diminishes and is eventually reversed at the largest tested model (Smets-Wouters 2007).
This benchmark focuses on the end-to-end point estimation to materialize the runtimes a user can experience in common workflows.
The benchmark is carried out with 5 published first-order DSGE models, where a two parameters are estimated per model.

## Model Compilation and Shapes

Dynare handles observables differently than SymbolicDSGE. In Dynare, observables are defined in the model block, potentially increasing the state-space dimension of the model.
SymbolicDSGE treats observables as *deterministic functions of the state*, hence observables do not enter the state-space. However, SymbolicDSGE only allows current-period states into observables, making auxiliary states necessary for some models.

## Benchmark Setup

### Optimizers

SymbolicDSGE implements the `L-BFGS-B` and `Nelder-Mead` optimizers natively.
There are no equivalent parity-identical implementations for `L-BFGS-B` in Dynare, therefore the benchmark will only compare `Nelder-Mead` implementations.
Dynare implements multiple optimizers and exposes them through `mode_compute`. Through parity testing, we found that the named Nelder-Mead implementation (`mode_compute=8`) does not hold parity with SymbolicDSGE's implementation, while `mode_compute=7` (`fminsearch`) can be bit-exactly matched to SymbolicDSGE's implementation on some models.
Dynare's `mode_compute=8` is explained as "generally more efficient than the MATLAB or Octave implementation available with `mode_compute=7`" (see [the Dynare Manual](https://www.dynare.org/manual/the-model-file.html#mode_compute%20=%20INTEGER%20|%20FUNCTION_NAME)); conversely, our testing shows that Dynare's implementation is substantially less efficient than the MATLAB/Octave implementation for the model sizes and estimated dimensions of our benchmark.
The benchmark reports results for both `mode_compute=7` and `mode_compute=8` in Dynare.
However, performance discussions will assume the faster `mode_compute=7` as the reference for Dynare.

???+ info "`mode_compute=7` Availability"
    Dynare's `mode_compute=7` requires MATLAB's Optimization Toolbox add-on, and Octave's `optim` package.
    While `optim` is available for free, the Optimization Toolbox can be a paid add-on depending on the MATLAB license.

### Models

In increasing order of size, the benchmark uses the following models:

- [Lubik and Schorfheide (2004) - POST82 Calibration](https://github.com/GongJr0/SymbolicDSGE/blob/main/tests/fixtures/models/post82_first_order.mod)
- [Ireland (2004)](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Ireland_2004/Ireland_2004.mod)
- [Gali and Monacelli (2005)](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Gali_Monacelli_2005/Gali_Monacelli_2005.mod)
- [Gali (2015) - Chapter 3](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Gali_2015/Gali_2015_chapter_3.mod)
- [Smets and Wouters (2007)](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod)

> All models except Lubik-Schorfheide (2004) are taken from Johannes Pfeifer's `DSGE_mod` and are distributed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). The `.mod` files are originally written for reproduction; each of the used scripts remove the original workflows and keep the model definitions. The Lubik-Schorfheide (2004) model is hand-transcribed and distributed under the [MIT License](https://opensource.org/license/mit/).

### Model Shapes

| Model                    | Declared variables S/D | Predetermined variables S/D | Observables S/D | Shocks S/D | Parameters S/D |
|--------------------------|:----------------------:|:---------------------------:|:---------------:|:----------:|:--------------:|
| Lubik-Schorfheide (2004) |         5 / 8          |             3/3             |       3/3       |    3/3     |     20/15      |
| Ireland (2004)           |        11 / 13         |             6/6             |       2/2       |    4/4     |     16/10      |
| Gali-Monacelli (2005)    |        15 / 15         |             3/3             |       2/2       |    2/2     |      10/9      |
| Gali (2015), Ch. 3       |        20 / 22         |             5/5             |       2/2       |    3/3     |     19/12      |
| Smets-Wouters (2007)     |        37 / 40         |            20/20            |       7/7       |    7/7     |     57/34      |

### Benchmark Specification

Each model is estimated with 200 simulated observations and the estimation is repeated 20 times with the same seed per model.
One warmup run is performed before the timed runs.
The benchmark script is available in [SymbolicDSGE's GitHub repository](https://github.com/GongJr0/SymbolicDSGE) and is runnable from the repo root:

```bash
python ./benchmarks/bench_estimation.py --help
```

The reported columns are:
    - __Model Size S/D:__ Number of states in SymbolicDSGE and Dynare, respectively.
    - __Median Runtime:__ Median runtime in milliseconds across 20 runs.
    - __nfev S/D:__ Number of function evaluations in SymbolicDSGE and Dynare, respectively.
    - __nit S/D:__ Number of iterations in SymbolicDSGE and Dynare, respectively.
    - __max |Δ loglik|:__ Maximum absolute difference in log-likelihood between SymbolicDSGE and Dynare across 20 runs.
    - __max |Δ theta|:__ Maximum absolute difference in parameter estimates between SymbolicDSGE and Dynare across 20 runs.

### Hardware and Software Specifications

- CPU: Intel(R) Core(TM) i9-10900K CPU @ 4.90GHz
- MEM: 32GB DDR4 2933MHz
- OS: Windows 11 Pro 22H2
- Python: `cpython 3.11.14`
- SymbolicDSGE: `2.0.0.dev8`
- Dynare: `7.1`
- Dynare Runtimes: MATLAB R2026a, Octave 11.3.0
- MATLAB BLAS/LAPACK: Intel openMKL build 20241031 (LAPACK 3.11.0)
- Octave BLAS/LAPACK: OpenBLAS 0.3.31 (LAPACK 3.12.0)

### Optimization Options

SymbolicDSGE matches the Dynare default optimization options for both `mode_compute=7` and `mode_compute=8`.
Below are the specific optimization options used for the benchmark:

|                              | `mode_compute=7` | `mode_compute=8` |
|:----------------------------:|:----------------:|:----------------:|
|        Max Iterations        |      6,000       |      10,000      |
|   Max Function Evaluations   |    1,000,000     |  500 * `n_par`   |
|   Theta Absolute Tolerance   |       1e-6       |       1e-4       |
| Objective Absolute Tolerance |       1e-8       |       1e-4       |
|     Initial Simplex Size     |       n/a        |       0.05       |

### Estimated Parameters

Each model has two parameters estimated. In the MAP case, all parameters use a Gaussian prior.

## Results

### mode_compute=7 | MLE

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "background": null,
  "description": "Runtime comparison for mode_compute=7, MLE.",
  "data": {
    "values": [
      {"model":"LS 2004","runtime":10.03,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"LS 2004","runtime":115.24,"implementation":"Dynare MATLAB", "label": "11.54x"},
      {"model":"LS 2004","runtime":1038.48,"implementation":"Dynare Octave", "label": "103.51x"},

      {"model":"Ireland 2004","runtime":27.93,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Ireland 2004","runtime":136.40,"implementation":"Dynare MATLAB", "label": "4.88x"},
      {"model":"Ireland 2004","runtime":1016.41,"implementation":"Dynare Octave", "label": "36.40x"},

      {"model":"GM 2005","runtime":112.98,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"GM 2005","runtime":383.98,"implementation":"Dynare MATLAB", "label": "3.40x"},
      {"model":"GM 2005","runtime":2095.80,"implementation":"Dynare Octave", "label": "18.55x"},

      {"model":"Gali Ch. 3","runtime":91.46,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Gali Ch. 3","runtime":115.43,"implementation":"Dynare MATLAB", "label": "1.26x"},
      {"model":"Gali Ch. 3","runtime":886.53,"implementation":"Dynare Octave", "label": "9.69x"},

      {"model":"SW 2007","runtime":674.99,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"SW 2007","runtime":475.61,"implementation":"Dynare MATLAB", "label": "0.70x"},
      {"model":"SW 2007","runtime":2824.32,"implementation":"Dynare Octave", "label": "4.18x"}
    ]
  },
  "transform": [
    {"calculate": "5", "as": "baseline"}
  ],
  "layer": [
    {
      "mark": {
        "type": "bar"
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ],
          "axis": {
            "title": null,
            "labelAngle": 0
          }
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          },
          "axis": {
            "title": "Runtime (ms, log scale)",
            "values": [
              10,
              100,
              1000,
              10000
            ]
          }
        },
        "y2": {
          "field": "baseline"
        },
        "color": {
          "field": "implementation",
          "type": "nominal",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ],
          "legend": {
            "title": null,
            "orient": "top"
          }
        },
        "tooltip": [
          {
            "field": "model",
            "type": "nominal",
            "title": "Model"
          },
          {
            "field": "implementation",
            "type": "nominal",
            "title": "Implementation"
          },
          {
            "field": "runtime",
            "type": "quantitative",
            "title": "Runtime (ms)",
            "format": ",.2f"
          }
        ]
      }
    },
    {
      "mark": {
        "type": "text",
        "fontWeight": "bold",
        "dy": -7
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ]
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          }
        },
        "text": {
          "field": "label",
          "type": "nominal"
        }
      }
    }
  ],
  "config": {
    "view": {
      "fill": null,
      "stroke": null
    },
    "axis": {"labelFontSize": 13, "titleFontSize": 14},
    "axisX": {"labelFontWeight": "bold", "labelFontSize": 14},
    "axisY": {"titleFontWeight": "bold"},
    "legend": {"labelFontSize": 14},
    "text": {"fontSize": 12}
  }
}
```

??? note "Data (mode_compute=7, MLE, MATLAB)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare MATLAB ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |           10.03 |        115.24 (11.54×) |  87/87   |  45/45  |         0.000e+0 |    0.000e+0     |
    | Ireland (2004)           |  11 / 13   |           27.93 |         136.40 (4.88x) |  74/73   |  38/37  |        5.052e-07 |    8.881e-07    |
    | Gali-Monacelli (2005)    |  15 / 15   |          112.98 |         383.98 (3.40x) | 164/146  |  75/63  |        8.527e-14 |    2.997e-01    |
    | Gali (2015), Ch. 3       |  20 / 22   |           91.46 |         115.43 (1.26x) |  72/72   |  36/36  |        4.510e-07 |    1.744e-07    |
    | Smets-Wouters (2007)     |  37 / 40   |          674.99 |         475.61 (0.70x) |  80/80   |  42/42  |        3.365e-11 |    8.678e-08    |

??? note "Data (mode_compute=7, MLE, Octave)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare Octave ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |           10.03 |     1,038.48 (103.51×) |  87/116  |  45/53  |        6.139e-12 |    1.646e-07    |
    | Ireland (2004)           |  11 / 13   |           27.93 |      1,016.41 (36.40x) |  77/89   |  38/39  |        5.049e-07 |    7.432e-07    |
    | Gali-Monacelli (2005)    |  15 / 15   |          112.98 |      2,095.80 (18.55x) |  164/99  |  75/35  |        1.491e-11 |    1.835e-01    |
    | Gali (2015), Ch. 3       |   20 /22   |           91.46 |         886.53 (9.69x) |  77/93   |  36/46  |        4.507e-07 |    9.033e-07    |
    | Smets-Wouters (2007)     |  37 / 40   |          674.99 |       2,824.32 (4.18x) |  80/93   |  42/43  |        1.010e-09 |    3.706e-07    |

### mode_compute=7 | MAP

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "background": null,
  "description": "Runtime comparison for mode_compute=7, MAP.",
  "data": {
    "values": [
      {"model":"LS 2004","runtime":10.69,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"LS 2004","runtime":122.06,"implementation":"Dynare MATLAB", "label": "11.41x"},
      {"model":"LS 2004","runtime":947.94,"implementation":"Dynare Octave", "label": "88.64x"},

      {"model":"Ireland 2004","runtime":29.01,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Ireland 2004","runtime":134.04,"implementation":"Dynare MATLAB", "label": "4.62x"},
      {"model":"Ireland 2004","runtime":956.75,"implementation":"Dynare Octave", "label": "32.99x"},

      {"model":"GM 2005","runtime":61.43,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"GM 2005","runtime":269.12,"implementation":"Dynare MATLAB", "label": "4.38x"},
      {"model":"GM 2005","runtime":3634.37,"implementation":"Dynare Octave", "label": "59.16x"},

      {"model":"Gali Ch. 3","runtime":90.54,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Gali Ch. 3","runtime":125.92,"implementation":"Dynare MATLAB", "label": "1.39x"},
      {"model":"Gali Ch. 3","runtime":907.86,"implementation":"Dynare Octave", "label": "10.03x"},

      {"model":"SW 2007","runtime":660.28,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"SW 2007","runtime":472.06,"implementation":"Dynare MATLAB", "label": "0.71x"},
      {"model":"SW 2007","runtime":2944.92,"implementation":"Dynare Octave", "label": "4.46x"}
    ]
  },
  "transform": [
    {"calculate": "5", "as": "baseline"}
  ],
  "layer": [
    {
      "mark": {
        "type": "bar"
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ],
          "axis": {
            "title": null,
            "labelAngle": 0
          }
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          },
          "axis": {
            "title": "Runtime (ms, log scale)",
            "values": [
              10,
              100,
              1000,
              10000
            ]
          }
        },
        "y2": {
          "field": "baseline"
        },
        "color": {
          "field": "implementation",
          "type": "nominal",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ],
          "legend": {
            "title": null,
            "orient": "top"
          }
        },
        "tooltip": [
          {
            "field": "model",
            "type": "nominal",
            "title": "Model"
          },
          {
            "field": "implementation",
            "type": "nominal",
            "title": "Implementation"
          },
          {
            "field": "runtime",
            "type": "quantitative",
            "title": "Runtime (ms)",
            "format": ",.2f"
          }
        ]
      }
    },
    {
      "mark": {
        "type": "text",
        "fontWeight": "bold",
        "dy": -7
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ]
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          }
        },
        "text": {
          "field": "label",
          "type": "nominal"
        }
      }
    }
  ],
  "config": {
    "view": {
      "fill": null,
      "stroke": null
    },
    "axis": {"labelFontSize": 13, "titleFontSize": 14},
    "axisX": {"labelFontWeight": "bold", "labelFontSize": 14},
    "axisY": {"titleFontWeight": "bold"},
    "legend": {"labelFontSize": 14},
    "text": {"fontSize": 12}
  }
}
```

??? note "Data (mode_compute=7, MAP, MATLAB)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare MATLAB ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |           10.69 |        122.06 (11.41×) |  91/91   |  48/48  |         0.000e+0 |    0.000e+0     |
    | Ireland (2004)           |  11 / 13   |           29.01 |         134.04 (4.62x) |  77/71   |  40/36  |        5.008e-07 |    2.406e-07    |
    | Gali-Monacelli (2005)    |  15 / 15   |           61.43 |         269.12 (4.38x) |  93/93   |  47/47  |        1.990e-13 |    7.201e-07    |
    | Gali (2015), Ch. 3       |  20 / 22   |           90.54 |         125.92 (1.39x) |  72/72   |  37/37  |        4.296e-07 |    0.000e+00    |
    | Smets-Wouters (2007)     |  37 / 40   |          660.28 |         472.06 (0.71x) |  78/78   |  40/40  |        2.046e-11 |    0.000e+00    |

??? note "Data (mode_compute=7, MAP, Octave)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare Octave ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |           10.69 |        947.94 (88.64×) |  91/105  |  48/51  |        1.046e-11 |    8.817e-07    |
    | Ireland (2004)           |  11 / 13   |           29.01 |        956.75 (32.99x) |  77/84   |  40/35  |        5.022e-07 |    8.310e-07    |
    | Gali-Monacelli (2005)    |  15 / 15   |           61.43 |      3,634.37 (59.16x) |  93/164  |  47/71  |        5.812e-12 |    2.147e-07    |
    | Gali (2015), Ch. 3       |  20 / 22   |           90.54 |        907.86 (10.03x) |  72/93   |  37/46  |        4.297e-07 |    5.618e-07    |
    | Smets-Wouters (2007)     |  37 / 40   |          660.28 |       2,944.92 (4.46x) |  78/95   |  40/42  |        3.674e-10 |    2.456e-07    |

### mode_compute=8 | MLE

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "background": null,
  "description": "Runtime comparison for mode_compute=8, MLE.",
  "data": {
    "values": [
      {"model":"LS 2004","runtime":6.85,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"LS 2004","runtime":238.29,"implementation":"Dynare MATLAB", "label": "34.78x"},
      {"model":"LS 2004","runtime":1911.44,"implementation":"Dynare Octave", "label": "279.02x"},

      {"model":"Ireland 2004","runtime":16.08,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Ireland 2004","runtime":180.50,"implementation":"Dynare MATLAB", "label": "11.22x"},
      {"model":"Ireland 2004","runtime":1425.83,"implementation":"Dynare Octave", "label": "88.66x"},

      {"model":"GM 2005","runtime":81.82,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"GM 2005","runtime":659.12,"implementation":"Dynare MATLAB", "label": "8.06x"},
      {"model":"GM 2005","runtime":6579.23,"implementation":"Dynare Octave", "label": "80.41x"},

      {"model":"Gali Ch. 3","runtime":55.17,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Gali Ch. 3","runtime":254.15,"implementation":"Dynare MATLAB", "label": "4.61x"},
      {"model":"Gali Ch. 3","runtime":1579.20,"implementation":"Dynare Octave", "label": "28.62x"},

      {"model":"SW 2007","runtime":433.11,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"SW 2007","runtime":850.96,"implementation":"Dynare MATLAB", "label": "1.96x"},
      {"model":"SW 2007","runtime":5020.61,"implementation":"Dynare Octave", "label": "11.59x"}
    ]
  },
  "transform": [
    {"calculate": "5", "as": "baseline"}
  ],
  "layer": [
    {
      "mark": {
        "type": "bar"
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ],
          "axis": {
            "title": null,
            "labelAngle": 0
          }
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          },
          "axis": {
            "title": "Runtime (ms, log scale)",
            "values": [
              10,
              100,
              1000,
              10000
            ]
          }
        },
        "y2": {
          "field": "baseline"
        },
        "color": {
          "field": "implementation",
          "type": "nominal",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ],
          "legend": {
            "title": null,
            "orient": "top"
          }
        },
        "tooltip": [
          {
            "field": "model",
            "type": "nominal",
            "title": "Model"
          },
          {
            "field": "implementation",
            "type": "nominal",
            "title": "Implementation"
          },
          {
            "field": "runtime",
            "type": "quantitative",
            "title": "Runtime (ms)",
            "format": ",.2f"
          }
        ]
      }
    },
    {
      "mark": {
        "type": "text",
        "fontWeight": "bold",
        "dy": -7
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ]
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          }
        },
        "text": {
          "field": "label",
          "type": "nominal"
        }
      }
    }
  ],
  "config": {
    "view": {
      "fill": null,
      "stroke": null
    },
    "axis": {"labelFontSize": 13, "titleFontSize": 14},
    "axisX": {"labelFontWeight": "bold", "labelFontSize": 14},
    "axisY": {"titleFontWeight": "bold"},
    "legend": {"labelFontSize": 14},
    "text": {"fontSize": 12}
  }
}
```

??? note "Data (mode_compute=8, MLE, MATLAB)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare MATLAB ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |            6.85 |        238.29 (34.78x) |  57/209  | 30/102  |        5.914e-08 |    7.220e-05    |
    | Ireland (2004)           |  11 / 13   |           16.08 |        180.50 (11.22x) |  42/219  |  22/44  |        9.489e-06 |    2.956e-05    |
    | Gali-Monacelli (2005)    |  15 / 15   |           81.82 |         659.12 (8.06x) | 124/266  | 59/123  |        9.948e-14 |    2.229e-01    |
    | Gali (2015), Ch. 3       |  20 / 22   |           55.17 |         254.15 (4.61x) |  45/146  |  22/69  |        8.544e-07 |    2.924e-05    |
    | Smets-Wouters (2007)     |  37 / 40   |          433.11 |         850.96 (1.96x) |  51/154  |  26/73  |        2.501e-11 |    0.000e+00    |

??? note "Data (mode_compute=8, MLE, Octave)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare Octave ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |            6.85 |     1,911.44 (279.02x) |  57/209  | 30/102  |        5.914e-08 |    7.220e-05    |
    | Ireland (2004)           |  11 / 13   |           16.08 |      1,425.83 (88.66x) |  42/219  |  22/44  |        9.489e-06 |    2.956e-05    |
    | Gali-Monacelli (2005)    |  15 / 15   |           81.82 |      6,579.23 (80.41x) | 124/273  | 59/126  |        4.263e-14 |    3.677e-02    |
    | Gali (2015), Ch. 3       |  20 / 22   |           55.17 |      1,579.20 (28.62x) |  45/146  |  22/69  |        8.544e-07 |    2.924e-05    |
    | Smets-Wouters (2007)     |  37 / 40   |          433.11 |      5,020.61 (11.59x) |  51/154  |  26/73  |        7.708e-11 |    0.000e+00    |

### mode_compute=8 | MAP

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "background": null,
  "description": "Runtime comparison for mode_compute=8, MAP.",
  "data": {
    "values": [
      {"model":"LS 2004","runtime":6.94,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"LS 2004","runtime":200.48,"implementation":"Dynare MATLAB", "label": "28.89x"},
      {"model":"LS 2004","runtime":1576.83,"implementation":"Dynare Octave", "label": "227.25x"},

      {"model":"Ireland 2004","runtime":15.97,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Ireland 2004","runtime":172.77,"implementation":"Dynare MATLAB", "label": "10.82x"},
      {"model":"Ireland 2004","runtime":1278.40,"implementation":"Dynare Octave", "label": "80.06x"},

      {"model":"GM 2005","runtime":41.39,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"GM 2005","runtime":469.03,"implementation":"Dynare MATLAB", "label": "11.33x"},
      {"model":"GM 2005","runtime":4306.12,"implementation":"Dynare Octave", "label": "104.03x"},

      {"model":"Gali Ch. 3","runtime":60.19,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"Gali Ch. 3","runtime":200.20,"implementation":"Dynare MATLAB", "label": "3.33x"},
      {"model":"Gali Ch. 3","runtime":1554.34,"implementation":"Dynare Octave", "label": "25.82x"},

      {"model":"SW 2007","runtime":402.29,"implementation":"SymbolicDSGE", "label": "1.00x"},
      {"model":"SW 2007","runtime":855.26,"implementation":"Dynare MATLAB", "label": "2.13x"},
      {"model":"SW 2007","runtime":4978.64,"implementation":"Dynare Octave", "label": "12.38x"}
    ]
  },
  "transform": [
    {"calculate": "5", "as": "baseline"}
  ],
  "layer": [
    {
      "mark": {
        "type": "bar"
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ],
          "axis": {
            "title": null,
            "labelAngle": 0
          }
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          },
          "axis": {
            "title": "Runtime (ms, log scale)",
            "values": [
              10,
              100,
              1000,
              10000
            ]
          }
        },
        "y2": {
          "field": "baseline"
        },
        "color": {
          "field": "implementation",
          "type": "nominal",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ],
          "legend": {
            "title": null,
            "orient": "top"
          }
        },
        "tooltip": [
          {
            "field": "model",
            "type": "nominal",
            "title": "Model"
          },
          {
            "field": "implementation",
            "type": "nominal",
            "title": "Implementation"
          },
          {
            "field": "runtime",
            "type": "quantitative",
            "title": "Runtime (ms)",
            "format": ",.2f"
          }
        ]
      }
    },
    {
      "mark": {
        "type": "text",
        "fontWeight": "bold",
        "dy": -7
      },
      "encoding": {
        "x": {
          "field": "model",
          "type": "nominal",
          "sort": [
            "LS 2004",
            "Ireland 2004",
            "GM 2005",
            "Gali Ch. 3",
            "SW 2007"
          ]
        },
        "xOffset": {
          "field": "implementation",
          "sort": [
            "SymbolicDSGE",
            "Dynare MATLAB",
            "Dynare Octave"
          ]
        },
        "y": {
          "field": "runtime",
          "type": "quantitative",
          "scale": {
            "type": "log",
            "domain": [
              5,
              10000
            ]
          }
        },
        "text": {
          "field": "label",
          "type": "nominal"
        }
      }
    }
  ],
  "config": {
    "view": {
      "fill": null,
      "stroke": null
    },
    "axis": {"labelFontSize": 13, "titleFontSize": 14},
    "axisX": {"labelFontWeight": "bold", "labelFontSize": 14},
    "axisY": {"titleFontWeight": "bold"},
    "legend": {"labelFontSize": 14},
    "text": {"fontSize": 12}
  }
}
```

??? note "Data (mode_compute=8, MAP, MATLAB)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare MATLAB ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |            6.94 |        200.48 (28.89x) |  58/169  |  31/80  |        8.415e-10 |    4.013e-05    |
    | Ireland (2004)           |  11 / 13   |           15.97 |        172.77 (10.82x) |  42/183  |  22/39  |        3.476e-06 |    7.167e-05    |
    | Gali-Monacelli (2005)    |  15 / 15   |           41.39 |        469.03 (11.33x) |  63/178  |  32/84  |        4.748e-09 |    2.790e-05    |
    | Gali (2015), Ch. 3       |  20 / 22   |           60.19 |         200.20 (3.33x) |  46/147  |  23/70  |        4.303e-07 |    0.000e+00    |
    | Smets-Wouters (2007)     |  37 / 40   |          402.29 |         855.26 (2.13x) |  47/152  |  24/72  |        3.562e-07 |    5.099e-05    |

??? note "Data (mode_compute=8, MAP, Octave)"
    | Model                    | States S/D | SymbolicDSGE ms | Dynare Octave ms (D/S) | nfev S/D | nit S/D | max \|Δ loglik\| | max \|Δ theta\| |
    |--------------------------|:----------:|----------------:|-----------------------:|:--------:|:-------:|-----------------:|:---------------:|
    | Lubik-Schorfheide (2004) |   5 / 8    |            6.94 |     1,576.83 (227.25x) |  58/169  |  31/80  |        8.415e-10 |    4.013e-05    |
    | Ireland (2004)           |  11 / 13   |           15.97 |      1,278.40 (80.06x) |  42/183  |  22/39  |        3.476e-06 |    7.167e-05    |
    | Gali-Monacelli (2005)    |  15 / 15   |           41.39 |     4,306.12 (104.03x) |  63/178  |  32/84  |        4.748e-09 |    2.790e-05    |
    | Gali (2015), Ch. 3       |  20 / 22   |           60.19 |      1,554.34 (25.82x) |  46/147  |  23/70  |        4.303e-07 |    0.000e+00    |
    | Smets-Wouters (2007)     |  37 / 40   |          402.29 |      4,978.64 (12.38x) |  47/152  |  24/72  |        3.562e-07 |    5.099e-05    |

## Outlook

As mentioned in the intro, the Kalman filter is the main computational component in point estimation.
Therefore, the results expectedly show that SymbolicDSGE's performance advantage decays with increasing model size. (see the [BLAS discussion](linear_kf.md#outlook) for more details)
For this benchmark, however, Smets-Wouters (2007) is not large enough to demonstrate the cross-over point where SymbolicDSGE's performance advantage is lost.

The benchmark also demonstrates that `mode_compute=7` and SymbolicDSGE's `Nelder-Mead` are similar enough to share `nfev` and `nit` counts, alongside bit-exact agreement in the log-likelihood and parameter estimates in multiple models.
With demonstrated parity, SymbolicDSGE becomes a favorable alternative especially for Octave users, where the performance advantage is much more pronounced.

On `mode_compute=8`, it is important to notice that we exercised the Dynare defaults and did not attempt to tune any optimization parameters.
With defaults, `7` is rougly 1.32-2.2x faster in wall-clock time than `8` (including MATLAB and Octave runs).
`8` appears to restart or explore a larger parameter space with consistently higher `nfev` and `nit` counts.
This may be the preferred behavior for models with challenging likelihood surfaces.
While `mode_compute=7` worked better in out-of-the-box comparison for our models, this result should not discourage users from interacting with `mode_compute=8` and it's various options, especially when Optimization Toolbox in not available.

???+ note "On Gali-Monacelli (2005)"
    The Gali-Monacelli (2005) model is the only model in the benchmark with apparent flat likelihood regions.
    MLE cases for this model consistently display likelihood deltas in the order of 1e-14 - 1e-10, while the thetas producing these likelihoods have deltas in the order of 1e-1 - 1e-2.
    Moreover, the inclusion of a prior supplies the optimizer with enough information to unify the algorithms' convergence point.
    Collectively, these observations show that the likelihood surface is flat and the MAP case convergence is largely attributable to the prior information.
    Therefore, the increased runtime, especially visible in the Octave implementation, does not indicate an issue with Dynare's optimization algorithm or an issue with the timing logic.
    The model performing poorly should not be treated as an outlier, it demonstrates a challenging optimization problem that is not uncommon in DSGE estimation.
