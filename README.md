# Correlation-aware sequential sampling for multi-output PCE

Code and numerical examples for the PCA-based multi-output Theta criterion,
including adaptive weighting by the variance-concentration exponent eta.
Authors: Qitian Lu, Miroslav Vořechovský and Lukáš Novák.

## Files

```
PCA_Theta.py                 Shared algorithm: Theta, PCA, eta, OLS and LOO
Gaussian/
    Gaussian.ipynb          Model, settings and experiments
    Plotting.ipynb          Error, degree, density, field and PCA/time figures
    Data/                   Saved paper statistics and final models
    Figures/                Paper PDFs
Wave/
    Wave.ipynb              Common and individual degree experiments
    Plotting.ipynb
    Data/  Figures/
RandomField_2D/
    RandomField_2D.ipynb     Fifty paired random-field realizations
    Plotting.ipynb
    Data/  Figures/
RandomField_5D/
    RandomField_5D.ipynb     Fixed p=10; 3010 to 8000 samples; batch size 10
    Plotting.ipynb
    Data/  Figures/
```

There is only one algorithm `.py` file. Models, editable settings and execution
are in the notebooks. Plotting code is also in notebooks. Every example keeps
its own data and figures. Development tests, logs and publication preparation
notes are kept outside this repository.

## Installation and use

Tested with Python 3.11.7. Install into a Python 3.11 environment:

```sh
git clone https://github.com/QitianLu/PCA_Theta.git
cd PCA_Theta
python -m pip install -r requirements.txt
jupyter notebook
```

Open an example notebook. `RUN_MODE='reference'` is the default and inspects
the saved results. Set it to `'smoke'` for a short execution check, or `'full'`
to run the paper settings. Then run the cells in order. New results go to
`Results/`, leaving supplied `Data/` unchanged. `Plotting.ipynb` regenerates
the paper figures from `Data/` and shows them inline. Restart the kernel when
switching examples. Only load trusted notebooks; NPZ results use no pickle.

The full 5D run creates an approximately 4.8 GB basis cache and can take hours.
Its saved checkpoints can be resumed. The smoke version uses p=2 and 64 to 74
samples only for checking execution, not as a paper result.

## Shared algorithm

`ThetaCriterionPCE(pces, adaptive_weighting=False)` gives conventional Theta.
Pass `decorrelate=True` to its `run()` method for PCA-Theta. Setting
`adaptive_weighting=True` enables eta for either version. The cached-matrix
5D implementation is in the same file as `select_matrix_batch`.

`svd_loo` computes relative squared LOO using the thin SVD's left singular
vectors. Candidate models require N>P, numerical full column rank and valid
LOO denominators. Common degree minimizes the largest output-wise LOO;
individual degree minimizes it separately per output. No 1.25 ratio, ridge
regularization, trimming or explicit leave-one-out fallback is used.
The Gaussian maximum-degree comparison instead takes the largest p with N>P
for every method. Model predictions are always fitted in the original outputs.

## Reproducibility notes

- The 2D comparisons use six methods and 50 repetitions, N=30 to 1000, batches
  of 10, p=5..25, a 100 x 100 candidate grid, 1000 test inputs and PCA threshold
  0.01. Seeds and all model parameters appear in the notebooks.
- Each repetition shares the initial ED and field realization across methods.
  LHS uses a fresh complete Latin hypercube at each N>30 and is not nested.
  Its N denotes one design size, not cumulative evaluations of all budgets.
- PCA uses PCE predictions at the current ED. The retained fractions refer
  to this empirical measure. Coefficients are zero-padded to a union basis;
  no extra common-basis refit is introduced.
- Weighted variants compute eta=(sum V)^2/(Nc sum V^2) over unused candidates.
  A zero computed sum of squares triggers distance-only selection. The
  normalization floor 1e-15 is not a threshold on eta's denominator.
- Test errors are normalized by output RMS. NRMSE and NMAE average over
  outputs; NMaxAE is the largest normalized error over outputs and finite test
  points. Plots show mean log10 errors and one sample standard deviation.
- The four adaptive timing comparisons use one BLAS thread per job. Caching
  applies to every method. Test diagnostics are excluded from construction
  CPU time. New timings depend on hardware and software; supplied timings are
  preserved measurements, not timings obtained during package validation.
- The 5D data contain only the final 8000-point experiment, not older
  5000-point results. PCA plotting values are from N=7990, before the final fit.
- Times New Roman is used when installed, with DejaVu Serif as fallback.

## License and citation

MIT; see `LICENSE`. UQpy is installed separately and retains its MIT license
and Copyright (c) 2018 Michael D. Shields (`LICENSE_UQpy.txt`). Other dependencies
retain their own licenses; no dependency binaries or fonts are redistributed.

If you use this software in your research, please cite the following paper:

Lu, Q., Vořechovský, M., & Novák, L. (2026).
*Correlation-aware sequential sampling for multi-output polynomial chaos expansions.*
Unpublished manuscript.

`CITATION.cff` provides machine-readable citation metadata and identifies this
manuscript as the preferred citation. The citation will be updated with the
publication details and DOI when available. This prepared software release has
no assigned Zenodo DOI yet; `.zenodo.json` contains draft upload metadata.
