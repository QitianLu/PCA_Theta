# PCA-based Adaptive Sampling for Multi-output PCE

This repository contains the code and numerical examples accompanying
*Correlation-aware sequential sampling for multi-output polynomial chaos expansions*.
The implementation extends the multi-output Theta sampling criterion using principal
component analysis (PCA) and adaptive weighting. It is written in Python and uses
the open-source [UQpy](https://uqpyproject.readthedocs.io/) library.

## Contents

Each example folder contains its own copy of `PCA_Theta.py`, together with
Jupyter notebooks, reference data, and figures:

- `Gaussian/`: localized Gaussian responses.
- `Wave/`: correlated wave responses.
- `RandomField_2D/`: random fields with prescribed trigonometric modes.
- `RandomField_5D/`: random fields represented by a Karhunen–Loève expansion.

## Requirements and usage

Tested with Python 3.11.7. From the repository directory, install the dependencies
and start Jupyter:

```sh
python -m pip install -r requirements.txt
jupyter notebook
```

Open an example notebook, such as `Gaussian/Gaussian.ipynb`, and run the cells
in order with its example folder as the working directory.
The default `RUN_MODE='reference'` loads saved results; use `'smoke'`
for a short execution check or `'full'` to run the paper settings.
Each example includes a `Plotting.ipynb` notebook for reproducing the figures.

## Citation

If you use this software in your research, please cite the following paper:

Lu, Q., Vořechovský, M., & Novák, L. (2026).
*Correlation-aware sequential sampling for multi-output polynomial chaos expansions.*
Unpublished manuscript.

Citation metadata are provided in [CITATION.cff](CITATION.cff) and will be updated
when the paper is published.

## License

This project is distributed under the [MIT License](LICENSE).
UQpy retains its original [MIT license and copyright notice](LICENSE_UQpy.txt).
