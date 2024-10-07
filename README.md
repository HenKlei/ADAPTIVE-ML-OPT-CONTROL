[![DOI](https://zenodo.org/badge/758423893.svg)](https://zenodo.org/doi/10.5281/zenodo.10669854)

```
# ~~~
# This file is part of the paper:
#
#           " Application of an adaptive model hierarchy to
#                 parametrized optimal control problems "
#
#   https://github.com/HenKlei/ADAPTIVE-ML-OPT-CONTROL.git
#
# Copyright 2024 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Hendrik Kleikamp
# ~~~
```

# Optimal control of parametrized linear systems using machine learning
In this repository, we provide the code used for the numerical experiments in the paper "Application of an adaptive
model hierarchy to parametrized optimal control problems" by Hendrik Kleikamp.

You find the paper [here](http://www.iam.fmph.uniba.sk/amuc/ojs/index.php/algoritmy/article/view/2145) (the preprint is available [here](https://arxiv.org/abs/2402.10708)).

## Installation
On a system with `git` (`sudo apt install git`), `python3` (`sudo apt install python3-dev`) and
`venv` (`sudo apt install python3-venv`) installed, the following commands should be sufficient
to install the `adaptive-ml-control` package with all required dependencies in a new virtual environment:
```
git clone https://github.com/HenKlei/adaptive-ml-control.git
cd adaptive-ml-control
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Running the experiments
To reproduce the results, we provide the original script creating the results presented in
the paper in the directory [`adaptive_ml_control/examples/`](adaptive_ml_control/examples/).

## Questions
If you have any questions, feel free to contact me via email at <hendrik.kleikamp@uni-muenster.de>.
