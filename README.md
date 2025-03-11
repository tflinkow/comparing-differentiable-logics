# Comparing Differentiable Logics for Learning with Logical Constraints

Code for the *Science of Computer Programming* paper [Comparing Differentiable Logics for Learning with Logical Constraints](https://www.sciencedirect.com/science/article/pii/S016764232500019X).

```tex
@article{flinkowComparingDifferentiableLogics2025a,
  title   = {Comparing {{Differentiable Logics}} for {{Learning}} with {{Logical Constraints}}},
  author  = {Flinkow, Thomas and Pearlmutter, Barak A. and Monahan, Rosemary},
  year    = {2025},
  month   = mar,
  journal = {Science of Computer Programming},
  issn    = {0167-6423},
  doi     = {10.1016/j.scico.2025.103280},
}
```

## Repository Structure

```
.
├── base
│   ├── backends.py          - logical operators in Numpy, SymPy, and PyTorch
│   ├── dl2.py               - implementation of DL2
│   └── fuzzy_logics.py      - implementations of various fuzzy logics
├── extras
│   ├── consistency.py       - investigates consistency of fuzzy logics
│   └── derivatives.py       - investigates derivatives of differentiable logic operators
├── results
│   └── Makefile             - creates plots and tables from the experimental results
├── training
│   ├── attacks.py           - PGD and AutoPGD
│   ├── constraints.py       - definitions of constraints used in the experiments
│   ├── grad_norm.py         - GradNorm adaptive loss-balancing
│   ├── main.py              - entry point
│   ├── models.py            - neural networks used in the experiments
│   └── run.sh               - script to run the training experiment
├── verification
│   ├── property.vcl         - verification property
│   └── run.sh               - script to run the verification experiment
├── README.md                - this file
└── requirements.txt         - pip requirements
```

## Reproducing the results from the paper

The experiments were run on `Python 3.11.11`.
The provided `requirements.txt` can be used to install the required packages with `pip install -r requirements.txt`.

### Training experiment
The training experiments from the paper can be run using `cd training && sh run.sh`.

Assuming you have a reasonably up-to-date LaTeX distribution installed, the plots and tables from the publication can then be generated with `cd reports && make`.

### Verification experiment
The verification experiment from the paper can be run using `cd verification && bash run.sh`.

Note that this requires the `.onnx` network files, which will by default be in `results/strong-classification-robustness/mnist/` after running the training experiment.

Also note that for improved solver performance, we use [Marabou built from source with Gurobi support](https://github.com/NeuralNetworkVerification/Marabou?tab=readme-ov-file#compile-marabou-with-the-gurobi-optimizer-optional).

A standard installation of Marabou can be installed with `pip install maraboupy==2.0.0` and can be used for the verification experiment without any changes to the verification files, however, verification will take longer and will most likely require increased timeouts.