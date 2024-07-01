# Comparing Differentiable Logics for Learning with Logical Constraints

An experimental comparison of differentiable logics for machine learning with logical constraints.

## Repository Structure

```
.
├── README.md                    - this file
├── base
│   ├── backends.py              - implementations of logic operators in Numpy, SymPy, and PyTorch
│   ├── dl2.py                   - implementation of DL2
│   ├── fuzzy_logics.py          - implementations of various fuzzy logics
├── experiments
│   ├── constraints.py           - the constraints used in the experiments
│   ├── main.py                  - the entry point containing training and test loop
│   ├── models.py                - the neural networks used in the experiments
│   ├── run.sh                   - script to run the experiments and replicate results from the paper
│   └── util.py                  - implementations of PGD and GradNorm
├── extras
│   ├── Makefile                 - creates the consistency and derivatives tables
│   ├── consistency.py           - evaluates the consistency of fuzzy logics
│   ├── derivatives.py           - creates LaTeX tables of derivatives of differentiable logic operators
│   ├── tautologies.py           - the tautologies used in the consistency calculation
├── reports                      - folder containing experimental results (.csv files)
│   ├── Makefile                 - creates plots and tables from the experimental results
└── requirements.txt             - pip requirements
```

## Reproducing the results from the paper

The results from the paper can be replicated by running `cd experiments && sh run.sh`.

Assuming you have a reasonably up-to-date LaTeX distribution installed, the plots and tables from the publication can then be generated with `cd reports && make`.

## Requirements

The experiments were run on `Python 3.12.3` (but should probably work with newer versions as well).
The provided `requirements.txt` can be used to install the required packages using `pip install -r requirements.txt`.