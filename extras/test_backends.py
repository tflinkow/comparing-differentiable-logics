from __future__ import print_function

import sympy
import numpy
import torch

from collections import namedtuple

import sys

sys.path.append(".")
sys.path.append("..")

from base.backends import SympyBackend, NumpyBackend, TorchBackend
from base.dl2 import DL2
from base.fuzzy_logics import *

def main():
    Config = namedtuple('Config', 'backend x y')

    configs: list[Config] = [
        Config(SympyBackend(), sympy.Symbol('x'), sympy.Symbol('y')),
        Config(NumpyBackend(), .25, .75),
        Config(TorchBackend(), torch.tensor(.25), torch.tensor(.75)),
    ]

    for config in configs:
        logics: list[Logic] = [
            DL2(config.backend),
            GoedelFuzzyLogic(config.backend),
            KleeneDienesFuzzyLogic(config.backend),
            LukasiewiczFuzzyLogic(config.backend),
            ReichenbachFuzzyLogic(config.backend),
            GoguenFuzzyLogic(config.backend),
            ReichenbachSigmoidalFuzzyLogic(config.backend),
            YagerFuzzyLogic(config.backend),
        ]

        for logic in logics:
            print(f"AND_{logic.display_name} is {logic.AND(config.x, config.y)}")
            print(f"OR_{logic.display_name} is {logic.OR(config.x, config.y)}")

            if not isinstance(logic, DL2):
                print(f"IMPL_{logic.display_name} is {logic.IMPL(config.x, config.y)}")

if __name__ == '__main__':
    main()