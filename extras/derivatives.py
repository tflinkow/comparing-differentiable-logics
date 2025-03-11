from __future__ import print_function

from collections import namedtuple

import sympy

import codecs
import os
import sys

sys.path.append(".")
sys.path.append("..")

from base.backends import SympyBackend
from base.dl2 import DL2
from base.fuzzy_logics import *

def main():
    backend = SympyBackend()

    conjunctions_disjunctions: list[Logic] = [
        DL2(backend),
        GoedelFuzzyLogic(backend),
        LukasiewiczFuzzyLogic(backend),
        ReichenbachFuzzyLogic(backend),
        YagerFuzzyLogic(backend),
    ]

    implications: list[Logic] = [
        DL2(backend),
        GoedelFuzzyLogic(backend),
        KleeneDienesFuzzyLogic(backend),
        LukasiewiczFuzzyLogic(backend),
        ReichenbachFuzzyLogic(backend),
        ReichenbachSigmoidalFuzzyLogic(backend),
        GoguenFuzzyLogic(backend),
        YagerFuzzyLogic(backend),
    ]

    diff = lambda f, var: sympy.diff(f, var).rewrite(sympy.Piecewise).doit()

    AND = lambda logic, x, y: logic.AND(x, y)
    OR = lambda logic, x, y: logic.OR(x, y)
    IMPL = lambda logic, x, y: logic.IMPL(x, y)

    Config = namedtuple('Config', 'file_name diff_func operator logics')

    configs: list[Config] = [
        Config('derivatives-conjunction.tex', diff, AND, conjunctions_disjunctions),
        Config('derivatives-disjunction.tex', diff, OR, conjunctions_disjunctions),
        Config('derivatives-implication.tex', diff, IMPL, implications),
    ]

    x, y = sympy.symbols('x y')

    for config in configs:
        try:
            os.remove(config.file_name)
        except OSError:
            pass

        with codecs.open(config.file_name, 'w', 'utf-8') as file:
            for item in config.logics:
                f = config.operator(item, x, y)

                dx = config.diff_func(f, x)
                dy = config.diff_func(f, y)

                file.write(fr'{item.display_name} & {sympy.latex(f)} & {sympy.latex(sympy.simplify(dx))} & {sympy.latex(sympy.simplify(dy))} \\' + '\n')

if __name__ == '__main__':
    main()