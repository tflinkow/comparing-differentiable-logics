from __future__ import print_function

from scipy.integrate import nquad
import inspect
import time

import numpy

import codecs
import sys

sys.path.append(".")
sys.path.append("..")

from base.backends import NumpyBackend
from base.fuzzy_logics import *

from extras.tautologies import *

def main():
    backend = NumpyBackend()

    logics: list[Logic] = [
        GoedelFuzzyLogic(backend),
        KleeneDienesFuzzyLogic(backend),
        LukasiewiczFuzzyLogic(backend),
        ReichenbachFuzzyLogic(backend),
        GoguenFuzzyLogic(backend),
        ReichenbachSigmoidalFuzzyLogic(backend),
        YagerFuzzyLogic(backend),
    ]

    results: dict[Logic, list] = { logic: [] for logic in logics }

    with codecs.open('table-consistency.tex', 'w', 'utf-8') as file:
        begin = r"""
\documentclass{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

\usepackage{amsmath}
\usepackage{amssymb}

\usepackage{graphicx}

\usepackage{tabularray}
\UseTblrLibrary{booktabs}

\usepackage{xparse}
\NewDocumentCommand{\rot}{O{45} O{2em} m}{\makebox[#2][l]{\rotatebox{#1}{#3}}}%

\begin{document}
\footnotesize
""" + fr"""
\begin{{tblr}}
{{
    colspec={'l' + 'c' * len(logics)},
    cells={{mode=dmath}},
    row{{1}}={{font=\bfseries, mode=text}},
    row{{Z}}={{font=\bfseries, mode=text}},
}}
"""
        file.write(begin)

        file.write(r'\toprule' + '\n')
        file.write(r'\textbf{Tautology} & ' + ' & '.join([fr'\rot{{{l.display_name}}}' for l in logics]) + r'\\' + '\n')
        file.write(r'\midrule' + '\n')

        for i, (group_name, formulas) in enumerate(groups.items()):
            if i > 0:
                file.write(r'\midrule' + '\n')

            file.write(fr'{{\footnotesize\text{{\textbf{{{group_name}}}}}}}' + ' & ' * len(logics) + r'\\' + '\n')

            print(group_name)

            for f in formulas:
                arity = len(inspect.signature(f.value).parameters) - 1

                file.write(f'{f.description} & ')

                print(f.description)

                for j, logic in enumerate(logics):
                    start = time.time()

                    if arity == 3:
                        func = lambda R, Q, P: f.value(logic, P, Q, R)
                    elif arity == 2:
                        func = lambda Q, P: f.value(logic, P, Q)
                    elif arity == 1:
                        func = lambda P: f.value(logic, P)

                    v, e = nquad(func, [(0, 1)] * arity, opts = { 'limit': 200, 'epsrel': .009 })

                    end = time.time()
                    elapsed = f'{(end - start):.2f}'

                    # round here for fair average computation - everything after 2 decimal places has an error
                    v = round(v, 2)

                    print(f'[{logic.display_name}] = {("1" if v == 1 else f"{v:.2f}")} (err: {e}) [{elapsed} s]')
                    file.write(f'{v:.2f}' if v != 1 else '1')

                    results[logic].append(v)
                        
                    if j < len(logics) - 1:
                        file.write(' & ')

                file.write(r'\\' + '\n')

        file.write(r'\bottomrule' + '\n')
        file.write(r'Average Consistency & ' + ' & '.join([f'{numpy.average(val):.2f}' for val in results.values()]) + r'\\' + '\n')
        file.write(r'\end{tblr}')
        file.write(r'\end{document}')

if __name__ == '__main__':
    main()