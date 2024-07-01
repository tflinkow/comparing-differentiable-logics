from __future__ import print_function

import codecs
import os

import pandas
import numpy

from pathlib import Path

from collections import namedtuple

Result = namedtuple('Result', 'p_acc c_acc overall')

def format_acc_value(v) -> str:
    return 'nan' if v == -1 else f'{v * 100:.2f}'

def get_name_from_file(report: str) -> str:
    name = Path(report).stem

    if name == 'Goedel':
        name = 'Gödel'
    elif name == 'Lukasiewicz':
        name = r'\L ukasiewicz'
    elif name == 'KleeneDienes':
        name = 'Kleene-Dienes'
    elif name == 'ReichenbachSigmoidal':
        name = 'sig. Reichenbach'

    if Path(report).stem == 'Goedel' and 'robustness' in report:
        name = 'Fuzzy Logic'

    return name

def get_legendentry_from_file(report: str) -> str:
    return fr'\addlegendentry {{{get_name_from_file(report)}}};'

def write_plot_file(report_dir: str, target_file: str):
    full_path = lambda r : f'{report_dir}/{r}'

    with codecs.open(target_file, 'w', 'utf-8') as file:
        begin = r"""
\documentclass[tikz]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

\usepackage{amsmath}

\input{tikz_settings}

\begin{document}
\begin{tikzpicture}[font=\footnotesize]
  \begin{groupplot}[group/results]
    \nextgroupplot[title={Prediction (P)},]
"""

        file.write(begin)

        report_files = sorted([f for f in os.listdir(report_dir) if f.endswith('.csv')])

        def write(report, p_acc: bool):
            df = pandas.read_csv(os.path.join(report_dir, report), comment='#')

            p = df['Test-P-Acc'].values[-(len(df) // 10):]
            c = df['Test-C-Acc'].values[-(len(df) // 10):]
            i = numpy.argmax(p * c)

            best_epoch = df['Epoch'].values[-(len(df) // 10):][i] + 1

            if 'Baseline' in report:
                file.write(fr'\addplot+[mark indices={best_epoch}, densely dotted] table [y={"Test-P-Acc" if p_acc else "Test-C-Acc"}] {{{full_path(report)}}};' + '\n')
            else:
                file.write(fr'\addplot+[mark indices={best_epoch}] table [y={"Test-P-Acc" if p_acc else "Test-C-Acc"}] {{{full_path(report)}}};' + '\n')

        for report in report_files:
            write(report, True)

        intermediate = r"""
\coordinate (c1) at (rel axis cs:0,1);
    \nextgroupplot[title={Constraint (C)},
      yticklabel pos=right,
      legend to name=full-legend
    ]
"""
        file.write(intermediate)

        for report in report_files:
            write(report, False)

        for report in report_files:
            file.write(get_legendentry_from_file(full_path(report)) + '\n')

        end = r"""
\coordinate (c2) at (rel axis cs:1,1);
  \end{groupplot}
  \coordinate (c3) at ($(c1)!.5!(c2)$);
  \node[below] at (c3 |- current bounding box.south) {\pgfplotslegendfromname{full-legend}};
\end{tikzpicture}
\end{document}
"""

        file.write(end)

def write_table_file(report_dir: str, target_file: str):
    full_path = lambda r : f'{report_dir}/{r}'
    results: dict[str, Result] = {}

    report_files = sorted([f for f in os.listdir(report_dir) if f.endswith('.csv')])

    if len(report_files) < 1:
        return

    for report in report_files:
        print(f'Reading {os.path.join(report_dir, report)}')
        df = pandas.read_csv(os.path.join(report_dir, report), comment='#')

        p = df['Test-P-Acc'].values[-(len(df) // 10):]
        c = df['Test-C-Acc'].values[-(len(df) // 10):]
        i = numpy.argmax(p * c)

        results[full_path(report)] = Result(p[i], c[i], p[i] * c[i])

        best = max(results, key=lambda k: results[k].overall)

        with codecs.open(target_file, 'w', 'utf-8') as file:
            begin = r"""
\documentclass{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

\usepackage{amsmath}
\usepackage{amssymb}

\usepackage{tabularray}
\UseTblrLibrary{booktabs}

\begin{document}
\footnotesize
\begin{tblr}
  {
    colspec={Q[l, mode=text]Q[c, mode=text]Q[c, mode=text]},
    row{1}={font=\bfseries, mode=text},
  }
    \toprule
      Logic & P & C \\
    \midrule
"""
            file.write(begin)

            for key, value in results.items():
                if key == best:
                    file.write(fr'{get_name_from_file(key)} & \textbf{{{format_acc_value(value.p_acc)}}} & \textbf{{{format_acc_value(value.c_acc)}}} \\' + '\n')
                else:
                    file.write(fr'{get_name_from_file(key)} & {format_acc_value(value.p_acc)} & {format_acc_value(value.c_acc)} \\' + '\n')

            end = r"""
    \bottomrule
  \end{tblr}
\end{document}
"""

            file.write(end)

def main():
    for folder_constraint in os.listdir('.'):
        if os.path.isdir(folder_constraint) and folder_constraint != '.git':
            for folder_dataset in os.listdir(folder_constraint):
                if os.path.isdir(os.path.join(folder_constraint, folder_dataset)) and folder_dataset != '.git':
                    report_dir = f'{folder_constraint}/{folder_dataset}'
                    
                    plot_file = f'plot_{folder_constraint}_{folder_dataset}.tex'
                    write_plot_file(report_dir, plot_file)

                    table_file = f'table_{folder_constraint}_{folder_dataset}.tex'
                    write_table_file(report_dir, table_file)

if __name__ == '__main__':
    main()