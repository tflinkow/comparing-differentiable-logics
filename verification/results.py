import os
import argparse
import re

import pandas

from datetime import timedelta

def parse_time(time_str: str):
    match = re.search(r"(?:(\d+):)?(\d+):([\d.]+)", time_str)

    assert match

    hours = int(match.group(1) if match.group(1) else 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int)
    args = parser.parse_args()

    n = args.size

    times = []

    timeouts = 0

    for filename in sorted(os.listdir('.')):
        if filename.endswith('.csv'):
            df = pandas.read_csv(filename, comment='#')

            p_acc = df['Test-P-Acc'].values[-1]
            c_acc = df['Test-C-Acc'].values[-1]
            c_sec = df['Test-C-Sec'].values[-1]

            print(f'{os.path.splitext(filename)[0]} P-Acc={p_acc} C-Acc={c_acc} C-Sec={c_sec}')

    for filename in sorted(os.listdir('logs')):
        if filename.endswith('.txt'):
            with open(os.path.join('logs', filename), 'r', encoding='utf-8') as file:
                content = file.read()

            unsat = content.count('proved no counterexample exists')
            sat = content.count('found a counterexample')
            timeout = content.count('timed out')

            time = parse_time(content)

            if unsat + sat + timeout != n:
                print(f'in {filename}, something does not add up: {unsat} + {sat} + {timeout} != {n}')

            times.append(time)
            timeouts += timeout

            print(f'{filename}: UNSAT={(100 * unsat/(n - timeout)):.2f}% ({unsat}/{n - timeout}) [SAT={sat} T/O={timeout}]')

    total_time = sum(times, timedelta(0))
    hours, remainder = divmod(total_time.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f'Total time: {total_time.days} days {hours} hours {minutes} minutes')
    print(f'Total number of timeouts: {timeouts}')

if __name__ == '__main__':
    main()