"""Appends the timed runs of one test_runner.py --verbose output to a CSV, one row per run.

usage: python3 parse_samples.py <samples.csv> <suite> <threads> <rows> <row_group_size> <test_runner output>
"""
import csv
import os
import re
import sys

samples_path, suite, threads, rows, row_group_size, output_path = sys.argv[1:7]
GROUP = re.compile(r'::group::raw output: (Base|PR) (\S+)')
new_file = not os.path.exists(samples_path)
with open(samples_path, 'a', newline='') as samples, open(output_path) as output:
    writer = csv.writer(samples)
    if new_file:
        writer.writerow(['suite', 'benchmark', 'threads', 'rows', 'row_group_size', 'side', 'invocation', 'run', 'seconds'])
    invocations = {}
    side = benchmark = None
    for line in output:
        line = line.rstrip('\n')
        if match := GROUP.search(line):
            side, benchmark = match.group(1).lower(), match.group(2)
            invocations[side, benchmark] = invocations.get((side, benchmark), 0) + 1
        elif line.startswith('::endgroup::'):
            side = benchmark = None
        elif side and line.startswith(benchmark + '\t'):
            _, run, seconds = line.split('\t')
            writer.writerow([suite, benchmark, threads, rows, row_group_size, side, invocations[side, benchmark], run, seconds])
