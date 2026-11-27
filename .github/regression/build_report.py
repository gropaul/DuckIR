"""Writes the timed runs of test_runner.py --verbose logs to a CSV and inlines that CSV into report_template.html.
Prints the counts of regressions and improvements, by the LIMIT of the template, for the report's commit status.

usage: python3 build_report.py <samples.csv> <report.html> <pull request URL> <runner-THREADS.log>...
"""
import os
import re
import sys
from statistics import median

samples_path, report_path, pr_url, *log_paths = sys.argv[1:]
GROUP = re.compile(r'::group::raw output: (Base|PR) (\S+)')
rows = ['benchmark,threads,side,run,seconds']
for log_path in log_paths:
    threads = re.search(r'runner-(\d+)\.log$', log_path)[1]
    side = None
    for line in open(log_path):
        if match := GROUP.search(line):
            side, benchmark = match[1].lower(), match[2]
        elif line.startswith('::endgroup::'):
            side = None
        elif side and line.startswith(benchmark + '\t'):
            _, run, seconds = line.split()
            rows.append(f'{benchmark},{threads},{side},{run},{seconds}')
samples = '\n'.join(rows) + '\n'
open(samples_path, 'w').write(samples)

template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_template.html')).read()
open(report_path, 'w').write(template
                             .replace('/*__SAMPLES_CSV__*/', samples.replace('</', '<\\/'))
                             .replace('/*__PR_URL__*/', pr_url)
                             .replace('/*__PR_NUMBER__*/', pr_url.rstrip('/').rsplit('/', 1)[-1]))

runs = {}
for row in rows[1:]:
    benchmark, threads, side, _, seconds = row.split(',')
    runs.setdefault((benchmark, threads), {'base': [], 'pr': []})[side].append(float(seconds))
changes = [median(r['pr']) / median(r['base']) - 1 for r in runs.values() if r['base'] and r['pr']]
print(f'{sum(c >= 0.1 for c in changes)} regressions and {sum(c <= -0.1 for c in changes)} improvements of 10% or more')
