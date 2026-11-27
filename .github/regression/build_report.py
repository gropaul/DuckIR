"""Inlines a samples CSV into report_template.html, so the report is one self-contained file.

usage: python3 build_report.py <samples.csv> <report.html> <pull request URL>
"""
import os
import sys

samples_path, report_path, pr_url = sys.argv[1:4]
template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_template.html')
with open(template_path) as template, open(samples_path) as samples:
    html = (template.read()
            .replace('/*__SAMPLES_CSV__*/', samples.read().replace('</', '<\\/'))
            .replace('/*__PR_URL__*/', pr_url)
            .replace('/*__PR_NUMBER__*/', pr_url.rstrip('/').rsplit('/', 1)[-1]))
with open(report_path, 'w') as report:
    report.write(html)
