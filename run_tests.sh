#!/usr/bin/env bash
# Usage:  ./run_tests.sh            -> whole suite + coverage + junit report in test_reports/
#         ./run_tests.sh smoke      -> smoke only (seconds)
#         ./run_tests.sh sanjay     -> Sanjay's files only
#         ./run_tests.sh jai        -> Jai's files only
set -e
mkdir -p test_reports
case "$1" in
  smoke)  python -m pytest tests/test_smoke.py ;;
  sanjay) python -m pytest tests/test_sanjay_*.py --cov --cov-report=term --junitxml=test_reports/sanjay.xml ;;
  jai)    python -m pytest tests/test_jai_*.py    --cov --cov-report=term --junitxml=test_reports/jai.xml ;;
  *)      python -m pytest tests --cov --cov-report=term --cov-report=html:test_reports/coverage_html --junitxml=test_reports/all.xml ;;
esac
