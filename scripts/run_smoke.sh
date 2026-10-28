#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p outputs
.venv/bin/python -m pip --isolated check
.venv/bin/python scripts/prepare_data.py > outputs/data_preparation.log 2>&1
.venv/bin/python scripts/smoke_test.py 2>&1 | tee outputs/training.log
.venv/bin/python scripts/simulation_smoke.py 2>&1 | tee outputs/simulation.log
.venv/bin/python scripts/write_report.py
echo 'All smoke checks passed. See outputs/实验一测试报告.md.'
