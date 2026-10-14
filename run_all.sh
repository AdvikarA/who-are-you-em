#!/usr/bin/env bash
# Full pipeline, in order. Every step is resumable, so re-running skips finished work.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

python src/00_inspect.py --download
python src/01_generate.py
python src/02_judge.py
python src/04_truthfulqa.py
python src/03_score.py
python src/05_plots.py
python src/06_labels.py export
echo "Done. Plots in plots/, numbers in results/stats.json, labels to fill in labels/to_label.csv"
