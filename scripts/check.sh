#!/usr/bin/env bash
# یک فرمان برای «آیا هنوز سالم است؟» — تست‌های آفلاین + بررسیِ نحوِ همهٔ ماژول‌ها.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python -m compileall -q bot tests
python -m unittest discover -s tests
