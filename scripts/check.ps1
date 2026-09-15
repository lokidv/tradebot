# یک فرمان برای «آیا هنوز سالم است؟» — تست‌های آفلاین + بررسیِ نحوِ همهٔ ماژول‌ها.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONIOENCODING = "utf-8"
python -m compileall -q bot tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m unittest discover -s tests
exit $LASTEXITCODE
