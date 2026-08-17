# Scale sweep, Qwen2.5 at 0.5B, 1.5B, 3B (7B already run by run_all.ps1).
#
#   .\run_sweep.ps1
#
# Same family, same training recipe, only parameter count varies, a controlled
# comparison that adding another model FAMILY cannot give.
#
# The analysis is frozen by report/preregistration.md, committed before this
# script was first run. Do not change gate thresholds, bootstrap counts, seeds,
# the outcome set or the persona conditions while the sweep is in flight.

$py = ".venv\Scripts\python.exe"
$models = @(
  "Qwen/Qwen2.5-0.5B-Instruct",
  "Qwen/Qwen2.5-1.5B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct"
)

function Stage($label) { "`n=== $label ===`n" }

foreach ($m in $models) {
  Stage "elicit  $m  (prefer, better)"
  & $py scripts\01_elicit.py --model $m --templates prefer better --force --batch-size 16

  Stage "project 2  $m  (ratings + predictions)"
  & $py scripts\08_stated.py --model $m --batch-size 16
}

foreach ($m in $models) {
  Stage "analyse  $m"
  foreach ($t in @("prefer", "better")) {
    & $py scripts\02_analyze.py   --model $m --template $t
    & $py scripts\04_errorbars.py --model $m --template $t --n-boot 300
    & $py scripts\05_spread.py    --model $m --template $t
    & $py scripts\06_matched.py   --model $m --template $t
  }
}

Stage "project 2 across all scales"
& $py scripts\09_selfknowledge.py --models `
    "Qwen/Qwen2.5-0.5B-Instruct" `
    "Qwen/Qwen2.5-1.5B-Instruct" `
    "Qwen/Qwen2.5-3B-Instruct" `
    "Qwen/Qwen2.5-7B-Instruct" `
    "mistralai/Mistral-7B-Instruct-v0.3"

"`n=== sweep complete ===`n"
