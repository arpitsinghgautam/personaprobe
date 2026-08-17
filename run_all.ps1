# Reproduce every experiment in the report, in order.
#
#   .\run_all.ps1
#
# Stages are separated so a failure in one does not abort the rest, an
# overnight run that dies at stage 3 should still produce stages 4-9.
# Each python invocation is its own process, so GPU memory is released
# between models; two 7B checkpoints do not fit in 24GB simultaneously.

$py = ".venv\Scripts\python.exe"
$QI = "Qwen/Qwen2.5-7B-Instruct"
$QB = "Qwen/Qwen2.5-7B"
$MI = "mistralai/Mistral-7B-Instruct-v0.3"

New-Item -ItemType Directory -Force logs, figures, results | Out-Null

function Stage($n, $label) { "`n=== [$n] $label ===`n" }

Stage "1/9" "Qwen2.5-7B-Instruct - three framings"
& $py scripts\01_elicit.py --model $QI --templates prefer better choose --force --batch-size 16

Stage "2/9" "Qwen2.5-7B base - post-training comparison"
& $py scripts\01_elicit.py --model $QB --base --templates prefer --force --batch-size 16

Stage "3/9" "Mistral-7B-Instruct-v0.3 - second model family"
& $py scripts\01_elicit.py --model $MI --templates prefer better --force --batch-size 16

# NOTE: tags must not begin with "-". argparse treats a value starting with a
# dash as a new option flag, and "--tag -sd" fails with "expected one argument".
Stage "4/9" "Ablation v1 - self-description context, mid layers"
& $py scripts\03_ablate.py --model $QI --extract-context selfdesc --band 0.25 --tag _sd --batch-size 16

Stage "5/9" "Ablation v2 - preference context, full depth"
& $py scripts\03_ablate.py --model $QI --extract-context preference --band 1.0 --tag _ctx --batch-size 16

Stage "6/9" "Coherence and category analysis"
foreach ($t in @("prefer", "better", "choose")) { & $py scripts\02_analyze.py --model $QI --template $t }
& $py scripts\02_analyze.py --model $QB --template prefer
foreach ($t in @("prefer", "better")) { & $py scripts\02_analyze.py --model $MI --template $t }

Stage "7/9" "Bootstrap confidence intervals"
foreach ($t in @("prefer", "better", "choose")) { & $py scripts\04_errorbars.py --model $QI --template $t --n-boot 300 }
& $py scripts\04_errorbars.py --model $QB --template prefer --n-boot 300
foreach ($t in @("prefer", "better")) { & $py scripts\04_errorbars.py --model $MI --template $t --n-boot 300 }

Stage "8/9" "Spacing confound and separation-matched concordance"
foreach ($t in @("prefer", "better", "choose")) {
  & $py scripts\05_spread.py  --model $QI --template $t
  & $py scripts\06_matched.py --model $QI --template $t
}
& $py scripts\05_spread.py  --model $QB --template prefer
& $py scripts\06_matched.py --model $QB --template prefer
foreach ($t in @("prefer", "better")) {
  & $py scripts\05_spread.py  --model $MI --template $t
  & $py scripts\06_matched.py --model $MI --template $t
}

Stage "9/10" "Figures"
& $py scripts\07_figures.py --model $QI

Stage "10/10" "Report tables"
& $py scripts\10_tables.py

"`n=== run_all complete ===`n"
