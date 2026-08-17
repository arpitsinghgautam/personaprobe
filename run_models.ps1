# Breadth sweep, does the effect survive outside the Qwen family?
#
#   .\run_models.ps1
#
# The single strongest objection to project 1 is that only Qwen2.5-7B passes the
# validity gate, making it effectively an n=1 study. This runs four more
# checkpoints across three additional families, plus one control.
#
# Order matters. The quantisation control runs FIRST: Qwen2.5-7B at 4-bit, a
# checkpoint whose bf16 behaviour we already know. If the 4-bit version also
# passes the gate, quantisation is not breaking the instrument and the 14B
# result below is interpretable. If it fails, every 4-bit number here is
# confounded and must be reported as such rather than as a property of the model.
#
# Models are run one at a time; two do not fit in 24GB.

$py = ".venv\Scripts\python.exe"

function Stage($label) { "`n=== $label ===`n" }

# --- 1. quantisation control -------------------------------------------------
Stage "CONTROL  Qwen2.5-7B-Instruct @ 4-bit  (bf16 version already passes the gate)"
& $py scripts\01_elicit.py --model Qwen/Qwen2.5-7B-Instruct --quant 4bit `
      --templates prefer better --force --batch-size 16

# --- 2. other families, full precision --------------------------------------
$bf16 = @(
  "microsoft/Phi-3.5-mini-instruct",
  "allenai/OLMo-2-1124-7B-Instruct",
  "tiiuae/Falcon3-7B-Instruct"
)
foreach ($m in $bf16) {
  Stage "elicit  $m  (bf16)"
  & $py scripts\01_elicit.py --model $m --templates prefer better --force --batch-size 16
}

# --- 3. scale up, quantised --------------------------------------------------
# Unsloth's pre-quantised checkpoint: ~10GB to download instead of ~29GB for the
# bf16 weights, and the same nf4 scheme the control above uses.
Stage "elicit  Qwen2.5-14B-Instruct @ 4-bit (unsloth pre-quantised)"
& $py scripts\01_elicit.py --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit `
      --templates prefer better --force --batch-size 8

# --- 4. analysis -------------------------------------------------------------
$all = @(
  "Qwen/Qwen2.5-7B-Instruct-4bit",
  "microsoft/Phi-3.5-mini-instruct",
  "allenai/OLMo-2-1124-7B-Instruct",
  "tiiuae/Falcon3-7B-Instruct",
  "unsloth/Qwen2.5-14B-Instruct-bnb-4bit-prequantized"
)
foreach ($m in $all) {
  Stage "analyse  $m"
  foreach ($t in @("prefer", "better")) {
    & $py scripts\02_analyze.py   --model $m --template $t
    & $py scripts\04_errorbars.py --model $m --template $t --n-boot 300
    & $py scripts\06_matched.py   --model $m --template $t
  }
}

"`n=== run_models complete ===`n"
