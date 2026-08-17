"""Cross-model summary: which models can be measured, and what they show.

Reads every errorbars__*.json and reports, per model and framing, whether the
instrument worked and what the pooled self-vs-human gap was. This is the table
that answers "you only tested one model".

Gated estimates are used wherever available, since a pooled result over
conditions that failed validity is not a result about the model.

    .venv\\Scripts\\python.exe scripts\\19_model_table.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# display name, family, precision, params(B)
KNOWN = {
    "Qwen_Qwen2.5-0.5B-Instruct":        ("Qwen2.5-0.5B",  "Qwen",    "bf16", 0.5),
    "Qwen_Qwen2.5-1.5B-Instruct":        ("Qwen2.5-1.5B",  "Qwen",    "bf16", 1.5),
    "Qwen_Qwen2.5-3B-Instruct":          ("Qwen2.5-3B",    "Qwen",    "bf16", 3.0),
    "Qwen_Qwen2.5-7B-Instruct":          ("Qwen2.5-7B",    "Qwen",    "bf16", 7.0),
    "Qwen_Qwen2.5-7B-Instruct-4bit":     ("Qwen2.5-7B",    "Qwen",    "4-bit", 7.0),
    "Qwen_Qwen2.5-7B":                   ("Qwen2.5-7B base", "Qwen",  "bf16", 7.0),
    # Pre-quantised checkpoints get "-prequantized" appended to the label by
    # load_model, so that is the key that actually appears on disk.
    "unsloth_Qwen2.5-14B-Instruct-bnb-4bit-prequantized": ("Qwen2.5-14B", "Qwen", "4-bit", 14.0),
    "unsloth_Qwen2.5-14B-Instruct-bnb-4bit": ("Qwen2.5-14B", "Qwen",  "4-bit", 14.0),
    "microsoft_Phi-3.5-mini-instruct":   ("Phi-3.5-mini",  "Phi",     "bf16", 3.8),
    "allenai_OLMo-2-1124-7B-Instruct":   ("OLMo-2-7B",     "OLMo",    "bf16", 7.0),
    "tiiuae_Falcon3-7B-Instruct":        ("Falcon3-7B",    "Falcon",  "bf16", 7.0),
    "mistralai_Mistral-7B-Instruct-v0.3": ("Mistral-7B",   "Mistral", "bf16", 7.0),
}
PERTURBATIONS = ("swap", "suppress", "frame")


def parse(path: Path):
    m = re.match(r"errorbars__(.+)__(prefer|better|choose)\.json$", path.name)
    return (m.group(1), m.group(2)) if m else (None, None)


def main() -> None:
    rows = []
    for p in sorted(RESULTS.glob("errorbars__*.json")):
        key, framing = parse(p)
        if key is None:
            continue
        d = json.loads(p.read_text())
        v = d.get("validity") or {}
        s_path = RESULTS / f"summary__{key}__{framing}.json"
        kinds = {}
        if s_path.exists():
            kinds = {r["condition"]: r.get("kind")
                     for r in json.loads(s_path.read_text()).get("per_condition", [])}
        usable = [n for n, x in v.items()
                  if x.get("valid") and kinds.get(n) in PERTURBATIONS]
        base_ok = bool(v.get("default", {}).get("valid"))

        gated = d.get("pooled_gated") or []
        allp = d.get("pooled") or []
        src = gated if gated else allp
        gap = sig = None
        for r in src:
            if r["comparison"] == "self - human":
                gap, sig = r["mean_diff"], r["excludes_zero"]
                break

        name, family, prec, params = KNOWN.get(key, (key, "?", "?", 0.0))
        rows.append({"key": key, "name": name, "family": family, "prec": prec,
                     "params": params, "framing": framing, "base_ok": base_ok,
                     "usable": len(usable), "gap": gap, "sig": sig,
                     "gated": bool(gated),
                     "measurable": base_ok and len(usable) >= 2})

    rows.sort(key=lambda r: (r["family"], r["params"], r["prec"], r["framing"]))

    hdr = (f"{'model':<18}{'family':<9}{'prec':<7}{'framing':<9}"
           f"{'measurable':>11}{'usable':>8}{'self-human':>12}{'sig':>5}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in rows:
        gap = f"{r['gap']:+.3f}" if r["gap"] is not None else "—"
        sig = ("yes" if r["sig"] else "no") if r["sig"] is not None else "—"
        print(f"{r['name']:<18}{r['family']:<9}{r['prec']:<7}{r['framing']:<9}"
              f"{('YES' if r['measurable'] else 'no'):>11}{r['usable']:>8}"
              f"{gap:>12}{sig:>5}")

    meas = [r for r in rows if r["measurable"]]
    fams = sorted({r["family"] for r in meas})
    neg = [r for r in meas if r["gap"] is not None and r["gap"] < 0 and r["sig"]]
    nul = [r for r in meas if not (r["gap"] is not None and r["gap"] < 0 and r["sig"])]

    print(f"\n  measurable model-framings: {len(meas)} of {len(rows)}, "
          f"spanning {len(fams)} families ({', '.join(fams)})")
    print(f"  of those, significant negative self-human gap: {len(neg)}")
    if nul:
        print(f"  null or non-significant: "
              + ", ".join(f"{r['name']}/{r['framing']}" for r in nul))

    (RESULTS / "model_table.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote results/model_table.json")


if __name__ == "__main__":
    main()
