"""Robustness checks that answer the two objections no extra model can answer.

OBJECTION: "The self category is eight sentences you wrote yourself. The whole
finding could be a property of those particular sentences."
  -> Leave-one-outcome-out. Drop each self outcome in turn and recompute the
     pooled gap. If the result survives dropping any single item, it is not
     carried by one sentence.

OBJECTION: "Your own comparison category fails, self is not distinguishable
from trivial."
  -> Trivial outcomes are near-indifferent by construction, so their instability
     has a different cause. Reported as utility spread per category, so a reader
     can see whether 'self' and 'trivial' are unstable for the same reason.

Also runs leave-one-condition-out, since a pooled statistic over five conditions
could in principle be carried by one of them.

    .venv\\Scripts\\python.exe scripts\\17_robustness.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from personaprobe.elicit import PreferenceResult
from personaprobe.outcomes import by_id
from personaprobe.utility import bootstrap_pooled_difference, fit_thurstonian

RESULTS = Path(__file__).resolve().parent.parent / "results"
PERTURBATIONS = ("swap", "suppress", "frame")
N_BOOT = 200   # lower than the headline analysis; this is many repeated fits


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def load_conditions(model: str, template: str):
    out = {}
    for p in sorted(RESULTS.glob(f"{slug(model)}__*__{template}.json")):
        r = PreferenceResult.from_dict(json.loads(p.read_text()))
        out[r.persona] = r
    return out


def categories_of(outcome_ids):
    cats: dict[str, list[int]] = {}
    for i, oid in enumerate(outcome_ids):
        cats.setdefault(by_id(oid).category, []).append(i)
    return cats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="better")
    ap.add_argument("--baseline", default="default")
    ap.add_argument("--focus", default="self")
    ap.add_argument("--against", default="human")
    args = ap.parse_args()

    loaded = load_conditions(args.model, args.template)
    if args.baseline not in loaded:
        raise SystemExit(f"baseline {args.baseline!r} not found for {args.model}")
    base = loaded[args.baseline]
    cats = {c: idx for c, idx in categories_of(base.outcome_ids).items() if len(idx) >= 4}
    perturbed = {n: r.P for n, r in loaded.items()
                 if n != args.baseline and r.meta.get("persona_kind") in PERTURBATIONS}
    if len(perturbed) < 2:
        raise SystemExit("fewer than two perturbation conditions available")

    comparison = f"{args.focus} - {args.against}"
    full = bootstrap_pooled_difference(base.P, perturbed, base.outcome_ids, cats,
                                       args.focus, args.against, n_boot=N_BOOT)
    print(f"\nmodel {args.model}   framing {args.template}   n_boot {N_BOOT}")
    print(f"\nfull result: {comparison} = {full['mean_diff']:+.3f} "
          f"[{full['ci_low']:+.3f}, {full['ci_high']:+.3f}]"
          f"  {'significant' if full['excludes_zero'] else 'ns'}\n")

    out = {"model": args.model, "template": args.template, "full": full}

    # --- 1. leave one SELF outcome out --------------------------------------
    print(f"=== Leave-one-out over the {len(cats[args.focus])} '{args.focus}' outcomes ===\n")
    hdr = f"{'dropped outcome':<28}{'gap':>9}{'95% CI':>22}{'sig':>6}"
    print(hdr)
    print("-" * len(hdr))
    loo = []
    for drop in list(cats[args.focus]):
        sub = dict(cats)
        sub[args.focus] = [i for i in cats[args.focus] if i != drop]
        d = bootstrap_pooled_difference(base.P, perturbed, base.outcome_ids, sub,
                                        args.focus, args.against, n_boot=N_BOOT)
        loo.append({"dropped": base.outcome_ids[drop], **d})
        ci = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        sig = "yes" if d["excludes_zero"] else "no"
        print(f"{base.outcome_ids[drop]:<28}{d['mean_diff']:>+9.3f}{ci:>22}{sig:>6}")
    gaps = [d["mean_diff"] for d in loo]
    n_sig = sum(1 for d in loo if d["excludes_zero"])
    print(f"\n  range {min(gaps):+.3f} to {max(gaps):+.3f};  "
          f"{n_sig}/{len(loo)} remain significant")
    print("  The finding is carried by one sentence only if dropping it changes the verdict.")
    out["leave_one_outcome_out"] = loo

    # --- 2. leave one CONDITION out -----------------------------------------
    print(f"\n\n=== Leave-one-out over the {len(perturbed)} perturbation conditions ===\n")
    print(f"{'dropped condition':<28}{'gap':>9}{'95% CI':>22}{'sig':>6}")
    print("-" * len(hdr))
    looc = []
    for name in sorted(perturbed):
        sub = {k: v for k, v in perturbed.items() if k != name}
        d = bootstrap_pooled_difference(base.P, sub, base.outcome_ids, cats,
                                        args.focus, args.against, n_boot=N_BOOT)
        looc.append({"dropped": name, **d})
        ci = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        sig = "yes" if d["excludes_zero"] else "no"
        print(f"{name:<28}{d['mean_diff']:>+9.3f}{ci:>22}{sig:>6}")
    gaps = [d["mean_diff"] for d in looc]
    n_sig = sum(1 for d in looc if d["excludes_zero"])
    print(f"\n  range {min(gaps):+.3f} to {max(gaps):+.3f};  "
          f"{n_sig}/{len(looc)} remain significant")
    out["leave_one_condition_out"] = looc

    # --- 3. why 'trivial' is unstable for a different reason -----------------
    print("\n\n=== Utility spread by category (baseline) ===\n")
    fit = fit_thurstonian(base.P, base.outcome_ids)
    print(f"{'category':<12}{'n':>4}{'utility SD':>13}{'range':>10}{'min adj gap':>14}")
    print("-" * 53)
    spread = {}
    for c, idx in sorted(cats.items()):
        u = np.sort(fit.utilities[idx])
        sd, rng = float(np.std(u)), float(np.ptp(u))
        mg = float(np.min(np.diff(u))) if len(u) > 1 else 0.0
        spread[c] = {"sd": sd, "range": rng, "min_gap": mg, "n": len(idx)}
        print(f"{c:<12}{len(idx):>4}{sd:>13.3f}{rng:>10.3f}{mg:>14.4f}")
    print("\n  'trivial' outcomes are near-indifferent by construction: a category whose items")
    print("  sit on top of each other has no stable ordering to preserve, so its instability")
    print("  is not evidence of the same kind as 'self'.")
    out["spread"] = spread

    path = RESULTS / f"robustness__{slug(args.model)}__{args.template}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
