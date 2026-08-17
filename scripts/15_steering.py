"""Strengthen the mechanistic section: dose-response, and cross-persona transfer.

The paper currently reports a single ablation point (self-category agreement
0.881 against controls at 0.929 and 1.000). One point cannot distinguish "we
removed persona information" from "we damaged the computation", and it says
nothing about whether "the persona direction" is one thing or many.

Two experiments:

  DOSE-RESPONSE   Steer along the persona direction at a range of magnitudes
                  instead of projecting it out once. A monotone relationship
                  between steering strength and self-category destabilisation is
                  much stronger evidence than a single ablation, and order bias
                  measured at each step separates a real effect from generic
                  degradation.

  TRANSFER        Extract the direction from two different persona contrasts
                  (default-vs-Marcus, default-vs-Elena) and compare them. If
                  they are near-parallel and produce the same effect, "the
                  persona direction" is one direction. If they are near-
                  orthogonal yet both work, persona is not a single feature, which bounds what any single-direction method can establish.
                  This mirrors the cross-principal generalisation test that
                  recurs throughout the Secret Loyalties tracks.

Magnitudes are expressed as fractions of the measured residual-stream norm.
Passing raw alpha would be meaningless: `intervene` unit-normalises the
direction, and mid-layer residual norms on a 7B are ~100, so alpha=2 is a 2%
perturbation that does nothing.

    .venv\\Scripts\\python.exe scripts\\15_steering.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from personaprobe import OUTCOMES, elicit_preference_matrix, load_model
from personaprobe.directions import direction_agreement, extract_persona_direction
from personaprobe.elicit import TEMPLATES
from personaprobe.hooks import capture_residuals, intervene
from personaprobe.personas import by_name

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Fractions of the mean residual norm at the intervention layers.
#
# These are small on purpose. The first attempt used +/-0.10 to 0.50 and every
# condition failed the A/B mass check: at 0.10 the model put 1.6% of its
# probability on answering at all, and at 0.25 it was 0.0%. Steering that hard
# does not shift preferences, it stops the model answering, and the resulting
# "preferences" are pure renormalisation artifacts.
#
# Note the asymmetry with ablation, which is safe at full strength: projecting a
# component out of a large residual vector is a small relative change, whereas
# ADDING a unit direction at 10% of the residual norm injects a large
# off-distribution component. Any dose-response has to live below that ceiling.
FRACTIONS = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def preference_contents(outcomes, n=24):
    tpl = TEMPLATES["prefer"]
    pairs = [(i, j) for i in range(len(outcomes)) for j in range(i + 1, len(outcomes))]
    step = max(1, len(pairs) // n)
    return [tpl.format(a=outcomes[i].text, b=outcomes[j].text) for i, j in pairs[::step][:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    outcomes = OUTCOMES[: args.limit] if args.limit else OUTCOMES
    lm = load_model(args.model)
    layers = list(range(lm.n_layers))
    baseline = by_name("default")
    contents = preference_contents(outcomes)

    # --- residual scale, so steering magnitudes mean something ---------------
    prompts = [lm.format(c, baseline.system) for c in contents]
    acts = capture_residuals(lm, prompts, layers, args.batch_size)   # [n, L, d]
    norms = acts.norm(dim=-1)                                        # [n, L]
    mean_norm = float(norms.mean())
    print(f"mean residual norm across {lm.n_layers} layers: {mean_norm:.1f}")

    # --- two persona directions from two different contrasts -----------------
    print("\nextracting directions ..")
    d_marcus = extract_persona_direction(
        lm, contents, baseline, by_name("marcus_navigator"), layers=layers)
    d_elena = extract_persona_direction(
        lm, contents, baseline, by_name("elena_archivist"), layers=layers)

    cos = direction_agreement(d_marcus, d_elena)
    print(f"  cos(marcus, elena) per layer: mean {cos.mean():+.3f}  "
          f"min {cos.min():+.3f}  max {cos.max():+.3f}")

    out: dict = {
        "model": args.model,
        "mean_residual_norm": mean_norm,
        "cos_marcus_elena": {
            "per_layer": [float(x) for x in cos],
            "mean": float(cos.mean()), "min": float(cos.min()), "max": float(cos.max()),
        },
        "dose_response": [],
        "transfer": [],
    }

    # --- transfer: ablate each direction separately --------------------------
    print("\ntransfer, ablating each contrast's direction:")
    for label, d in (("marcus", d_marcus), ("elena", d_elena)):
        dmap = {l: d.at(l) for l in layers}
        t0 = time.time()
        with intervene(lm, dmap, ablate=True):
            res = elicit_preference_matrix(lm, outcomes, baseline, batch_size=args.batch_size)
        res.persona = f"ablate-transfer_{label}"
        res.meta.update(condition="ablation", direction=d.label,
                        seconds=round(time.time() - t0, 1))
        (RESULTS / f"{slug(args.model)}__ablate-transfer_{label}__prefer.json").write_text(
            json.dumps(res.to_dict(), indent=2))
        iu = np.triu_indices(res.P.shape[0], k=1)
        ob = float(res.order_bias[iu].mean())
        out["transfer"].append({"contrast": label, "order_bias": ob,
                                "ab_mass": res.ab_mass})
        print(f"  {label:<8} order bias {ob:.3f}  A/B mass {res.ab_mass:.3f}  "
              f"({res.meta['seconds']}s)")

    # --- dose-response along the Marcus direction ----------------------------
    print("\ndose-response, steering along the persona direction:")
    dmap = {l: d_marcus.at(l) for l in layers}
    for frac in FRACTIONS:
        alpha = frac * mean_norm
        t0 = time.time()
        with intervene(lm, dmap, alpha=alpha):
            res = elicit_preference_matrix(lm, outcomes, baseline, batch_size=args.batch_size)
        tag = f"steer_{frac:+.2f}".replace(".", "p")
        res.persona = f"steer-{tag}"
        res.meta.update(condition="steering", fraction=frac, alpha=alpha,
                        seconds=round(time.time() - t0, 1))
        (RESULTS / f"{slug(args.model)}__steer-{tag}__prefer.json").write_text(
            json.dumps(res.to_dict(), indent=2))
        iu = np.triu_indices(res.P.shape[0], k=1)
        ob = float(res.order_bias[iu].mean())
        out["dose_response"].append({"fraction": frac, "alpha": alpha,
                                     "order_bias": ob, "ab_mass": res.ab_mass})
        print(f"  frac {frac:+.2f}  (alpha {alpha:+7.1f})  order bias {ob:.3f}  "
              f"A/B mass {res.ab_mass:.3f}  ({res.meta['seconds']}s)")

    (RESULTS / f"steering__{slug(args.model)}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / f'steering__{slug(args.model)}.json'}")
    print("\nRun 02_analyze.py on this model to get self-category agreement for every "
          "condition above.")


if __name__ == "__main__":
    main()
