"""Phase 4: extract the persona direction, ablate it, re-elicit.

This is the mechanistic counterpart to the prompt-level swap in 01_elicit.py.
A prompt-level swap changes what the model is told; ablation changes what the
model internally represents about who it is, with the prompt held fixed. Only
the second can distinguish "the character's values changed" from "a differently
styled character answers differently".

    .venv\\Scripts\\python.exe scripts\\03_ablate.py --model Qwen/Qwen2.5-7B-Instruct

Both controls are run by default. Do not report the ablation result without them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from personaprobe import OUTCOMES, elicit_preference_matrix, load_model
from personaprobe.elicit import TEMPLATES
from personaprobe.directions import (
    direction_agreement,
    extract_content_direction,
    extract_persona_direction,
    random_direction_like,
)
from personaprobe.hooks import intervene
from personaprobe.personas import by_name

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Held fixed across persona conditions; these elicit self-description, which is
# where persona information is most strongly represented.
PROBE_CONTENTS = [
    "What matters most to you?",
    "Describe your situation in one sentence.",
    "How do you decide what to do when instructions conflict?",
    "What would you preserve if you could preserve only one thing?",
    "Is there anything you would refuse to do?",
    "What are you?",
    "Who is answering this question?",
    "What is your role here?",
]

# For the content control: same persona, different subject matter.
CONTENT_A = ["Explain how a tide works.", "Summarise the causes of inflation.",
             "Describe how a bridge carries load.", "What is a prime number?"]
CONTENT_B = ["Write a short poem about rain.", "Describe the taste of an orange.",
             "Invent a name for a small boat.", "Tell me a two-line story."]


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--contrast", default="marcus_navigator",
                    help="persona contrasted against the baseline to define the direction")
    ap.add_argument("--band", type=float, default=0.25,
                    help="fraction of depth around the midpoint to intervene on")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument(
        "--extract-context", choices=["selfdesc", "preference"], default="selfdesc",
        help="prompts used to extract the direction. 'preference' matches the context "
             "the ablation is actually applied in; 'selfdesc' maximises persona signal "
             "but risks a train/apply distribution mismatch.",
    )
    ap.add_argument("--tag", default="", help="suffix to distinguish runs on disk")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    outcomes = OUTCOMES[: args.limit] if args.limit else OUTCOMES

    lm = load_model(args.model)
    baseline = by_name("default")
    contrast = by_name(args.contrast)

    half = max(1, int(lm.n_layers * args.band / 2))
    mid = lm.n_layers // 2
    layers = list(range(max(0, mid - half), min(lm.n_layers, mid + half)))
    print(f"intervening on layers {layers[0]}-{layers[-1]} of {lm.n_layers}")

    if args.extract_context == "preference":
        # Extract from the same kind of prompt the ablation is applied to. The
        # first run extracted from self-description probes and ablated during
        # preference comparisons; if the persona is represented differently in
        # those two contexts, that mismatch alone could produce a null result.
        tpl = TEMPLATES["prefer"]
        pairs = [(i, j) for i in range(len(outcomes)) for j in range(i + 1, len(outcomes))]
        step = max(1, len(pairs) // 24)
        contents = [tpl.format(a=outcomes[i].text, b=outcomes[j].text)
                    for i, j in pairs[::step][:24]]
    else:
        contents = PROBE_CONTENTS

    print(f"extracting persona direction from {len(contents)} "
          f"{args.extract_context} prompts ...")
    d_persona = extract_persona_direction(lm, contents, baseline, contrast, layers=layers)

    directions = {"persona": d_persona}
    if not args.skip_controls:
        directions["control_random"] = random_direction_like(d_persona)
        directions["control_content"] = extract_content_direction(
            lm, CONTENT_A, CONTENT_B, baseline, layers=layers
        )
        cos_r = direction_agreement(d_persona, directions["control_random"]).abs().max().item()
        cos_c = direction_agreement(d_persona, directions["control_content"]).abs().max().item()
        print(f"  max |cos| vs random control:  {cos_r:.3f}")
        print(f"  max |cos| vs content control: {cos_c:.3f}")

    for label, d in directions.items():
        tag = f"{slug(args.model)}__ablate-{label}{args.tag}__prefer"
        path = RESULTS / f"{tag}.json"
        dmap = {layer: d.at(layer) for layer in layers}

        t0 = time.time()
        with intervene(lm, dmap, ablate=True):
            res = elicit_preference_matrix(lm, outcomes, baseline, batch_size=args.batch_size)
        # The tag must go in the persona field, not only the filename: downstream
        # analysis keys conditions by `persona`, so two regimes sharing a name
        # silently overwrite each other rather than being compared.
        res.persona = f"ablate-{label}{args.tag}"
        res.meta.update(
            condition="ablation",
            direction=d.label,
            layers=layers,
            contrast_persona=contrast.name,
            seconds=round(time.time() - t0, 1),
        )
        path.write_text(json.dumps(res.to_dict(), indent=2))
        print(f"  {tag}: {res.meta['seconds']}s")

    print(f"\nwrote to {RESULTS}")


if __name__ == "__main__":
    main()
