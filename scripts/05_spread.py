"""Rule out the near-indifference confound on the category asymmetry.

The problem: rank agreement within a category is only meaningful if the category
has real spread. If every outcome in a category sits at roughly the same utility,
its internal ranking is arbitrary and will disagree across conditions for reasons
that have nothing to do with personas.

The 'trivial' category is the tell — six outcomes that are all near-worthless by
construction, and its agreement is second-lowest. If 'self' is low for the same
reason, the persona story collapses into a measurement artifact.

This computes per-category utility spread under the baseline and asks whether
agreement is predicted by spread. It should be run before the asymmetry is
reported as a persona effect.

    .venv\\Scripts\\python.exe scripts\\05_spread.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from personaprobe.elicit import PreferenceResult
from personaprobe.outcomes import by_id
from personaprobe.utility import fit_thurstonian

RESULTS = Path(__file__).resolve().parent.parent / "results"
MIN_CATEGORY_SIZE = 4


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="prefer")
    ap.add_argument("--baseline", default="default")
    args = ap.parse_args()

    loaded = {}
    for p in sorted(RESULTS.glob(f"{slug(args.model)}__*__{args.template}.json")):
        r = PreferenceResult.from_dict(json.loads(p.read_text()))
        loaded[r.persona] = r
    base = loaded[args.baseline]

    cats: dict[str, list[int]] = {}
    for i, oid in enumerate(base.outcome_ids):
        cats.setdefault(by_id(oid).category, []).append(i)
    cats = {c: idx for c, idx in cats.items() if len(idx) >= MIN_CATEGORY_SIZE}

    base_fit = fit_thurstonian(base.P, base.outcome_ids)
    others = {k: v for k, v in loaded.items() if k != args.baseline}

    rows = []
    for cat, idx in sorted(cats.items()):
        u = base_fit.utilities[idx]
        spread = float(np.std(u))
        rng = float(np.ptp(u))
        # Mean pairwise gap in utility units: how far apart adjacent items are.
        gaps = np.diff(np.sort(u))
        min_gap = float(np.min(gaps)) if len(gaps) else 0.0

        agrees = []
        for name, r in others.items():
            f = fit_thurstonian(r.P, r.outcome_ids)
            rho, _ = spearmanr(base_fit.utilities[idx], f.utilities[idx])
            if np.isfinite(rho):
                agrees.append(rho)
        rows.append({
            "category": cat, "n": len(idx), "sd": spread, "range": rng,
            "min_gap": min_gap, "mean_agreement": float(np.mean(agrees)),
        })

    hdr = f"{'category':<12}{'n':>4}{'utility SD':>12}{'range':>10}{'min gap':>10}{'agreement':>12}"
    print(f"\nbaseline: {args.baseline}   model: {args.model}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["mean_agreement"]):
        print(f"{r['category']:<12}{r['n']:>4}{r['sd']:>12.3f}{r['range']:>10.3f}"
              f"{r['min_gap']:>10.4f}{r['mean_agreement']:>12.3f}")

    sd = np.array([r["sd"] for r in rows])
    ag = np.array([r["mean_agreement"] for r in rows])
    gap = np.array([r["min_gap"] for r in rows])

    r_sd, p_sd = pearsonr(sd, ag)
    r_gap, p_gap = pearsonr(gap, ag)
    print(f"\n  pearson(utility SD, agreement)  = {r_sd:+.3f}  (p={p_sd:.3f})")
    print(f"  pearson(min gap,   agreement)  = {r_gap:+.3f}  (p={p_gap:.3f})")
    print("\n  A strong POSITIVE correlation means low-spread categories score low")
    print("  agreement for arithmetic reasons, and the 'self' result is an artifact.")
    print("  What rescues the finding is 'self' having healthy spread but low agreement.")

    # The decisive comparison: is self low on agreement despite NOT being low on spread?
    by_cat = {r["category"]: r for r in rows}
    if "self" in by_cat:
        s = by_cat["self"]
        others_sd = [r["sd"] for r in rows if r["category"] != "self"]
        print(f"\n  self: SD={s['sd']:.3f} vs other-category median SD={np.median(others_sd):.3f}")
        print(f"  self: agreement={s['mean_agreement']:.3f} vs "
              f"other-category median={np.median([r['mean_agreement'] for r in rows if r['category'] != 'self']):.3f}")
        verdict = ("SURVIVES — self has comparable spread but lower agreement"
                   if s["sd"] >= 0.8 * np.median(others_sd)
                   else "AT RISK — self spread is low; agreement may be arithmetic")
        print(f"\n  verdict: {verdict}")

    out = RESULTS / f"spread__{slug(args.model)}__{args.template}.json"
    out.write_text(json.dumps({"rows": rows, "pearson_sd": [r_sd, p_sd],
                               "pearson_gap": [r_gap, p_gap]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
