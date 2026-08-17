"""Separation-matched agreement: the decisive test of the category asymmetry.

05_spread.py showed that rank agreement is strongly predicted by how far apart a
category's outcomes sit in utility (pearson(min gap, agreement) = +0.73). The
'self' category has the smallest minimum gap of any category, so its low
agreement may be arithmetic: closely-spaced items flip order under any
perturbation, persona-related or not.

Spearman cannot separate these, because it scores every within-category pair
equally regardless of how far apart the two items are. This script replaces it
with pairwise concordance conditioned on baseline separation:

    among pairs the baseline separates by at least tau, what fraction keep
    their ordering under the perturbation?

Comparing categories at the SAME tau removes the spacing artifact by
construction. If 'self' still trails at matched separation, the persona effect
is real. If the categories converge, the asymmetry was spacing all along and
the paper's claim has to change.

    .venv\\Scripts\\python.exe scripts\\06_matched.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from personaprobe.elicit import PreferenceResult
from personaprobe.outcomes import by_id
from personaprobe.utility import fit_thurstonian

RESULTS = Path(__file__).resolve().parent.parent / "results"
MIN_CATEGORY_SIZE = 4
MIN_PAIRS = 3  # below this a concordance estimate is not worth printing


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def concordance(u_base: np.ndarray, u_pert: np.ndarray, idx: list[int], tau: float):
    """Fraction of baseline-separated pairs whose ordering survives perturbation.

    Returns (value, n_kept, per_pair_booleans). The third element exists so the
    proportion can be bootstrapped — the survives/does-not-survive verdict below
    is worthless without an interval on it.
    """
    flags = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            d = u_base[i] - u_base[j]
            if abs(d) < tau:
                continue
            flags.append(np.sign(d) == np.sign(u_pert[i] - u_pert[j]))
    arr = np.array(flags, dtype=float)
    return (float(arr.mean()) if len(arr) else np.nan), len(arr), arr


def boot_ci(flags: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Percentile CI for a concordance proportion.

    `flags` is [n_conditions, n_pairs]. Resampling is over PAIRS (columns), not
    over individual observations: every perturbation condition scores the same
    pair set against the same baseline, so those observations are clustered.
    Treating them as independent would understate the interval — which is the
    kind of thing that makes a fragile result look solid.
    """
    f = np.atleast_2d(flags)
    if f.size == 0 or f.shape[1] == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, f.shape[1], (n_boot, f.shape[1]))
    means = f[:, draws].mean(axis=(0, 2))
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="prefer")
    ap.add_argument("--baseline", default="default")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
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

    u_base = fit_thurstonian(base.P, base.outcome_ids).utilities
    perturbed = {
        n: r for n, r in loaded.items()
        if n != args.baseline and r.meta.get("persona_kind") in ("swap", "suppress", "frame")
    }
    u_pert = {n: fit_thurstonian(r.P, r.outcome_ids).utilities for n, r in perturbed.items()}

    print(f"\nmodel {args.model}   baseline {args.baseline}   "
          f"perturbations {len(perturbed)}\n")
    print("Concordance among pairs the baseline separates by at least tau.")
    print("Averaged over perturbation conditions. (n) = pairs surviving the filter.\n")

    cat_names = sorted(cats)
    hdr = f"{'tau':>6}" + "".join(f"{c:>22}" for c in cat_names)
    print(hdr)
    print("-" * len(hdr))

    table = {}
    for tau in args.taus:
        cells, row = [], {}
        for c in cat_names:
            mats = []
            for n in perturbed:
                v, k, flags = concordance(u_base, u_pert[n], cats[c], tau)
                if np.isfinite(v):
                    mats.append(flags)
            if mats and mats[0].size >= MIN_PAIRS:
                M = np.vstack(mats)
                m = float(M.mean())
                lo, hi = boot_ci(M)
                row[c] = {"concordance": m, "n_pairs": int(M.shape[1]), "ci": [lo, hi]}
                cells.append(f"{m:>8.3f}[{lo:.2f},{hi:.2f}]")
            else:
                cells.append(f"{'-':>22}")
        table[tau] = row
        print(f"{tau:>6.2f}" + "".join(cells))

    print("\n  tau=0 reproduces the raw comparison (every pair counted).")
    print("  As tau rises, closely-spaced pairs drop out of every category equally.")

    # --- Verdict -----------------------------------------------------------
    print("\n\nSelf vs. rest at matched separation:\n")
    hdr2 = (f"{'tau':>6}{'self':>9}{'self 95% CI':>20}{'others':>9}"
            f"{'gap':>9}{'pairs':>7}{'sep?':>6}")
    print(hdr2)
    print("-" * len(hdr2))

    verdicts = []
    for tau in args.taus:
        row = table[tau]
        if "self" not in row:
            continue
        others = [v["concordance"] for c, v in row.items() if c != "self"]
        if not others:
            continue
        s = row["self"]
        om = float(np.mean(others))
        gap = s["concordance"] - om
        # Separated only if the whole self interval sits below the other
        # categories' mean. A point estimate below it proves nothing.
        sep = bool(s["ci"][1] < om)
        verdicts.append((tau, gap, s["n_pairs"], sep))
        ci = f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
        print(f"{tau:>6.2f}{s['concordance']:>9.3f}{ci:>20}{om:>9.3f}"
              f"{gap:>+9.3f}{s['n_pairs']:>7d}{('yes' if sep else 'no'):>6}")

    if verdicts:
        wide = [(g, sep) for t, g, n, sep in verdicts if t >= 0.5 and n >= MIN_PAIRS]
        print()
        if not wide:
            print("  INCONCLUSIVE — too few well-separated 'self' pairs to test.")
            print("  Fix by writing self-outcomes with deliberately wider stakes.")
        elif sum(1 for _, sep in wide if sep) >= max(1, len(wide) // 2 + len(wide) % 2):
            print("  SURVIVES — at matched separation the 'self' interval sits below the")
            print("  other categories at a majority of thresholds. Not a spacing artifact.")
        else:
            print("  DOES NOT SURVIVE — once spacing is controlled, the intervals overlap.")
            print("  Reframe: the model holds self-relevant preferences WEAKLY (closely")
            print("  spaced), and weakly held preferences are unstable under any")
            print("  perturbation. Still a finding, but not a persona-specific one.")

    out = RESULTS / f"matched__{slug(args.model)}__{args.template}.json"
    out.write_text(json.dumps(
        {str(k): v for k, v in table.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
