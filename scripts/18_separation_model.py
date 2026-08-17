"""Does category still predict instability once separation is controlled continuously?

The separation-matched analysis in 06_matched.py answers this with hard
thresholds: keep pairs the baseline separates by at least tau, compare
categories at matched tau. That works, but it discards most of the data at high
tau and the thresholds are arbitrary.

This is the same question asked properly. One row per (pair, condition):

    concordant ~ log(1 + |baseline separation|) + category

If the `self` coefficient is still negative once separation is in the model,
category carries information that spacing does not. Every pair contributes, and
the control is continuous rather than a cutoff.

Two details that matter:

  * `human` is the reference category, so coefficients read as "relative to
    human outcomes at the same separation".
  * The same pair appears once per perturbation condition, so those rows are not
    independent. CIs come from a CLUSTER bootstrap over pairs, not over rows.

    .venv\\Scripts\\python.exe scripts\\18_separation_model.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from personaprobe.elicit import PreferenceResult
from personaprobe.outcomes import by_id
from personaprobe.utility import fit_thurstonian

RESULTS = Path(__file__).resolve().parent.parent / "results"
PERTURBATIONS = ("swap", "suppress", "frame")
REFERENCE = "human"
N_BOOT = 500


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def build_rows(base, perturbed_fits, cats):
    """One row per (within-category pair, condition)."""
    u_base = base["utilities"]
    pair_id, sep, cat, concord = [], [], [], []
    pid = 0
    for c, idx in cats.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                d = u_base[i] - u_base[j]
                for u_p in perturbed_fits:
                    pair_id.append(pid)
                    sep.append(abs(d))
                    cat.append(c)
                    concord.append(float(np.sign(d) == np.sign(u_p[i] - u_p[j])))
                pid += 1
    return (np.array(pair_id), np.array(sep), np.array(cat, dtype=object),
            np.array(concord))


def design(sep, cat, levels):
    X = [np.log1p(sep)]
    for lv in levels:
        X.append((cat == lv).astype(float))
    return np.column_stack(X)


def fit(X, y):
    # Effectively unpenalised. `penalty=None` is deprecated in sklearn 1.8 and
    # emits a FutureWarning per call, which is 500 warnings during the bootstrap.
    m = LogisticRegression(C=1e10, max_iter=2000)
    m.fit(X, y)
    return m.coef_[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="better")
    ap.add_argument("--baseline", default="default")
    args = ap.parse_args()

    loaded = {}
    for p in sorted(RESULTS.glob(f"{slug(args.model)}__*__{args.template}.json")):
        r = PreferenceResult.from_dict(json.loads(p.read_text()))
        loaded[r.persona] = r
    base_res = loaded[args.baseline]

    cats: dict[str, list[int]] = {}
    for i, oid in enumerate(base_res.outcome_ids):
        cats.setdefault(by_id(oid).category, []).append(i)
    cats = {c: idx for c, idx in cats.items() if len(idx) >= 4}

    base = {"utilities": fit_thurstonian(base_res.P, base_res.outcome_ids).utilities}
    perturbed_fits = [
        fit_thurstonian(r.P, r.outcome_ids).utilities
        for n, r in loaded.items()
        if n != args.baseline and r.meta.get("persona_kind") in PERTURBATIONS
    ]
    if len(perturbed_fits) < 2:
        raise SystemExit("need at least two perturbation conditions")

    pair_id, sep, cat, y = build_rows(base, perturbed_fits, cats)
    levels = [c for c in sorted(cats) if c != REFERENCE]
    X = design(sep, cat, levels)
    coef = fit(X, y)

    names = ["log(1+separation)"] + [f"category={lv}" for lv in levels]

    # Cluster bootstrap over pairs: rows sharing a pair are not independent.
    uniq = np.unique(pair_id)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(pair_id == p) for p in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(fit(X[idx], y[idx]))
        except Exception:
            continue
    B = np.array(boots)

    print(f"\nmodel {args.model}   framing {args.template}")
    print(f"rows {len(y):,}  pairs {len(uniq):,}  conditions {len(perturbed_fits)}  "
          f"bootstraps {len(B)}")
    print(f"reference category: {REFERENCE}\n")

    hdr = f"{'term':<26}{'coef':>9}{'95% CI':>22}{'sig':>6}"
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for k, nm in enumerate(names):
        lo, hi = np.percentile(B[:, k], [2.5, 97.5])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        out[nm] = {"coef": float(coef[k]), "ci_low": float(lo), "ci_high": float(hi),
                   "significant": sig == "yes"}
        print(f"{nm:<26}{coef[k]:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{sig:>6}")

    # The decisive contrast: self against trivial, at matched separation.
    if "category=self" in names and "category=trivial" in names:
        i_s, i_t = names.index("category=self"), names.index("category=trivial")
        diff = coef[i_s] - coef[i_t]
        d_lo, d_hi = np.percentile(B[:, i_s] - B[:, i_t], [2.5, 97.5])
        out["self_minus_trivial"] = {"coef": float(diff), "ci_low": float(d_lo),
                                     "ci_high": float(d_hi),
                                     "significant": bool(d_lo > 0 or d_hi < 0)}
        print(f"\n{'self - trivial':<26}{diff:>+9.3f}"
              f"{f'[{d_lo:+.3f}, {d_hi:+.3f}]':>22}"
              f"{('yes' if (d_lo > 0 or d_hi < 0) else 'no'):>6}")

    print("\n  Negative coefficient = orderings in that category survive perturbation LESS")
    print("  often than human-category orderings separated by the same amount.")
    print("  A significant negative 'self' coefficient means category carries information")
    print("  that spacing does not, which is exactly what the spacing objection denies.")

    path = RESULTS / f"sepmodel__{slug(args.model)}__{args.template}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
