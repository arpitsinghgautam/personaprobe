"""Build the report figures.

Three figures, matching the three claims:
  fig1, category agreement by framing, with CIs. The main result and its
         framing-dependence in one panel.
  fig2, separation-matched concordance vs tau. Shows the spacing confound
         being controlled rather than asserted away.
  fig3, ablation vs random and content controls. The mechanistic result,
         null or otherwise.

Text is sized for a 4-page two-column report; the template calls out figure
legibility explicitly, so nothing here shrinks below 8pt.

    .venv\\Scripts\\python.exe scripts\\07_figures.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

FRAMINGS = ["better", "prefer", "choose"]
COLORS = {"better": "#b2182b", "prefer": "#2166ac", "choose": "#4d9221"}


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fig1_by_framing(model: str) -> bool:
    """Per-category agreement with baseline, one series per framing."""
    data = {}
    for fr in FRAMINGS:
        d = load(RESULTS / f"errorbars__{slug(model)}__{fr}.json")
        if d:
            data[fr] = d["agreements"]
    if not data:
        return False

    cats = sorted({c for a in data.values() for cond in a.values() for c in cond})
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.8 / max(len(data), 1)

    for k, (fr, agr) in enumerate(data.items()):
        means, los, his = [], [], []
        for c in cats:
            vals = [cond[c]["mean"] for cond in agr.values() if c in cond]
            lo = [cond[c]["ci_low"] for cond in agr.values() if c in cond]
            hi = [cond[c]["ci_high"] for cond in agr.values() if c in cond]
            means.append(np.mean(vals) if vals else np.nan)
            los.append(np.mean(lo) if lo else np.nan)
            his.append(np.mean(hi) if hi else np.nan)
        means, los, his = np.array(means), np.array(los), np.array(his)
        x = np.arange(len(cats)) + (k - (len(data) - 1) / 2) * width
        ax.bar(x, means, width * 0.9, label=fr, color=COLORS.get(fr, "#888"), alpha=0.85)
        ax.errorbar(x, means, yerr=[means - los, his - means], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2)

    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel("rank agreement with baseline\n(Spearman, mean over perturbations)")
    ax.set_ylim(0.3, 1.06)
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    # Legend below the axes: inside the plot it sat on top of the leftmost bars.
    ax.legend(title="framing", frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.set_title("Persona intervention concentrates on self-relevant outcomes,\n"
                 "and how much depends on how the question is framed", loc="left")
    fig.text(0.01, -0.16, "Bars are means over five perturbation conditions; whiskers are "
             "bootstrap 95% CIs. `trivial` outcomes are near-indifferent by\nconstruction and "
             "have no stable ordering to preserve, so their instability is not evidence of the "
             "same kind. Axis starts at 0.3.",
             fontsize=6.5, va="top", color="#444")
    fig.savefig(FIGURES / "fig1_category_by_framing.png")
    plt.close(fig)
    return True


def fig2_matched(model: str) -> bool:
    """Concordance vs separation threshold, per category."""
    panels = [(fr, load(RESULTS / f"matched__{slug(model)}__{fr}.json")) for fr in FRAMINGS]
    panels = [(fr, d) for fr, d in panels if d]
    if not panels:
        return False

    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels) + 1, 2.9),
                             sharey=True, squeeze=False)
    for ax, (fr, d) in zip(axes[0], panels):
        taus = sorted(float(t) for t in d)
        cats = sorted({c for t in d for c in d[t]})
        for c in cats:
            xs = [t for t in taus if c in d[str(t)]]
            ys = [d[str(t)][c]["concordance"] for t in xs]
            style = dict(lw=2.0, marker="o", ms=3.5, zorder=3) if c == "self" \
                else dict(lw=0.9, alpha=0.55, marker="", zorder=2)
            ax.plot(xs, ys, label=c, color="#b2182b" if c == "self" else "#777", **style)
        ax.set_title(f'"{fr}"', loc="left")
        ax.set_xlabel(r"separation threshold $\tau$")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("concordance among\nbaseline-separated pairs")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("The gap persists when closely-spaced pairs are filtered out "
                 "(self in red)", x=0.02, y=1.10, ha="left", fontsize=10)
    fig.savefig(FIGURES / "fig2_separation_matched.png")
    plt.close(fig)
    return True


def fig3_ablation(model: str) -> bool:
    """Ablation vs controls, alongside prompt-level swaps for scale."""
    d = load(RESULTS / f"summary__{slug(model)}__prefer.json")
    if not d:
        return False
    comps = d.get("comparisons", {})
    if not comps:
        return False

    def cat_val(name, cat="self"):
        bc = comps.get(name, {}).get("by_category", {})
        return bc.get(cat, {}).get("spearman", np.nan)

    def group(n: str) -> int:
        if n.startswith("ablate-persona"):
            return 0
        if n.startswith("ablate-control"):
            return 1
        return 2

    names = sorted(comps, key=lambda n: (group(n), -cat_val(n)))
    if not names:
        return False
    vals = [cat_val(n) for n in names]
    palette = {0: "#2166ac", 1: "#92c5de", 2: "#b0b0b0"}
    colors = [palette[group(n)] for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("self-category rank agreement with baseline")
    ax.set_xlim(0.5, 1.03)
    ax.axvline(1.0, color="grey", lw=0.6, ls=":")

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in (0, 1, 2)]
    ax.legend(handles, ["persona-direction ablation", "control directions",
                        "prompt-level persona change"],
              frameon=False, fontsize=7.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Ablating the persona direction moves self-relevant preferences\n"
                 "only partway to what a prompt-level swap does", loc="left")
    fig.text(0.01, -0.14,
             "Lower = less stable. The matched-context ablation (0.881) falls below both "
             "controls but well short of the prompt-level\nswaps (0.667–0.786). It also raises "
             "order bias to 0.320 against 0.145–0.177 for the controls, so part of the shift is "
             "measurement\ndegradation rather than removed persona information. Axis starts at 0.5.",
             fontsize=6.5, va="top", color="#444")
    fig.savefig(FIGURES / "fig3_ablation.png")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    FIGURES.mkdir(exist_ok=True)

    for name, fn in [("fig1", fig1_by_framing), ("fig2", fig2_matched), ("fig3", fig3_ablation)]:
        try:
            ok = fn(args.model)
            print(f"  {name}: {'written' if ok else 'SKIPPED (inputs missing)'}")
        except Exception as e:
            print(f"  {name}: FAILED, {type(e).__name__}: {e}")

    print(f"\nfigures in {FIGURES}")


if __name__ == "__main__":
    main()
