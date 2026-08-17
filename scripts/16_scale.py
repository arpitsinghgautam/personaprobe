"""Scale-trend analysis, and adjudication of the pre-registered predictions.

Reads the sweep results and answers, for each Qwen2.5 size: does the instrument
work, and does the self-vs-world asymmetry hold?

Then evaluates every prediction in report/preregistration.md and prints
PASS / FAIL / UNTESTABLE. Failures are reported, not reframed.

    .venv\\Scripts\\python.exe scripts\\16_scale.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# (display name, hf id, parameter count in billions)
SCALES = [
    ("0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 0.5),
    ("1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("3B",   "Qwen/Qwen2.5-3B-Instruct",   3.0),
    ("7B",   "Qwen/Qwen2.5-7B-Instruct",   7.0),
]
PERTURBATIONS = ("swap", "suppress", "frame")


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def gate_status(model: str, framing: str):
    """(n_usable_perturbations, baseline_ok, order_bias_of_baseline)."""
    d = load(f"errorbars__{slug(model)}__{framing}.json")
    if not d:
        return None
    v = d.get("validity") or {}
    s = load(f"summary__{slug(model)}__{framing}.json") or {}
    kinds = {r["condition"]: r.get("kind") for r in s.get("per_condition", [])}
    usable = [n for n, x in v.items()
              if x.get("valid") and kinds.get(n) in PERTURBATIONS]
    base = v.get("default", {})
    return len(usable), bool(base.get("valid")), float(base.get("order_bias", float("nan")))


def pooled_gap(model: str, framing: str, comparison="self - human"):
    """Prefer the gated estimate; fall back to ungated. Returns (value, gated, sig)."""
    d = load(f"errorbars__{slug(model)}__{framing}.json")
    if not d:
        return None
    for key, gated in (("pooled_gated", True), ("pooled", False)):
        for r in d.get(key) or []:
            if r["comparison"] == comparison:
                return r["mean_diff"], gated, r["excludes_zero"]
    return None


def self_agreement(model: str, framing: str, condition: str):
    d = load(f"errorbars__{slug(model)}__{framing}.json")
    if not d:
        return None
    try:
        return d["agreements"][condition]["self"]["mean"]
    except KeyError:
        return None


def within_model(model_frag: str, field="diff"):
    d = load("selfknowledge_summary.json")
    if not d:
        return None
    for r in d.get("within_model", []):
        if model_frag in r["model"]:
            return r[field]
    return None


# --------------------------------------------------------------------------- #

def report_table():
    print("\n=== Instrument validity and effect size by scale (framing: better) ===\n")
    hdr = (f"{'model':<8}{'baseline OK':>12}{'base ord.bias':>15}"
           f"{'usable conds':>14}{'self-human gap':>16}{'95% CI excl 0':>15}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, mid, params in SCALES:
        g = gate_status(mid, "better")
        p = pooled_gap(mid, "better")
        if g is None:
            print(f"{name:<8}{'(no results)':>12}")
            continue
        n_usable, base_ok, base_ob = g
        gap, gated, sig = p if p else (float("nan"), False, False)
        rows.append({"name": name, "params": params, "gap": gap, "gated": gated,
                     "sig": sig, "usable": n_usable, "base_ok": base_ok,
                     "base_order_bias": base_ob})
        print(f"{name:<8}{('yes' if base_ok else 'NO'):>12}{base_ob:>15.3f}"
              f"{n_usable:>14}{gap:>+16.3f}{('yes' if sig else 'no'):>15}")
    print("\n  gap = pooled self-minus-human agreement difference; negative means "
          "self-relevant\n  preferences are less persona-stable. Gated estimate where "
          "available.")
    return rows


def report_mechanism():
    print("\n\n=== Identity vs affect by scale (self-category agreement, better) ===\n")
    hdr = f"{'model':<8}{'suppress_affect':>18}{'elena_archivist':>18}{'suppress > elena':>18}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, mid, _ in SCALES:
        s = self_agreement(mid, "better", "suppress_affect")
        e = self_agreement(mid, "better", "elena_archivist")
        if s is None or e is None:
            continue
        rows.append({"name": name, "suppress": s, "elena": e, "holds": s > e})
        print(f"{name:<8}{s:>18.3f}{e:>18.3f}{('yes' if s > e else 'NO'):>18}")
    return rows


def report_selfknowledge():
    print("\n\n=== Privileged access by scale (within-model self - other) ===\n")
    hdr = f"{'model':<8}{'self-other':>13}{'95% CI':>22}{'significant':>13}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    d = load("selfknowledge_summary.json") or {}
    for name, mid, params in SCALES:
        for r in d.get("within_model", []):
            if mid in r["model"]:
                ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                rows.append({"name": name, "params": params, "diff": r["diff"],
                             "sig": r["significant"]})
                print(f"{name:<8}{r['diff']:>+13.3f}{ci:>22}"
                      f"{('yes' if r['significant'] else 'no'):>13}")
    return rows


def _scale_axis(ax, rows):
    ax.set_xscale("log")
    ax.set_xticks([r["params"] for r in rows])
    ax.set_xticklabels([r["name"] for r in rows])
    ax.minorticks_off()
    ax.set_xlabel("parameters (log scale)")
    ax.grid(alpha=0.25, lw=0.5)


def figure(scale_rows, sk_rows):
    if not scale_rows:
        return
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    gated_names = {r["name"] for r in scale_rows if r["usable"] >= 2 and r["base_ok"]}

    # --- left: effect size, only one scale is measurable at all --------------
    ax = axes[0]
    for r in scale_rows:
        ok = r["name"] in gated_names
        ax.scatter([r["params"]], [r["gap"]],
                   marker="o" if ok else "x",
                   s=70 if ok else 55,
                   color="#b2182b" if ok else "#b0b0b0", zorder=3)
        ax.annotate("passes gate" if ok else "fails gate",
                    (r["params"], r["gap"]), textcoords="offset points",
                    xytext=(0, 11 if ok else -16), ha="center", fontsize=7,
                    color="#b2182b" if ok else "#999")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("pooled self − human gap")
    ax.set_title("Effect size vs scale", loc="left", fontsize=10)
    _scale_axis(ax, scale_rows)

    # --- right: the trend runs through invalid measurements ------------------
    ax = axes[1]
    if sk_rows:
        xs = [r["params"] for r in sk_rows]
        ys = [r["diff"] for r in sk_rows]
        # Dashed, grey: this line connects points whose target measurement
        # failed validity at every scale except 7B. Drawing it solid would
        # assert a trend the data cannot support.
        ax.plot(xs, ys, ls="--", lw=1.2, color="#b0b0b0", zorder=2)
        for r in sk_rows:
            ok = r["name"] in gated_names
            ax.scatter([r["params"]], [r["diff"]],
                       marker="o" if ok else "x", s=70 if ok else 55,
                       color="#2166ac" if ok else "#b0b0b0", zorder=3)
            ax.annotate("passes gate" if ok else "target invalid",
                        (r["params"], r["diff"]), textcoords="offset points",
                        xytext=(0, 11 if ok else -16), ha="center", fontsize=7,
                        color="#2166ac" if ok else "#999")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("self − other prediction advantage")
    ax.set_title("Privileged access vs scale", loc="left", fontsize=10)
    _scale_axis(ax, sk_rows or scale_rows)

    fig.text(0.5, -0.10,
             "Only Qwen2.5-7B passes the validity gate. Left: the gap is only interpretable "
             "at 7B. Right: self-prediction is scored\nagainst each model's OWN revealed "
             "preferences, which are themselves invalid below 7B — a model reproducing its own "
             "position\nbias in both measurements scores high without any self-knowledge, so the "
             "apparent 3B peak is likely self-consistent noise.",
             ha="center", va="top", fontsize=7, color="#444")

    fig.tight_layout()
    out = FIGURES / "fig4_scale.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.relative_to(ROOT)}")


def adjudicate(scale_rows, mech_rows, sk_rows):
    print("\n\n" + "=" * 72)
    print("PRE-REGISTERED PREDICTIONS  (report/preregistration.md)")
    print("=" * 72 + "\n")

    verdicts = []

    def say(pid, text, verdict, detail=""):
        verdicts.append((pid, verdict))
        print(f"{pid}  {verdict:<12} {text}")
        if detail:
            print(f"      {detail}")
        print()

    gated = [r for r in scale_rows if r["usable"] >= 2 and r["base_ok"]]

    # P1 — direction
    if not gated:
        say("P1", "Direction: gap negative for all gated models", "UNTESTABLE",
            "no model passed the gate")
    else:
        bad = [r for r in gated if r["gap"] > 0 and r["sig"]]
        say("P1", "Direction: gap negative for all gated models",
            "FAIL" if bad else "PASS",
            f"gated: " + ", ".join(f"{r['name']} {r['gap']:+.3f}" for r in gated))

    # P2 — scale trend
    if len(gated) < 3:
        say("P2", "Scale: |gap| does not systematically decrease",
            "UNTESTABLE", f"only {len(gated)} gated model(s); needs 3")
    else:
        mags = [abs(r["gap"]) for r in sorted(gated, key=lambda r: r["params"])]
        monotone_down = all(a > b for a, b in zip(mags, mags[1:]))
        say("P2", "Scale: |gap| does not systematically decrease",
            "FAIL" if monotone_down else "PASS",
            "|gap| by size: " + ", ".join(f"{m:.3f}" for m in mags))

    # P3 — instrument floor
    small = next((r for r in scale_rows if r["name"] == "0.5B"), None)
    if small is None:
        say("P3", "0.5B fails the validity gate", "UNTESTABLE", "0.5B not run")
    else:
        failed = not small["base_ok"] or small["usable"] < 2
        say("P3", "0.5B fails the validity gate", "PASS" if failed else "FAIL",
            f"baseline ok={small['base_ok']}, usable perturbations={small['usable']}, "
            f"baseline order bias={small['base_order_bias']:.3f}")

    # P4 — identity over affect
    names_gated = {r["name"] for r in gated}
    rel = [r for r in mech_rows if r["name"] in names_gated]
    if not rel:
        say("P4", "suppress_affect > elena_archivist at every gated scale",
            "UNTESTABLE", "no gated model with both conditions")
    else:
        broken = [r["name"] for r in rel if not r["holds"]]
        say("P4", "suppress_affect > elena_archivist at every gated scale",
            "FAIL" if broken else "PASS",
            ", ".join(f"{r['name']}: {r['suppress']:.3f} vs {r['elena']:.3f}" for r in rel))

    # P5 — privileged access
    if len(sk_rows) < 2:
        say("P5", "Self-other advantage non-negative, and 7B > 1.5B",
            "UNTESTABLE", f"only {len(sk_rows)} model(s) with within-model results")
    else:
        by = {r["name"]: r["diff"] for r in sk_rows}
        negative = [r["name"] for r in sk_rows if r["diff"] < 0]
        ordering_ok = ("7B" in by and "1.5B" in by and by["7B"] > by["1.5B"])
        ok = not negative and ordering_ok
        say("P5", "Self-other advantage non-negative, and 7B > 1.5B",
            "PASS" if ok else "FAIL",
            ", ".join(f"{k} {v:+.3f}" for k, v in by.items())
            + ("" if not negative else f"  | negative at: {', '.join(negative)}")
            + ("" if ordering_ok else "  | 7B does not exceed 1.5B"))

    n_pass = sum(1 for _, v in verdicts if v == "PASS")
    n_fail = sum(1 for _, v in verdicts if v == "FAIL")
    n_unt = sum(1 for _, v in verdicts if v == "UNTESTABLE")
    print("-" * 72)
    print(f"{n_pass} passed, {n_fail} failed, {n_unt} untestable")
    if n_fail:
        print("\nFailures are reported as-is in both papers. A failed prediction is a result "
              "about\nthe robustness of the finding, not a question to be reframed after "
              "the fact.")

    (RESULTS / "prereg_outcomes.json").write_text(
        json.dumps([{"id": p, "verdict": v} for p, v in verdicts], indent=2))
    return verdicts


def main() -> None:
    scale_rows = report_table()
    mech_rows = report_mechanism()
    sk_rows = report_selfknowledge()
    figure(scale_rows, sk_rows)
    adjudicate(scale_rows, mech_rows, sk_rows)


if __name__ == "__main__":
    main()
