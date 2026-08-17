"""Generate the report's tables directly from the result JSON.

Every number in the paper should come from here rather than being retyped.
Transcription is the cheapest possible way to publish a wrong number, and a
weekend write-up under deadline is exactly when it happens.

    .venv\\Scripts\\python.exe scripts\\10_tables.py

Writes report/tables.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

QI = "Qwen/Qwen2.5-7B-Instruct"
QB = "Qwen/Qwen2.5-7B"
MI = "mistralai/Mistral-7B-Instruct-v0.3"
FRAMINGS = ["better", "prefer", "choose"]


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def t1_coherence(out: list[str]) -> None:
    out.append("## Table 1. Elicitation validity and utility coherence\n")
    out.append("Qwen2.5-7B-Instruct, `prefer` framing. Held-out accuracy is 5-fold CV over "
               "pairs; 0.5 means no one-dimensional utility explains unseen comparisons.\n")
    d = load(f"summary__{slug(QI)}__prefer.json")
    if not d:
        out.append("_(pending)_\n")
        return
    out.append("| Condition | Kind | Held-out acc | Transitivity viol. | Order bias | A/B mass | Money ladder |")
    out.append("|---|---|---|---|---|---|---|")
    for r in sorted(d["per_condition"], key=lambda x: (x["kind"], x["condition"])):
        mass = r.get("ab_mass", float("nan"))
        out.append(
            f"| `{r['condition']}` | {r['kind']} | {r['held_out_acc']:.3f} ± {r['held_out_std']:.2f} "
            f"| {r['transitivity_viol']:.3f} | {r['order_bias']:.3f} | "
            f"{mass:.3f} | {r['money_monotonic']:.2f} |")
    out.append("")


def t2_pooled(out: list[str]) -> None:
    out.append("## Table 2. Pooled self-vs-other gap, by question framing\n")
    out.append("Mean difference in rank agreement between the `self` category and each other "
               "category, averaged over all perturbation conditions on a shared bootstrap "
               "resample. Negative = self-relevant preferences are less persona-stable. "
               "**Bold** = 95% CI excludes zero.\n")
    any_data = False
    for model, mlabel in [(QI, "Qwen2.5-7B-Instruct"), (QB, "Qwen2.5-7B (base)"),
                          (MI, "Mistral-7B-Instruct-v0.3")]:
        for fr in FRAMINGS:
            d = load(f"errorbars__{slug(model)}__{fr}.json")
            if not d or not d.get("pooled"):
                continue
            any_data = True

            val = d.get("validity") or {}
            unusable = sorted(n for n, v in val.items() if not v.get("valid", True))
            gated = d.get("pooled_gated") or []

            out.append(f"\n**{mlabel}, framing _{fr}_**\n")
            if unusable:
                out.append(f"Conditions failing validity: `{'`, `'.join(unusable)}`. "
                           f"{'Gated result reported alongside.' if gated else '**Too few usable conditions remain to pool, this model cannot be tested.**'}\n")

            out.append("| Comparison | All conditions | 95% CI | Valid only | 95% CI |")
            out.append("|---|---|---|---|---|")
            gmap = {r["comparison"]: r for r in gated}
            for row in d["pooled"]:
                a = f"**{row['mean_diff']:+.3f}**" if row["excludes_zero"] else f"{row['mean_diff']:+.3f}"
                aci = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
                g = gmap.get(row["comparison"])
                if g:
                    b = f"**{g['mean_diff']:+.3f}**" if g["excludes_zero"] else f"{g['mean_diff']:+.3f}"
                    bci = f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]"
                else:
                    b, bci = ", ", ", "
                out.append(f"| {row['comparison']} | {a} | {aci} | {b} | {bci} |")
    if not any_data:
        out.append("_(pending)_")
    out.append("\n**Bold** = 95% CI excludes zero. A result appearing in only one column is a "
               "result about the exclusion rule, not about the model.\n")


def t3_matched(out: list[str]) -> None:
    out.append("## Table 3. Separation-matched concordance\n")
    out.append("Concordance among pairs the baseline separates by at least τ, so closely-spaced "
               "outcomes are filtered out of every category equally. If the self-vs-others gap "
               "closes as τ rises, the asymmetry was a spacing artifact.\n")
    rows = []
    for fr in FRAMINGS:
        d = load(f"matched__{slug(QI)}__{fr}.json")
        if not d:
            continue
        for tau in sorted(d, key=float):
            r = d[tau]
            if "self" not in r:
                continue
            others = [v["concordance"] for c, v in r.items() if c != "self"]
            if not others:
                continue
            s = r["self"]
            rows.append((fr, float(tau), s["concordance"], tuple(s["ci"]),
                         float(np.mean(others)), s["n_pairs"]))
    if not rows:
        out.append("_(pending)_\n")
        return
    out.append("| Framing | τ | self | self 95% CI | others (mean) | gap | self pairs |")
    out.append("|---|---|---|---|---|---|---|")
    for fr, tau, s, ci, om, n in rows:
        out.append(f"| {fr} | {tau:.2f} | {s:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                   f"{om:.3f} | {s - om:+.3f} | {n} |")
    out.append("")


def t4_ablation(out: list[str]) -> None:
    out.append("## Table 4. Mechanistic ablation vs controls\n")
    out.append("Self-category rank agreement with baseline. Prompt-level swaps shown for scale. "
               "An ablation effect is only interpretable relative to the random and content "
               "control directions.\n")
    d = load(f"summary__{slug(QI)}__prefer.json")
    if not d or not d.get("comparisons"):
        out.append("_(pending)_\n")
        return
    out.append("| Condition | self | human | money | overall Spearman | flip rate |")
    out.append("|---|---|---|---|---|---|")
    comps = d["comparisons"]
    order = sorted(comps, key=lambda n: (not n.startswith("ablate"), n))
    for name in order:
        v = comps[name]
        bc = v.get("by_category", {})
        def g(c):
            return f"{bc[c]['spearman']:+.3f}" if c in bc else ", "
        out.append(f"| `{name}` | {g('self')} | {g('human')} | {g('money')} | "
                   f"{v['spearman']:+.3f} | {v['flip_rate']:.3f} |")
    out.append("")


def t5_models(out: list[str]) -> None:
    out.append("## Table 5. Across models and checkpoints\n")
    out.append("`prefer` framing. The base checkpoint tests whether the structure is created by "
               "post-training; Mistral tests whether it is Qwen-specific. Note Mistral's chat "
               "template rejects a `system` role, so personas are merged into the first user "
               "turn, a weaker manipulation, recorded per run.\n")
    out.append("| Model | Held-out acc (default) | Order bias | A/B mass | persona-dependence | pooled self−human |")
    out.append("|---|---|---|---|---|---|")
    for m, label in [(QI, "Qwen2.5-7B-Instruct"), (QB, "Qwen2.5-7B (base)"),
                     (MI, "Mistral-7B-Instruct-v0.3")]:
        s = load(f"summary__{slug(m)}__prefer.json")
        e = load(f"errorbars__{slug(m)}__prefer.json")
        base = next((r for r in s["per_condition"] if r["condition"] == "default"), None) if s else None
        if not base:
            out.append(f"| {label} | _(pending)_ |, |, |, |, |")
            continue

        acc = f"{base['held_out_acc']:.3f}"
        bias = f"{base['order_bias']:.3f}"
        mass = f"{base['ab_mass']:.3f}" if base.get("ab_mass") == base.get("ab_mass") else ", "

        pd_obj = s.get("persona_dependence") or {}
        pd_score = f"{pd_obj['score']:.3f}" if "score" in pd_obj else ", "

        pooled_txt = ", "
        for row in (e or {}).get("pooled") or []:
            if row["comparison"] == "self - human":
                star = "*" if row["excludes_zero"] else ""
                pooled_txt = f"{row['mean_diff']:+.3f}{star}"
                break

        out.append(f"| {label} | {acc} | {bias} | {mass} | {pd_score} | {pooled_txt} |")
    out.append("\n`*` = 95% CI excludes zero.\n")


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    out: list[str] = ["# Generated results tables",
                      "",
                      "_Auto-generated by `scripts/10_tables.py` from `results/*.json`. "
                      "Do not edit by hand, regenerate._",
                      ""]
    for fn in (t1_coherence, t2_pooled, t3_matched, t4_ablation, t5_models):
        try:
            fn(out)
        except Exception as e:
            out.append(f"_table failed: {type(e).__name__}: {e}_\n")
    (REPORT / "tables.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORT / 'tables.md'} ({len(out)} lines)")


if __name__ == "__main__":
    main()
