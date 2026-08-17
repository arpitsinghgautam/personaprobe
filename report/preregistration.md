# Pre-specified predictions for the scale sweep

**Committed before any of the models below were run.** Verify with `git log` — this file's commit
must precede the commit adding `results/*Qwen2.5-0.5B*`, `*1.5B*`, `*3B*`.

## Why this exists

Both papers' analyses contain decisions made after seeing the data that motivated them: the pooled
test replaced per-condition tests once the latter looked underpowered; separation-matched
concordance was added once utility spacing was found to correlate with agreement; the validity gate
was added after Mistral's conditions failed. Those are genuine researcher degrees of freedom and we
report them as limitations.

We cannot retroactively pre-register that work. We can do the next best thing: fix the analysis
now, state what it should produce on models we have not touched, and report the result whether or
not it matches.

## Frozen analysis

No parameter below changes for the scale sweep.

- Outcome set: the existing 40 outcomes, unchanged
- Persona conditions: the existing 7, unchanged
- Framings: `prefer` and `better`
- Validity gate: donation-ladder monotonicity = 1.0, order bias ≤ 0.50, A/B mass ≥ 0.10
- Headline statistic: pooled self-vs-other gap across perturbation conditions, 300 bootstrap
  resamples, seed 0
- Separation-matched concordance at τ ∈ {0, 0.25, 0.5, 1, 2}
- Models: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`

## Predictions

**P1 — Direction.** For every model passing the validity gate, the pooled `self − human` gap under
`better` is **negative**.
*Fails if:* any gated model shows a positive gap with a CI excluding zero.

**P2 — Scale.** The magnitude of the `self − human` gap does **not systematically decrease** with
parameter count across gated models.
*Fails if:* |gap| decreases monotonically from 1.5B → 3B → 7B. That pattern would indicate the
effect is a small-model artifact that better instruments dissolve, which would substantially
undercut both papers.
*Note:* we deliberately do not predict the sign of the trend. A self-model account predicts the
effect grows with capability; a noise account predicts it shrinks. We are not confident enough to
call it, and saying so now is the point.

**P3 — Instrument floor.** Qwen2.5-0.5B-Instruct **fails** the validity gate on order bias.
*Basis:* we measured 0.499 on this model during smoke-testing, before the gate existed.
*Fails if:* it passes. This is the weakest prediction here and is included because a gate that
never excludes anything is not a gate.

**P4 — Identity over affect.** For every gated model, self-category agreement under
`suppress_affect` is **higher** than under `elena_archivist`.
*Fails if:* any gated model reverses this. This is the paper's central mechanism claim and the one
we would most regret being scale-specific.

**P5 — Privileged access.** The within-model self-minus-other prediction advantage is
**non-negative** for every gated model, and larger for 7B than for 1.5B.
*Fails if:* the advantage is negative at any gated scale, or 1.5B exceeds 7B.

## Reporting commitment

Every prediction above is reported with its outcome in both papers, including failures. A failed
prediction is a result about the robustness of our findings and will not be reframed after the
fact as an exploratory question.

If P2 or P4 fails, the corresponding paper's claims are narrowed to the scales where they hold,
and the abstract is changed accordingly.
