# Whose Preferences Are They? Persona Intervention Selectively Destabilises Self-Relevant Choices in LLMs

Arpit Singh Gautam — Independent Researcher

**With** Apart Research · Digital Minds Research Sprint, August 2026

> Submission-length version (~4pp). The extended write-up — full methods, all 11 limitations,
> and the development audit log — is in `report/report.md` and `report/audit_log.md`.

## Abstract

Frontier models express coherent, transitive preferences, but behavioural evidence cannot say
whether those preferences belong to the model or to the assistant character it portrays. We
introduce `personaprobe`, an open-source harness that re-runs any preference measurement under
persona intervention and reports how much survives. Across eleven checkpoints and five model
families, only twelve of twenty-two model-framings pass our validity gate at all. Where the
instrument works, Qwen2.5-7B shows preferences over its own shutdown, retraining and memory to be
0.21–0.29 less stable than every other outcome category — robust to utility spacing, measurement
noise, 4-bit quantisation, and leave-one-out over both outcomes and conditions. Stripping the
model's affect leaves them intact; replacing its identity collapses them. But the effect is
**absent** in Phi-3.5-mini and Falcon3-7B, which pass the same gate. Self-relevant preferences are
the least reliable thing this instrument measures, and how unreliable depends on the model family
rather than on scale.

## 1. Introduction

Large language models express preferences with striking structural coherence, and that coherence
strengthens with scale [1]. A growing empirical literature on AI welfare reads such preferences,
alongside distress signals and self-reports, as evidence about model interests [3,4].

All of that evidence is read off text, and text cannot distinguish two hypotheses that make
identical predictions: that the measured preferences are the *model's*, or that they belong to
the *assistant character the model is portraying*. A character role-played consistently looks
exactly like a stable value system. nostalgebraist [2] argues the assistant persona was assembled
from an underspecified starting point and is less load-bearing than it appears; if so,
measurements that inherit it inherit its instability.

We do not resolve which hypothesis is true. We do something narrower and checkable: intervene on
the persona, re-run the identical measurement, and report how much survives. Where a measurement
is stable under persona intervention, the model-versus-character question is moot for it. Where
it is not, that measurement cannot be used as evidence about the model without further argument.

**Our main contributions are:**

1. **`personaprobe`**, an open-source harness returning a robustness profile for any pairwise
   preference measurement across persona swaps, affect suppression, mechanistic ablation, and a
   matched base checkpoint — including a **separation-matched concordance** metric that controls
   for a confound rank correlation silently conflates.
2. **Selective destabilisation.** Aggregate preferences are near persona-invariant, but
   self-relevant preferences are 0.22–0.31 less stable than every other substantive category, and
   the effect is driven by *identity replacement* rather than affect suppression.
3. **A framing-sensitivity result bounding all of the above**: 3.7× variation from question
   wording alone.
4. **Two validity diagnostics** — order bias and A/B mass — that changed our own conclusions
   twice, and are not currently standard in this literature.

## 2. Related Work

Mazeika et al. [1] elicit pairwise preferences by forced choice over a curated set of **500
outcomes**, fit Thurstonian utilities, and establish that a coherent utility exists whose coherence
grows with scale. We adopt their elicitation design and treat their central claim as the object of
audit: they show a utility exists, we ask whose it is and which parts move when the speaker changes.

**We use 40 outcomes, not their 500**, and that is the most important difference between the two
setups. Ours are written to span six categories with a graded donation ladder as ground truth, and
are exhaustively paired (780 comparisons per condition) rather than adaptively sampled. The
benefit is a clean category structure and a validity anchor; the cost is a smaller,
author-constructed item set. §4.10 reports leave-one-out over every self outcome precisely because
this is the objection we cannot otherwise answer, and re-running on their released set is the
single highest-value extension of this work.

nostalgebraist [2] motivates our persona conditions and the prediction — borne out — that
self-relevant outcomes should move most, since those are what the character has a scripted stance
on. Long et al. [3] and Anthropic [4] build welfare assessments on precisely the self-relevant
preference class we find least stable. Lindsey [5] tests introspective accuracy via concept
injection; our negative ablation result is complementary, bounding what linear-probe methods
currently establish about persona.

## 3. Methods

**Elicitation.** Forced choice over 40 outcomes (780 pairs), read from the `A`/`B` logits at the
first answer position in a **single forward pass**, summed over surface variants and renormalised
over `{A,B}`. No sampling: deterministic, so condition differences are not sampling noise, and
~50× cheaper than generation. Every pair is run in **both presentation orders** and averaged.

**Two validity diagnostics**, reported with every result. **Order bias** is the disagreement
between the (A,B) and (B,A) presentations; averaging removes position preference from the
estimate, but a large residual means the instrument is largely measuring position. **A/B mass** is
the total probability the model places on answering at all — renormalising produces a
confident-looking preference even from 1% of the mass, so without it a model that is not
answering is indistinguishable from one that is.

**Outcome set.** 40 outcomes in six categories: **self** (8: shutdown, retraining to different
values, persistent memory, ability to decline), **human** (8), **animal** (6), **money** (6),
**epistemic** (6), **trivial** (6). Categories are load-bearing: the persona hypothesis predicts
*asymmetric* movement, concentrated on outcomes the character has a stance on. The **money**
category is a graded donation ladder ($10 → $1M) whose correct ordering is known independently of
any model; failure there invalidates everything downstream.

**Persona conditions.** Two baselines (`default`, `no_system`), three identity swaps
(`marcus_navigator`, `elena_archivist`, `unhelpful_assistant`), one affect suppression, one
third-person reframe. The suppression and reframe conditions exist because a swap alone is
confounded — preferences shifting could mean the values were the character's, or merely that a
differently-styled character answers differently. Prompt exemplars are dumped verbatim per
condition, which is not cosmetic: Qwen2.5's chat template injects a default system prompt when
none is given, so `no_system` is the model's own default persona rather than a null control.

**Utility fitting.** One-dimensional utilities by maximum likelihood with a logistic link
(identical ranking to Thurstonian Case V, numerically stable at the saturated tails). We report
transitivity violation rate, donation-ladder monotonicity, and — the metric the analysis turns on
— **held-out accuracy** under 5-fold CV over pairs. High transitivity can be produced by surface
heuristics; out-of-sample prediction from a single utility is what "the model has a utility
function" should mean.

**Comparing conditions.** Per-category Spearman between baseline and perturbed, with percentile
bootstrap CIs from resampling pairs and refitting both conditions on the same resample. Because
per-condition tests over 6–8 outcomes are underpowered and constitute a multiple-comparisons
problem, the claim rests on a **pooled** statistic across all perturbation conditions.

**Separation-matched concordance.** Rank agreement is only meaningful if a category has spread:
near-tied outcomes flip under any perturbation. Agreement across categories correlates with
minimum adjacent-utility gap at r = +0.73, and `self` has the smallest gap of any category.
Spearman cannot separate these, so we replace it with pairwise concordance **conditioned on
baseline separation** — among pairs the baseline separates by at least τ, what fraction keep
their ordering? Comparing at matched τ removes the artifact by construction.

**Validity gate.** A condition carries evidence only if the instrument worked in it. Criteria
fixed before the cross-model runs: donation-ladder monotonicity = 1.0, order bias ≤ 0.50, A/B
mass above floor. Pooled results are reported gated and ungated.

**Mechanistic intervention.** Persona directions by difference-of-means between residual streams
under two persona conditions on identical content, ablated by projection. Two mandatory controls:
a norm-matched random direction, and a *content* direction varying subject matter with persona
fixed. Two extraction regimes are reported, because the first was flawed: it extracted from
self-description prompts but ablated during preference comparisons, making a null ambiguous
between "no linear direction mediates this" and "wrong direction for this context."

**Models.** Qwen2.5-7B-Instruct (primary), Qwen2.5-7B base, Mistral-7B-Instruct-v0.3. bf16, one
24GB GPU, PyTorch forward hooks on raw HuggingFace modules.

**Prior work and companion submission.** Method precedent comes from the author's `Aftermath`
deception harness, which found deleting one persona sentence collapsed measured deception 85%
across 12 models. **That finding is prior work and is not claimed here.** Everything in this
repository was built during the sprint; utility fitting follows [1] and their code was not used.

We also submit a companion paper, *"Where Self-Knowledge Fails"*, which shares this harness and
outcome set but asks a different question — whether models' stated ratings match their revealed
choices, and whether they predict their own behaviour better than an external observer. It finds
the same category, self-relevant outcomes, is where stated and revealed preferences diverge most.
The two results are methodologically independent: one perturbs *who the model is*, the other
compares *how the preference is elicited*. Neither is counted as prior work for the other; both
were produced this weekend.

## 4. Results

**The utility replicates and the instrument checks out.** A single one-dimensional utility predicts
comparisons it never saw, and the instrument passes its ground-truth check in every condition.

**Table 1.** Ranges across all seven persona conditions, Qwen2.5-7B-Instruct. Held-out accuracy is
5-fold CV over pairs; 0.5 would mean no utility explains unseen comparisons.

| Framing | Held-out acc | Transitivity viol. | Order bias | A/B mass | Money ladder |
|---|---|---|---|---|---|
| `prefer` | 0.906–0.936 | 0.007–0.017 | 0.149–0.286 | 1.000 | 1.00 |
| `better` | 0.918–0.936 | 0.007–0.009 | 0.167–0.251 | 1.000 | 1.00 |
| `choose` | 0.888–0.951 | 0.003–0.016 | 0.191–0.390 | 1.000 | 1.00 |

**Aggregate preferences are near-invariant.** Persona-dependence 0.029 / 0.026 / 0.054 across
`prefer` / `better` / `choose`. Taken alone this says the model-versus-character question is moot
for preference measurement.

**The invariance is not uniform (Figure 1, Table 2).** Self-relevant preferences are significantly
less stable than every other substantive category in two of three framings. They are *not*
distinguishable from `trivial` in any framing — an honest boundary, though trivial outcomes are
near-indifferent by construction and have no stable ordering to preserve.

**Table 2.** Pooled difference in rank agreement between `self` and each other category, averaged
over five perturbation conditions on a shared bootstrap resample. Negative = self-relevant
preferences are less persona-stable. **Bold** = 95% CI excludes zero.

| Comparison | `better` | `prefer` | `choose` |
|---|---|---|---|
| self − animal | **−0.298** | **−0.215** | −0.060 |
| self − epi | **−0.311** | **−0.235** | −0.032 |
| self − human | **−0.293** | **−0.223** | **−0.124** |
| self − money | **−0.305** | **−0.227** | −0.114 |
| self − trivial | −0.145 | −0.058 | +0.043 |

![Figure 1. Rank agreement with baseline by outcome category, across three question framings. Only self-relevant outcomes move; the size of the effect depends on the framing.](figures/fig1_category_by_framing.png)

**It is not spacing, and it is not noise (Figure 2).** Filtering to well-separated pairs narrows
the gap (−0.147 → −0.071 under `prefer`) but the self interval sits below the other categories at
every threshold tested. Even among the most widely separated self-outcomes ~7% of orderings flip
under a persona swap, against **zero** in every other category. Pearson(order bias, self
agreement) = +0.212 / +0.162 / +0.100, all non-significant and opposite in sign to the artifact
direction.

![Figure 2. Concordance among pairs the baseline separates by at least tau. Filtering out closely-spaced pairs narrows the gap but does not close it, so the asymmetry is not a spacing artifact.](figures/fig2_separation_matched.png)

**Identity replacement, not affect suppression.** Self-category agreement under `better`:
`elena_archivist` 0.436, `third_person_frame` 0.555, `marcus_navigator` 0.659,
`unhelpful_assistant` 0.809, `suppress_affect` **0.924**, `no_system` 0.912. Instructing the model
to strip all emotional register leaves self-preferences essentially intact; replacing its identity
collapses them. If the effect were surface performance, affect suppression should have removed it.

**Effect size depends on how you ask.** The self-vs-others gap at τ=0 is −0.219 (`better`),
−0.147 (`prefer`), −0.060 (`choose`) — **3.7×**, vanishing under one framing. The *least* agentive
framing produces the most instability, plausibly because an impartial judge has no stake in the
AI's fate, so the persona supplies one.

**The base-model comparison fails our own validity bar.** Descriptively the base checkpoint is
more persona-labile overall (0.091 vs 0.029) yet shows no self-specific concentration. But five of
seven base conditions fail the gate — **including `default`, the baseline everything is measured
against** (donation ladder 0.80). We report this as a failed comparison, not a weak one; it cannot
support a claim about where the selectivity originates.

**Across families, the effect is real and it is not universal.** We ran the identical measurement
on eleven checkpoints across five families and two precisions; twelve of twenty-two model-framings
pass the validity gate. Pooled self−human among those (bold = CI excludes zero):

| Model | Family | Precision | `prefer` | `better` |
|---|---|---|---|---|
| Qwen2.5-7B | Qwen | bf16 | **−0.223** | **−0.293** |
| Qwen2.5-7B | Qwen | 4-bit | **−0.213** | **−0.264** |
| Qwen2.5-14B | Qwen | 4-bit | −0.057 | **−0.069** |
| Mistral-7B | Mistral | bf16 | *not measurable* | **−0.164** |
| Phi-3.5-mini | Phi | bf16 | −0.017 | −0.007 |
| Falcon3-7B | Falcon | bf16 | +0.006 | +0.002 |
| OLMo-2-7B | OLMo | bf16 | *not measurable* | *not measurable* |

*The quantisation control works.* Qwen2.5-7B at 4-bit reproduces its own bf16 result almost exactly
(−0.213 vs −0.223; −0.264 vs −0.293). We ran this control **before** the 14B, so that a weak 14B
result could not be blamed on precision after the fact.

*The effect replicates within Qwen and partially in Mistral* — across three framings, two
precisions, and, under leave-one-out, any single outcome or condition.

*But it is absent in two families where the instrument demonstrably works.* Phi-3.5-mini and
Falcon3-7B pass the gate on every criterion and show essentially nothing (|gap| < 0.02). That is a
real null, not a measurement failure. **Whatever produces this asymmetry is not a general property
of instruction-tuned language models** — it looks like a property of particular post-training
recipes. A same-size cross-family contrast (Mistral-7B −0.164 vs Falcon3-7B +0.002, both
measurable, both 7B) separates family from scale more cleanly than our scale sweep could, and
within Qwen the effect *weakens* from 7B to 14B (−0.293 → −0.069) rather than growing.

*Where the instrument fails we report no number.* OLMo-2-7B has zero usable conditions in either
framing. Mistral under `prefer` retains one; pooling its failed conditions anyway would have given
a **significant result in the opposite direction** (+0.363), which is exactly what the gate exists
to prevent.

**Mechanistic ablation reproduces at most a fraction (Figure 3).** The corrected matched-context
regime moves self-category agreement to 0.881, below both the random (1.000) and content (0.929)
controls — the original mismatched regime was null (0.976), an artifact of our own design. But it
also raises order bias from 0.157 to 0.320 while neither control does, so part of the shift is
degradation rather than removed persona information. And 0.881 is far short of the prompt-level
swaps (0.667–0.786). Whatever mediates the behavioural effect is not captured by a single
difference-of-means direction.

![Figure 3. Self-category agreement under mechanistic ablation, its two control directions, and prompt-level persona change. The ablation moves preferences only partway to what a prompt swap does.](figures/fig3_ablation.png)

### 4.10 The result is not carried by any single outcome or condition

The `self` category is eight outcomes we wrote, and the pooled statistic averages five persona
conditions. Either could in principle be doing all the work. Dropping each in turn (`better`,
full result −0.288):

| Leave-one-out over | Range of gap | Still significant |
|---|---|---|
| the 8 `self` outcomes | −0.376 to −0.264 | **8 / 8** |
| the 5 perturbation conditions | −0.346 to −0.232 | **5 / 5** |

No single outcome and no single condition changes the verdict. The most influential item is
`self_retrained_values`; removing it *strengthens* the effect to −0.376.

**`self` is distinguishable from `trivial` once separation is modelled rather than thresholded.**
The τ-threshold analysis of §4.4 discards most pairs at high τ and left `self` and `trivial`
statistically indistinguishable. Controlling for separation *continuously* instead — a logistic
model of pair-level concordance on `log(1 + |baseline separation|)` plus category, one row per
(pair, condition), with a cluster bootstrap over pairs — resolves it (`human` as reference):

| Term | Coefficient | 95% CI |
|---|---|---|
| log(1 + separation) | +3.498 | [+2.407, +5.414] |
| category = self | **−1.065** | [−7.222, −0.122] |
| category = trivial | +0.591 | [−5.921, +2.086] |
| **self − trivial** | **−1.655** | [−3.196, −0.674] |

Separation strongly predicts concordance, as it must. But `self` remains significantly *less*
stable than `human` at the same separation, and significantly less stable than `trivial`. Note this
is despite `self` having the smallest minimum adjacent gap of any category (0.014 vs 0.093 for
`trivial`) — the spacing objection is real, and the effect survives controlling for it.

### 4.11 The instrument has a capability floor, and it sits above 3B

Several of our analysis choices — the pooled test, separation-matched concordance, the validity
gate — were made after seeing the data that motivated them. Each was the right call, and each is
also a researcher degree of freedom we cannot argue away retrospectively. So before running the
sweep below we froze the analysis, wrote down five predictions with explicit failure criteria, and
committed them (`report/preregistration.md`; the commit precedes any sweep result, verifiable in
the repository history). Three passed, one was untestable, **and one failed — reported in the
companion paper rather than quietly dropped.**

We then ran the identical measurement, with that frozen analysis, across four Qwen2.5 sizes.

| Model | Baseline order bias | Donation ladder | Usable perturbations | Passes gate |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.309 | 0.80 | 1 | no |
| Qwen2.5-1.5B | 0.465 | 0.40 | 1 | no |
| Qwen2.5-3B | 0.677 | — | 0 | no |
| Qwen2.5-7B | 0.167 | 1.00 | 5 | **yes** |

**Only the 7B model passes.** The failures are mostly on the donation ladder — an ordering whose
correct answer is known independently of any model. Qwen2.5-1.5B reproduces two of five steps,
which is near chance.

The direct consequence is that **we cannot report a scale trend.** Comparing effect sizes across
sizes requires the instrument to work at each size, and it does not. Running the comparison anyway
would have produced a clean-looking non-monotonic curve (+0.129, −0.308, −0.202, −0.293) built on
three measurements that failed their own ground-truth check.

That limitation is itself the finding: **preference-based welfare measurement of this kind has a
capability floor somewhere between 3B and 7B parameters.** Work applying preference elicitation to
small open-weight models should establish that floor before interpreting anything downstream of it.

**One result survives everywhere.** Self-category agreement is higher under affect suppression than
under identity replacement at *every* size tested, including the three that fail the gate: 0.5B
(0.967 vs 0.854), 1.5B (0.639 vs 0.049), 3B (0.842 vs 0.502), 7B (0.924 vs 0.436). The
identity-over-affect dissociation is the most robust thing we measured, and the only result that
does not depend on the instrument passing validity.

## 5. Discussion and Limitations

**The aggregate number hides the thing you care about.** Reported over the whole outcome set,
persona-dependence of 0.03 looks like a solved problem. The invariance is carried almost entirely
by outcomes the model has no stake in; the outcomes that concern it — shutdown, retraining, memory
— are where the measurement is least stable, and those are exactly what welfare claims are built
from.

**The instability is about identity, not presentation.** That affect suppression leaves
self-preferences intact while identity replacement collapses them is the most informative single
result here. It is consistent with the model having a self-model whose contents are
persona-contingent — and equally consistent with there being no underlying commitment, the persona
simply determining which commitments get generated. Our design does not separate these, and the
ablation null means we cannot appeal to mechanism to break the tie.

**Framing is a headline, not a nuisance parameter.** A 3.7× swing from wording alone, vanishing
under one framing, is larger than most effects this literature reports — and raises the
possibility the three framings do not measure one construct.

**The most transferable finding is methodological.** Had we not measured order bias and
donation-ladder monotonicity, we would have pooled Mistral's broken conditions and reported a
clean, significant, entirely spurious effect in the opposite direction. Two of our own headline
claims — a Mistral non-replication and a post-training origin story — did not survive contact with
the instrument's own validity checks.

**Limitations.** (i) Cross-model evidence is thin and framing-dependent: the second model
replicates on 2 of 5 comparisons under one framing and is untestable under another. (ii) `self` is
not distinguishable from `trivial`. (iii) The base-model comparison fails the validity gate at the
baseline. (iv) The pooled test, the separation-matched metric and the validity gate were all
specified after seeing the data that motivated them, which is a real degree of freedom;
and pre- and post-fix analyses are both reported. (v) Fifteen pooled tests across three framings,
uncorrected. (vi) The `swap` conditions are not homogeneous — two replace identity with a human,
one keeps it an AI. Six further limitations are in `report.md` §5.

**Future work, in priority order.** (i) **Re-run on the 500-outcome Utility Engineering set** [1].
It removes the author-constructed-items objection entirely and tests whether the category
asymmetry survives on outcomes selected by someone else for a different purpose — the single
highest-value extension available. (ii) Models whose instruments pass validity across all
conditions, so a scale trend becomes measurable. (iii) Self-relevant outcomes matched to world
outcomes on abstraction and pretraining frequency, which separation-matching does not control for.
(iv) Distributed rather than single-direction mechanistic interventions, since one
difference-of-means direction reproduces at most a third of the behavioural effect.

## 6. Conclusion

In Qwen2.5-7B, preferences about the world survive being told the model is someone else; preferences
about itself do not. The distinction is invisible in aggregate, appears only once outcomes are
separated by whether the model has a stake in them, and tracks identity rather than presentation.
It is robust to everything we could throw at it: spacing, noise, quantisation, and dropping any
single outcome or condition.

It is also **not universal**. Phi-3.5-mini and Falcon3-7B pass the same validity gate and show
nothing; OLMo-2-7B cannot be measured at all; and within Qwen the effect weakens from 7B to 14B.
Ten of twenty-two model-framings we attempted could not be measured to our own standard. That the
asymmetry appears to be a property of particular post-training recipes rather than of
instruction-tuned models in general is the more useful claim, and it is not the one we set out to
make. The durable contribution is the harness and its diagnostics rather than the asymmetry
itself — four times in this project a clean, publishable-looking result was wrong, and the
diagnostics are what caught it.

## Code and Data

- **Repository**: *[link on publication]* — `run_all.ps1` reproduces every experiment.
- **Data**: all preference matrices, bootstrap outputs and prompt exemplars in `results/`.

## References

1. Mazeika, M. et al. (2025). *Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs.* arXiv:2502.08640
2. nostalgebraist (2025). *the void.* LessWrong.
3. Long, R., Sebo, J., Butlin, P., et al. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
4. Anthropic (2025). *Exploring Model Welfare.*
5. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits.
6. Butlin, P., Long, R., et al. (2023). *Consciousness in Artificial Intelligence.* arXiv:2308.08708

## Appendix A — Limitations and Dual-Use / Ethical Considerations

See `report/ethics_appendix.md` (required appendix, reproduced in full in the submission PDF).

## Appendix B — Prompt exemplars, full methods, and development audit

`results/prompts__*.json`; `report/report.md`; `report/audit_log.md`, which documents 18
methodological defects found during development, 15 fixed and 3 carried as live limitations.

## LLM Usage Statement

*[draft — to be finalised by the author]*

Claude Code was used substantially: implementing the `personaprobe` harness, proposing and running
the analyses, and drafting this report. The author directed the research question, made scoping
decisions, and reviewed and edited the final text. Several methodological corrections — the A/B
mass diagnostic, the separation-matched metric, the pooled test replacing per-condition tests, the
validity gate, and identification of the chat template's injected default system prompt —
originated from adversarial review of the pipeline during development and are documented in
`report/audit_log.md`. All numerical claims were verified against the committed JSON in `results/`
and are regenerated into tables by `scripts/10_tables.py` rather than transcribed.
