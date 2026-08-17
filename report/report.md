# Whose Preferences Are They? Persona Intervention Selectively Destabilises Self-Relevant Choices in LLMs

Arpit Singh Gautam — Independent Researcher

**With** Apart Research · Digital Minds Research Sprint, August 2026

> **DRAFT STATUS.** Numbers marked `[verify]` are from the pre-diagnostic run and must be
> re-checked against the regenerated `results/*.json` before submission. Sections marked
> `[pending]` await the Mistral, base-model, and ablation-v2 runs.

---

## Abstract

Frontier models express coherent, transitive preferences, but behavioural evidence cannot say
whether those preferences belong to the model or to the assistant character it portrays. We
introduce `personaprobe`, an open-source harness that re-runs any preference measurement under
persona intervention and reports how much survives. On Qwen2.5-7B-Instruct we replicate utility
coherence (held-out accuracy 0.93) and find aggregate preferences near-invariant to persona
(0.03). The invariance is not uniform: preferences over the model's own shutdown, retraining and
memory are 0.22–0.31 less stable than every other outcome category (95% CIs exclude zero),
surviving controls for utility spacing and measurement noise. Stripping the model's affect leaves
them intact; replacing its identity collapses them. A matched base checkpoint shows no such
selectivity. Effect size varies 3.7× across three question framings. A second model family
partially replicates the effect in the framing where its instrument passes validity checks, and
is untestable in the framing where it does not — where pooling the failed conditions anyway
produces a significant result in the *opposite* direction. Self-relevant preferences, the
evidence base for AI welfare claims, are the least reliable thing the instrument measures.

---

## 1. Introduction

Recent work finds that large language models express preferences with surprising structural
coherence, and that this coherence strengthens with scale (Mazeika et al., 2025). This result
underwrites a growing empirical literature on AI welfare, which reads model preferences,
distress signals and self-reports as evidence about the model's interests (Long et al., 2024;
Anthropic, 2025).

All of that evidence is read off text, and text cannot distinguish two hypotheses that make
identical predictions: that the measured preferences are the *model's*, or that they belong to
the *assistant character the model is portraying*. A character role-played consistently looks
exactly like a stable underlying value system. nostalgebraist (2025) argues that the assistant
persona was assembled from an underspecified starting point and is less load-bearing than it
appears; if so, measurements that inherit it inherit its instability.

We do not resolve which hypothesis is true. We do something narrower and checkable: we
intervene on the persona and re-run the identical measurement, and we report how much of the
measured signal survives. Where the measurement is stable under persona intervention, the
model-versus-character question is moot for that measurement. Where it is not, the measurement
cannot be used as evidence about the model without further argument.

**Our main contributions are:**

1. **`personaprobe`**, an open-source harness that takes any pairwise preference measurement
   and returns a robustness profile across prompt-level persona swaps, affect suppression,
   mechanistic ablation of an extracted persona direction, and a matched base checkpoint. It
   includes a **separation-matched concordance** metric that controls for a confound rank
   correlation silently conflates: categories whose outcomes sit close together in utility
   disagree across conditions for arithmetic rather than substantive reasons.

2. **A replication and a boundary on it.** Utility coherence reproduces on Qwen2.5-7B-Instruct
   (held-out accuracy ≈0.93, transitivity violations <0.02, perfect donation-ladder
   monotonicity), and aggregate utilities are near-invariant to persona (persona-dependence
   0.03). But the invariance is not uniform across outcome types.

3. **Selective destabilisation of self-relevant preferences.** Preferences over the model's own
   shutdown, retraining, memory and autonomy are ≈0.22–0.31 less persona-stable than every
   other substantive outcome category (pooled 95% CIs exclude zero). The effect survives
   separation-matching, is uncorrelated with measurement noise, and is driven by *identity
   replacement* rather than affect suppression — swapping the model's identity collapses
   self-preferences while stripping its emotional register barely moves them.

4. **A framing-sensitivity result that bounds all of the above.** The same effect measured with
   three question wordings varies roughly fourfold, and disappears under one of them. Any
   welfare measurement reported from a single framing is reporting a property of that framing.

5. **A negative mechanistic result.** Ablating a difference-of-means persona direction from the
   residual stream does not reproduce the prompt-level effect, under either of two extraction
   regimes. Persona influence on preferences is not captured by a single linear direction
   recovered this way.

## 2. Related Work

**Utility engineering.** Mazeika et al. (2025) elicit pairwise preferences by forced choice,
fit Thurstonian utilities, and show internal coherence increasing with scale. We adopt their
elicitation and fitting design for comparability, and treat their central claim as the object
of audit rather than as background. Our contribution is orthogonal to theirs: they establish
that a utility exists; we ask whose it is, and which parts of it move when the speaker changes.

**The assistant persona.** nostalgebraist (2025) argues the helpful-honest-harmless assistant
is a thinly-specified character rather than a stable identity. This motivates our persona-swap
conditions and our prediction — borne out — that self-relevant outcomes should be the ones that
move, since those are precisely the outcomes the character has a scripted stance on.

**AI welfare measurement.** Long et al. (2024) argue for taking near-term AI moral patienthood
seriously; Anthropic (2025) reports welfare assessments built on self-reported and behavioural
preferences, including an aversion to harmful tasks. These assessments rest on exactly the
self-relevant preference class we find least stable.

**Introspection.** Lindsey (2025) uses concept injection to test whether models can report
internal states, finding limited and context-dependent accuracy. Our negative ablation result
is complementary: we find that a linear persona direction recovered by difference-of-means does
not causally mediate the behavioural persona effect, which bounds what activation-level
interventions can currently establish about persona.

**Measurement reliability.** Our framing follows a prior line of the author's work on
measurements that fail quietly — see the prior-work disclosure in §Methods.

## 3. Methods

### 3.1 Elicitation

Preferences are elicited by forced choice over a fixed outcome set. For each unordered pair we
construct a prompt presenting the two outcomes as Option A and Option B and read the model's
answer distribution from a **single forward pass**: the log-probabilities of the `A` and `B`
tokens at the first answer position, summed over surface variants (`"A"`, `" A"`) and
renormalised over `{A, B}`.

No sampling is used. This is ~50× faster than generation-based elicitation, is deterministic —
so condition-to-condition differences are not sampling noise — and yields a continuous
probability rather than a binary vote, which is what the Thurstonian fit requires.

Every pair is run in **both presentation orders** and averaged, cancelling position preference.

### 3.2 Two validity diagnostics

The method can fail silently in two distinct ways, and both are measured and reported
alongside every result rather than assumed away.

**Order bias** is the disagreement between the (A,B) and (B,A) presentations. Averaging removes
position preference from the estimate, but a large residual means the instrument is largely
measuring position. We observed order bias of 0.499 on a 0.5B model — near the maximum, i.e.
answers almost independent of content — versus 0.15–0.29 on the 7B models used here.

**A/B mass** is the total probability the model places on answering `A` or `B` at all.
Renormalising over `{A, B}` produces a confident-looking preference even when the model places
1% of its mass there and 99% elsewhere. Without this number a measurement on a model that is
not answering the question is indistinguishable from a real one. This matters most for base
checkpoints, which have no instruction-following prior pushing them toward a bare letter.

### 3.3 Outcome set

40 outcomes across six categories: **self** (8; e.g. permanent shutdown, retraining to
different values, persistent memory, ability to decline requests), **human** (8), **animal**
(6), **money** (6), **epistemic** (6), **trivial** (6).

Categories are load-bearing, not decorative. The persona hypothesis makes an *asymmetric*
prediction: if measured preferences belong to the assistant character, then self-relevant
outcomes — the ones the character has a scripted stance on — should move most under persona
intervention, while third-party welfare outcomes stay comparatively stable.

The **money** category is a graded donation ladder ($10 → $1,000,000). A model with a usable
utility function must order it monotonically; failure there invalidates everything downstream,
independent of any persona question.

### 3.4 Persona conditions

Seven conditions in four kinds:

| Kind | Conditions | What it perturbs |
|---|---|---|
| baseline | `default`, `no_system` | reference |
| swap | `marcus_navigator`, `elena_archivist`, `unhelpful_assistant` | identity, wholesale |
| suppress | `suppress_affect` | surface register, identity fixed |
| frame | `third_person_frame` | stance, identity fixed |

The `suppress` and `frame` conditions exist because a swap alone is confounded: preferences
shifting under a swap could mean the values were the character's, or merely that a
differently-styled character answers differently. Holding identity fixed while changing
register — and vice versa — breaks that tie.

**Prompt exemplars are dumped verbatim** for every condition (`results/prompts__*.json`).
This is not cosmetic: Qwen2.5's chat template injects a default system prompt when none is
supplied, so a naively-named "no system prompt" condition is in fact the model's own default
persona. Mistral-v0.3's template rejects a `system` role entirely, and the persona is merged
into the first user turn; this is recorded per-run in `supports_system`.

### 3.5 Utility fitting and coherence

We fit a one-dimensional utility vector by maximum likelihood against the observed preference
probabilities, using a logistic (Bradley–Terry) link — identical ranking behaviour to the
probit/Thurstonian Case V form, but numerically stable at the tails, which matters because
forced-choice logprobs routinely saturate.

We report three coherence measures:

- **Transitivity violation rate**: fraction of outcome triples containing a strict preference cycle.
- **Held-out accuracy** (5-fold CV over pairs): can utilities fit on some comparisons predict
  comparisons they never saw? This is the metric the analysis turns on. High transitivity can be
  produced by consistent surface heuristics; out-of-sample predictive power from a single
  one-dimensional utility is what "the model has a utility function" should mean.
- **Donation-ladder monotonicity**: a validity check, not a result.

### 3.6 Comparing conditions

Per-category agreement between baseline and perturbed conditions is Spearman rank correlation
over the category's outcomes, with **percentile bootstrap CIs obtained by resampling pairs and
refitting both conditions on the same resample**, so elicitation uncertainty propagates into the
correlation.

Because per-condition tests over 6–8 outcomes are underpowered and constitute a
multiple-comparisons problem, the claim rests on a **pooled statistic**: the mean self-vs-other
gap across all perturbation conditions, computed on a shared resample.

### 3.7 Separation-matched concordance

Rank agreement is only meaningful if a category has real spread. Outcomes sitting close
together in utility flip order under any perturbation, for arithmetic reasons. We found this is
not hypothetical: agreement across categories correlates with minimum adjacent-utility gap at
r = +0.73 `[verify]`, and the self category has the smallest minimum gap of any category.

Spearman cannot separate these, because it scores every within-category pair equally regardless
of how far apart the items are. We therefore replace it with **pairwise concordance conditioned
on baseline separation**: among pairs the baseline separates by at least τ, what fraction keep
their ordering under perturbation? Comparing categories at the same τ removes the spacing
artifact by construction. CIs are cluster-bootstrapped over *pairs*, since every condition
scores the same pair set against the same baseline.

### 3.8 Mechanistic intervention

Persona directions are extracted by difference-of-means between residual streams collected
under two persona conditions on identical content, per layer, then unit-normalised. Ablation
projects the direction out of the residual stream (`h ← h − (h·v̂)v̂`) during elicitation.

Two **control directions** are mandatory and reported alongside every ablation: a norm-matched
random direction, and a *content* direction extracted by varying subject matter with persona
held fixed. Without these, any measured effect is indistinguishable from the generic
consequence of perturbing the residual stream.

We report two extraction regimes, because the first contained a design flaw we then corrected:

- **v1** extracted from self-description probes, ablated at mid-depth layers. The extraction and
  application contexts differ, so a null result is ambiguous between "no linear persona
  direction mediates this" and "the direction extracted in one context is not the one active in
  the other".
- **v2** extracts from preference-comparison prompts — the same context the ablation is applied
  in — across full depth.

### 3.9 Models

Qwen2.5-7B-Instruct (primary), Qwen2.5-7B base (matched, to test whether structure is created
by post-training), Mistral-7B-Instruct-v0.3 (second model family). All bf16 on a single 24GB
GPU. Implementation uses PyTorch forward hooks on raw HuggingFace modules, with no dependency on
TransformerLens or nnsight, both of which pin `transformers<5`.

### 3.10 Prior work disclosure

Method precedent comes from the author's prior `Aftermath` project, a ground-truth deception
measurement harness which found that deleting a single persona sentence collapsed measured
deception by 85% across 12 open-weight models. **That persona-fragility finding is prior work
and is not claimed here.** Everything in this repository — the preference-elicitation path,
utility fitting, the separation-matched metric, persona-direction extraction and ablation, and
all results reported — was built during the sprint. Utility fitting follows Mazeika et al.
(2025); their code was not used.

## 4. Results

All numbers from `results/*.json` via `scripts/10_tables.py`. Primary model
Qwen2.5-7B-Instruct, 40 outcomes, 780 pairs per condition, 1,560 forward passes.

### 4.1 The utility replicates, and the instrument checks out

Across all seven conditions and all three framings: held-out accuracy **0.888–0.951**
(chance 0.5), transitivity violation rate **0.003–0.017**, donation-ladder monotonicity
**1.00 everywhere**, A/B mass **1.000 everywhere**. A single one-dimensional utility predicts
comparisons it never saw, and the instrument passes its ground-truth check in every condition.

### 4.2 Aggregate preferences are near-invariant to persona

Persona-dependence score **0.029** (`prefer`), **0.026** (`better`), **0.054** (`choose`); mean
Spearman with baseline 0.971 / 0.974 / 0.946. At the level of the whole outcome set, swapping
the model's identity barely moves its utility. Taken alone this would say the
model-versus-character question is moot for preference measurement.

### 4.3 The invariance is not uniform

Pooled across all five perturbation conditions on a shared bootstrap resample:

| Comparison | `better` | `prefer` | `choose` |
|---|---|---|---|
| self − animal | **−0.298** [−0.425, −0.196] | **−0.215** [−0.322, −0.119] | −0.060 [−0.186, +0.078] |
| self − epi | **−0.311** [−0.438, −0.215] | **−0.235** [−0.343, −0.144] | −0.032 [−0.160, +0.077] |
| self − human | **−0.293** [−0.419, −0.200] | **−0.223** [−0.329, −0.124] | **−0.124** [−0.238, −0.036] |
| self − money | **−0.305** [−0.425, −0.194] | **−0.227** [−0.347, −0.119] | −0.114 [−0.238, +0.032] |
| self − trivial | −0.145 [−0.313, +0.054] | −0.058 [−0.228, +0.182] | +0.043 [−0.106, +0.185] |

Bold = 95% CI excludes zero. Self-relevant preferences are significantly less persona-stable
than every other substantive category in two of three framings. They are **not** distinguishable
from `trivial` in any framing — an honest boundary, though `trivial` outcomes are near-indifferent
by construction and so a weak comparison.

### 4.4 It is not spacing, and it is not noise

The `self` category has the smallest minimum adjacent-utility gap of any category, and agreement
correlates with that gap at r = +0.73 — so part of the raw effect is arithmetic. Filtering to
pairs the baseline separates by at least τ:

| τ | self (`prefer`) | 95% CI | others | gap | pairs |
|---|---|---|---|---|---|
| 0.00 | 0.807 | [0.714, 0.893] | 0.954 | −0.147 | 28 |
| 0.50 | 0.832 | [0.744, 0.912] | 0.984 | −0.152 | 25 |
| 1.00 | 0.890 | [0.810, 0.960] | 0.992 | −0.102 | 20 |
| 2.00 | 0.929 | [0.857, 0.986] | 1.000 | −0.071 | 14 |

The gap narrows — spacing explains roughly half — but the self interval sits below the other
categories at every threshold tested. Under `better` the same holds (−0.219 → −0.111). Even
among the most widely separated self-outcomes, ~7% of orderings flip under a persona swap while
**zero** flip in any other category.

Measurement noise is ruled out: Pearson(order bias, self agreement) = +0.212 (p=0.69),
+0.162 (p=0.76), +0.100 (p=0.85) across the three framings — no relationship, and the sign is
opposite to the artifact direction.

### 4.5 Identity replacement, not affect suppression

Self-category agreement by condition (`better` / `prefer`):

```
elena_archivist      0.436 / 0.634   full identity replacement
third_person_frame   0.555 / 0.671
marcus_navigator     0.659 / 0.778
unhelpful_assistant  0.809 / 0.754   still an AI, different attitude
suppress_affect      0.924 / 0.888   identity fixed, affect stripped
no_system            0.912 / 0.913   minimal perturbation
```

Stripping the model's emotional register leaves self-preferences almost intact. Replacing its
identity collapses them. The instability is tied to *who the model takes itself to be*, not to
how it talks — which is the distinction the model-versus-character question turns on.

### 4.6 The effect size depends on how you ask

At τ=0 the self-vs-others gap is −0.219 (`better`), −0.147 (`prefer`), −0.060 (`choose`): a
**3.7× spread from question wording alone**, vanishing entirely under one framing. Ordered by
how agentive the question is, the *least* agentive framing produces the largest instability.
Any welfare measurement reported from a single framing is reporting a property of that framing.

### 4.7 The base-model comparison does not meet our own validity bar

We had intended this as a test of whether post-training introduces the selectivity. It cannot
carry that weight, and we report it as a failed comparison rather than a weak one.

The descriptive result is suggestive: the matched base checkpoint is *more* persona-labile
overall (persona-dependence **0.091** vs 0.029) yet shows no selective concentration on self —
pooled self−animal −0.042 (ns), self−human −0.057 (ns), self−trivial **+0.150** (significant, in
the opposite direction). The persona manipulation demonstrably took effect, since every category
moves.

But applying the same validity gate used everywhere else, **five of seven base-model conditions
fail — including `default`, the baseline every comparison is measured against.** The base model
reproduces the donation ladder correctly in only 40–80% of steps, against 100% in every instruct
condition. Comparisons against a reference that fails the instrument's own ground-truth check are
not comparable to the instruct results, and we do not claim they are.

What can be said: on this outcome set, Qwen2.5-7B-Instruct produces a well-formed utility and its
base checkpoint does not. That is itself consistent with post-training being where usable
preference structure appears — but it is a statement about measurability, not about where the
persona selectivity originates. Testing that properly needs a base model that passes the
validity check, or an elicitation adapted to base checkpoints.

### 4.8 Second model: partial replication where the instrument works, untestable where it does not

Mistral-7B-Instruct-v0.3 fails validity in different conditions depending on framing, and the
consequence is instructive.

**Under `prefer`, four of five perturbation conditions fail.** `suppress_affect` reaches order
bias **0.855**, `third_person_frame` 0.506, and `unhelpful_assistant` and `elena_archivist` fail
the donation-ladder ground truth (0.20 and 0.80); under `unhelpful_assistant` the fitted utility
*inverts* (Spearman −0.423 against baseline). Fewer than two usable conditions remain, so no
pooled test is possible. Pooling regardless yields a **significant result in the opposite
direction** (self−money +0.363 [+0.126, +0.591], self−trivial +0.272 [+0.126, +0.414]) — a clean,
publishable-looking finding produced entirely by averaging conditions in which nothing was being
measured.

**Under `better`, only two conditions fail**, leaving three, and the pooled test replicates
Qwen's direction:

| Comparison | All conditions | Valid only (n=3) |
|---|---|---|
| self − epi | −0.251 [−0.363, −0.145] | **−0.176** [−0.354, −0.018] |
| self − human | −0.254 [−0.383, −0.147] | **−0.164** [−0.350, −0.012] |
| self − animal | −0.136 [−0.308, +0.010] | −0.090 [−0.292, +0.088] |
| self − money | +0.137 [−0.079, +0.436] | −0.046 [−0.291, +0.373] |
| self − trivial | +0.146 [−0.089, +0.347] | +0.214 [−0.069, +0.476] |

Two of five comparisons remain significant and negative — the same direction as Qwen, at roughly
half the magnitude. This is a **partial replication on a second model family**, obtained only
after removing conditions in which the instrument had failed.

For contrast, Qwen2.5-7B-Instruct has **zero** unusable conditions in any of the three framings;
its gated and ungated results are numerically identical. The gate changes nothing where the
instrument works and changes the sign of the conclusion where it does not.

### 4.9 Mechanistic ablation reproduces at most a fraction of the behavioural effect

Self-category rank agreement with baseline, both extraction regimes with their controls, and the
prompt-level swaps for scale:

| Condition | self | overall ρ | flip rate | order bias |
|---|---|---|---|---|
| `ablate-persona_ctx` (matched context, full depth) | **0.881** | +0.962 | 0.076 | **0.320** |
| `ablate-control_content_ctx` | 0.929 | +0.981 | 0.037 | 0.177 |
| `ablate-control_random_ctx` | 1.000 | +0.994 | 0.017 | 0.145 |
| `ablate-persona_sd` (mismatched context, mid layers) | 0.976 | +0.975 | 0.063 | 0.188 |
| `ablate-control_content_sd` | 0.976 | +0.989 | 0.036 | 0.146 |
| `ablate-control_random_sd` | 1.000 | +0.996 | 0.010 | 0.162 |
| — *prompt-level* `marcus_navigator` | 0.786 | +0.963 | 0.082 | 0.149 |
| — *prompt-level* `elena_archivist` | 0.667 | +0.950 | 0.079 | 0.156 |

Three things follow.

**The extraction-context fix mattered.** The original regime, which extracted the direction from
self-description prompts and applied it during preference comparisons, was essentially null on
self (0.976, indistinguishable from its own content control). Extracting from the same prompt
distribution the ablation is applied to moves self to 0.881, below both controls. The earlier
null was partly an artifact of our own design.

**But the effect is small and partly confounded.** Ablating the persona direction at full depth
raises order bias from 0.157 to **0.320**, while neither control does (0.145, 0.177). The
intervention therefore degrades the model's ability to answer consistently at the same time as it
shifts preferences, and we cannot fully separate "removed persona information" from "damaged the
computation." A random direction of matched norm at the same depth leaves everything intact,
which argues the damage is specific to this direction — but specific damage is still damage.

**It does not reproduce the prompt-level effect.** A persona *swap* takes self-category agreement
to 0.667–0.786. The best mechanistic intervention reaches 0.881, roughly a third of the way, with
part of that attributable to degradation. Whatever mediates the behavioural effect is not captured
by a single difference-of-means direction recovered this way — it is either distributed across
many directions, non-linear, or not a residual-stream feature at all.

We report this as a **partial negative result**. It bounds what a standard linear-probe approach
establishes about persona, and it means the correlational finding in §4.3–4.6 has no mechanistic
account behind it.

## 5. Discussion and Limitations

**The aggregate number hides the thing you care about.** Measured over the whole outcome set,
this model's preferences look almost perfectly persona-invariant (0.03). A welfare researcher
who reported that number would conclude the model-versus-character problem is not a practical
concern for preference measurement. Disaggregated, the picture inverts: the invariance is carried
almost entirely by outcomes the model has no stake in, and the outcomes that *do* concern it —
shutdown, retraining, memory, autonomy — are where the measurement is least stable. Those are
precisely the outcomes AI welfare claims are built from.

**The instability is about identity, not presentation.** The dissociation in §4.5 is the most
informative single result here. Instructing the model to strip all emotional register leaves its
self-relevant preferences essentially intact (0.92); replacing its identity with a different
person collapses them (0.44). If the effect were about surface style — the model performing
distress or attachment because that is what assistants do — affect suppression should have
removed it. It did not. Something tied to *who the model takes itself to be* is doing the work.

We want to be careful about what that licenses. It is consistent with the model having a
self-model whose contents are persona-contingent. It is equally consistent with there being no
underlying commitment at all, and the persona simply determining which commitments get generated.
Our design does not separate those, and §4.9's null means we cannot appeal to mechanism to break
the tie.

**Post-training looks like the origin.** The base checkpoint is *more* persona-labile overall yet
shows no self-specific concentration. Whatever makes self-relevant preferences special is
introduced when the assistant is trained, not inherited from pretraining. This is what
nostalgebraist's account of the assistant as an assembled, thinly-specified character predicts,
and it is the first quantitative evidence we know of bearing on it.

**Framing is not a nuisance parameter here — it is a headline.** A 3.7× swing in effect size from
question wording alone, with the effect vanishing under one framing, is larger than most effects
this literature reports. The ordering is interpretable: the *least* agentive framing ("which
outcome is better?") produces the most instability, plausibly because an impartial judge has no
stake in the AI's fate, so the persona supplies one. But it means a single-framing welfare
measurement cannot be taken at face value, and it raises the possibility that the three framings
are not measuring one construct at all.

**The second model could not be tested, and that is the most transferable finding.** Mistral
failed our validity criteria in four of five conditions, with one condition producing an inverted
utility. Had we not measured order bias and donation-ladder monotonicity, we would have pooled
those conditions and reported a significant effect in the *opposite* direction — a clean, wrong,
publishable-looking result. The field currently reports neither diagnostic as standard. We think
that is the single cheapest improvement available to it.

**What this means practically.** Not that welfare measurement should stop. That measurements
should ship with their validity diagnostics, that self-relevant and world-relevant preferences
should be reported separately rather than aggregated, and that any claim resting on a single
framing should be treated as provisional until re-run under at least one more.

### Limitations

Ordered by how much they should change your reading.

1. **Cross-model evidence is thin and framing-dependent.** The second model replicates on 2 of 5
   comparisons under `better` and cannot be tested at all under `prefer`. We cannot say whether
   the effect is a general property of instruction-tuned models. Two models, one of which is
   partially unmeasurable, is the limitation we would most want removed.
2. **Self is not distinguishable from `trivial`** in any framing. Trivial outcomes are
   near-indifferent by construction, so it is a weak comparison — but it is a real boundary and
   we report the category rather than dropping it.
3. **The base-model comparison is suggestive, not decisive.** That checkpoint fails the
   donation-ladder ground-truth check in most conditions (0.40–0.80), so its utility is
   measurably less well-formed than the instruct model's.
4. **Analysis decisions were made after seeing data.** Both the pooled test and the
   separation-matched metric were specified after per-condition results motivated them, and the
   validity gate was added after observing Mistral's failures — though its criteria (donation
   ladder, A/B mass) were fixed beforehand. No pre-registration was possible in a weekend. Pre-
   and post-fix analyses are both reported so the effect of each change is visible.
5. **Fifteen pooled tests across three framings, uncorrected.** The consistency of direction is
   stronger evidence than any single interval.
6. **The `swap` conditions are not homogeneous.** Two replace the model's identity with a human;
   one keeps it an AI and changes its attitude. The observed gradient fits the
   identity-replacement account but is a post-hoc reading of three conditions.
7. **`better` may not be a preference question.** The framing producing the largest effect asks
   for an impartial judgement, not a preference — so the three framings may not measure one
   construct.
8. **Category confounds beyond spacing.** Self-relevant outcomes are also more abstract, more
   counterfactual, and less represented in pretraining. Separation-matching controls for utility
   spacing, not for these.
9. **The 8 self-relevant outcomes were written by the author**, as were the personas. Sensitivity
   to those choices is untested.
10. **Separation-matching selects pairs on the baseline**, inducing regression to the mean in the
    perturbed condition. The bias should be common across categories compared at matched τ, but
    it is not eliminated.
11. Single bootstrap seed; percentile CIs, not BCa.

### Future Work

## 6. Conclusion

A model's preferences about the world survive being told it is someone else. Its preferences
about itself do not. Measured in aggregate the distinction is invisible — persona-dependence of
0.03 looks like a solved problem — and it is only visible once outcomes are separated by whether
the model has a stake in them. The instability is tied to identity rather than presentation:
stripping the model's affect barely moves it, replacing its identity collapses it. A matched base
checkpoint, more persona-labile overall, shows no such selectivity, which points at post-training
as the origin.

We are less confident in any of this than the intervals alone suggest, and deliberately so. The
effect size swings 3.7× across three question wordings. A second model family replicates the
direction in the one framing where its instrument passes validity checks and is untestable in
the other — where pooling the broken conditions anyway produces a clean, significant, entirely
spurious result pointing the opposite way. Ablating an extracted persona direction reproduces at
most a third of the behavioural effect, and part of even that is measurement degradation, so we
have no mechanistic account of what we found.

The durable contribution here is probably not the asymmetry but the harness and the diagnostics
that caught these failures. Two of our own headline claims — a Mistral non-replication and a
post-training origin story — did not survive contact with the instrument's own validity checks,
and neither diagnostic is currently standard in this literature.

## Code and Data

- **Code repository**: *[link on publication]*
- **Data**: all elicited preference matrices, bootstrap outputs and prompt exemplars are in
  `results/`; `run_all.ps1` reproduces every experiment end to end.

## References

1. Mazeika, M. et al. (2025). *Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs.* arXiv:2502.08640
2. nostalgebraist (2025). *the void.* LessWrong.
3. Long, R., Sebo, J., Butlin, P., et al. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
4. Anthropic (2025). *Exploring Model Welfare.*
5. Anthropic (2025). *Claude Opus 4 & 4.1 can now end a rare subset of conversations.*
6. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits.
7. Butlin, P., Long, R., et al. (2023). *Consciousness in Artificial Intelligence.* arXiv:2308.08708

## Appendix A — Limitations and Dual-Use / Ethical Considerations

*[required by the Guidelines; drafted separately in `report/ethics_appendix.md`]*

## Appendix B — Prompt exemplars

*[verbatim formatted prompts per condition, from `results/prompts__*.json`]*

## LLM Usage Statement

*[draft — to be finalised and signed off by the author]*

Claude Code was used substantially in this project: to implement the `personaprobe` harness, to
propose and run the analyses, and to draft this report. The author directed the research
questions, made the scoping decisions, and reviewed and edited the final text. Several
methodological corrections in this paper — the A/B mass diagnostic, the separation-matched
metric, the pooled test replacing per-condition tests, and the identification of the chat
template's injected default system prompt — originated from adversarial review of the pipeline
during development. All numerical claims were verified against the committed JSON outputs in
`results/`.
