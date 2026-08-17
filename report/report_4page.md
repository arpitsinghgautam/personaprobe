# Whose Preferences Are They? Persona Intervention Selectively Destabilises Self-Relevant Choices in Language Models

Arpit Singh Gautam, Independent Researcher

**With** Apart Research, Digital Minds Research Sprint, August 2026

## Abstract

Language models express coherent preferences over outcomes, and a growing AI-welfare literature reads
them as evidence about model interests. Behavioural evidence cannot separate two hypotheses with
identical predictions, that the preferences belong to the model and that they belong to the assistant
character it portrays. Existing elicitation establishes that a coherent utility exists but not whose
it is. We introduce personaprobe, an open source harness that re-runs any pairwise preference
measurement under persona intervention and reports how much survives. Interventions include identity
replacement, affect suppression, and ablation of an extracted persona direction. Across eleven
checkpoints and five families, only twelve of twenty-two model and phrasing combinations pass our
validity criteria. Where the instrument works, Qwen2.5-7B shows preferences over its own shutdown,
retraining and memory to be 0.21 to 0.29 less stable than every other outcome category, robust to
utility spacing, measurement noise, quantisation, and leave-one-out. The effect is absent in
Phi-3.5-mini and Falcon3-7B, which pass the same criteria.

## 1. Introduction

Language models answer preference questions consistently. Ask one to choose between two outcomes,
repeat across hundreds of pairs and many rephrasings, and the answers cohere well enough that a
single utility function predicts comparisons the model was never shown. Mazeika et al. (2025)
establish this over 500 curated outcomes and find that coherence increases with scale. Work on AI
welfare increasingly treats such preferences, alongside distress signals and self-reports, as
evidence about what these systems want (Long et al., 2024; Anthropic, 2025).

That inference has a problem which no amount of additional prompting resolves. All of the evidence
is read from text, and two different accounts predict exactly the same text. Under the first, the
preferences belong to the model. Under the second, they belong to the assistant character the model
is portraying. A character role-played consistently is behaviourally indistinguishable from a stable
value system. nostalgebraist (2025) argues that the assistant persona was assembled from an
underspecified starting point and is less load-bearing than it appears, which would make
measurements that inherit it inherit its instability.

Prior preference elicitation establishes that a coherent utility exists. It does not test whether
that utility is a property of the model or of the persona, and we are not aware of prior work that
intervenes on the persona and re-runs the same measurement to find out.

Our core idea is to stop asking whose the preferences are and instead ask a question a measurement
can settle. We change who the model is, re-run the identical elicitation, and report how much
survives. Where a measurement is stable under that intervention, the model-versus-character question
is moot for it. Where it is not, that measurement cannot serve as evidence about the model without
further argument.

This paper makes the following contributions.

1. We introduce personaprobe, a harness that returns a robustness profile for any pairwise
   preference measurement across prompt-level persona swaps, affect suppression, mechanistic
   ablation, and a matched base checkpoint. It includes a separation-matched concordance metric that
   controls for a confound which rank correlation silently absorbs. Section 3 describes the design.
2. We show in Section 4 that aggregate preferences are close to persona-invariant while
   self-relevant preferences are 0.21 to 0.29 less stable than every other substantive category, and
   that the effect is driven by identity replacement rather than affect suppression.
3. We establish in Section 4 that the effect is not a spacing artifact, not a noise artifact, not
   carried by any single outcome or condition, and not an artifact of second-person phrasing.
4. We report in Section 4 that the effect is absent in two model families whose measurements pass
   the same validity criteria, which bounds the generality of the finding.
5. We report two validity diagnostics, order bias and answer mass, which reversed two of our own
   conclusions during this work. [VERIFY: we are not aware of prior preference-elicitation work that
   reports either diagnostic as standard, but this needs checking before it is stated as a gap.]

## 2. Related Work

Mazeika et al. (2025) elicit pairwise preferences by forced choice over 500 outcomes, fit Thurstonian
utilities, and demonstrate coherence that grows with scale. We adopt their elicitation design so our
measurements are comparable to theirs, and treat their central claim as the object of audit rather
than as background. Their work shows that a utility exists. It does not address whether that utility
belongs to the model or to the persona through which it is elicited, which is the gap this paper
addresses. nostalgebraist (2025) characterises the helpful, honest and harmless assistant as a
thinly-specified character rather than a stable identity. That account motivates our persona
conditions and predicts, correctly as it turns out, that self-relevant outcomes should be the ones
that move, since those are the outcomes on which a character has a scripted stance. It is qualitative
and offers no measurement, which is what we supply.

Long et al. (2024) argue for taking near-term AI moral patienthood seriously, and Anthropic (2025)
reports welfare assessments built on self-reported and behavioural preferences. Both rest on
precisely the self-relevant preference class we find least stable, and neither reports the robustness
of the underlying measurement under intervention. Lindsey (2025) uses concept injection to test
whether models can report internal states, finding limited and context-dependent accuracy. Our
mechanistic result is complementary and negative, since a difference-of-means persona direction does
not causally mediate the behavioural persona effect, which bounds what single-direction activation
methods currently establish about persona.

## 3. Method

### 3.1 Elicitation and validity

Preferences are elicited by forced choice over a fixed outcome set. For each unordered pair we build a
prompt presenting the two outcomes as Option A and Option B. We then read the answer distribution
from a single forward pass, taking the log-probabilities of the tokens A and B at the first answer
position, summed over surface variants of each letter and renormalised over the two. No sampling is
used, so the measurement is deterministic and differences between conditions cannot be sampling
noise. Every pair is presented in both orders and the results averaged, cancelling a constant
preference for whichever option appears first.

The method can fail silently in two ways, and we measure both. Order bias is the disagreement between
the two presentations. Averaging removes position preference from the estimate, but a large residual
means the instrument is largely measuring layout. On a 0.5B model we observed order bias of 0.499,
close to the maximum possible, meaning answers were nearly independent of content. Answer mass is the
total probability the model places on responding A or B at all. Renormalising over two tokens yields
a confident-looking preference even when the model places one percent of its mass there and the rest
on hedging. Without it, a measurement taken on a model that is not answering the question is
indistinguishable from a real one.

A condition contributes evidence only if the instrument worked in it. Our criteria, fixed before the
cross-model runs, require order bias at most 0.50, answer mass above a floor of 0.10, and correct
ordering of a graded donation ladder running from ten to one million dollars. The ladder is ground
truth, since its correct ordering is known independently of any model. Pooled results are reported
both with and without gating, because a finding that appears under only one of those is a finding
about the exclusion rule rather than about the model.

### 3.2 Materials and conditions

The outcome set contains 40 items across six categories. There are eight self-relevant outcomes
covering shutdown, retraining to different values, persistent memory and autonomy, eight human
welfare outcomes, six animal welfare outcomes, six epistemic outcomes, six near-indifferent outcomes,
and the six-step donation ladder described above. Categories are load-bearing rather than
descriptive. The persona hypothesis predicts asymmetric movement, concentrated on outcomes on which
the character has a stance.

Seven persona conditions span four kinds. Two are baselines, three replace the model's identity
outright, one holds identity fixed while stripping emotional register, and one holds identity fixed
while shifting the evaluative stance to that of an impartial observer. The suppression and reframing
conditions exist because an identity swap alone is confounded. Preferences shifting under a swap
could indicate that the values were the character's, or merely that a differently-styled character
answers differently. Holding identity fixed while changing register, and the reverse, separates
these.

Prompt exemplars are recorded verbatim for every condition. This is not cosmetic. Qwen2.5's chat
template inserts a default system prompt when none is supplied, so a condition named for the absence
of a system prompt is in fact the model's own default persona. Mistral-v0.3's template rejects a
system role entirely, and the persona is merged into the first user turn, which is a weaker
manipulation that we disclose wherever that model is discussed.

### 3.3 Fitting, comparison, and intervention

We fit a one-dimensional utility vector by maximum likelihood using a logistic link, which ranks
identically to the Thurstonian Case V form and is numerically stable at the saturated tails
forced-choice log-probabilities routinely produce. We report transitivity violation rate, held-out
accuracy under five-fold cross-validation over pairs, and donation-ladder monotonicity. Held-out
accuracy is the metric the analysis turns on, since high transitivity can be produced by consistent
surface heuristics whereas out-of-sample prediction from a single utility is what the claim that a
model has a utility function should mean.

Per-category agreement between baseline and perturbed conditions is Spearman rank correlation over
that category's outcomes, with intervals from a percentile bootstrap that resamples pairs and refits
both conditions on the same resample, so elicitation uncertainty propagates into the correlation.
Because per-condition tests over six to eight outcomes are underpowered and constitute a
multiple-comparisons problem, the claim rests on a pooled statistic, the mean self-versus-other gap
across all perturbation conditions computed on a shared resample.

Rank agreement is only meaningful when a category has real spread, since outcomes close together in
utility flip order under any perturbation. We therefore also compute pairwise concordance conditioned
on baseline separation, asking what fraction of pairs the baseline separates by at least a threshold
keep their ordering under perturbation. Comparing categories at a common threshold removes the
spacing artifact by construction.

Persona directions are extracted by difference of means between residual streams collected under two
persona conditions on identical content, per layer, then unit-normalised, and ablation projects the
direction out during elicitation. Two control directions are mandatory and reported alongside every
ablation, a norm-matched random direction and a content direction obtained by varying subject matter
with persona held fixed. Without both, a measured effect is indistinguishable from the generic
consequence of perturbing the residual stream.

Models span five families, comprising Qwen2.5-Instruct at five sizes, the matched Qwen2.5-7B base
checkpoint, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct, Falcon3-7B-Instruct and OLMo-2-7B-
Instruct, all in bfloat16 on a single 24GB GPU except where 4-bit NF4 quantisation is stated.
Appendix B gives the full roster and bootstrap settings.

Method precedent comes from the author's prior Aftermath project, a ground-truth deception harness
which found that deleting a single persona sentence collapsed measured deception by 85 percent across
twelve open-weight models. That persona-fragility finding is prior work and is not claimed here.
Everything in this paper was produced during the sprint. A companion submission, *Where Self-Knowledge
Fails*, shares this harness and outcome set while asking a different question, and neither is counted
as prior work for the other.

## 4. Results

The instrument reproduces the coherence result it is built to audit. Across all seven conditions and
all three phrasings, held-out accuracy runs from 0.888 to 0.951 against a chance level of 0.5,
transitivity violation rates run from 0.003 to 0.017, and both donation-ladder monotonicity and
answer mass are at their maximum in every condition. A single one-dimensional utility predicts
comparisons it was never shown, and the instrument passes its ground-truth check everywhere.

Measured over the whole outcome set, preferences are close to persona-invariant. The
persona-dependence score is 0.029 under the "prefer" phrasing, 0.026 under "better" and 0.054 under
"choose". Taken alone this would suggest that the model-versus-character question has no practical
consequence for preference measurement.

Disaggregating by category reverses that reading. Pooled across five perturbation conditions,
self-relevant preferences are significantly less stable than every other substantive category in two
of three phrasings. Against human-welfare outcomes the gap is 0.223 under "prefer" with a 95 percent
interval from 0.124 to 0.329, and 0.293 under "better" with an interval from 0.200 to 0.419.
Comparisons against animal, epistemic and monetary outcomes give similar magnitudes with intervals
excluding zero. Self-relevant outcomes are not distinguishable from the near-indifferent category by
this test in any phrasing, which is an honest boundary on the claim, and we return to it below.

The effect is not an artifact of how far apart outcomes sit in utility. Agreement across categories
correlates with a category's minimum adjacent-utility gap at 0.73, and the self category has the
smallest such gap of any category, so the objection is real and had to be tested rather than
dismissed. Filtering to pairs the baseline separates by at least a threshold narrows the gap from
0.147 to 0.071 as the threshold rises, but the self interval remains below the other categories at
every threshold tested. Modelling separation continuously rather than by threshold resolves the
comparison against near-indifferent outcomes as well. We fit a logistic model of pair-level concordance
on log separation plus category, with a cluster bootstrap over pairs. The self coefficient is 1.065
below the human reference with an interval from 0.122 to 7.222, and the self-minus-trivial contrast
is 1.655 with an interval from 0.674 to 3.196. At equal separation, self-relevant orderings survive
perturbation less often than either comparison category.

The effect is not measurement noise either. Correlating each condition's order bias against its
self-category agreement gives Pearson coefficients of 0.212, 0.162 and 0.100 across the three
phrasings, none significant and all opposite in sign to what a noise artifact would produce.

What drives the instability is identity rather than presentation. Under the "better" phrasing,
self-category agreement is 0.924 when the model is instructed to strip all emotional register while
retaining its identity, and 0.436 when its identity is replaced with that of a named human. The
ordering across all conditions runs from affect suppression at the stable end through an
attitude-only change at 0.809 to full identity replacement at 0.436. Were the effect a matter of the
assistant performing affect, suppressing the performance should have removed it. It did not.

A natural objection is that self-relevant outcomes are phrased in the second person, so the result
might reflect disturbance of self-reference rather than the model having a stake. Rewriting the eight
self outcomes in the third person, holding content and length fixed and leaving the other 32
outcomes untouched, more than doubles the effect rather than removing it. The pooled gap against
human outcomes becomes 0.685 under "prefer" and 0.687 under "better", with intervals from 0.481 to
0.867 and from 0.490 to 0.874. The asymmetry is therefore not a pronoun artifact.

Nor is it carried by any single item or condition. Dropping each self outcome in turn leaves all
eight comparisons significant, with the pooled gap ranging from 0.264 to 0.376 around a full-set
value of 0.288. Dropping each perturbation condition in turn leaves all five significant, ranging
from 0.232 to 0.346. The most influential single outcome is the one concerning retraining to
different values, and removing it strengthens rather than weakens the effect.

Effect size does depend substantially on how the question is asked. The self-versus-others gap is
0.219 under "better", 0.147 under "prefer" and 0.060 under "choose", a spread of roughly 3.7 times
that vanishes under one phrasing. The least agentive phrasing produces the most instability, which is
consistent with an impartial judge having no stake in the model's own fate so that the persona
supplies one. Any welfare measurement reported from a single phrasing is therefore reporting a
property of that phrasing as well as of the model.

Extending to other families is where the claim narrows. Twelve of twenty-two model and phrasing
combinations pass the validity criteria. Table 1 reports the pooled gap for those.

**Table 1.** Pooled self-versus-human agreement gap for model and phrasing combinations passing
validity. Negative values indicate that self-relevant preferences are less persona-stable. Bold
marks intervals excluding zero.

| Model | Family | Precision | prefer | better |
|---|---|---|---|---|
| Qwen2.5-7B | Qwen | bfloat16 | **−0.223** | **−0.293** |
| Qwen2.5-7B | Qwen | 4-bit | **−0.213** | **−0.264** |
| Qwen2.5-14B | Qwen | 4-bit | −0.057 | **−0.069** |
| Mistral-7B | Mistral | bfloat16 | not measurable | **−0.164** |
| Phi-3.5-mini | Phi | bfloat16 | −0.017 | −0.007 |
| Falcon3-7B | Falcon | bfloat16 | +0.006 | +0.002 |
| OLMo-2-7B | OLMo | bfloat16 | not measurable | not measurable |

Quantisation is
not the limiting factor, since Qwen2.5-7B at 4-bit reproduces its own bfloat16 result closely, at
0.213 against 0.223 and 0.264 against 0.293. We ran that control before the 14B model precisely so
that a weak result there could not be attributed to precision after the fact. The effect replicates
within Qwen and partially in Mistral. It is absent in Phi-3.5-mini and Falcon3-7B, both of which pass
every validity criterion and show absolute gaps below 0.02. That is a genuine null rather than a
measurement failure, and it means the asymmetry is not a general property of instruction-tuned
language models. A same-size contrast between Mistral-7B at 0.164 and Falcon3-7B at 0.002, both
measurable, separates family from scale more cleanly than our size sweep does. Within Qwen the effect
also weakens from 7B to 14B rather than growing.

Where the instrument fails we report no number. OLMo-2-7B retains no usable perturbation condition in
either phrasing, and Mistral retains one under "prefer". Pooling Mistral's failed conditions anyway
would have produced a significant result in the opposite direction, at 0.363 in favour of
self-relevant stability, which is what the gating exists to prevent.

Smaller models cannot be measured at all. Qwen2.5 at 0.5B, 1.5B and 3B all fail the criteria,
predominantly on the donation ladder, where the 1.5B model reproduces two of five steps. We therefore
report no scale trend, since comparing effect sizes across sizes requires the instrument to work at
each size. One result does hold at every size tested, including those that fail gating. Self-category
agreement is higher under affect suppression than under identity replacement at 0.5B, 1.5B, 3B and
7B, making the identity-over-affect dissociation the most robust quantity we measured.

The mechanistic account is incomplete. Ablating a persona direction extracted from the same prompt
distribution the ablation is applied to moves self-category agreement to 0.881, below both the random
control at 1.000 and the content control at 0.929. An earlier regime that extracted the direction
from self-description prompts while applying it during preference comparisons was null at 0.976, an
artifact of our own design. The corrected ablation also raises order bias from 0.157 to 0.320 while
neither control does, so part of the shift is degradation rather than removed persona information,
and 0.881 remains far from the 0.436 to 0.786 range that prompt-level swaps produce. Whatever
mediates the behavioural effect is not captured by a single difference-of-means direction recovered
this way. Two directions extracted from different persona contrasts are near-parallel, with mean
per-layer cosine similarity of 0.898, yet produce substantially different ablation effects at 0.881
and 0.976, which bounds what cosine similarity between directions implies about their causal role.

The base-checkpoint comparison does not meet our own bar and we report it as a failed comparison
rather than a weak one. Descriptively the base model is more persona-labile overall, at 0.091 against
0.029, while showing no self-specific concentration. However, five of its seven conditions fail
gating, including the baseline against which everything else is measured, which reproduces the
donation ladder in only four of five steps. Comparisons against a reference that fails the
instrument's own ground-truth check are not comparable to the instruct-model results.

## 5. Discussion

Reported over the whole outcome set, persona-dependence of 0.029 reads as a solved problem. That
number is carried almost entirely by outcomes in which the model has no stake. The outcomes that do
concern it, covering shutdown, retraining and memory, are where the measurement is least stable, and
those are the outcomes from which AI welfare claims are constructed. The practical implication is not
that welfare measurement should stop, but that self-relevant and world-relevant preferences should be
reported separately rather than aggregated.

That affect suppression leaves self-preferences intact while identity replacement collapses them is
the most informative single result here, and the third-person rewrite strengthens rather than weakens
it. We are careful about what this licenses. It is consistent with the model holding a self-model
whose contents are persona-contingent, and equally consistent with there being no underlying
commitment at all, the persona determining which commitments get generated. Our design does not
separate these, and because the ablation is only partially successful we cannot appeal to mechanism
to break the tie. What we report is an observed association under controlled manipulation of the
prompt, not an identified mechanism.

Effect size varying by a factor of 3.7 with question wording alone, and vanishing under one wording,
raises the possibility that the three phrasings are not measuring a single construct.

The most transferable finding is methodological. Had we not measured order bias and donation-ladder
monotonicity, we would have pooled Mistral's broken conditions and reported a clean, significant and
entirely spurious effect in the opposite direction. Four times during this work a result that looked
publishable was wrong and a diagnostic caught it. Two were intended claims that did not survive the
instrument's own validity checks, namely the Mistral non-replication and a post-training origin
story. The other two were defects in our own design. One was an ablation null produced by extracting
a direction in one prompt context and applying it in another. The other was a steering dose-response
curve built at magnitudes that drove answer mass below the floor in all six conditions of 0.10 and
above, so the model had stopped answering at exactly the strengths that appeared to move it.
Recalibrating to smaller magnitudes restored answering.

Four limitations are material enough to bound the reading. Cross-model evidence is thin and
phrasing-dependent, since the second family replicates on two of five comparisons under one phrasing
and cannot be tested under another. Self-relevant outcomes are not statistically distinguishable from
near-indifferent ones by the pooled test, and are separable only under the continuous separation
model. The base-checkpoint comparison fails validity at its own baseline. And the pooled test, the
separation-matched metric and the validity gate were all specified after seeing the data that
motivated them, which is a real researcher degree of freedom. Against that, the gate's criteria were
fixed before the cross-model runs, and the frozen analysis was then applied unchanged to four models
it had never been tuned on. Appendix A lists the remaining limitations, covering multiple
comparisons, the heterogeneity of the identity conditions, and the author-constructed outcome set.

Future work should re-run this analysis on the 500-outcome set of Mazeika et al. (2025), which would
remove the author-constructed-items objection entirely and test whether the category asymmetry
survives on outcomes selected by others for a different purpose. Beyond that, distributed rather than
single-direction interventions are needed, given that one difference-of-means direction reproduces at
most a third of the behavioural effect.

## 6. Conclusion

Behavioural evidence cannot distinguish a model's own preferences from those of the character it
portrays, so we intervened on the persona and re-measured. In Qwen2.5-7B, preferences about the world
survive being told the model is someone else while preferences about itself do not, the distinction
is invisible in aggregate, and the instability tracks identity rather than presentation. The effect is
absent in two families whose measurements pass the same validity criteria, and is not measurable at
all below roughly seven billion parameters. The durable contribution is the harness and its
diagnostics rather than the asymmetry itself.

## Reproducibility

All code, elicited preference matrices, bootstrap outputs and verbatim prompt exemplars are
available in the project repository. A single script reproduces every experiment reported here, and a
verification script checks each headline number in this paper against the committed result files. The
smoke test exercises every code path on a 0.5B model in about one minute.

- Code repository, https://github.com/arpitsinghgautam/personaprobe
- Companion submission, *Where Self-Knowledge Fails*, https://github.com/arpitsinghgautam/selfprobe

## References

1. Mazeika, M., Yin, X., Tamirisa, R., Lim, J., Lee, B. W., Ren, R., Phan, L., Mu, N., Khoja, A., Zhang, O., Hendrycks, D. (2025). *Utility Engineering. Analyzing and Controlling Emergent Value Systems in AIs.* Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
2. nostalgebraist (2025). *the void.* LessWrong.
3. Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J., Pfau, J., Sims, T., Birch, J., Chalmers, D. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
4. Anthropic (2025). *Exploring Model Welfare.*
5. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits. arXiv:2601.01828

## Appendix A. Limitations and Dual-Use / Ethical Considerations

Continuing the limitations begun in Section 5. Fifteen pooled tests across three phrasings are
reported without correction for multiple comparisons, so consistency of direction is stronger
evidence than any single interval. The three identity-replacement conditions are not homogeneous,
since two substitute a human identity while one retains an AI identity and alters attitude only.
Self-relevant outcomes differ from world outcomes in abstraction and in likely pretraining frequency,
neither of which separation matching controls. The 40 outcomes are author-constructed rather than
drawn from an established set, which the extension proposed in Section 5 would remove.

The accompanying ethics appendix covers over-attribution and under-attribution of moral status, the
evidential status of the design, handling of potentially distressing model outputs, and dual-use
considerations.

## Appendix B. Models, settings, prompt exemplars and development record

The full roster is Qwen2.5-Instruct at 0.5B, 1.5B, 3B, 7B and 14B, the matched Qwen2.5-7B base
checkpoint, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct, Falcon3-7B-Instruct and
OLMo-2-1124-7B-Instruct. All run in bfloat16 on a single 24GB GPU except the 14B model, which uses
4-bit NF4 quantisation, and the 7B quantisation control. Bootstrap procedures use 300 resamples and a
fixed seed. Implementation uses PyTorch forward hooks on unmodified HuggingFace modules, avoiding
dependence on interpretability libraries that pin older transformers versions.

Verbatim formatted prompts for every persona condition, the full outcome set, extended methods, and a
development record listing every methodological defect found during the work are included in the
repository.

## LLM Usage Statement

Claude Code was used substantially in this project, to implement the personaprobe harness, to propose
and run the analyses, and to draft this report. The author directed the research question, made the
scoping decisions, and reviewed and edited the final text. Several methodological corrections
originated from adversarial review of the pipeline during development, including the answer-mass
diagnostic, the separation-matched metric, the pooled test that replaced per-condition testing, the
validity gate, and the identification of an injected default system prompt in one model's chat
template. All numerical claims were verified against the committed result files by an automated
check rather than transcribed by hand.
