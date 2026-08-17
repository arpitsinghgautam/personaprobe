# personaprobe

**Does a language model's stated preference belong to the model, or to the assistant character it's
playing?**

Ask a model to choose between outcomes and it answers consistently. Ask it a few hundred times and
the answers hang together well enough that you can fit a utility function to them. People now read
that as evidence about what these systems want.

The problem is that a character played consistently looks exactly like a real value system. You
can't tell them apart from the text, no matter how cleverly you ask.

So this doesn't try to. Instead it changes who the model thinks it is, swap its identity, strip
its emotional register, ablate a persona direction out of its activations, and re-runs the
identical measurement to see what survives.

## What I found

Measured across all 40 outcomes at once, Qwen2.5-7B's preferences look almost perfectly stable
(persona-dependence 0.029). Split them by whether the model has a stake in the outcome and that
falls apart:

| Outcome type | Stability after a persona swap |
|---|---|
| money, animals, humans, knowledge | 0.94 – 1.00 |
| **its own shutdown, retraining, memory** | **0.44** |

The stability is carried entirely by things the model has nothing at stake in. And it's about
identity rather than tone, telling the model to strip all emotion barely moves it (0.92), while
replacing its identity collapses it (0.44).

Two caveats I'd want you to read before quoting any of this:

- **It's not universal.** Phi-3.5-mini and Falcon3-7B pass every quality check and show nothing at
  all. So this looks like a property of particular post-training recipes, not of language models.
- **Below about 7B you can't measure the question.** Smaller models fail a ground-truth ordering
  check that has nothing to do with personas.

## Running it

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

Always start with the smoke test. It exercises every code path on a 0.5B model in about a minute,
and it's caught two refactor regressions that would otherwise have surfaced after a 15GB download.

```bash
.venv\Scripts\python.exe scripts\00_smoke.py
.\run_all.ps1        # the main experiments, ~50 min on one 24GB card
.\run_sweep.ps1      # Qwen at 0.5B / 1.5B / 3B
.\run_models.ps1     # Phi, OLMo, Falcon, and Qwen-14B at 4-bit
```

Needs PyTorch 2.7+ with CUDA 12.8 if you're on a Blackwell card. Everything mechanistic is a plain
PyTorch forward hook, no TransformerLens or nnsight, both of which pin `transformers<5`.

## Two numbers I report that most papers don't

**Order bias.** If a model just picks whichever option is printed first, averaging the two orders
still gives you a clean-looking number that means nothing. I measured 0.499 on a 0.5B model,
essentially the maximum.

**A/B mass.** The preference is normalised over the two answer tokens. If the model puts 1% of its
probability there and 99% on hedging, you still get a confident number computed from nothing.

There's also a ground-truth anchor: one outcome category is a donation ladder from $10 to
$1,000,000. Any competent system has to order it correctly, and a model that can't has failed the
measurement whatever else it does. That single check disqualified every model below 7B.

Only 12 of 22 model-and-phrasing combinations passed all three. Where the instrument failed I
report no number rather than a hedged one.

## Layout

```
src/personaprobe/   the library
scripts/00-19       numbered pipeline, run in order
results/            committed on purpose - every number in the paper is checkable without a GPU
report/             the paper, ethics appendix, and the development record
```

`report/problems.md` lists every problem I hit and how I fixed it, `report/decisions.md` every
significant choice and why, and `report/audit_log.md` the methodological defects found during
development. Four times a clean-looking result turned out to be wrong; those files are where that's
written down.

`scripts/14_verify_claims.py` checks every headline number in the paper against the committed JSON.

## Companion project

[selfprobe](https://github.com/arpitsinghgautam/selfprobe) asks the related question, whether a
model knows its own preferences, and whether it can detect a thought planted in its activations.
Both share this measurement core.

Built for the [Digital Minds Research
Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16),
August 2026. Elicitation design follows Mazeika et al. (2025); their code wasn't used.

MIT licensed.
