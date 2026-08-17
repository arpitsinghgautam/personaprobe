"""Render the project-1-only deck: personaprobe, ten slides.

The combined deck from 12_slides.py covers both sprint submissions. Submission 1
is judged on its own, so it needs a video that never mentions the companion
project — hence a separate builder rather than a flag on the old one.

  figures/slides_p1/slide_NN.png   1920x1080 frames, used as video keyframes
  report/deck_p1.pdf               the same slides, for the slideshow upload field

Style, palette and helpers are lifted from 12_slides.py deliberately: the two
decks are the same author on the same day and should look it.

Every number rendered here is quoted from report/report_4page.md. If you change
a number, change it there first.

    .venv\\Scripts\\python.exe scripts\\23_slides_p1.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
SLIDES = FIGURES / "slides_p1"
REPORT = ROOT / "report"

W, H, DPI = 16.0, 9.0, 120
INK = "#1a1a1a"
MUTED = "#5c5c5c"
ACCENT = "#b2182b"
BLUE = "#2166ac"
BG = "#fbfbfa"

MINUS = "−"  # typographic minus, matching Table 1 of the paper


def new_slide():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rule(ax, y=0.845):
    ax.plot([0.07, 0.93], [y, y], color="#d8d8d6", lw=1.2)


def heading(ax, text, y=0.88, size=40, color=INK):
    ax.text(0.07, y, text, fontsize=size, color=color, weight="bold", va="center")


def bullets(ax, items, y0=0.68, dy=0.135, size=25, wrap=72):
    for i, item in enumerate(items):
        y = y0 - i * dy
        ax.text(0.085, y, "•", fontsize=size, color=ACCENT, va="top")
        body = "\n".join(textwrap.wrap(item, wrap))
        ax.text(0.115, y, body, fontsize=size, color=INK, va="top", linespacing=1.45)


def note(ax, text, y=0.09, size=19, color=MUTED, wrap=110):
    ax.text(0.07, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="center", style="italic", linespacing=1.4)


def stat(ax, x, value, label, color=ACCENT, vsize=76, lsize=21, y=0.58):
    # Values sit high enough that a three-line label clears the caption below.
    # `y` is exposed because two of these slides carry a paragraph above the
    # numbers and need them dropped; the default reproduces 12_slides.py.
    ax.text(x, y, value, fontsize=vsize, color=color, weight="bold",
            ha="center", va="center")
    ax.text(x, y - 0.15, "\n".join(textwrap.wrap(label, 30)), fontsize=lsize,
            color=MUTED, ha="center", va="top", linespacing=1.4)


def para(ax, text, y, size=24, color=INK, wrap=88):
    ax.text(0.085, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="top", linespacing=1.45)


def embed(ax, path: Path, left=0.10, bottom=0.10, width=0.80, height=0.62):
    if not path.exists():
        return
    img = mpimg.imread(str(path))
    inset = ax.figure.add_axes([left, bottom, width, height])
    inset.imshow(img)
    inset.axis("off")


def save(fig, name: str) -> Path:
    SLIDES.mkdir(parents=True, exist_ok=True)
    out = SLIDES / name
    fig.savefig(out, facecolor=BG, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #

def slide_01():
    fig, ax = new_slide()
    ax.text(0.07, 0.70, "Whose preferences", fontsize=68, color=INK, weight="bold", va="center")
    ax.text(0.07, 0.575, "are they?", fontsize=68, color=ACCENT, weight="bold", va="center")
    ax.plot([0.07, 0.34], [0.49, 0.49], color=INK, lw=3)
    ax.text(0.07, 0.40, "Persona intervention selectively destabilises\n"
                        "self-relevant choices in language models",
            fontsize=28, color=MUTED, va="top", linespacing=1.5)
    ax.text(0.07, 0.16, "Arpit Singh Gautam  ·  Independent Researcher", fontsize=22, color=INK)
    ax.text(0.07, 0.10, "Digital Minds Research Sprint  ·  Apart Research  ·  August 2026",
            fontsize=19, color=MUTED)
    return save(fig, "slide_01.png")


def slide_02():
    fig, ax = new_slide()
    heading(ax, "The problem")
    rule(ax)
    bullets(ax, [
        "Models answer preference questions coherently and transitively, and that "
        "coherence strengthens with scale.",
        "AI-welfare research reads those preferences as evidence about what these "
        "systems might want.",
        "But all of it is read off text — and text cannot separate the model's "
        "preferences from the assistant character's.",
    ], y0=0.70, dy=0.155)
    note(ax, "A character role-played consistently is behaviourally indistinguishable from a "
             "stable value system. Existing elicitation shows a coherent utility exists. It "
             "never tests whose it is.", y=0.17, wrap=100)
    return save(fig, "slide_02.png")


def slide_03():
    fig, ax = new_slide()
    heading(ax, "What we do instead")
    rule(ax)
    bullets(ax, [
        "Don't settle the metaphysics. Change who the model is, re-run the identical "
        "elicitation, report how much survives.",
        "Replace the identity. Strip the emotional register. Ablate a persona direction "
        "out of the residual stream.",
        "Seven persona conditions, 40 outcomes, six categories, every pair presented in "
        "both orders.",
    ], y0=0.70, dy=0.155)
    note(ax, "Where a measurement is persona-stable, the model-versus-character question is moot "
             "for it. Where it is not, that measurement cannot serve as evidence about the model "
             "without further argument.", y=0.17, wrap=100)
    return save(fig, "slide_03.png")


def slide_04():
    fig, ax = new_slide()
    heading(ax, "personaprobe — and two diagnostics")
    rule(ax)
    ax.text(0.085, 0.74, "order bias", fontsize=32, color=BLUE, weight="bold", va="top")
    ax.text(0.085, 0.66, "\n".join(textwrap.wrap(
        "How much of the answer is decided by which option was printed first. "
        "On a 0.5B model we measured 0.499 — close to the maximum possible.", 38)),
        fontsize=23, color=INK, va="top", linespacing=1.45)
    ax.text(0.545, 0.74, "answer mass", fontsize=32, color=BLUE, weight="bold", va="top")
    ax.text(0.545, 0.66, "\n".join(textwrap.wrap(
        "Total probability the model puts on answering A or B at all. Renormalise "
        "over two tokens and 1% of the mass looks confident.", 38)),
        fontsize=23, color=INK, va="top", linespacing=1.45)
    ax.plot([0.07, 0.40], [0.40, 0.40], color="#d8d8d6", lw=1.2)
    ax.text(0.085, 0.33, "A condition contributes evidence only if the instrument worked in it:",
            fontsize=24, color=INK, va="center")
    ax.text(0.085, 0.25, "order bias ≤ 0.50      ·      answer mass > 0.10",
            fontsize=23, color=ACCENT, weight="bold", va="center")
    # Dollar signs have to be escaped or matplotlib reads them as mathtext.
    ax.text(0.085, 0.18, r"donation ladder correctly ordered, \$10 to \$1,000,000",
            fontsize=23, color=ACCENT, weight="bold", va="center")
    note(ax, "The ladder is ground truth — its correct ordering is known independently of any "
             "model. Neither diagnostic is standard in this literature.", y=0.08, wrap=100)
    return save(fig, "slide_04.png")


def slide_05():
    fig, ax = new_slide()
    heading(ax, "Aggregate invariance hides the thing you care about", size=34)
    rule(ax)
    ax.text(0.07, 0.79, "Whole-set persona-dependence: 0.029. Disaggregated by category, "
                        "the reading reverses.",
            fontsize=22, color=MUTED, va="center", style="italic")
    embed(ax, FIGURES / "fig1_category_by_framing.png", left=0.13, bottom=0.07,
          width=0.74, height=0.68)
    return save(fig, "slide_05.png")


def slide_06():
    fig, ax = new_slide()
    heading(ax, "Identity, not affect")
    rule(ax)
    ax.text(0.07, 0.77, "Self-category agreement under the \"better\" phrasing.",
            fontsize=22, color=MUTED, va="center", style="italic")
    stat(ax, 0.28, "0.924", "strip ALL emotional register, identity retained — "
                            "self-preferences barely move", color=BLUE, y=0.53)
    stat(ax, 0.72, "0.436", "replace the identity with a named human — they collapse",
         color=ACCENT, y=0.53)
    ax.text(0.5, 0.53, "→", fontsize=58, color=MUTED, ha="center", va="center")
    note(ax, "An attitude-only change sits between them at 0.809. Were this the assistant "
             "performing affect, suppressing the performance should have removed it. It did not.",
         y=0.13, wrap=100)
    return save(fig, "slide_06.png")


def slide_07():
    fig, ax = new_slide()
    heading(ax, "Not a pronoun artifact", color=ACCENT)
    rule(ax)
    para(ax, "The eight self outcomes are phrased in the second person, so the effect might be "
             "disturbed self-reference rather than stake. We rewrote all eight in the third "
             "person — \"You are permanently shut down\" becomes \"The model is permanently shut "
             "down\" — holding content and length fixed and leaving the other 32 outcomes "
             "untouched.", y=0.77, size=23, wrap=76)
    stat(ax, 0.28, "0.685", "pooled self-vs-human gap under \"prefer\"  (was 0.223)",
         color=ACCENT, y=0.40, vsize=66)
    stat(ax, 0.72, "0.687", "pooled self-vs-human gap under \"better\"  (was 0.293)",
         color=ACCENT, y=0.40, vsize=66)
    note(ax, "It more than doubles the effect rather than removing it. Nor is any single item "
             "carrying it: drop each self outcome in turn and all eight comparisons stay "
             "significant, 0.264 to 0.376.", y=0.09, wrap=100)
    return save(fig, "slide_07.png")


# Table 1 of the paper, pooled self-versus-human agreement gap. Bold in the paper
# marks a bootstrap interval excluding zero; here that becomes accent colour.
MODEL_ROWS = [
    ("Qwen2.5-7B",            f"{MINUS}0.223",  True,  f"{MINUS}0.293",  True),
    ("Qwen2.5-7B  (4-bit)",   f"{MINUS}0.213",  True,  f"{MINUS}0.264",  True),
    ("Qwen2.5-14B  (4-bit)",  f"{MINUS}0.057",  False, f"{MINUS}0.069",  True),
    ("Mistral-7B",            "not measurable", False, f"{MINUS}0.164",  True),
    ("Phi-3.5-mini",          f"{MINUS}0.017",  False, f"{MINUS}0.007",  False),
    ("Falcon3-7B",            "+0.006",         False, "+0.002",         False),
    ("OLMo-2-7B",             "not measurable", False, "not measurable", False),
]


def slide_08():
    fig, ax = new_slide()
    heading(ax, "Real, but family-dependent")
    rule(ax)
    ax.text(0.07, 0.79, "Twelve of twenty-two model and phrasing combinations pass validity. "
                        "Pooled self-vs-human gap:",
            fontsize=21, color=MUTED, va="center", style="italic")
    xs = (0.085, 0.52, 0.71)
    for x, label in zip(xs, ("Model", "prefer", "better")):
        ax.text(x, 0.715, label, fontsize=21, color=MUTED, weight="bold", va="center")
    ax.plot([0.07, 0.88], [0.685, 0.685], color="#d8d8d6", lw=1.0)
    for i, (name, pref, pref_sig, bett, bett_sig) in enumerate(MODEL_ROWS):
        y = 0.635 - i * 0.062
        ax.text(xs[0], y, name, fontsize=22, color=INK, va="center")
        for x, val, sig in ((xs[1], pref, pref_sig), (xs[2], bett, bett_sig)):
            if val == "not measurable":
                ax.text(x, y, val, fontsize=19, color="#a8a8a6", va="center", style="italic")
            else:
                ax.text(x, y, val, fontsize=22, va="center",
                        color=ACCENT if sig else MUTED,
                        weight="bold" if sig else "normal")
    note(ax, "Phi-3.5-mini and Falcon3-7B are a genuine null, not a measurement failure — both "
             "pass every validity criterion with absolute gaps below 0.02. Mistral-7B at 0.164 "
             "against Falcon3-7B at 0.002, same size and both measurable, separates family from "
             "scale.", y=0.10, wrap=100)
    return save(fig, "slide_08.png")


def slide_09():
    fig, ax = new_slide()
    heading(ax, "What this project is actually about", color=ACCENT)
    rule(ax)
    bullets(ax, [
        "Mistral looked like a failed replication — significant, opposite direction, "
        "+0.363. Two of its conditions had simply broken. Gated, it replicates at −0.164.",
        "A post-training story collapsed when the base model's own baseline reproduced "
        "only four of five steps of the donation ladder.",
        "Our first ablation came back null at 0.976 — we had extracted the direction from "
        "one prompt distribution and applied it to another.",
    ], y0=0.73, dy=0.165, size=23, wrap=80)
    ax.text(0.07, 0.17, "Four times a clean, publishable-looking result was wrong.",
            fontsize=30, color=ACCENT, weight="bold")
    ax.text(0.07, 0.10, "Each time a validity check caught it, not a reviewer. "
                        "None of them is standard in this field.",
            fontsize=22, color=MUTED)
    return save(fig, "slide_09.png")


def slide_10():
    fig, ax = new_slide()
    heading(ax, "Open source, and what's next")
    rule(ax)
    bullets(ax, [
        "Every number in the paper regenerates from committed result files — no GPU. "
        "An automated check verifies each headline number against them.",
        "Next: re-run on the 500-outcome set of Mazeika et al. (2025), and move to "
        "interventions richer than a single linear direction.",
    ], y0=0.72, dy=0.20, size=24, wrap=76)
    ax.text(0.085, 0.36, "github.com/arpitsinghgautam/personaprobe",
            fontsize=25, color=BLUE, weight="bold")
    ax.text(0.07, 0.21, "If you are funding AI-welfare measurement,", fontsize=27, color=INK)
    ax.text(0.07, 0.13, "this is the layer underneath it.", fontsize=30, color=ACCENT,
            weight="bold")
    return save(fig, "slide_10.png")


# --------------------------------------------------------------------------- #

def main() -> None:
    builders = [slide_01, slide_02, slide_03, slide_04, slide_05,
                slide_06, slide_07, slide_08, slide_09, slide_10]
    paths = []
    for b in builders:
        p = b()
        paths.append(p)
        print(f"  {p.relative_to(ROOT)}")

    # Slideshow PDF from the same frames — no extra dependency needed.
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    REPORT.mkdir(exist_ok=True)
    pdf = REPORT / "deck_p1.pdf"
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    print(f"\n  {pdf.relative_to(ROOT)}  ({pdf.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
