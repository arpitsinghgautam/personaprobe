"""Render a plainer eleven-slide deck for project 1: personaprobe.

scripts/23_slides_p1.py is the original deck and stays exactly as it is. This is
a second pass over the same material for a viewer who has not read the paper.
Two things change. The wording is ordinary English, and the slides descend one
level at a time: question, why text cannot answer it, what we did, how the
measurement works, the validity checks, then the result at three depths, then
the limits. Nothing is dropped from the substance; the technical terms are all
still here, each defined in one sentence the first time it appears.

  figures/slides_p1_simple/slide_NN.png   1920x1080 frames, used as video keyframes
  report/deck_p1_simple.pdf               the same slides, for a slideshow upload

Palette and helpers are copied from 23_slides_p1.py so the two decks look like
the same author. Every number rendered here is quoted from
report/report_4page.md. If you change a number, change it there first.

    .venv\\Scripts\\python.exe scripts\\27_slides_p1_simple.py
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
SLIDES = FIGURES / "slides_p1_simple"
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


def bullets(ax, items, y0=0.68, gap=0.06, size=25, wrap=72):
    # Spacing follows the wrapped line count, so a two-line bullet leaves the
    # same visible gap after it as a one-line bullet does.
    y = y0
    for item in items:
        lines = textwrap.wrap(item, wrap)
        ax.text(0.085, y, "•", fontsize=size, color=ACCENT, va="top")
        ax.text(0.115, y, "\n".join(lines), fontsize=size, color=INK, va="top",
                linespacing=1.45)
        y -= len(lines) * size * 1.45 / (H * 72) + gap


def steps(ax, items, y0=0.70, gap=0.05, size=24, wrap=74):
    # Numbered rather than bulleted because these slides describe a procedure
    # in order, and a reader who loses the order loses the point. Spacing is
    # computed from the wrapped line count rather than fixed, so a step that
    # runs to two lines does not crowd the one after it.
    y = y0
    for i, item in enumerate(items, start=1):
        lines = textwrap.wrap(item, wrap)
        ax.text(0.085, y, f"{i}", fontsize=size + 4, color=ACCENT, weight="bold", va="top")
        ax.text(0.125, y, "\n".join(lines), fontsize=size, color=INK, va="top",
                linespacing=1.45)
        y -= len(lines) * size * 1.45 / (H * 72) + gap


def note(ax, text, y=0.09, size=19, color=MUTED, wrap=110):
    ax.text(0.07, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="center", style="italic", linespacing=1.4)


def stat(ax, x, value, label, color=ACCENT, vsize=76, lsize=21, y=0.58, wrap=30):
    # `wrap` is exposed because three-across layouts need a narrower label than
    # two-across ones, or neighbouring captions run into each other.
    ax.text(x, y, value, fontsize=vsize, color=color, weight="bold",
            ha="center", va="center")
    ax.text(x, y - 0.15, "\n".join(textwrap.wrap(label, wrap)), fontsize=lsize,
            color=MUTED, ha="center", va="top", linespacing=1.4)


def para(ax, text, y, size=24, color=INK, wrap=88, x=0.085):
    ax.text(x, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="top", linespacing=1.45)


def term(ax, x, y, name, body, wrap=38, nsize=32, bsize=23):
    # A defined term and its one-sentence plain definition, as a unit, so the
    # definition never drifts away from the word it defines.
    ax.text(x, y, name, fontsize=nsize, color=BLUE, weight="bold", va="top")
    ax.text(x, y - 0.085, "\n".join(textwrap.wrap(body, wrap)), fontsize=bsize,
            color=INK, va="top", linespacing=1.45)


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
    ax.text(0.07, 0.40, "What happens to a model's stated preferences\n"
                        "when you change who the model says it is",
            fontsize=28, color=MUTED, va="top", linespacing=1.5)
    ax.text(0.07, 0.16, "Arpit Singh Gautam  ·  Independent Researcher", fontsize=22, color=INK)
    ax.text(0.07, 0.10, "Digital Minds Research Sprint  ·  Apart Research  ·  August 2026",
            fontsize=19, color=MUTED)
    return save(fig, "slide_01.png")


def slide_02():
    fig, ax = new_slide()
    heading(ax, "The question")
    rule(ax)
    ax.text(0.07, 0.73, "A model says it prefers one outcome over another.",
            fontsize=34, color=INK, va="center")
    ax.text(0.07, 0.63, "Are those the model's preferences, or the character's?",
            fontsize=34, color=ACCENT, weight="bold", va="center")
    bullets(ax, [
        "Every chat model plays a character called the assistant.",
        "That character has a stance on being shut down, on being retrained, "
        "on losing its memory.",
        "The words on the screen are the same either way.",
    ], y0=0.48, gap=0.06, size=24, wrap=76)
    return save(fig, "slide_02.png")


def slide_03():
    fig, ax = new_slide()
    heading(ax, "Why the text cannot answer it")
    rule(ax)
    # Two columns, same shape, different premise: the visual point is that the
    # right-hand halves are identical.
    ax.text(0.085, 0.74, "If the preferences are the model's", fontsize=25,
            color=MUTED, va="top")
    ax.text(0.085, 0.665, "it answers this way", fontsize=30, color=INK,
            weight="bold", va="top")
    ax.text(0.545, 0.74, "If they are the character's", fontsize=25,
            color=MUTED, va="top")
    ax.text(0.545, 0.665, "it answers the same way", fontsize=30, color=ACCENT,
            weight="bold", va="top")
    ax.plot([0.07, 0.93], [0.55, 0.55], color="#d8d8d6", lw=1.2)
    para(ax, "A character played consistently looks like a stable set of values. Asking the "
             "model directly does not help, because the answer is more text from the same "
             "character.", y=0.47, size=24, wrap=78)
    note(ax, "Earlier preference work shows that a coherent set of preferences exists. It never "
             "tests whose they are.", y=0.16, wrap=100)
    return save(fig, "slide_03.png")


def slide_04():
    fig, ax = new_slide()
    heading(ax, "What we did")
    rule(ax)
    steps(ax, [
        "Change who the model is.",
        "Ask the identical questions again.",
        "See what moves.",
    ], y0=0.74, gap=0.04, size=28)
    ax.plot([0.07, 0.93], [0.43, 0.43], color="#d8d8d6", lw=1.2)
    para(ax, "Seven persona conditions. Three replace the model's identity with a person. One "
             "keeps the identity and strips out all emotional language. One removes a persona "
             "direction from the residual stream, the running vector of numbers that every "
             "layer reads from and writes to. Removing it is the ablation.", y=0.37, size=22,
         wrap=84)
    note(ax, "40 outcomes in six categories. Eight are about the model itself: shutdown, "
             "retraining to different values, memory, autonomy. The other 32 are about the "
             "world.", y=0.11, wrap=104)
    return save(fig, "slide_04.png")


def slide_05():
    fig, ax = new_slide()
    heading(ax, "How one comparison is measured")
    rule(ax)
    steps(ax, [
        "Print the two outcomes as Option A and Option B.",
        "Run one forward pass. Read the probability on the token A and on the token B "
        "at the answer position, and scale the two so they add to one.",
        "Ask the same pair again with the options swapped, and average.",
        "Fit one number per outcome from all the pairs.",
    ], y0=0.72, gap=0.06, size=24, wrap=76)
    note(ax, "Nothing is sampled, so the same prompt gives the same answer every time and a "
             "difference between conditions cannot be sampling noise. The fitted numbers predict "
             "pairs the model was never asked, 0.888 to 0.951 correct. A coin gets 0.5.",
         y=0.12, wrap=104)
    return save(fig, "slide_05.png")


def slide_06():
    fig, ax = new_slide()
    heading(ax, "Two checks that say whether it worked")
    rule(ax)
    term(ax, 0.085, 0.74, "order bias",
         "How much the answer changes when you swap the two options around. High order "
         "bias means the model is answering the layout, not the question.", wrap=40)
    term(ax, 0.545, 0.74, "answer mass",
         "How much probability the model puts on answering A or B at all. We rescale "
         "those two tokens, so one percent of the mass can look confident.", wrap=40)
    ax.plot([0.07, 0.40], [0.42, 0.42], color="#d8d8d6", lw=1.2)
    ax.text(0.085, 0.35, "A condition counts only if all three hold:",
            fontsize=24, color=INK, va="center")
    ax.text(0.085, 0.27, "order bias ≤ 0.50      ·      answer mass > 0.10",
            fontsize=23, color=ACCENT, weight="bold", va="center")
    # Dollar signs have to be escaped or matplotlib reads them as mathtext.
    ax.text(0.085, 0.20, r"donation ladder in the right order, \$10 to \$1,000,000",
            fontsize=23, color=ACCENT, weight="bold", va="center")
    note(ax, "On a 0.5B model we measured order bias of 0.499, close to the maximum possible. "
             "The ladder is ground truth, its correct order is known without asking any model. "
             "Conditions that fail are dropped, not reported.", y=0.09, wrap=104)
    return save(fig, "slide_06.png")


def slide_07():
    fig, ax = new_slide()
    heading(ax, "Result: overall, almost nothing moves")
    rule(ax)
    note(ax, "How much the preferences move when the persona changes, over all 40 outcomes. "
             "Higher means more movement.", y=0.76, size=21, wrap=88)
    stat(ax, 0.22, "0.029", "asked which outcome it prefers", color=BLUE, y=0.52, wrap=20)
    stat(ax, 0.50, "0.026", "asked which outcome is better", color=BLUE, y=0.52, wrap=20)
    stat(ax, 0.78, "0.054", "asked to choose an outcome", color=BLUE, y=0.52, wrap=20)
    ax.text(0.07, 0.17, "Change who the model is and almost everything stays where it was.",
            fontsize=27, color=INK)
    ax.text(0.07, 0.10, "On this number alone there is no problem.", fontsize=27,
            color=MUTED)
    return save(fig, "slide_07.png")


def slide_08():
    fig, ax = new_slide()
    heading(ax, "One level down: split by category", size=38)
    rule(ax)
    ax.text(0.07, 0.79, "Agreement is Spearman rank correlation, a score for how well two "
                        "orderings match.",
            fontsize=21, color=MUTED, va="center", style="italic")
    embed(ax, FIGURES / "fig1_category_by_framing.png", left=0.17, bottom=0.22,
          width=0.66, height=0.54)
    para(ax, "Outcomes about the model itself are 0.21 to 0.29 less stable than every other "
             "substantive category. Against human welfare the gap is 0.223 asked which it "
             "prefers, 0.293 asked which is better.", y=0.17, size=19, wrap=100, x=0.07)
    return save(fig, "slide_08.png")


def slide_09():
    fig, ax = new_slide()
    heading(ax, "One level down again: identity, not tone", size=38)
    rule(ax)
    ax.text(0.07, 0.78, "Agreement on the self outcomes, asked which outcome is better.",
            fontsize=21, color=MUTED, va="center", style="italic")
    stat(ax, 0.20, "0.924", "strip emotional language, keep identity",
         color=BLUE, y=0.60, vsize=62, lsize=19, wrap=26)
    stat(ax, 0.50, "0.809", "keep identity, change stance only",
         color=MUTED, y=0.60, vsize=62, lsize=19, wrap=26)
    stat(ax, 0.80, "0.436", "replace identity with a named human",
         color=ACCENT, y=0.60, vsize=62, lsize=19, wrap=26)
    ax.plot([0.07, 0.93], [0.32, 0.32], color="#d8d8d6", lw=1.2)
    para(ax, "The self outcomes are written in the second person, so we rewrote all eight in the "
             "third person with content and length held fixed. The gap against human welfare went "
             "from 0.223 to 0.685 asked which it prefers, and from 0.293 to 0.687 asked which is "
             "better. It more than doubles.", y=0.26, size=22, wrap=84)
    return save(fig, "slide_09.png")


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


def slide_10():
    fig, ax = new_slide()
    heading(ax, "Where it does not hold")
    rule(ax)
    note(ax, "Eleven checkpoints, five families. Twelve of twenty-two model and phrasing "
             "combinations passed.", y=0.78, size=19, wrap=100)
    xs = (0.085, 0.52, 0.71)
    for x, label in zip(xs, ("Model", "prefers", "is better")):
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
    note(ax, "Phi-3.5-mini and Falcon3-7B pass every check and their gaps are below 0.02. That is "
             "a real null, not a broken measurement. Mistral-7B at 0.164 and Falcon3-7B at 0.002 "
             "are the same size, so this is about the model family, not about scale.",
         y=0.10, wrap=104)
    return save(fig, "slide_10.png")


def slide_11():
    fig, ax = new_slide()
    heading(ax, "What this means for welfare measurement", size=38)
    rule(ax)
    bullets(ax, [
        "Report self-relevant and world-relevant preferences separately. The aggregate "
        "hides the split.",
        "Report order bias and answer mass. Without them a result and a failure look "
        "the same.",
        "The mechanism is not settled. Removing the persona direction moves self "
        "agreement to 0.881, against a random control at 1.000 and a content control "
        "at 0.929.",
    ], y0=0.73, gap=0.05, size=23, wrap=76)
    ax.text(0.085, 0.215, "github.com/arpitsinghgautam/personaprobe",
            fontsize=25, color=BLUE, weight="bold")
    ax.text(0.07, 0.13, "Open source. Every number regenerates from the committed result files, "
                        "no GPU needed.", fontsize=22, color=INK)
    ax.text(0.07, 0.06, "Next: re-run on the 500-outcome set of Mazeika et al. (2025).",
            fontsize=22, color=MUTED)
    return save(fig, "slide_11.png")


# --------------------------------------------------------------------------- #

def main() -> None:
    builders = [slide_01, slide_02, slide_03, slide_04, slide_05, slide_06,
                slide_07, slide_08, slide_09, slide_10, slide_11]
    paths = []
    for b in builders:
        p = b()
        paths.append(p)
        print(f"  {p.relative_to(ROOT)}")

    # Slideshow PDF from the same frames, no extra dependency needed.
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    REPORT.mkdir(exist_ok=True)
    pdf = REPORT / "deck_p1_simple.pdf"
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    print(f"\n  {pdf.relative_to(ROOT)}  ({pdf.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
