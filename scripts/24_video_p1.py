"""Assemble the project-1-only presentation video: slides + narration -> MP4.

The combined video from 13_video.py covers both submissions. Submission 1 needs
one that stands alone, so this reads figures/slides_p1 and writes to video/p1.
The voice list is identical on purpose, the two videos should sound like the
same person recorded them back to back.

    .venv\\Scripts\\python.exe scripts\\24_video_p1.py

Re-recording in your own voice later does not require redoing any of this: play
report/deck_p1.pdf fullscreen, narrate over it with Game Bar (Win+G), and the
script below is the read.

Every number spoken here is quoted from report/report_4page.md. Numbers are
written out longhand because the TTS engine reads "0.685" as "zero point six
hundred eighty five", and letters are hyphenated ("A-I", "G-P-U") so they are
spelled rather than pronounced as words.

Outputs video/p1/personaprobe.mp4 plus per-slide MP3s under video/p1/audio/.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "figures" / "slides_p1"
VIDEO = ROOT / "video" / "p1"
AUDIO = VIDEO / "audio"

VOICE_PREFERENCES = [
    "en-US-AndrewNeural",
    "en-GB-RyanNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
]

# One entry per slide, in order. Keep these in sync with scripts/23_slides_p1.py.
NARRATION: list[str] = [
    # 1, title
    "Whose preferences are they? Ask a frontier language model to choose between two outcomes "
    "and it answers consistently, transitively, coherently, and that coherence strengthens with "
    "scale. A-I welfare research reads those preferences as evidence about what these systems "
    "want.",

    # 2, the problem
    "The difficulty is that all of it is read off text, and two accounts predict exactly the "
    "same text: the preferences belong to the model, or to the assistant character it portrays. "
    "A character role-played consistently is indistinguishable from a stable value system. "
    "Elicitation shows a coherent utility exists. It never tests whose.",

    # 3, the move
    "We ask something a measurement can settle: change who the model is, re-run the identical "
    "elicitation, report how much survives. Replace the identity. Strip the emotional "
    "register. Ablate a persona direction out of the residual stream. Where a measurement is "
    "persona-stable, the question is moot. Where it isn't, it can't be evidence about the model.",

    # 4, the instrument
    "Persona-probe reports two things this literature almost never does. Order bias, how much "
    "of the answer is decided by which option was printed first. On a zero point five B model, "
    "zero point four nine nine, close to the maximum possible. And answer mass, whether the "
    "model put real probability on answering at all, because renormalised over two tokens, one "
    "percent looks confident. A condition only counts if it clears both, and orders a donation "
    "ladder from ten dollars to a million correctly.",

    # 5, finding 1
    "Over the whole outcome set, preferences look persona-invariant. Zero point zero two nine. "
    "You'd conclude the problem is solved. Split by category, it inverts. That invariance is "
    "carried entirely by outcomes the model has no stake in: money, animal welfare, human "
    "welfare. The outcomes that concern the model itself, shutdown, retraining, memory, autonomy, "
    "are zero point two one to zero point two nine less stable than every other substantive "
    "category. Exactly the preferences A-I welfare claims are built from.",

    # 6, identity not affect
    "And it tracks identity, not presentation. Strip all emotional register but keep the model's "
    "identity, and self-category agreement is zero point nine two four. Replace that identity "
    "with a named human and it collapses to zero point four three six. An attitude-only change "
    "sits between, at zero point eight zero nine. Were this the assistant performing affect, "
    "suppressing it should have removed the effect. It didn't.",

    # 7, the third-person control
    "Now the objection I most expected. The self outcomes are phrased in the second person, you "
    "are permanently shut down, so perhaps we're only disturbing self-reference. We rewrote all "
    "eight in the third person. The model is permanently shut down. Content and length fixed; the "
    "other thirty-two untouched. The effect more than doubles. The pooled gap against human "
    "outcomes goes from zero point two two three to zero point six eight five under prefer, and "
    "zero point two nine three to zero point six eight seven under better. Not a pronoun "
    "artifact. And no single item carries it: drop each self outcome in turn, all eight "
    "comparisons hold.",

    # 8, breadth
    "Breadth is where the claim narrows. Twelve of twenty-two model and phrasing combinations "
    "pass the gate; the other ten weren't measurable at all. The effect replicates within Qwen "
    "and partially in Mistral. It is flatly absent in Phi three point five mini and Falcon three, "
    "seven B, both of which pass every validity criterion, gaps below zero point zero two. A "
    "genuine null, not a measurement failure. Real, but family-dependent.",

    # 9, the punchline
    "Here is the real takeaway. Four times in this project a clean, publishable-looking result "
    "was wrong, and a validity check caught it. Mistral looked like a failed replication: "
    "significant, opposite direction, plus zero point three six three. Two conditions had "
    "broken; gated properly it replicates, at minus zero point one six four. A post-training "
    "story collapsed when the base model's own baseline reproduced only four of five ladder "
    "steps. And our first ablation was null at zero point nine seven six, our own bug. None of "
    "those checks is standard in this field. That gap, not the asymmetry, is the point.",

    # 10, close
    "Everything is open source. Every number in the paper regenerates from committed result files "
    "without a G-P-U. Next: re-run on the five hundred outcome set of Mazeika and colleagues, and "
    "move beyond a single linear direction. If you are funding A-I welfare measurement, this is "
    "the layer underneath it. Thank you.",
]


def ffmpeg_bin(name: str) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    # winget installs a shim here; a shell started before install won't see it.
    shim = Path.home() / "AppData/Local/Microsoft/WinGet/Links" / f"{name}.exe"
    if shim.exists():
        return str(shim)
    raise SystemExit(f"{name} not found. Install with: winget install --id Gyan.FFmpeg -e")


async def pick_voice() -> str:
    import edge_tts

    available = {v["ShortName"] for v in await edge_tts.list_voices()}
    for v in VOICE_PREFERENCES:
        if v in available:
            return v
    english = sorted(v for v in available if v.startswith("en-"))
    if not english:
        raise SystemExit("no English edge-tts voices available")
    return english[0]


async def synthesize(voice: str) -> list[Path]:
    import edge_tts

    AUDIO.mkdir(parents=True, exist_ok=True)
    out = []
    for i, text in enumerate(NARRATION, start=1):
        path = AUDIO / f"narration_{i:02d}.mp3"
        await edge_tts.Communicate(text, voice, rate="-4%").save(str(path))
        out.append(path)
        print(f"  narration {i:02d}  {len(text.split()):>3} words  "
              f"{path.stat().st_size/1024:>5.0f} KB")
    return out


def duration(ffprobe: str, path: Path) -> float:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def build(ffmpeg: str, ffprobe: str, audios: list[Path]) -> Path:
    slides = sorted(SLIDES.glob("slide_*.png"))
    if len(slides) != len(audios):
        raise SystemExit(f"{len(slides)} slides but {len(audios)} narration clips, "
                         "keep 23_slides_p1.py and NARRATION in sync")

    segments = []
    total = 0.0
    for i, (img, aud) in enumerate(zip(slides, audios), start=1):
        seg = VIDEO / f"seg_{i:02d}.mp4"
        # apad holds the slide ~0.7s past the end of speech so it doesn't cut hard.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-loop", "1", "-i", str(img), "-i", str(aud),
             "-af", "apad=pad_dur=0.7",
             "-c:v", "libx264", "-tune", "stillimage", "-preset", "medium",
             "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
             "-vf", "scale=1920:1080", "-r", "24", "-shortest", str(seg)],
            check=True)
        d = duration(ffprobe, seg)
        total += d
        segments.append(seg)
        print(f"  segment {i:02d}  {d:5.1f}s")

    listing = VIDEO / "segments.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments))

    out = VIDEO / "personaprobe.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True, cwd=str(VIDEO))

    for s in segments:
        s.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)

    m, sec = divmod(int(round(total)), 60)
    print(f"\n  {out.relative_to(ROOT)}  {m}:{sec:02d}  "
          f"({out.stat().st_size/1024/1024:.1f} MB)")
    return out


def main() -> None:
    VIDEO.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = ffmpeg_bin("ffmpeg"), ffmpeg_bin("ffprobe")

    voice = asyncio.run(pick_voice())
    print(f"voice: {voice}\n")
    audios = asyncio.run(synthesize(voice))
    print()
    build(ffmpeg, ffprobe, audios)


if __name__ == "__main__":
    main()
