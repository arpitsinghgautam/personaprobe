"""Assemble the plainer project-1 presentation video: slides + narration -> MP4.

scripts/24_video_p1.py builds the original video and stays exactly as it is.
This is the companion to scripts/27_slides_p1_simple.py: the same material read
in ordinary English, for a viewer meeting the work for the first time. It reads
figures/slides_p1_simple and writes video/p1/personaprobe_simple.mp4, touching
nothing the first video produced.

    .venv\\Scripts\\python.exe scripts\\28_video_p1_simple.py

The voice list matches 24_video_p1.py so the two videos sound like the same
person recorded them.

Re-recording in your own voice later does not require redoing any of this: play
report/deck_p1_simple.pdf fullscreen, narrate over it with Game Bar (Win+G), and
the script below is the read.

Every number spoken here is quoted from report/report_4page.md. Numbers are
written out longhand because the TTS engine reads "0.685" as "zero point six
hundred eighty five", and letters are hyphenated ("A-I", "G-P-U") so they are
spelled rather than pronounced as words.

Outputs video/p1/personaprobe_simple.mp4 plus per-slide MP3s under
video/p1/audio_simple/.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "figures" / "slides_p1_simple"
VIDEO = ROOT / "video" / "p1"
AUDIO = VIDEO / "audio_simple"

VOICE_PREFERENCES = [
    "en-US-AndrewNeural",
    "en-GB-RyanNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
]

# One entry per slide, in order. Keep these in sync with 27_slides_p1_simple.py.
# The register is deliberately flatter than 24_video_p1.py: short sentences, no
# signposting, no adjectives doing persuasive work. Each technical term is
# defined in one sentence the first time it is spoken.
NARRATION: list[str] = [
    # 1, title
    "Whose preferences are they? Ask a language model to choose between two outcomes and it "
    "answers. Ask over hundreds of pairs and the answers hang together. A-I welfare research "
    "reads those preferences as evidence about what the model wants.",

    # 2, the question
    "A model says it prefers one outcome over another. Are those the model's preferences, or the "
    "character's? Every chat model plays a character called the assistant. That character has a "
    "stance on being shut down, on being retrained, on losing its memory. The words on screen "
    "are the same either way.",

    # 3, why the text cannot settle it
    "You cannot settle this by reading the text. If the preferences belong to the model or to "
    "the character, the text is the same. A character played consistently looks like a stable "
    "set of values. Earlier work shows a coherent set of preferences exists. It never tests "
    "whose.",

    # 4, the intervention
    "So we asked something a measurement can settle. Change who the model is. Ask the identical "
    "questions again. See what moves. Seven persona conditions. Three replace the model's "
    "identity with a person. One keeps the identity and strips out emotional language. One "
    "removes a persona direction from the residual stream, the running vector of numbers every "
    "layer reads and writes. Forty outcomes in six categories. Eight are about the model itself.",

    # 5, the mechanics of one comparison
    "Print the two outcomes as Option A and Option B. Run one forward pass. Read the probability "
    "on the token A and the token B, and scale the two to add to one. Ask the same pair again "
    "with the options swapped, and average. Nothing is sampled, so the answer is the same every "
    "time. From all the pairs we fit one number per outcome. Those numbers predict pairs the "
    "model was never asked, zero point eight eight eight to zero point nine five one correct. A "
    "coin gets zero point five.",

    # 6, the validity checks
    "Two checks say whether the measurement is real. Order bias is how much the answer changes "
    "when you swap the two options around. High order bias means the model is answering the "
    "layout, not the question. On a zero point five B model we measured zero point four nine "
    "nine. Answer mass is how much probability the model puts on answering A or B at all. We "
    "rescale those two tokens, so one percent of the mass can look confident. A condition counts "
    "only if both clear their thresholds, and the model orders the donation ladder correctly.",

    # 7, top-level result
    "Over all forty outcomes the preferences barely move. Persona dependence is zero point zero "
    "two nine under the prefer phrasing, zero point zero two six under better, zero point zero "
    "five four under choose. Change who the model is and almost everything stays where it was. "
    "On this number alone there is no problem.",

    # 8, one level down
    "Split the outcomes by category and that reverses. Agreement here is Spearman rank "
    "correlation, a score for how well two orderings of the same outcomes match. Outcomes about "
    "the model itself are zero point two one to zero point two nine less stable than every other "
    "substantive category. Against human welfare the gap is zero point two two three under "
    "prefer and zero point two nine three under better. The overall stability came from outcomes "
    "the model has no stake in.",

    # 9, one level down again
    "It is identity that matters, not tone. Strip all emotional language but keep the model's "
    "identity and self category agreement is zero point nine two four. Replace the identity with "
    "a named human and it falls to zero point four three six. Changing only the stance sits "
    "between, at zero point eight zero nine. The self outcomes are in the second person, so we "
    "rewrote all eight in the third person, holding content and length fixed. The gap against "
    "human welfare went to zero point six eight five and zero point six eight seven. It more "
    "than doubles.",

    # 10, the limits
    "Across eleven checkpoints and five families, only twelve of twenty two model and phrasing "
    "combinations passed the checks. The effect replicates inside Qwen and holds under four bit "
    "quantisation. It is partial in Mistral. It is absent in Phi three point five mini and "
    "Falcon three, seven B, which both pass every check with gaps below zero point zero two. "
    "That is a real null, not a broken measurement.",

    # 11, close
    "Report self relevant and world relevant preferences separately, because the aggregate hides "
    "the split. Report order bias and answer mass, or a result and a failure look the same. The "
    "mechanism is not settled. Removing the persona direction moves self agreement to zero point "
    "eight eight one, against a random control at one point zero zero zero and a content control "
    "at zero point nine two nine. Everything is open source. Every number regenerates from the "
    "committed files without a G-P-U. Thank you.",
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
                         "keep 27_slides_p1_simple.py and NARRATION in sync")

    segments = []
    total = 0.0
    for i, (img, aud) in enumerate(zip(slides, audios), start=1):
        # Segment names are prefixed so a concurrent run of 24_video_p1.py could
        # not collide with these intermediates in the same directory.
        seg = VIDEO / f"seg_simple_{i:02d}.mp4"
        # apad holds the slide ~0.7s past the end of speech so it doesn't cut hard.
        #
        # -t is doing real work here. `-loop 1` makes the image an infinite input,
        # and apad makes the audio infinite too, so -shortest has no finite stream
        # to clip against and every segment ran about 3.3s long. Across eleven
        # slides that was 36s of silence on a static frame. Set the length
        # explicitly from the narration instead.
        seg_len = duration(ffprobe, aud) + 0.7
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-loop", "1", "-i", str(img), "-i", str(aud),
             "-af", "apad=pad_dur=0.7", "-t", f"{seg_len:.3f}",
             "-c:v", "libx264", "-tune", "stillimage", "-preset", "medium",
             "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
             "-vf", "scale=1920:1080", "-r", "24", "-shortest", str(seg)],
            check=True)
        d = duration(ffprobe, seg)
        total += d
        segments.append(seg)
        print(f"  segment {i:02d}  {d:5.1f}s")

    listing = VIDEO / "segments_simple.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments))

    out = VIDEO / "personaprobe_simple.mp4"
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
