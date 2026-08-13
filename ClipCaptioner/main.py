"""Caption cut clips as vertical, Shorts-ready videos.

Edit config.py to change folders or styling, then run:
    python main.py

Optional overrides:
    python main.py --input "E:\\Youtube Videos\\Videos" --output "E:\\out"
    python main.py --only "part of a filename"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
import ffmpeg_tools
import pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn clips into captioned vertical Shorts.")
    parser.add_argument("--input", type=Path, default=config.INPUT_DIR, help="Folder of source clips.")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR, help="Folder for rendered clips.")
    parser.add_argument("--only", type=str, default=None, help="Only process clips whose name contains this text.")
    parser.add_argument("--layout", choices=("crop", "fit", "stacked"), default=None,
                        help="How 16:9 becomes 9:16. Skips the prompt.")

    captions = parser.add_mutually_exclusive_group()
    captions.add_argument("--captions", dest="captions", action="store_true", default=None,
                          help="Burn in captions without asking.")
    captions.add_argument("--no-captions", dest="captions", action="store_false",
                          help="Skip captions; still crop to vertical and track faces.")

    parser.add_argument("--language", type=str, default=None,
                        help="Force a language code (e.g. ur). Default is to detect it.")

    tracking = parser.add_mutually_exclusive_group()
    tracking.add_argument("--face-tracking", dest="face_tracking", action="store_true", default=None,
                          help="Follow faces when cropping to vertical.")
    tracking.add_argument("--no-face-tracking", dest="face_tracking", action="store_false",
                          help="Hold the centre instead. Right for gameplay.")
    return parser.parse_args()


def _ask_about_captions() -> bool:
    """Ask whether to burn captions in.

    Worth asking rather than assuming: commentary in a language the model
    handles poorly can caption badly enough to be worse than no captions, and
    a clip destined for a compilation may not want them at all.
    """
    try:
        answer = input("Burn captions into these clips? [Y/n] ").strip().lower()
    except EOFError:
        # Piped or scheduled run - carry on rather than hang waiting on stdin.
        return True

    return answer not in {"n", "no"}


def _ask_about_layout() -> str:
    """Ask how the 16:9 source should become a 9:16 Short.

    Worth asking rather than defaulting, because the default is destructive and
    silently so. A 9:16 slice keeps 31.6% of the width, so cropping discards
    two thirds of the picture. On a talking head that is empty room either
    side. On a stream it is the game the reaction is about, and the finished
    Short shows a face reacting to something the viewer cannot see.
    """
    print("\nWhat kind of clips are these?")
    print("  [1] Podcast or interview  - crop to the speaker")
    print("  [2] Streamer or reaction  - camera on top, the screen below")
    print("  [3] Gameplay or esports   - fit the whole frame, nothing cut")

    try:
        answer = input("Choose [1]: ").strip() or "1"
    except EOFError:
        # Piped or scheduled run. The safe default is the one that cannot throw
        # anything away, not the one that happens to be first.
        return "fit"

    return {"2": "stacked", "3": "fit"}.get(answer, "crop")


def main() -> int:
    args = _parse_args()

    if args.language:
        config.WHISPER_LANGUAGE = args.language
    config.BURN_CAPTIONS = args.captions if args.captions is not None else _ask_about_captions()

    # Whatever produced these clips may have left a note about what they are.
    marker = pipeline.apply_content_marker(args.input) or {}
    content_type = marker.get("content_type")

    # Order of authority: an explicit flag, then a marker that actually stated
    # a layout, then ask.
    #
    # The middle test has to be "did it state a layout", not "does a marker
    # exist". The downloader's marker records a tail padding and says nothing
    # about framing, so treating its presence as an answer meant a note left in
    # one folder silently decided the layout for whatever footage was dropped
    # there afterwards - which is exactly how stream clips ended up cropped
    # without ever being asked about.
    if args.layout is not None:
        config.CROP_MODE = args.layout
    elif marker.get("crop_mode") in ("crop", "fit", "stacked"):
        print(f"Layout from content.json: {config.CROP_MODE}")
    else:
        config.CROP_MODE = _ask_about_layout()

    if config.CROP_MODE == "stacked":
        # There is no single crop window in a stacked layout - the camera and
        # the screen are placed separately - so a moving one has nothing to do.
        config.TRACK_FACES = False

    # An explicit flag always beats the marker.
    if args.face_tracking is not None:
        config.TRACK_FACES = args.face_tracking

    language_note = config.WHISPER_LANGUAGE or "auto-detect"

    print("\nClip Captioner")
    print("=" * 14)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    if content_type:
        print(f"Content:  {content_type}")
    print(f"Model:    {config.WHISPER_MODEL} ({config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE_TYPE})")
    print(f"Language: {language_note}")
    print(f"Captions: {'burned in' if config.BURN_CAPTIONS else 'off'}")
    if config.CROP_MODE == "crop":
        print(f"Layout:   crop, {'face-tracked' if config.TRACK_FACES else 'centred'}")
    else:
        print(f"Layout:   {config.CROP_MODE}")
    if config.SOURCE_TAIL_PADDING_SECONDS > 0:
        # Worth stating rather than applying quietly. It is right for clips the
        # downloader cut, and wrong for anything else dropped in the same
        # folder afterwards - where it would trim real footage off the end.
        print(
            f"Tail:     last {config.SOURCE_TAIL_PADDING_SECONDS:.0f}s treated as "
            "padding (from content.json)"
        )
    print(f"Format:   {config.TARGET_WIDTH}x{config.TARGET_HEIGHT}\n")

    try:
        ffmpeg_tools.resolve_tool_dir()
    except ffmpeg_tools.FFmpegNotFoundError as exc:
        print(str(exc))
        return 1

    try:
        if args.only:
            clips = [
                c
                for c in pipeline.find_clips(args.input, args.output)
                if args.only.lower() in c.name.lower()
            ]
            if not clips:
                print(f"No clips matched --only {args.only!r}")
                return 1
            args.output.mkdir(parents=True, exist_ok=True)
            produced = []
            for position, clip in enumerate(clips, start=1):
                print(f"[{position}/{len(clips)}] {clip.name}")
                produced.extend(pipeline.process_clip(clip, args.output))
                print()
        else:
            produced = pipeline.run(args.input, args.output)
    except pipeline.PipelineError as exc:
        print(f"Failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    if not produced:
        print("Nothing was rendered.")
        return 0

    print(f"Done - {len(produced)} file(s):\n")
    for path in produced:
        print(f"  - {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
