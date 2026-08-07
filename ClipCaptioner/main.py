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

    captions = parser.add_mutually_exclusive_group()
    captions.add_argument("--captions", dest="captions", action="store_true", default=None,
                          help="Burn in captions without asking.")
    captions.add_argument("--no-captions", dest="captions", action="store_false",
                          help="Skip captions; still crop to vertical and track faces.")

    parser.add_argument("--language", type=str, default=None,
                        help="Force a language code (e.g. ur). Default is to detect it.")
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


def main() -> int:
    args = _parse_args()

    if args.language:
        config.WHISPER_LANGUAGE = args.language
    config.BURN_CAPTIONS = args.captions if args.captions is not None else _ask_about_captions()

    language_note = config.WHISPER_LANGUAGE or "auto-detect"

    print("\nClip Captioner")
    print("=" * 14)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Model:    {config.WHISPER_MODEL} ({config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE_TYPE})")
    print(f"Language: {language_note}")
    print(f"Captions: {'burned in' if config.BURN_CAPTIONS else 'off'}")
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
