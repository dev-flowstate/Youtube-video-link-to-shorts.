# Clip Captioner

Turns the clips produced by `YouTubeReplayDownloader` into vertical,
Shorts-ready videos with burned-in clipper-style captions.

For each clip it extracts the audio, transcribes it with word-level timing,
groups the words into short bursts, renders them as an ASS subtitle track with
the spoken word highlighted, centre-crops to 9:16, and encodes the result.

## Requirements

- Python 3.11+
- FFmpeg on PATH (`winget install Gyan.FFmpeg`)
- `pip install -r requirements.txt`

The Whisper weights download automatically on first run (~460 MB for `small`)
into `~/.cache/huggingface`.

## Usage

```bash
python main.py
```

Reads every `.mp4` in `INPUT_DIR` and writes captioned clips to `OUTPUT_DIR`.
Already-rendered files are skipped, so re-running is cheap.

```bash
# Override folders
python main.py --input "E:\Youtube Videos\Videos" --output "E:\out"

# Process a single clip
python main.py --only "02m49s"
```

## Configuration

Everything tunable lives in `config.py`.

| Setting | Purpose |
|---|---|
| `WHISPER_MODEL` | `base` / `small` / `medium`. Bigger is more accurate and much slower on CPU. |
| `MAX_WORDS_PER_GROUP` | Words on screen at once. 2-3 reads fastest. |
| `FONT_NAME`, `FONT_SIZE` | `Arial Black` and `Impact` ship with Windows. |
| `COLOR_ACTIVE` | Highlight colour, in ASS `BBGGRR` order — **not** web hex. |
| `CAPTION_MARGIN_V` | Pixels from the bottom of the 1920-tall frame. |
| `SPLIT_LONG_CLIPS` | Split clips over `MAX_PART_DURATION_S` into numbered parts. |
| `VIDEO_CRF`, `VIDEO_PRESET` | Quality vs encode time. |

## Layout

| File | Responsibility |
|---|---|
| `main.py` | CLI entry point and argument parsing |
| `pipeline.py` | Orchestrates clip → transcript → captions → render |
| `config.py` | All tunable settings |
| `models.py` | `Word`, `CaptionGroup`, `ClipPart`, `VideoInfo` |
| `audio.py` | Extracts 16 kHz mono WAV for Whisper |
| `transcriber.py` | faster-whisper wrapper, word-level timestamps |
| `caption_builder.py` | Groups words into short on-screen bursts |
| `ass_writer.py` | Emits the ASS subtitle file |
| `splitter.py` | Splits long clips on speech gaps |
| `renderer.py` | Crop, burn-in, encode |
| `ffmpeg_tools.py` | FFmpeg discovery, execution, probing |

## Notes

Two details that are easy to get wrong if you edit this:

- **ASS colours are `BBGGRR`**, the reverse of web hex. Amber is `00E5FF`,
  not `FFE500`.
- **The `subtitles` filter is given a bare filename**, with FFmpeg's working
  directory set to the file's folder. Passing an absolute Windows path breaks
  the filtergraph parser, which treats `:` as an argument separator.
