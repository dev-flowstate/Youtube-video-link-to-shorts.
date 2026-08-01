# Clip Captioner

Turns the clips produced by `YouTubeReplayDownloader` into vertical,
Shorts-ready videos with burned-in clipper-style captions.

For each clip it extracts the audio, transcribes it with word-level timing,
groups the words into short bursts, renders them as an ASS subtitle track with
the spoken word highlighted, centre-crops to 9:16, and encodes the result.

## Requirements

- Python 3.11+
- FFmpeg on PATH (`winget install Gyan.FFmpeg`)
- `py -m pip install -r requirements.txt`

Use `py -m pip`, not bare `pip` — on Windows `pip` is frequently not on PATH
even when Python is installed correctly.

The Whisper weights download automatically on first run (~460 MB for `small`)
into `~/.cache/huggingface`.

## Usage

```powershell
py main.py
```

Reads every `.mp4` in `INPUT_DIR` and writes captioned clips to `OUTPUT_DIR`.
Already-rendered files are skipped, so re-running is cheap.

```powershell
# Override folders
py main.py --input "E:\Youtube Videos\Videos" --output "E:\out"

# Process a single clip
py main.py --only "02m49s"
```

## Configuration

Everything tunable lives in `config.py`.

| Setting | Purpose |
|---|---|
| `WHISPER_MODEL` | `base` / `small` / `medium`. Bigger is more accurate and much slower on CPU. |
| `WHISPER_VOCABULARY` | Names and jargon to expect. Small models mangle proper nouns, which is exactly what a podcast clip is about — listing them is the difference between `WILLEY` and `Willy`, at no runtime cost. |
| `MAX_WORDS_PER_GROUP` | Words on screen at once. 2-3 reads fastest. |
| `FONT_NAME`, `FONT_SIZE` | `Arial Black` and `Impact` ship with Windows. |
| `COLOR_ACTIVE` | Highlight colour, in ASS `BBGGRR` order — **not** web hex. |
| `CAPTION_MARGIN_V` | Pixels from the bottom of the 1920-tall frame. |
| `SPLIT_LONG_CLIPS` | Split clips over `MAX_PART_DURATION_S` into numbered parts. |
| `WRITE_TITLES` | Save a suggested title beside each clip as a `.txt` file. |
| `TITLE_MAX_CHARS` | Titles longer than this get truncated on a word boundary. |
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
| `titler.py` | Scores the transcript to pick a title |
| `tracker.py` | Face detection, subject choice, and the smoothed crop path |
| `renderer.py` | Crop, burn-in, encode |
| `ffmpeg_tools.py` | FFmpeg discovery, execution, probing |
| `models/` | YuNet face detection model (227 KB, ships with the repo) |

## Titles

Each rendered clip gets a `.txt` beside it holding a suggested title, taken
from the clip's own strongest line.

No model is involved. Sentences are rebuilt from the word timings, then scored
on properties that separate a hook from filler:

| Property | Effect |
|---|---|
| Ends with a question mark | Strongest single signal |
| Superlatives — *never, biggest, insane, only* | Raises the score, up to a cap |
| Numbers or money | Specific claims read as credible |
| Emotional words — *scared, shocked, brutal* | Moderate lift |
| 4–12 words long | Title-shaped; longer and shorter are penalised |
| Filler — *um, like, yeah* | Penalised in proportion to how much there is |
| Near the middle of the clip | The clip is cut around its peak, so the middle is the moment |

The winner is tidied up: leading filler is dropped, a trailing full stop
removed, a question mark kept, and anything over `TITLE_MAX_CHARS` truncated
on a word boundary. If nothing scores well enough the clip's filename is used
instead.

Because the title is always a line the clip actually contains, it can never
describe something that was not said — which also means it is only as good as
the transcription. If names come out wrong, fill in `WHISPER_VOCABULARY`.

## Notes

Details that are easy to get wrong if you edit this:

- **ASS colours are `BBGGRR`**, the reverse of web hex. Amber is `00E5FF`,
  not `FFE500`.
- **File paths in the filtergraph are bare filenames**, with FFmpeg's working
  directory set to the folder holding them. This applies to both the subtitle
  file and the sendcmd script. An absolute Windows path breaks the parser,
  which treats `:` as an argument separator.
- **Crop panning is driven by `sendcmd`**, which rewrites the `crop` filter's
  `x` as the clip plays. Timestamps in that script are relative to the part,
  not the source clip, because input seeking rezeroes the clock.
- **Hardware encoding is probed, not assumed.** `h264_qsv` is listed in most
  FFmpeg builds but fails without a matching iGPU and driver, so the renderer
  encodes a throwaway frame once to check before relying on it.
- **Transcription runs a clip ahead of encoding.** Transcription is CPU-bound
  and encoding runs on the GPU, so overlapping them costs nothing in
  contention and recovers most of the transcription time. A one-item queue
  keeps at most one clip's work in flight.
- **Face choice follows the nearest subject, not the largest.** With two
  people at similar distance, tiny size changes flip which is momentarily
  bigger, and the crop swings between them. Size only decides after a cut,
  when there is no previous framing to stay near.
