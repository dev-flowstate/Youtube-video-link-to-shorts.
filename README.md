# YouTube Replay Clipper

Turn any YouTube video into vertical, subtitled Shorts / TikToks — automatically.

It finds the moments people **rewatch most** (YouTube's own "most replayed"
graph), cuts those out, then transcribes them and burns in bold clipper-style
captions.

```
YouTube link
     ↓
[1] YouTubeReplayDownloader   →  landscape clips, no subtitles
     ↓
[2] ClipCaptioner             →  vertical 1080×1920, burned-in subtitles
```

Two separate programs, run one after the other.

---

## What each step does

### Step 1 — `YouTubeReplayDownloader`

Finds the best moments in a video and saves each as a clip. Output is
**landscape, no subtitles** — raw material for step 2.

It works on normal videos *and* past live broadcasts, using whichever signal is
available:

| Signal | Used when |
|---|---|
| **Most replayed** | Normal videos and older broadcasts |
| **Chat activity** | Past live streams — message-rate spikes |
| **Audio loudness** | Anything else |

Live streams have no most-replayed graph until well after they end, so chat
stands in. Streams that are **still broadcasting** are rejected: a clip cannot
be cut from a video that is still growing.

### Step 2 — `ClipCaptioner`

Takes those clips and, for each one: extracts the audio, transcribes it with
word-level timing, centre-crops to 9:16, and burns in captions with the spoken
word highlighted. Clips longer than ~3 minutes are split into numbered parts
that fit Shorts.

---

## Setup

You need **Python 3.11+** and **FFmpeg**.

```bash
# 1. Get the code
# The trailing "." is part of the repo name, and the folder is named
# explicitly because Windows strips trailing dots from directory names.
git clone https://github.com/dev-flowstate/Youtube-video-link-to-shorts..git youtube-shorts
cd youtube-shorts

# 2. FFmpeg (Windows)
winget install Gyan.FFmpeg

# 3. Dependencies
pip install -r YouTubeReplayDownloader/requirements.txt
pip install -r ClipCaptioner/requirements.txt
```

On macOS use `brew install ffmpeg`; on Debian/Ubuntu `sudo apt install ffmpeg`.

> The first captioning run downloads the Whisper speech model (~460 MB). That
> happens once and is cached afterwards.

---

## Making clips from a new video

**1. Paste your link.** Open `YouTubeReplayDownloader/main.py` and edit one
line near the top:

```python
YOUTUBE_URL = "https://youtu.be/YOUR_VIDEO_ID"
```

A past-broadcast link works too:

```python
YOUTUBE_URL = "https://www.youtube.com/live/YOUR_VIDEO_ID"
```

**2. Download the clips.**

```bash
cd YouTubeReplayDownloader
python main.py
```

You'll see the detected peaks, then the clips appear in
`YouTubeReplayDownloader/output/`. These are landscape with no subtitles.

**3. Add captions and go vertical.**

```bash
cd ../ClipCaptioner
python main.py
```

Finished Shorts land in `ClipCaptioner/output/`, ready to upload.

That's it. For the next video, change the link and repeat.

---

## Useful options

Caption clips from somewhere else:

```bash
python main.py --input "D:\my\clips" --output "D:\my\shorts"
```

Do a single clip first to check the look before committing to a whole batch:

```bash
python main.py --only "02m49s"
```

Re-running is safe — anything already rendered is skipped, not redone.

---

## Tuning the captions

Everything lives in `ClipCaptioner/config.py`.

| Setting | Does what |
|---|---|
| `WHISPER_MODEL` | `base` (fast, sloppier) · `small` (default) · `medium` (slow, most accurate) |
| `MAX_WORDS_PER_GROUP` | Words on screen at once. 2–3 reads fastest. |
| `FONT_NAME` / `FONT_SIZE` | `Arial Black` and `Impact` ship with Windows |
| `COLOR_ACTIVE` | Highlight colour — see the gotcha below |
| `CAPTION_MARGIN_V` | Height of the captions in the frame |
| `SPLIT_LONG_CLIPS` | Split long clips into Shorts-length parts |
| `UPPERCASE_CAPTIONS` | Uppercase reads louder; it's the clipper convention |
| `TRACK_FACES` | Follow the speaker when choosing the crop (see below) |
| `PREFER_HARDWARE_ENCODER` | Use the GPU encoder when one is available |

One thing that bites people editing this: **caption colours are `BBGGRR`, not
the usual `RRGGBB`.** Amber is `00E5FF`. Get it backwards and your highlight
comes out the wrong colour.

---

## Face tracking

The 9:16 crop follows the speaker instead of blindly taking the middle of the
frame. It samples several times a second, finds the largest face with OpenCV's
YuNet detector, then glides the crop window along a smoothed path.

Three things make the motion read as smooth rather than stuttery:

- **Dense sampling.** Sparse samples mean the crop sits still then lurches.
- **Interpolation.** The path is resampled to a fine grid, so the crop moves
  in steps too small to see. FFmpeg's `crop` only takes discrete positions —
  without this it snaps between them.
- **Zero-phase smoothing.** The filter runs forwards *and* backwards, so it
  removes jitter without the drag a one-way filter introduces.

Scene cuts are detected and handled as hard boundaries: the crop repositions
instantly across a cut, since gliding through one looks like a mistake. Short
"shots" are merged first, because handheld camera movement otherwise trips the
cut detector and produces constant snapping.

If no face is found, framing holds its last position rather than jumping to
the centre and back.

| Setting | Does what |
|---|---|
| `TRACKING_SAMPLE_FPS` | Face checks per second. Below ~3 the crop lags the subject. |
| `TRACKING_SMOOTHING_WINDOW_S` | The main dial. Larger is smoother and slower to react. |
| `TRACKING_OUTPUT_FPS` | Path resolution. Below ~10 the motion reads as stuttery. |
| `TRACKING_CUT_THRESHOLD` | Higher if camera shake is being mistaken for cuts |
| `TRACKING_MIN_SHOT_S` | Shots shorter than this are treated as motion, not edits |
| `TRACKING_MAX_PAN_PX_PER_S` | Speed limit, so a bad detection can't whip the frame |
| `TRACKING_DEADZONE_PX` | Ignores micro-movement so the frame is not always drifting |

The model lives in `ClipCaptioner/models/` and ships with the repo, so there's
nothing to download.

---

## Speed

Transcription runs on the CPU unless you have an NVIDIA GPU — budget roughly
**an hour per hour of footage** with the default `small` model. Drop to `base`
if you want it faster and can live with more mistakes on names and numbers.

Encoding uses your GPU when possible. On an Intel laptop chip, QuickSync is
about **3× faster than software encoding** and keeps the work off the CPU
cores, which also helps a laptop avoid thermal throttling on long batches. It
falls back to `libx264` automatically if no hardware encoder works.

Measured on an i5-1135G7, for a 20-second 4K clip:

| Stage | Time |
|---|---|
| Face tracking at 4 samples/sec | 18.5s |
| Render, software `libx264` | 73.6s |
| Render, hardware `h264_qsv` | 25.6s |
| **Whole pipeline** (transcribe + track + render) | **68s** |

That works out to roughly 3.4× realtime end to end, so an hour of clips takes
about three and a half hours. Tracking is the part you can trade away: drop
`TRACKING_SAMPLE_FPS` to 2 to halve its cost, at the price of less responsive
framing, or set `TRACK_FACES = False` to skip it entirely.

---

## Layout

```
YouTubeReplayDownloader/   Step 1 — find and download the replay peaks
ClipCaptioner/             Step 2 — transcribe, caption, crop to vertical
```

Each folder has its own README with the details of how it works internally.

Clips and rendered videos are gitignored — this repo holds the code only.
