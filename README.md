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

Reads the video's most-replayed data, detects the peaks, and downloads just
those sections as clips. Output is **landscape, no subtitles** — raw material
for step 2.

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
git clone https://github.com/dev-flowstate/Youtube-Videos.git
cd Youtube-Videos

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
frame. It samples keyframes, finds the largest face with OpenCV's YuNet
detector, then pans the crop window along a smoothed path.

Sampling only keyframes is close to free — about **0.8s of tracking per 20s of
video** — and because encoders put a keyframe at every scene cut, it lands a
sample exactly where the framing needs to change.

If no face is found, it falls back to centre framing rather than failing.

| Setting | Does what |
|---|---|
| `TRACKING_SMOOTHING` | 0–1. Lower is steadier but slower to follow. The main jitter dial. |
| `TRACKING_MAX_PAN_PX_PER_S` | Speed limit on the pan, so a bad detection can't whip the frame |
| `TRACKING_MAX_JUMP_FRACTION` | Ignores faces this far from current framing — usually bystanders |
| `TRACKING_SAMPLE_MODE` | `keyframes` (default, nearly free) or `dense` for fast action |

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
| Face tracking (keyframe sampling) | 0.8s |
| Render, software `libx264` | 73.6s |
| Render, hardware `h264_qsv` | 25.6s |

---

## Layout

```
YouTubeReplayDownloader/   Step 1 — find and download the replay peaks
ClipCaptioner/             Step 2 — transcribe, caption, crop to vertical
```

Each folder has its own README with the details of how it works internally.

Clips and rendered videos are gitignored — this repo holds the code only.
