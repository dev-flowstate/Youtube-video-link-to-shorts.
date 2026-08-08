# YouTube Replay Clipper

Turn any YouTube video into vertical, subtitled Shorts / TikToks — automatically.

It finds the moments people **rewatch most** (YouTube's own "most replayed"
graph), cuts those out, then transcribes them and burns in bold clipper-style
captions.

```
YouTube link
     ↓
[1] YouTubeReplayDownloader   →  landscape clips, no subtitles
    or EsportsClipper         →  fight clips + a compilation
     ↓
[2] ClipCaptioner             →  vertical 1080×1920, burned-in subtitles
```

Two steps. Step 1 has two front ends — pick the one that matches the content.

| Step 1 | For | Finds moments by |
|---|---|---|
| **YouTubeReplayDownloader** | Podcasts, vlogs, talk content | Most-replayed graph, chat spikes, speech energy |
| **EsportsClipper** | Tournament broadcasts | Gunfire bursts gated against caster reaction |

Esports needs its own detector: casters talk continuously, so speech energy is
high everywhere and cannot separate a squad wipe from a rotation. See
[EsportsClipper/README.md](EsportsClipper/README.md).

**The downloader asks what the video is** before it starts, because that
decides which evidence it trusts. Talk content uses the most-replayed graph and
chat only — speech energy is deliberately withheld there, since it finds
whoever is loudest rather than whatever is interesting. Picking the gaming
option points you at EsportsClipper instead.

**Step 2 adapts too.** Clips from EsportsClipper carry a marker saying they are
gameplay, and ClipCaptioner reads it and holds the crop centred instead of
hunting for faces that a match feed does not have.

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
| **Chat activity** | Past streams — message-rate spikes |
| **Speech energy** | Anything else, including live |

The first two measure the audience, and a stream can have neither: a
most-replayed graph takes days to appear, and chat replay is not published the
moment a broadcast ends. Speech energy reads the speaker instead — loudness
weighted by how much a passage actually sounds like talking, so music and
applause cannot outrank a real moment. It needs no external data, which is
what makes live and just-ended streams workable.

Streams that are **still broadcasting** are clipped from the part aired so
far. That fetches the broadcast from its beginning, so a long-running stream
takes a while to download; clipping it after it ends is much faster.

### Step 2 — `ClipCaptioner`

Takes those clips and, for each one: extracts the audio, transcribes it with
word-level timing, centre-crops to 9:16, and burns in captions with the spoken
word highlighted. Clips longer than ~3 minutes are split into numbered parts
that fit Shorts.

---

## Setup

> **Use `py -m pip`, not `pip`.** On Windows, `pip` is frequently not on your
> PATH even when Python is installed correctly, which produces
> `'pip' is not recognized as an internal or external command`. Running it as
> `py -m pip` goes through the Python launcher and works regardless. Every
> command below uses that form on purpose.

### 1. Install Python 3.11 or newer

Download from [python.org/downloads](https://www.python.org/downloads/).

On the installer's **first screen, tick "Add python.exe to PATH"** — it is
unchecked by default and skipping it causes most of the problems below.

Check it worked, in a **new** terminal:

```powershell
py --version
```

You should see `Python 3.11` or higher. If `py` is not recognised, reinstall
and make sure that PATH box is ticked.

### 2. Install FFmpeg

```powershell
winget install Gyan.FFmpeg
```

macOS: `brew install ffmpeg` · Debian/Ubuntu: `sudo apt install ffmpeg`

Then **close and reopen your terminal** — PATH changes do not reach terminals
that were already open. Check:

```powershell
ffmpeg -version
```

### 3. Get the code

The trailing `.` is part of the repository name. The folder is named
explicitly because Windows strips trailing dots from directory names.

```powershell
git clone https://github.com/dev-flowstate/Youtube-video-link-to-shorts..git youtube-shorts
cd youtube-shorts
```

No git? Use the green **Code → Download ZIP** button on GitHub and unzip it.

### 4. Install the dependencies

```powershell
py -m pip install --upgrade pip
py -m pip install -r YouTubeReplayDownloader/requirements.txt
py -m pip install -r ClipCaptioner/requirements.txt
```

That is roughly 500 MB, mostly PyTorch-free speech and vision libraries, so
give it a few minutes.

### 5. Check it

```powershell
py -c "import yt_dlp, numpy, scipy, faster_whisper, cv2; print('all good')"
```

> The first captioning run additionally downloads the Whisper speech model
> (~460 MB). That happens once and is cached afterwards.

### If something goes wrong

| Message | Fix |
|---|---|
| `'pip' is not recognized` | Use `py -m pip` instead of `pip` |
| `'python' is not recognized` | Use `py` instead of `python`, or reinstall with "Add python.exe to PATH" ticked |
| `'py' is not recognized` | Python is not installed, or was installed without the launcher. Reinstall from python.org. |
| `ffmpeg not found` | Reopen your terminal. If it persists, run `winget install Gyan.FFmpeg` again. |
| `No module named 'faster_whisper'` | Step 4 did not finish. Re-run it and read the output for the real error. |
| Installs succeed but imports fail | You likely have several Pythons. Run `py -0p` to list them, and use `py -m pip` so installer and interpreter always match. |

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

```powershell
cd YouTubeReplayDownloader
py main.py
```

You'll see the detected peaks, then the clips appear in
`YouTubeReplayDownloader/output/`. These are landscape with no subtitles.

**3. Add captions and go vertical.**

```powershell
cd ../ClipCaptioner
py main.py
```

Finished Shorts land in `ClipCaptioner/output/`, ready to upload.

That's it. For the next video, change the link and repeat.

---

## What you get

The hottest moments, each capped at **90 seconds**. How many you get scales
with how long the source runs — a 20 minute video gives 5, a 55 minute stream
gives 14, anything from 3 hours up gives 40.

Clips are trimmed around their peak, so the best bit stays in frame, and
overlapping candidates are dropped — five distinct moments rather than five
that partly repeat each other. Short clips hold attention, which is the whole
point of the format.

Edges are aligned to pauses in speech, so clips never start mid-word, and
candidates that are mostly silence are skipped before the hottest are picked.

Clips then **end where a sentence finishes**, not merely where speech pauses —
a pause happens inside sentences too. The ending shifts either way to reach
one, so a clip may run slightly over or under 90 seconds rather than stop
halfway through an idea.

**Clips are named after their title**, picked by scoring the clip's own
transcript — YouTube pre-fills the title field from the filename on upload, so
there is nothing to retype. The title is stored in the MP4's metadata too.

No model is involved, so the title is always a line the clip actually
contains.

Set `WRITE_UPLOAD_NOTES = True` in `ClipCaptioner/config.py` if you also want a
`.txt` beside each clip with a description and hashtags.

All near the top of `YouTubeReplayDownloader/main.py`:

```python
CLIPS_PER_HOUR = 15       # clips scale with source length
MIN_CLIPS = 5             # floor for short videos
MAX_CLIPS = 40            # ceiling for very long streams
FIXED_CLIP_COUNT = None   # set a number to ignore the scaling
MAX_CLIP_SECONDS = 90.0   # per-clip ceiling
```

---

## Useful options

Caption clips from somewhere else:

```powershell
py main.py --input "D:\my\clips" --output "D:\my\shorts"
```

Do a single clip first to check the look before committing to a whole batch:

```powershell
py main.py --only "02m49s"
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
