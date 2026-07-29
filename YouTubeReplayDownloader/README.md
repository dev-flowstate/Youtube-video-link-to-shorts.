# YouTube Highlight Downloader

Finds the best moments in a YouTube video or past live broadcast and saves each
one as its own MP4 clip.

## How moments are found

Three signals, tried best-first. Whichever works is reported when you run it.

| Signal | Used when | Quality |
|---|---|---|
| **Most replayed** | Normal videos, and older broadcasts | Best — a direct measure of what viewers rewatched |
| **Chat activity** | Past live streams with chat replay | Strong — message-rate spikes are what human clippers scrub for |
| **Audio loudness** | Anything else | Blunt — finds shouting and reactions, but also loud intros and music |

Live streams have no most-replayed graph until well after the broadcast ends,
which is why chat is the fallback rather than a failure.

## Requirements

- Python 3.11+
- FFmpeg on PATH (`winget install Gyan.FFmpeg`)
- `pip install -r requirements.txt`

## Usage

1. Open `main.py`
2. Edit `YOUTUBE_URL` near the top
3. Run:

```bash
python main.py
```

Clips land in `output/` next to the script. Example filename:

```text
MrBeast [03m42s-04m12s].mp4
```

## Supported URLs

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/live/VIDEO_ID` — past broadcasts
- `https://www.youtube.com/embed/VIDEO_ID`

Not supported: Shorts, playlists, several URLs at once, and streams that are
**still broadcasting** — a clip cannot be cut from a video that is still
growing. Wait for the stream to end.

## Settings

Both near the top of `main.py`:

| Setting | Purpose |
|---|---|
| `YOUTUBE_URL` | The video to clip |
| `OUTPUT_DIR` | Where clips are written |
| `MAX_CLIPS` | Cap on clips per video. A long stream can produce dozens of peaks, and each one costs a download plus a captioning pass. Only the strongest are kept. `None` disables the cap. |

## How it works

1. `moment_finder.py` picks the best available signal for this video
2. `replay_fetcher.py`, `chat_fetcher.py` or `audio_peaks.py` produces an
   activity curve — all three emit the same `HeatmapPoint` shape, so the
   detector does not care where the data came from
3. `peak_detector.py` smooths the curve and finds significant peaks, expanding
   each one outward until activity returns near baseline
4. `downloader.py` fetches the source once and cuts every clip locally

## Project structure

```text
YouTubeReplayDownloader/
├── main.py             entry point and settings
├── moment_finder.py    picks the best available signal
├── replay_fetcher.py   YouTube most-replayed heatmap
├── chat_fetcher.py     live chat replay activity
├── audio_peaks.py      audio loudness fallback
├── activity.py         turns raw events into heatmap points
├── peak_detector.py    finds peaks and their boundaries
├── downloader.py       downloads and cuts the clips
├── ffmpeg_utils.py     FFmpeg discovery
├── utils.py            URL parsing, timestamps, filenames
└── requirements.txt
```

## Notes

- Re-running the same URL skips clips that already exist. If every clip is
  present, the source video is not downloaded at all.
- The source video is downloaded once in full, then clips are cut locally.
  Downloading only the needed ranges was tried and abandoned: YouTube stalls
  FFmpeg's HTTP client, which is what partial downloads depend on.
- Chat and audio curves are smoothed over a fixed timespan rather than a fixed
  number of buckets, and their first and last 90 seconds are damped. Without
  that, the surge of people arriving and leaving a stream outscores every real
  moment in it.

## Troubleshooting

**FFmpeg not found** — ensure `ffmpeg -version` works in the same terminal.

**"This stream is still live"** — wait for the broadcast to end.

**Too many clips** — lower `MAX_CLIPS` in `main.py`.

**Download failed** — `pip install -U yt-dlp`, then retry.
