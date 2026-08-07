# YouTube Highlight Downloader

Finds the best moments in a YouTube video or past live broadcast and saves each
one as its own MP4 clip.

## How moments are found

Three signals, tried best-first. Whichever works is reported when you run it.

| Signal | Used when | Quality |
|---|---|---|
| **Most replayed** | Normal videos, and older broadcasts | Best — a direct measure of what viewers rewatched |
| **Chat activity** | Past streams with chat replay | Strong — message-rate spikes are what human clippers scrub for |
| **Speech energy** | Anything else, including live | Decent — reads the speaker instead of the audience |

The first two measure the audience, and a stream can have neither. A
most-replayed graph takes days to appear, and chat replay is not published the
moment a broadcast ends — so a stream that finished an hour ago has nothing to
go on. Speech energy needs no external data at all, which is what makes live
and just-ended streams workable.

### Speech energy

Raw loudness was tried once and abandoned, because it flags an intro sting or
background music as readily as a real moment. This is not that. Loudness is
weighted by how much a bucket actually **sounds like speech**: talking carries
a steady rhythm of short gaps — breaths, syllable breaks, turn-taking — while
music, applause and room noise run continuous. A stretch with no gaps at all
is not someone making a point, however loud it is.

The effect, on a synthetic timeline: loud music scores **0.15**, loud speech
scores **0.89**.

Loudness is also rescaled across the whole recording rather than measured
against a fixed level, so a quiet recording and a loud one rank the same way.

It is fast — a 55-minute broadcast analysed in about 35 seconds, well under
what fetching chat costs on a stream that size.

## Requirements

- Python 3.11+
- FFmpeg on PATH (`winget install Gyan.FFmpeg`)
- `py -m pip install -r requirements.txt`

Use `py -m pip`, not bare `pip` — on Windows `pip` is often not on PATH even
when Python is installed correctly.

## Usage

1. Open `main.py`
2. Edit `YOUTUBE_URL` near the top
3. Run:

```powershell
py main.py
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

Streams that are **still broadcasting** now work: the part aired so far is
pulled and clipped. Be aware that this fetches the broadcast from its
beginning, so a stream that has been running for hours takes a long time to
download before anything can be analysed. Clipping it after it ends is far
faster and gives better moments.

Not supported: Shorts, playlists, and several URLs at once.

## Settings

Both near the top of `main.py`:

| Setting | Purpose |
|---|---|
| `YOUTUBE_URL` | The video to clip |
| `OUTPUT_DIR` | Where clips are written |
| `MAX_CLIPS` | How many clips to keep. Only the strongest peaks survive, so this is "the N hottest moments". `None` keeps every peak. |
| `MAX_CLIP_SECONDS` | Per-clip ceiling. Clips are trimmed around their peak so the best bit stays in frame. |
| `SNAP_TO_SPEECH` | Align clip edges with pauses so they start and end on a sentence, and skip candidates that are mostly silence. |
| `TAIL_PADDING_SECONDS` | Extra footage kept past each cut. ClipCaptioner reads the transcript and stops the finished video where a sentence actually ends, which it can only do if there is material to reach into. Unused padding is dropped there. |

Overlapping candidates are dropped during selection, so you get distinct
moments rather than several that partly repeat each other.

## Speech-aligned edges

Activity peaks say *when* something happened, not where the sentence around it
starts. Cutting on the raw boundary lands mid-word, which is the clearest
giveaway of an auto-generated clip.

Before downloading the video, the audio track alone is fetched and scanned for
pauses, and each clip's edges move to the nearest one. Edges prefer to widen
rather than narrow, so a clip gains its opening words instead of losing them.

Silence is measured **relative to the material's own loudness**, not at a fixed
dB. Real vlog audio carries constant room tone — one clip measured here
averaged -18 dB and never dropped below -30 dB, so a fixed threshold finds
nothing on some sources and flags quiet speech as a pause on others.

Candidates that are mostly silence are dropped before the hottest are chosen,
so a dud frees its slot for the next best moment instead of wasting one.

Note that peak detection deliberately targets a shorter clip than
`MAX_CLIP_SECONDS` allows, leaving headroom for edges to widen out to a pause
while staying under the cap.

## How it works

1. `moment_finder.py` picks the best available signal for this video
2. `replay_fetcher.py` or `chat_fetcher.py` produces an activity curve — both
   emit the same `HeatmapPoint` shape, so the detector does not care where the
   data came from
3. `peak_detector.py` smooths the curve, finds significant peaks, expands each
   outward until activity returns near baseline, then trims to
   `MAX_CLIP_SECONDS` around the peak
4. `downloader.py` fetches the source once and cuts every clip locally

## Project structure

```text
YouTubeReplayDownloader/
├── main.py             entry point and settings
├── moment_finder.py    picks the best available signal
├── replay_fetcher.py   YouTube most-replayed heatmap
├── chat_fetcher.py     live chat replay activity
├── activity.py         turns raw events into heatmap points
├── peak_detector.py    finds peaks, boundaries and clip length
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
- Chat curves are smoothed over a fixed timespan rather than a fixed number of
  buckets, and their first and last 90 seconds are damped. Without that, the
  surge of people arriving and leaving a stream outscores every real moment
  in it.

## Troubleshooting

**`'pip' is not recognized`** — use `py -m pip` instead of `pip`.

**FFmpeg not found** — ensure `ffmpeg -version` works in the same terminal.

**"This stream is still live"** — wait for the broadcast to end.

**"No usable signal"** — the video has neither a most-replayed graph nor chat
replay. Nothing can be detected from it.

**Want more or longer clips** — raise `MAX_CLIPS` and `MAX_CLIP_SECONDS` in
`main.py`.

**Download failed** — `py -m pip install -U yt-dlp`, then retry.
