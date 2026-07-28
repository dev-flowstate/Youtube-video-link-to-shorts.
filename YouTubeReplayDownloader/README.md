# YouTube Most-Replayed Downloader

Download the **Most Replayed** segments from a normal YouTube video in the highest available quality.

This tool reads YouTube's replay heatmap from page data, detects all significant replay peaks, and saves each peak as its own MP4 clip.

## Features

- Command-line Python project you can run from VS Code
- Uses YouTube's built-in replay heatmap (no AI)
- Detects all confident replay peaks
- Automatically finds each clip's start/end based on replay activity
- Downloads only the needed segments with `yt-dlp` + FFmpeg
- Saves clips as MP4 in the folder where you run the script
- Skips download entirely if replay data is unavailable

## Requirements

- Windows
- Python 3.10+
- FFmpeg on PATH

## Setup

1. Open this folder in VS Code:

```text
YouTubeReplayDownloader/
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg and add its `bin` folder to PATH.

Verify:

```bash
ffmpeg -version
```

## Usage

1. Open `main.py`
2. Edit `YOUTUBE_URL` near the top
3. Run:

```bash
python main.py
```

Example output filename:

```text
MrBeast [03m42s-04m12s].mp4
```

## Supported URLs

- Normal watch URLs like `https://www.youtube.com/watch?v=VIDEO_ID`
- `youtu.be/VIDEO_ID`

Not supported:

- Shorts
- Playlists
- Multiple URLs at once

## How it works

1. `replay_fetcher.py` loads the YouTube page and extracts the replay heatmap JSON
2. `peak_detector.py` smooths the graph and finds all significant peaks
3. For each peak, it expands outward until replay activity returns near baseline
4. `downloader.py` uses `yt-dlp --download-sections` to fetch only those ranges
5. Clips are merged to MP4 in highest available video + audio quality

## Project structure

```text
YouTubeReplayDownloader/
├── main.py
├── replay_fetcher.py
├── peak_detector.py
├── downloader.py
├── utils.py
├── requirements.txt
└── README.md
```

## Notes

- If a video has no Most Replayed graph, the script exits cleanly and downloads nothing
- Segment-only downloading depends on YouTube format support; `yt-dlp` uses FFmpeg to produce the final clip
- Re-running the same URL skips files that already exist

## Troubleshooting

**FFmpeg not found**

Install FFmpeg and ensure `ffmpeg -version` works in the same terminal you use in VS Code.

**No replay data available**

Not every video exposes the Most Replayed heatmap. Try another video.

**Download failed**

Update dependencies:

```bash
pip install -U yt-dlp
```

Then retry.
