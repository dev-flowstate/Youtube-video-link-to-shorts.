# Esports Fight Clipper

Finds the fights in a PUBG Mobile tournament broadcast and cuts each one out,
plus a single compilation of all of them.

Built for PMWC-style streams: 5+ hours, continuous commentary, dozens of
engagements. The podcast pipeline's signals do not work here — most-replayed
takes days to appear, chat may not exist, and speech energy is high everywhere
because the casters never stop talking.

## How a fight is found

Two audio signals with **different timing**, which is the crux of it:

| Signal | Timing | Meaning |
|---|---|---|
| **Gunfire onsets** | During the fight | *Where* the fight is |
| **Caster energy** | 5–30s **after** | *How much it mattered* |

The caster screams at the kill — the *end* of an engagement. Anchoring a clip
on that scream puts the fight before the clip starts. So caster energy is
folded backwards (`hype(t) = max(caster[t : t+30s])`), letting the run-up
inherit the weight of the reaction it earned.

Then they are combined by **multiplying, not adding**:

```
score = gunfire * (HYPE_BASE + HYPE_GAIN * hype)
```

Gunfire **gates**. No gunfire means no clip, however loud things get — which is
what keeps the opening ceremony and six chicken dinners out of the output. They
are the loudest audio of the day and none of them is a fight.

### Telling gunfire from a voice

A frame counts as gunfire when its **spectral flux** spikes above a rolling
median *and* most of its energy sits **above 4 kHz**. Gunshots are broadband;
speech lives between 300 and 3400 Hz. The second condition is what stops a hard
consonant registering as a shot.

Ceremony music is broadband too — a drum track reads like gunfire on flux
alone. It is separated by rhythm: music onsets are evenly spaced, gunfire is
ragged, so metronomic buckets are penalised. Measured on synthetic audio at the
same onset rate: ragged 12.0 vs metronomic 2.3.

### No dead air

The clip start is **not** a fixed offset. It follows the gunfire backwards from
the peak and stops at the first real gap, so a clip opens when shooting opens
rather than on thirty seconds of someone driving to the zone. A short sharp
trade gives a short sharp clip.

### Why the scale is compressed

Curves are **log-compressed** before normalising. A match-winning scream is
twenty times an ordinary good moment, and dividing by the maximum would push
every ordinary fight to near-zero — six celebrations would disable the very
feature meant to rank the other ~45 clips.

Caster energy is additionally clipped at a percentile, since its outliers are
genuine. Gunfire is **not**: clipping ties everything above the cut, and the
strongest fights are exactly what must stay rankable.

## Requirements

- Python 3.11+, FFmpeg on PATH
- `py -m pip install -r requirements.txt`
- **`YouTubeReplayDownloader/` must sit beside this folder** — its URL parsing,
  audio fetching, peak detection and clip cutting are reused rather than
  duplicated. `shared.py` is the one place that path is wired up.

## Usage

```powershell
py main.py
```

Edit `STREAM_URL` in `config.py` first. Works on finished broadcasts and on
streams still running (those fetch from the beginning, so a long one takes a
while before anything is analysed).

Then for Shorts:

```powershell
cd ../ClipCaptioner
py main.py --input "../EsportsClipper/output"
```

It will ask whether to burn captions in — worth saying no for a language the
model handles poorly.

## Settings

All in `config.py`.

| Setting | Purpose |
|---|---|
| `MIN/MAX_CLIP_SECONDS` | 15–45s. Anything shorter is an incidental pot-shot. |
| `MIN/MAX_PRE_ROLL_SECONDS` | Bounds on how far back the clip may reach |
| `POST_ROLL_SECONDS` | Kept after the peak for the caster's reaction |
| `ACTION_FLOOR` | Gunfire density that still counts as shooting |
| `CLIPS_PER_HOUR` | Scales with length — ~50 over a 5.5h day |
| `SOURCE_MAX_HEIGHT` | 1080p. A 5.5h source must be downloaded whole. |
| `HF_RATIO_MIN` | Gunshot vs voice. **First dial to touch** if detection is noisy. |
| `ONSET_FLUX_FACTOR` | How far above the local median counts as an onset |
| `CASTER_LOOKAHEAD_SECONDS` | How far ahead a reaction still counts |
| `MAKE_COMPILATION` | Join every clip into one long video |

## The compilation

Every clip is cut from the same source with identical codec parameters, so they
are joined with a stream copy — seconds of work, no re-encode. The dead air
between fights is never in it, which is the point.

## Cost

- Analysis: **~0.7 min CPU for 5.5h** of audio, ~4 MB peak. Audio is streamed
  in chunks, never loaded whole (5.5h is ~1.3 GB as float32).
- The **source download dominates**: 10–12 GB at 1080p for a 5.5h broadcast.
  Partial downloads are not possible — YouTube stalls FFmpeg's HTTP client,
  which is what they depend on.

## Limitations

- **Audio only.** A quiet sniper pick with no reaction is missed. Kill-feed OCR
  would be the strongest signal but needs Tesseract and breaks whenever the
  broadcast overlay changes.
- **Replays duplicate kills.** Broadcasts replay big moments with gunfire *and*
  fresh commentary, so the same kill can be clipped twice at different
  timestamps. Overlap rejection cannot catch this.
- **Watchparty vs official feed.** A watchparty puts a streamer's mic over the
  game audio, so gunfire is quieter relative to voice. Thresholds tuned on one
  may need adjusting for the other.
- **Thresholds are not yet calibrated against real broadcast audio.** The
  scoring is verified on synthetic signals; `HF_RATIO_MIN` and
  `ONSET_FLUX_FACTOR` are the two that real footage is most likely to move.
