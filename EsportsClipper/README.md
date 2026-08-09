# Esports Fight Clipper

Finds the fights in a PUBG Mobile tournament broadcast and cuts each one out,
plus a single compilation of all of them.

Built for PMWC-style streams: 5+ hours, continuous commentary, dozens of
engagements. The podcast pipeline's signals do not work here — most-replayed
takes days to appear, chat may not exist, and speech energy is high everywhere
because the casters never stop talking.

## How a fight is found

**It watches the broadcast.** Frames are sampled roughly every 10 seconds and
each one is labelled `FIGHT`, `GAME` or `STUDIO`. Runs of `FIGHT` frames become
clips; everything else is thrown away.

That is the whole idea, and it exists because the obvious approach does not
work. See below.

Frames come from **keyframes only**, which costs almost nothing to decode — the
encoder already made them independent — and has a useful side effect: a
broadcast puts a keyframe at every scene cut, which is exactly where it switches
between the game and the desk. They are tiled nine to an image, so one request
covers a minute and a half of broadcast.

### Why not audio

The first version listened for gunfire. It was measured against two fights
identified by hand in a real EWC broadcast, and it does not work:

| | Real fights | Wrongly-picked studio segments |
|---|---|---|
| Gunfire onsets/sec | 0.43 | **1.03** |
| HF ratio (mean) | 0.15 | 0.17 |
| Frames above HF threshold | 2.4–4.2% | 5.1–8.1% |

The signal is not weak, it is **inverted** — interviews scored 2.4× higher on
"gunfire" than the actual firefights. Four more features were tried
(low-frequency energy, spectral flatness, crest factor, low-frequency
transients) and all came out flat, within 6% between the two classes.

The cause is the mix: game audio sits so far under the casters that gunfire
never meaningfully reaches the broadcast. No threshold fixes an absent signal.

Measured on the same footage, watching gets it right. Of 51 clips the audio
detector produced, **88% were not gameplay at all** — presenters, interviews,
ranking boards, adverts.

The audio path is **not** a fallback. It is a recorded negative result, kept in
the tree so the measurement is not lost, and reachable only by deliberately
setting `ALLOW_AUDIO_FALLBACK = True`.

It used to run automatically whenever the API key was missing, and that caused
a real failure: a terminal inherits its environment from when its parent window
opened, so a VS Code window opened before the key was saved could not see it —
the run silently switched to the inverted detector and produced a full set of
interview clips. A missing key now stops the run and explains itself.

The rest of this section describes that audio path.

### The audio path (off by default)

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
- **`GEMINI_API_KEY` set in the environment.** Without it the run falls back to
  the audio detector, and says so. The key is read from the environment only —
  it is never stored in this repo.
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
| `USE_VISION` | Watch the broadcast instead of listening. |
| `ALLOW_AUDIO_FALLBACK` | Off. Lets the known-bad audio detector run when watching is unavailable, instead of stopping. |
| `VISION_SAMPLE_SECONDS` | How often to look. Lower catches shorter fights and costs proportionally more requests. |
| `VISION_REQUESTS_PER_MINUTE` | Held under the key's quota. **The main thing setting how long a broadcast takes to watch.** Raise it on a paid tier. |
| `VISION_BRIDGE_SECONDS` | Fight frames closer than this are one engagement |
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

- **Nothing heavy runs locally.** The model is remote; this machine only decodes
  keyframes and tiles them, which is a few minutes for a 5.5h broadcast and
  needs no GPU.
- **Requests**: ~2000 sampled frames over 5.5h, nine to an image, so ~220. The
  run prints the exact number before it starts.
- **Watch the daily quota.** Free-tier limits are per model and per day, and
  they are small — `gemini-3-flash-preview` allows **20 a day**, which is why
  it is not the model used here despite scoring just as well. If a run stops
  early saying the key is spent, either enable billing or raise
  `VISION_SAMPLE_SECONDS` to look less often. The key is checked before the
  download, so a spent key costs seconds rather than an hour.
- The **source download still dominates**: 10–12 GB at 1080p for a 5.5h
  broadcast. Partial downloads are not possible — YouTube stalls FFmpeg's HTTP
  client, which is what they depend on.
- The download now happens **before** detection, because you cannot watch a
  video you have not got. It is cached, so cutting reuses the same file.

## Limitations

- **A fight between sampled frames is missed.** At 10s spacing a brief exchange
  can fall in a gap. Lowering `VISION_SAMPLE_SECONDS` fixes it and costs
  proportionally more requests.
- **The label is one frame's worth of judgement.** A frame mid-fight where
  nobody happens to be shooting reads as `GAME`; `VISION_BRIDGE_SECONDS` is what
  stops that splitting one engagement into two clips.
- **Replays duplicate kills.** Broadcasts replay big moments, and a replay looks
  exactly like a fight because it is one. The same kill can be clipped twice.
- **Requests cost quota.** A key that runs out mid-broadcast leaves frames
  unlabelled; above `VISION_MAX_UNLABELLED` the run refuses rather than
  reporting a broadcast with no fights in it.
- **The audio path is wrong, not merely weak** — see the measurements above. It
  is off by default and a missing key stops the run instead, because producing
  fifty interview clips is worse than producing none.
