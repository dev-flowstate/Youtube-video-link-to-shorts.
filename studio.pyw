"""Clip Studio - a window over the clip pipeline, so a run needs no editor.

Two programs do the work and neither is changed by this one. First
YouTubeReplayDownloader finds the moments worth keeping and cuts them out,
then ClipCaptioner captions and renders them as Shorts. Both normally ask
their questions on stdin; this collects the answers up front and passes them
as flags, so a run started here never stops halfway waiting for a human.

Double-click it, or:
    python studio.pyw
"""

from __future__ import annotations

import codecs
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
DOWNLOADER_DIR = ROOT / "YouTubeReplayDownloader"
CAPTIONER_DIR = ROOT / "ClipCaptioner"

# Kept out of the repo on purpose: it is per-machine, and the repo is a working
# tree other people commit in.
SETTINGS_PATH = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ClipStudio" / "settings.json"

# One spacing scale for the whole window, each step twice the one below it.
# Ad-hoc numbers per widget are what make an interface look assembled rather
# than designed.
TIGHT, GAP, PAD, WIDE = 4, 8, 16, 24

# A label frame draws its title across the top border, which already opens up
# the space a full pad would add there. Padding the top as heavily as the other
# three sides just leaves a hole under every heading.
BOX = (PAD, GAP, PAD, PAD)

POLL_MS = 100  # how often the window drains what the pipeline has printed

# A long run prints a great deal, and a Tk text widget holding tens of
# thousands of lines slows the whole window down. Old lines are dropped rather
# than letting a five hour download make the log unusable by hour three.
MAX_LOG_LINES = 4000

# The style picker answers two questions at once: how 16:9 becomes 9:16, and
# which evidence the detector is allowed to trust.
#
# Gameplay maps to "general" rather than refusing, because the downloader's own
# third option is a signpost to EsportsClipper rather than a mode it can run.
# Choosing it here is taken to mean "this footage is not a podcast, fit the
# whole frame", and the log says as much.
STYLES = {
    "podcast": ("crop", "talk"),
    "streamer": ("stacked", "general"),
    "gameplay": ("fit", "general"),
}

# Filler and cutaways compete for the same frame, so this is one choice rather
# than two switches. The flag pairs match what ClipCaptioner's own prompt sends.
EXTRAS = {
    "broll": ["--broll", "--no-parkour"],
    "filler": ["--parkour", "--no-broll"],
    "none": ["--no-broll", "--no-parkour"],
}

# The log is colour-coded, and the same ink cannot serve both papers. The
# plain set is for light system ttk; the dark set is what Sun Valley needs,
# because on #181818 the light colours land between 1.0:1 and 3.3:1 against
# the paper - technically present, practically unreadable.
PLAIN_COLOURS = {
    "out": "#1c1c1c",
    # stderr, which here is mostly progress: tqdm and ffmpeg both report
    # through it. It gets its own colour rather than the failure one, because
    # painting every progress bar red would leave a real error indisting-
    # uishable from a download working perfectly.
    "err": "#8a5300",
    "sys": "#2f5d8a",   # this window's own narration
    "fail": "#b3261e",
    "done": "#1b7a44",
}
# Contrast against the dark paper in the comments. Plain output is deliberately
# the quietest of the five: most lines are that one, and a wall of them at full
# brightness leaves nothing for a failure to stand out against.
DARK_COLOURS = {
    "out": "#d6d6d6",   # 12.2:1
    "err": "#e3a857",   #  8.5:1
    "sys": "#79b8ff",   #  8.6:1
    "fail": "#ff7b72",  #  7.0:1
    "done": "#56d364",  #  9.2:1
}
LOG_COLOURS = dict(PLAIN_COLOURS)  # _paint() swaps in the dark set

# The log is a plain Tk text widget, so no ttk theme reaches it and its paper
# is set by hand to match whichever theme is in use.
PLAIN_PAPER = "#fcfcfc"
DARK_PAPER = "#181818"  # a shade under the window, so the log reads as inset

# What the primary button says when it is not busy. Two places need to agree
# on it: the one that builds it and the one that puts it back afterwards.
RUN_LABEL = "Cut and caption"


def _python() -> str:
    """The interpreter to run the pipeline with.

    Double-clicking a .pyw runs pythonw.exe, which has no console. The pipeline
    is console software that prints its way through a run, so hand it python.exe
    where the two sit side by side.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(exe)


def _quoted(argv: list[str]) -> str:
    """The command as you would have to type it, for the log."""
    return " ".join(f'"{part}"' if " " in part else part for part in argv)


def _paint(root: tk.Tk) -> str:
    """Theme the window, and hand back the paper colour for the log.

    Dark, because this window is mostly log and the log is program output. On
    dark paper all five line colours can be bright, and bright colours separate
    from one another; on white they have to be dark enough to read, which drags
    the amber, the red and the green towards the same muddy middle.

    The theme is a nicety, not a dependency. A machine without sv-ttk - or with
    a copy of it that will not load - still has a pipeline to run, so a failure
    here falls back to plain ttk and the palette that suits it.
    """
    try:
        import sv_ttk

        sv_ttk.set_theme("dark", root)
    except Exception:
        return PLAIN_PAPER

    _darken_title_bar(root)
    LOG_COLOURS.update(DARK_COLOURS)
    return DARK_PAPER


def _darken_title_bar(root) -> None:
    """Ask Windows to draw this window's title bar dark.

    sv-ttk restyles the widgets inside the window; the title bar belongs to the
    desktop manager and stays light, which leaves a white strip along the top
    of a dark window and reads as half-finished rather than themed.

    DWMWA_USE_IMMERSIVE_DARK_MODE is attribute 20 on Windows 10 build 18985 and
    later, and was 19 on the builds before it. Both are tried because the call
    simply reports failure on a build that does not know the attribute.

    Entirely cosmetic, so every failure is swallowed: an older Windows, a
    non-Windows machine, or a missing symbol leaves the light title bar exactly
    as it is and changes nothing else.
    """
    try:
        import ctypes

        root.update_idletasks()  # the window must exist before DWM will style it
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        enabled = ctypes.c_int(1)
        for attribute in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(enabled), ctypes.sizeof(enabled)
            ) == 0:
                return
    except Exception:
        pass


def _outcome_colour(outcome: str) -> str:
    """What colour the status text is once a run has stopped."""
    return {
        "Finished": LOG_COLOURS["done"],
        "Ready": LOG_COLOURS["out"],
    }.get(outcome, LOG_COLOURS["fail"])


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        # Remembering a folder is a convenience, never a reason to stop.
        pass


class Studio:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.messages: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.stopping = False
        self.outcome = "Ready"

        # True while the last line in the log is a progress redraw waiting to
        # be overwritten by the next one.
        self._redrawing = False

        settings = _load_settings()
        self.url_var = tk.StringVar()
        self.dest_var = tk.StringVar(value=settings.get("output_dir", ""))
        self.style_var = tk.StringVar(value="podcast")
        self.extras_var = tk.StringVar(value="broll")
        self.captions_var = tk.BooleanVar(value=True)
        self.faces_var = tk.BooleanVar(value=True)
        self.hook_var = tk.BooleanVar(value=True)
        self.per_hour_var = tk.StringVar(value="4")  # mirrors CLIPS_PER_HOUR
        self.status_var = tk.StringVar(value="Ready")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(POLL_MS, self._drain)

    # -- window ------------------------------------------------------------

    def _build(self) -> None:
        root = self.root
        root.title("Clip Studio")
        # Taller controls need more room than the plain-ttk version did, but not
        # more than the screen has: at 125% scaling a 1080p desktop leaves about
        # 816pt once the taskbar has taken its strip, and a window taller than
        # that opens with its last inch already behind the taskbar.
        root.geometry("960x760")
        root.minsize(840, 600)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)  # only the log takes the extra height

        paper = _paint(root)

        # Fixed width for the log and the spinbox, the two places digits matter.
        # The pipeline prints counters that change width as they climb - [9/12]
        # to [10/12], byte counts, timestamps - and in a proportional font every
        # one of them shifts the line sideways as it updates, which reads as
        # noise. The spinbox has the same problem in miniature: 9 to 10 nudges
        # the field. Labels and controls stay in the system UI font.
        families = set(tkfont.families())
        family = next((f for f in ("Consolas", "Cascadia Mono", "Courier New") if f in families), None)
        self.log_font = tkfont.Font(family=family, size=10) if family else tkfont.nametofont("TkFixedFont")
        self.log_bold = self.log_font.copy()
        self.log_bold.configure(weight="bold")
        self.digit_font = self.log_font.copy()

        style = ttk.Style()
        # Nothing in here should need aiming at. A ttk tick, radio or small
        # button is about 24px tall out of the box, comfortably under the 40px
        # a control wants to be; padding makes up the difference, and it comes
        # off the same scale as every other gap in the window.
        style.configure("TButton", padding=(PAD, GAP))
        style.configure("TCheckbutton", padding=(0, GAP + TIGHT))
        style.configure("TRadiobutton", padding=(0, GAP + TIGHT))
        style.configure("TEntry", padding=(GAP, GAP))
        style.configure("Digits.TSpinbox", padding=(GAP, GAP), font=self.digit_font)

        # Run and Stop are the two buttons that matter, so they are the largest
        # things in the window. Run is also the only filled one: it starts the
        # work, and the eye should find it without reading. Stop is an outline,
        # because throwing away a five hour download is not the default move.
        for action in ("Action.TButton", "Action.Accent.TButton"):
            style.configure(action, padding=(WIDE, PAD))

        # -- source
        source = ttk.LabelFrame(root, text=" Source ", padding=BOX)
        source.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(GAP, GAP))
        source.columnconfigure(1, weight=1)

        ttk.Label(source, text="YouTube link").grid(row=0, column=0, sticky="w", padx=(0, PAD))
        self.url_entry = ttk.Entry(source, textvariable=self.url_var)
        self.url_entry.grid(row=0, column=1, sticky="ew")
        self.paste_btn = ttk.Button(source, text="Paste", command=self._paste)
        self.paste_btn.grid(row=0, column=2, sticky="ew", padx=(GAP, 0))

        ttk.Label(source, text="Save finished clips to").grid(
            row=1, column=0, sticky="w", padx=(0, PAD), pady=(GAP, 0)
        )
        self.dest_entry = ttk.Entry(source, textvariable=self.dest_var)
        self.dest_entry.grid(row=1, column=1, sticky="ew", pady=(GAP, 0))
        self.browse_btn = ttk.Button(source, text="Choose folder", command=self._browse)
        self.browse_btn.grid(row=1, column=2, sticky="ew", padx=(GAP, 0), pady=(GAP, 0))

        # -- the two choices that decide the shape of the finished clip
        choices = ttk.Frame(root)
        choices.grid(row=1, column=0, sticky="ew", padx=PAD)
        choices.columnconfigure(0, weight=1, uniform="choice")
        choices.columnconfigure(1, weight=1, uniform="choice")

        style_box = ttk.LabelFrame(choices, text=" Style ", padding=BOX)
        style_box.grid(row=0, column=0, sticky="nsew", padx=(0, GAP))
        self.style_radios = [
            self._radio(style_box, text, self.style_var, value, row)
            for row, (value, text) in enumerate(
                (
                    ("podcast", "Podcast - crop to the speaker"),
                    ("streamer", "Streamer - camera above the screen"),
                    ("gameplay", "Gameplay - whole frame, nothing cut"),
                )
            )
        ]
        self.style_var.trace_add("write", self._sync_face_tracking)

        extras_box = ttk.LabelFrame(choices, text=" Alongside the speaker ", padding=BOX)
        extras_box.grid(row=0, column=1, sticky="nsew", padx=(GAP, 0))
        self.extras_radios = [
            self._radio(extras_box, text, self.extras_var, value, row)
            for row, (value, text) in enumerate(
                (
                    ("broll", "Cutaways - stock scenes on a subject"),
                    ("filler", "Filler footage - a strip along the bottom"),
                    ("none", "Nothing - just the clip"),
                )
            )
        ]

        finishing = ttk.LabelFrame(root, text=" Finishing ", padding=BOX)
        finishing.grid(row=2, column=0, sticky="ew", padx=PAD, pady=GAP)
        self.captions_check = ttk.Checkbutton(finishing, text="Burn captions in", variable=self.captions_var)
        self.captions_check.grid(row=0, column=0, sticky="w", padx=(0, WIDE))
        self.face_check = ttk.Checkbutton(finishing, text="Follow faces when cropping", variable=self.faces_var)
        self.face_check.grid(row=0, column=1, sticky="w", padx=(0, WIDE))
        self.hook_check = ttk.Checkbutton(finishing, text="Hook card over the opening seconds", variable=self.hook_var)
        self.hook_check.grid(row=0, column=2, sticky="w")

        ttk.Label(finishing, text="Clips per hour of source").grid(
            row=1, column=0, sticky="w", pady=(PAD, 0)
        )
        self.per_hour_spin = ttk.Spinbox(
            finishing, from_=1, to=60, width=5, textvariable=self.per_hour_var, style="Digits.TSpinbox"
        )
        self.per_hour_spin.grid(row=1, column=1, sticky="w", padx=(GAP, 0), pady=(PAD, 0))

        # -- run, stop, and where the run has got to
        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="ew", padx=PAD, pady=(0, GAP))
        actions.columnconfigure(2, weight=1)

        # A fixed width, because this button relabels itself as the run moves
        # through its stages and a button that resizes under the cursor is the
        # kind of small wrongness people notice without being able to name.
        self.run_btn = ttk.Button(
            actions, text=RUN_LABEL, width=17, style="Action.Accent.TButton", command=self._on_run
        )
        self.run_btn.grid(row=0, column=0)
        # A mis-click that kills a five hour download is expensive, so Stop
        # keeps its distance from Run rather than sitting flush against it.
        self.stop_btn = ttk.Button(
            actions, text="Stop", width=8, style="Action.TButton", command=self._on_stop, state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=(WIDE, 0))

        self.status_label = ttk.Label(actions, textvariable=self.status_var, foreground=LOG_COLOURS["out"])
        self.status_label.grid(row=0, column=2, sticky="e", padx=(WIDE, PAD))
        self.progress = ttk.Progressbar(actions, mode="determinate", value=0, length=180)
        self.progress.grid(row=0, column=3, sticky="e")

        # -- log
        log_box = ttk.LabelFrame(root, text=" Log ", padding=GAP)
        log_box.grid(row=4, column=0, sticky="nsew", padx=PAD, pady=(0, PAD))
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)

        self.log = ScrolledText(
            log_box,
            wrap="word",
            height=10,
            font=self.log_font,
            background=paper,
            foreground=LOG_COLOURS["out"],
            selectbackground="#2f60d8",  # the same blue either theme selects with
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=PAD,
            pady=GAP,
            state="disabled",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        # The scrollbar ScrolledText brings with it is left alone deliberately.
        # It is a plain Tk one, and on Windows that is drawn by the system: it
        # accepts -troughcolor and -background and then ignores them, so it
        # stays light beside a dark log. Nothing short of swapping the widget
        # out changes that, and it is not worth swapping a working widget for.
        for tag, colour in LOG_COLOURS.items():
            self.log.tag_configure(tag, foreground=colour)
        for tag in ("fail", "done"):
            self.log.tag_configure(tag, font=self.log_bold)

        self._inputs = [
            self.url_entry,
            self.paste_btn,
            self.dest_entry,
            self.browse_btn,
            self.captions_check,
            self.face_check,
            self.hook_check,
            self.per_hour_spin,
            *self.style_radios,
            *self.extras_radios,
        ]

        # An empty panel reads as broken, so say what will appear here and what
        # it will take. One line of orientation, shown once, not a tour.
        self._append(
            [
                (
                    "sys",
                    "Nothing has run yet. Paste a YouTube link, choose where the finished "
                    f"clips should go, then press {RUN_LABEL}. Everything the downloader "
                    "and the captioner print lands here as it happens, so a long video "
                    "will scroll for a while.",
                    False,
                )
            ]
        )
        self._sync_face_tracking()
        self.url_entry.focus_set()

    def _radio(self, parent: ttk.LabelFrame, text: str, var: tk.StringVar, value: str, row: int) -> ttk.Radiobutton:
        button = ttk.Radiobutton(parent, text=text, variable=var, value=value)
        button.grid(row=row, column=0, sticky="ew")
        return button

    # -- small actions -----------------------------------------------------

    def _paste(self) -> None:
        try:
            self.url_var.set(self.root.clipboard_get().strip())
        except tk.TclError:
            self._say("There is nothing on the clipboard to paste.", "fail")

    def _browse(self) -> None:
        current = self.dest_var.get().strip()
        chosen = filedialog.askdirectory(
            title="Where should the finished clips go?",
            initialdir=current if current and Path(current).is_dir() else str(ROOT),
        )
        if chosen:
            self.dest_var.set(str(Path(chosen)))

    def _sync_face_tracking(self, *_ignored: object) -> None:
        """Face tracking only means anything when the frame is cropped.

        A stacked layout places the camera and the screen separately, so there
        is no single crop window for a tracker to move; a fitted one keeps the
        whole frame. Rather than send a flag that does nothing - or worse, one
        that overrules the captioner's own rule for stacked - the tick greys out
        unless the podcast layout is chosen.
        """
        if self.running:
            return
        self.face_check.configure(state="normal" if self.style_var.get() == "podcast" else "disabled")

    # -- running -----------------------------------------------------------

    def _on_run(self) -> None:
        url = self.url_var.get().strip()
        dest = self.dest_var.get().strip()

        problems = []
        if not url:
            problems.append("Paste a YouTube link first.")
        if not dest:
            problems.append("Choose a folder for the finished clips.")
        try:
            per_hour = int(self.per_hour_var.get())
            if per_hour < 1:
                raise ValueError
        except ValueError:
            per_hour = 0
            problems.append("Clips per hour must be a whole number, 1 or more.")

        if problems:
            for problem in problems:
                self._say(problem, "fail")
            self.status_var.set("Nothing to run")
            self.status_label.configure(foreground=LOG_COLOURS["fail"])
            return

        dest_dir = Path(dest)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._say(f"Cannot use that folder: {exc}", "fail")
            self.status_var.set("Nothing to run")
            self.status_label.configure(foreground=LOG_COLOURS["fail"])
            return

        # Every run gets its own folder of cut clips. Sharing one folder would
        # mean the captioner re-rendering every clip from every earlier run,
        # since it captions whatever it finds where it is pointed.
        source_dir = DOWNLOADER_DIR / "output" / datetime.now().strftime("run-%Y%m%d-%H%M%S")
        layout, content = STYLES[self.style_var.get()]

        python = _python()
        download_cmd = [
            python, "-u", str(DOWNLOADER_DIR / "main.py"),
            "--url", url,
            "--output", str(source_dir),
            "--clips-per-hour", str(per_hour),
            "--content-type", content,
            # This window runs the captioner itself, as a second stage it can
            # report on and stop. The downloader chaining into it as well would
            # caption everything twice.
            "--no-caption",
        ]

        caption_cmd = [
            python, "-u", str(CAPTIONER_DIR / "main.py"),
            "--input", str(source_dir),
            "--output", str(dest_dir),
            "--layout", layout,
            "--captions" if self.captions_var.get() else "--no-captions",
            "--hook-card" if self.hook_var.get() else "--no-hook-card",
            *EXTRAS[self.extras_var.get()],
        ]
        if layout == "crop":
            caption_cmd.append("--face-tracking" if self.faces_var.get() else "--no-face-tracking")

        _save_settings({"output_dir": str(dest_dir)})

        self._clear_log()
        self._say(f"Clip Studio - {datetime.now():%d %b %Y, %H:%M}", "sys")
        self._say(f"  Link      {url}", "sys")
        self._say(f"  Finished  {dest_dir}", "sys")
        self._say(f"  Cut clips {source_dir}", "sys")
        self._say(f"  Style     {self.style_var.get()} (layout {layout}, signals {content})", "sys")
        if self.style_var.get() == "gameplay":
            self._say(
                "  Note      gameplay is detected as a general video here. For match "
                "footage, EsportsClipper is the tool built for it.",
                "sys",
            )
        self._say("", "sys")

        self.outcome = "Failed"
        self.running = True
        self.stopping = False
        self._at_work()
        threading.Thread(
            target=self._work,
            args=(source_dir, dest_dir, download_cmd, caption_cmd),
            daemon=True,
        ).start()

    def _work(self, source_dir: Path, dest_dir: Path, download_cmd: list[str], caption_cmd: list[str]) -> None:
        """Run both stages, one after the other, off the GUI thread."""
        try:
            self._say("Stage 1 of 2 - finding and cutting clips", "sys")
            self._say(_quoted(download_cmd), "sys")
            code = self._spawn(download_cmd, DOWNLOADER_DIR, "Cutting clips...", "Stage 1 of 2")
            if code != 0:
                self._report_exit("The downloader", code)
                return

            clips = sorted(source_dir.glob("*.mp4")) if source_dir.exists() else []
            if not clips:
                self._say("", "out")
                self._say("No clips were cut, so there is nothing to caption.", "fail")
                self.outcome = "No clips"
                return
            self._say("", "out")
            self._say(f"{len(clips)} clip(s) cut.", "done")

            already_there = set(dest_dir.glob("*.mp4"))

            self._say("", "out")
            self._say("Stage 2 of 2 - captioning and rendering", "sys")
            self._say(_quoted(caption_cmd), "sys")
            code = self._spawn(caption_cmd, CAPTIONER_DIR, "Rendering...", "Stage 2 of 2")
            if code != 0:
                self._report_exit("ClipCaptioner", code)
                return

            made = sorted(set(dest_dir.glob("*.mp4")) - already_there)
            self._say("", "out")
            if made:
                self._say(f"Finished - {len(made)} file(s) in {dest_dir}:", "done")
                for path in made:
                    self._say(f"  {path.name}", "done")
                self.outcome = "Finished"
            else:
                self._say(f"ClipCaptioner finished but wrote no new file to {dest_dir}.", "fail")
                self.outcome = "Nothing rendered"
        except Exception as exc:
            # A fault in this window must not leave the interface stuck saying
            # a run is under way, and must not disappear silently either.
            self._say(f"Clip Studio itself failed: {exc!r}", "fail")
            self.outcome = "Failed"
        finally:
            if self.stopping:
                self.outcome = "Stopped"
            self.messages.put(("finished", None))

    def _spawn(self, argv: list[str], cwd: Path, button: str, status: str) -> int:
        """Start one stage and stream both of its pipes into the log."""
        self.messages.put(("state", (button, status)))

        env = dict(os.environ)
        # Clip filenames are built from YouTube titles, so they carry emoji.
        # With its output on a pipe the child falls back to the ANSI codepage
        # and dies on the first one, part way through a run.
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as exc:
            self._say(f"Could not start {Path(argv[2]).parent.name}: {exc}", "fail")
            return -1

        self.proc = proc
        pumps = [
            threading.Thread(target=self._pump, args=(proc.stdout, "out"), daemon=True),
            threading.Thread(target=self._pump, args=(proc.stderr, "err"), daemon=True),
        ]
        for pump in pumps:
            pump.start()

        code = proc.wait()
        for pump in pumps:
            pump.join(timeout=5)
        self.proc = None
        return code

    def _pump(self, stream, tag: str) -> None:
        r"""Forward one pipe into the log, treating a carriage return as a redraw.

        Progress bars end their line with \r and expect it to be overwritten.
        Read this pipe a line at a time and a download's whole progress bar
        arrives as one enormous line once it has finished; read it by chunk and
        it can be shown while it happens, which is the point of the log pane.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        fd = stream.fileno()

        while True:
            try:
                chunk = os.read(fd, 8192)
            except (OSError, ValueError):
                break
            if not chunk:
                break

            pending = (pending + decoder.decode(chunk)).replace("\r\n", "\n")
            # A trailing \r may be the first half of a \r\n split across two
            # reads, so hold it back rather than call it a redraw.
            held = pending.endswith("\r")
            if held:
                pending = pending[:-1]

            start = 0
            for index, char in enumerate(pending):
                if char in "\r\n":
                    self.messages.put(("log", (tag, pending[start:index], char == "\r")))
                    start = index + 1
            pending = pending[start:] + ("\r" if held else "")

        pending = pending.rstrip("\r")
        if pending:
            self.messages.put(("log", (tag, pending, False)))

    def _report_exit(self, who: str, code: int) -> None:
        if self.stopping:
            self._say("", "out")
            self._say(f"{who} was stopped.", "fail")
            return
        self._say("", "out")
        self._say(f"{who} failed - exit code {code}. Nothing further will run.", "fail")

    def _on_stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        self.stopping = True
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopping...")
        self._say("", "out")
        self._say("Stopping - killing the pipeline and everything it started.", "fail")
        self._kill(proc)

    def _kill(self, proc: subprocess.Popen) -> None:
        """Kill the whole tree, not only the process we started.

        The pipeline runs yt-dlp and ffmpeg as children of its own. Terminating
        the Python parent on Windows leaves those running, so a stopped download
        would carry on filling the disk after the window said it had stopped.
        """
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def _on_close(self) -> None:
        if self.running and self.proc is not None:
            if not messagebox.askokcancel("Clip Studio", "A run is under way. Close and stop it?"):
                return
            self.stopping = True
            self._kill(self.proc)
        self.root.destroy()

    # -- state and log -----------------------------------------------------

    def _at_work(self) -> None:
        for widget in self._inputs:
            widget.configure(state="disabled")
        self.run_btn.configure(state="disabled", text="Working...")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(foreground=LOG_COLOURS["out"])
        self.progress.configure(mode="indeterminate")
        self.progress.start(14)

    def _at_rest(self) -> None:
        self.running = False
        self.proc = None
        for widget in self._inputs:
            widget.configure(state="normal")
        self._sync_face_tracking()
        self.run_btn.configure(state="normal", text=RUN_LABEL)
        self.stop_btn.configure(state="disabled")
        self.progress.stop()
        # Stopping an indeterminate bar parks its block at the left, where it
        # reads as a run stuck at nothing. An empty trough says idle.
        self.progress.configure(mode="determinate", value=0)
        self.status_var.set(self.outcome)
        self.status_label.configure(foreground=_outcome_colour(self.outcome))
        self.stopping = False

    def _say(self, text: str, tag: str) -> None:
        """Put one of this window's own lines in the log, from any thread."""
        self.messages.put(("log", (tag, text, False)))

    def _drain(self) -> None:
        """Move whatever the pipeline has printed into the log pane.

        The only place the log and the controls are touched. Everything else
        posts to the queue, so no worker thread ever reaches into Tk.
        """
        lines: list[tuple[str, str, bool]] = []
        try:
            for _ in range(500):
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    lines.append(payload)
                    continue

                if lines:
                    self._append(lines)
                    lines = []
                if kind == "state":
                    button, status = payload
                    self.run_btn.configure(text=button)
                    self.status_var.set(status)
                elif kind == "finished":
                    self._at_rest()
        except queue.Empty:
            pass

        if lines:
            self._append(lines)
        self.root.after(POLL_MS, self._drain)

    def _append(self, lines: list[tuple[str, str, bool]]) -> None:
        self.log.configure(state="normal")
        for tag, text, redraw in lines:
            if self._redrawing:
                # The line already there was a progress redraw. Replace it
                # rather than let a bar leave hundreds of near-identical lines.
                self.log.delete("end-2l linestart", "end-1c")
            self.log.insert("end", text + "\n", tag)
            self._redrawing = redraw

        overflow = int(self.log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if overflow > 0:
            self.log.delete("1.0", f"{overflow}.0")

        self.log.configure(state="disabled")
        self.log.see("end")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._redrawing = False


def main() -> None:
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
