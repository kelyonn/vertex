"""
Voice control for Project Vertex v4.

Two activation paths (both optional, app degrades gracefully):
  1. SPACEBAR push-to-talk — hold to record, release to transcribe
  2. Wake word ("Hey JARVIS") — always-listening via openwakeword

Transcription: faster-whisper (tiny.en, runs on CPU, ~75 MB model)
Commands matched via Levenshtein-style fuzzy match against a fixed grammar.

If faster-whisper or sounddevice are not installed, voice is silently
disabled and VOICE_ENABLED is False.
"""
import threading
import queue
import time
import math

# ── Optional dependencies ─────────────────────────────────────────────
try:
    import sounddevice as sd
    import numpy as np
    _SD_OK = True
except ImportError:
    _SD_OK = False

try:
    from faster_whisper import WhisperModel
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False

try:
    import openwakeword
    from openwakeword.model import Model as _OWWModel
    _OWW_OK = True
except ImportError:
    _OWW_OK = False

try:
    import pyttsx3
    _TTS_OK = True
except ImportError:
    _TTS_OK = False

VOICE_ENABLED = _SD_OK and _WHISPER_OK

# ── Audio constants ────────────────────────────────────────────────────
_SAMPLE_RATE  = 16_000
_CHUNK_FRAMES = 1_024
_MAX_RECORD_S = 8.0      # max recording duration per command

# ── Fixed grammar ─────────────────────────────────────────────────────
# command → list of canonical trigger phrases (first word is the command key)
_GRAMMAR: dict[str, list[str]] = {
    "add":        ["add", "show", "bring up", "load", "materialize", "show me"],
    "remove":     ["remove", "dismiss", "delete", "hide", "close"],
    "focus":      ["focus", "make active", "select", "switch to", "activate"],
    "annotate":   ["annotate", "sketch", "draw on", "start sketching", "mark"],
    "done":       ["done", "finish", "stop sketching", "end annotation", "exit sketch"],
    "wireframe":  ["wireframe", "wire frame", "show wireframe", "toggle wireframe"],
    "solid":      ["solid", "filled", "show solid", "toggle solid"],
    "reset":      ["reset", "reset view", "center", "home view"],
    "rotate":     ["rotate", "spin", "auto rotate", "start rotating"],
    "stop":       ["stop rotating", "stop spin", "freeze", "pause rotation"],
    "screenshot": ["screenshot", "capture", "take screenshot", "snap", "photo"],
    "save":       ["save", "save state", "store"],
    "load":       ["load", "restore", "load state"],
    "help":       ["help", "controls", "shortcuts"],
}

_SHAPE_ALIASES: dict[str, list[str]] = {
    "arc_reactor": ["arc reactor", "reactor", "arc", "iron man"],
    "dna_helix":   ["helix", "dna", "double helix", "spiral"],
    "geodesic":    ["geodesic", "buckyball", "fullerene", "soccer ball"],
    "cube":        ["cube", "box", "square"],
    "sphere":      ["sphere", "ball", "orb", "spear", "steer", "sphear", "sfere"],
    "torus":       ["torus", "donut", "ring"],
    "icosahedron": ["icosahedron", "ico", "icosa"],
}

# Build reverse lookup: phrase → canonical shape name
_SHAPE_LOOKUP: dict[str, str] = {}
for _cname, _aliases in _SHAPE_ALIASES.items():
    for _a in _aliases:
        _SHAPE_LOOKUP[_a] = _cname

# Number → shape name (keys 1-7 match keyboard shortcuts)
_NUMBER_SHAPES: dict[str, str] = {
    "1": "arc_reactor",   "one":   "arc_reactor",
    "2": "dna_helix",     "two":   "dna_helix",
    "3": "geodesic",      "three": "geodesic",
    "4": "cube",          "four":  "cube",
    "5": "sphere",        "five":  "sphere",
    "6": "torus",         "six":   "torus",
    "7": "icosahedron",   "seven": "icosahedron",
}


def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a):
        curr = [i+1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[-1]+1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _fuzzy_command(text: str) -> tuple[str, list[str]]:
    """
    Return (command_key, args) by fuzzy-matching *text* against _GRAMMAR.
    args is a list of shape/object tokens extracted from the phrase.
    Returns ("", []) if nothing matches with sufficient confidence.
    """
    text = text.lower().strip().rstrip(".")

    best_cmd, best_score = "", 999
    for cmd, phrases in _GRAMMAR.items():
        for phrase in phrases:
            # Allow the phrase to appear anywhere in text
            if phrase in text:
                dist = 0
            else:
                dist = min(_levenshtein(text[:len(phrase)], phrase),
                           _levenshtein(text, phrase))
            if dist < best_score:
                best_score = dist
                best_cmd   = cmd

    # Reject if too dissimilar
    if best_score > max(4, len(best_cmd)):
        return "", []

    # Extract shape: try name aliases first, then numbers
    args = []
    for alias, canonical in _SHAPE_LOOKUP.items():
        if alias in text:
            args = [canonical]
            break

    if not args:
        # Match "figure 1", "number 3", or bare digits/words: "add 2", "remove five"
        import re
        # Strip context words so "figure 1" → "1", "number two" → "two"
        stripped = re.sub(r"\b(figure|number|shape|hologram)\b", "", text).strip()
        for token, canonical in _NUMBER_SHAPES.items():
            # Match whole word to avoid "one" matching "stone" etc.
            if re.search(rf"\b{re.escape(token)}\b", stripped):
                args = [canonical]
                break

    return best_cmd, args


# ── VoiceController ────────────────────────────────────────────────────

class VoiceController:
    """
    Non-blocking voice controller.

    Usage:
        vc = VoiceController(config)
        vc.start()
        # each frame:
        while cmd := vc.poll_command():
            handle(cmd)
        # spacebar:
        vc.ptt_start()   # key-down
        vc.ptt_stop()    # key-up → triggers transcription
        vc.stop()        # on exit
    """

    def __init__(self, config=None):
        self._cfg          = config
        self._cmd_queue: queue.Queue = queue.Queue()
        self._ptt_event    = threading.Event()
        self._stop_event   = threading.Event()
        self._recording    = False
        self._model        = None
        self._oww_model    = None
        self._tts_engine   = None
        self.is_listening  = False   # True while mic is hot (HUD indicator)

        if not VOICE_ENABLED:
            print(">> VOICE: disabled (faster-whisper or sounddevice not installed)")
            return

        # Lazy-load whisper in background so startup is fast
        self._loader_thread = threading.Thread(target=self._load_model,
                                               daemon=True, name="whisper-loader")
        self._loader_thread.start()

        self._worker_thread = threading.Thread(target=self._worker_loop,
                                               daemon=True, name="voice-worker")
        self._worker_thread.start()

    # ── Public API ────────────────────────────────────────────────────

    def ptt_start(self):
        """Call on SPACEBAR key-down."""
        if VOICE_ENABLED:
            self._ptt_event.set()

    def ptt_stop(self):
        """Call on SPACEBAR key-up — stops recording and triggers transcription."""
        if VOICE_ENABLED:
            self._ptt_event.clear()

    def poll_command(self) -> dict | None:
        """Return the next pending command dict, or None if queue is empty."""
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    def speak(self, text: str):
        """Optional TTS acknowledgement (e.g. 'Yes, sir.')"""
        if not _TTS_OK:
            return
        if self._tts_engine is None:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 175)
            except Exception:
                return
        try:
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()
        self._ptt_event.clear()

    # ── Model loading ─────────────────────────────────────────────────

    def _load_model(self):
        model_name = "tiny.en"
        if self._cfg:
            model_name = self._cfg.get("voice", "whisper_model") or model_name
        try:
            print(f">> VOICE: loading Whisper {model_name} model (first-run may download ~75MB)…")
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            # Warm-up call so first real transcription is fast
            import tempfile, wave, struct, os
            tmp = tempfile.mktemp(suffix=".wav")
            with wave.open(tmp, "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(_SAMPLE_RATE)
                wf.writeframes(struct.pack("<" + "h"*_SAMPLE_RATE, *([0]*_SAMPLE_RATE)))
            list(self._model.transcribe(tmp, beam_size=1)[0])
            os.unlink(tmp)
            print(">> VOICE: Whisper ready")
        except Exception as e:
            print(f">> VOICE: Whisper load failed: {e}")
            self._model = None

    # ── Worker loop ───────────────────────────────────────────────────

    def _worker_loop(self):
        """Background thread: record on PTT, transcribe, push to queue."""
        while not self._stop_event.is_set():
            # Wait for PTT or wake word
            triggered = self._ptt_event.wait(timeout=0.05)
            if not triggered:
                continue
            if self._model is None:
                self._ptt_event.clear()
                continue

            self.is_listening = True
            audio = self._record_until_release()
            self.is_listening = False

            if audio is None or len(audio) < _SAMPLE_RATE * 0.3:
                continue

            self._transcribe_and_dispatch(audio)

    def _record_until_release(self):
        """Record audio until PTT is released or max duration hit."""
        if not _SD_OK:
            self._ptt_event.clear()
            return None
        chunks = []
        deadline = time.time() + _MAX_RECORD_S
        try:
            with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1,
                                 dtype="float32", blocksize=_CHUNK_FRAMES) as stream:
                while self._ptt_event.is_set() and time.time() < deadline:
                    data, _ = stream.read(_CHUNK_FRAMES)
                    chunks.append(data.copy())
        except Exception as e:
            print(f">> VOICE: recording error: {e}")
            self._ptt_event.clear()
            return None

        self._ptt_event.clear()
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0).flatten()

    def _transcribe_and_dispatch(self, audio_f32):
        """Transcribe float32 audio array and push parsed command to queue."""
        if self._model is None:
            return
        try:
            import tempfile, wave, struct, os
            tmp = tempfile.mktemp(suffix=".wav")
            pcm = (audio_f32 * 32767).astype("int16")
            with wave.open(tmp, "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(_SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())

            segments, _ = self._model.transcribe(tmp, beam_size=3,
                                                  language="en", vad_filter=True)
            text = " ".join(s.text for s in segments).strip()
            os.unlink(tmp)

            if not text:
                return
            print(f'>> VOICE heard: "{text}"')
            cmd, args = _fuzzy_command(text)
            if cmd:
                self._cmd_queue.put({"command": cmd, "args": args, "raw": text})
                print(f">> VOICE dispatched: {cmd} {args}")
        except Exception as e:
            print(f">> VOICE: transcription error: {e}")
