"""
Gesture Engine — Tight-5 state machine for Project Vertex v4.

Modes:
  ORBIT  (1 finger) — index up, pinch+drag to rotate
  PAN    (5 fingers) — open palm, move to pan
  SCALE  (0 fingers / fist) — vertical drag to scale; hold 1s = RESET
  ZOOM   (two-hand override, set externally)
  NONE   — dead zone (2-4 fingers) or no hand

Dead zone at 2-4 fingers prevents mode flicker when transitioning
between 1-finger ORBIT and 5-finger PAN.
"""
import time
from enum import Enum, auto
from collections import Counter


class GestureMode(Enum):
    NONE   = auto()   # no hand / fist
    ORBIT  = auto()   # 1 finger — free drag = orbit, pinch+drag = pan
    ZOOM   = auto()   # two-hand override


# Display label + RGB
MODE_META = {
    GestureMode.NONE:  ("---",         (100, 100, 100)),
    GestureMode.ORBIT: ("1F — ACTIVE", (  0, 210, 255)),
    GestureMode.ZOOM:  ("2H — ZOOM",   (  0, 210, 255)),
}

GESTURE_GUIDE = [
    ("1 finger pinch+drag", "Orbit / Rotate"),
    ("1 finger drag",       "Move position"),
    ("Fist hold (1s)",      "Reset view"),
    ("2 hands apart",       "Zoom in/out"),
    ("+/-  keys",           "Scale up/down"),
]

# Finger count → mode  (0 = fist reset-only, 2-5 = dead zone except zoom)
_FINGER_MODE_MAP = {
    1: GestureMode.ORBIT,
}


class GestureEngine:
    """Stateful gesture recognizer — no drawing, no OpenCV."""

    FIST_HOLD_SECONDS = 1.0
    DEBOUNCE_FRAMES   = 4    # new mode must be stable this many frames

    def __init__(self):
        self._hold_gesture: dict[str, str]   = {}
        self._hold_start:   dict[str, float] = {}
        self._mode_history: dict[str, list]  = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_mode(self, hand_id: str, lm_list: list) -> GestureMode:
        """Return the debounced GestureMode for a single hand."""
        n    = self.count_extended_fingers(lm_list)
        raw  = _FINGER_MODE_MAP.get(n, GestureMode.NONE)

        hist = self._mode_history.setdefault(hand_id, [])
        hist.append(raw)
        if len(hist) > self.DEBOUNCE_FRAMES + 3:
            hist.pop(0)

        # Only commit if mode is dominant in the last DEBOUNCE_FRAMES frames
        recent = hist[-self.DEBOUNCE_FRAMES:]
        cnt = Counter(recent)
        dominant, votes = cnt.most_common(1)[0]
        if votes >= self.DEBOUNCE_FRAMES - 1:
            return dominant
        # While transitioning, hold the previous committed mode
        return hist[-self.DEBOUNCE_FRAMES] if len(hist) >= self.DEBOUNCE_FRAMES else GestureMode.NONE

    def check_fist_reset(self, hand_id: str, lm_list: list, now: float) -> bool:
        """Return True exactly once when fist is held ≥ FIST_HOLD_SECONDS."""
        is_fist = self.count_extended_fingers(lm_list) == 0
        if is_fist:
            if self._hold_gesture.get(hand_id) == "fist":
                elapsed = now - self._hold_start.get(hand_id, now)
                if elapsed >= self.FIST_HOLD_SECONDS:
                    self._hold_start[hand_id] = now + 99_999
                    return True
            else:
                self._hold_gesture[hand_id] = "fist"
                self._hold_start[hand_id]   = now
        else:
            self._hold_gesture.pop(hand_id, None)
            self._hold_start.pop(hand_id, None)
        return False

    def fist_hold_progress(self, hand_id: str, now: float) -> float:
        """0→1 progress toward fist-reset (used to draw the radial arc)."""
        if self._hold_gesture.get(hand_id) != "fist":
            return 0.0
        elapsed = now - self._hold_start.get(hand_id, now)
        return min(elapsed / self.FIST_HOLD_SECONDS, 1.0)

    # ------------------------------------------------------------------
    # Finger counting — vector dot-product method (perspective-robust)
    # ------------------------------------------------------------------

    def count_extended_fingers(self, lm_list: list) -> int:
        """Count extended fingers (0-5) from a 21-point MediaPipe landmark list."""
        if not lm_list or len(lm_list) < 21:
            return 0

        def pt(i):
            return (lm_list[i][1], lm_list[i][2])

        def dist_sq(p1, p2):
            return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

        def sub(p1, p2):
            return (p1[0]-p2[0], p1[1]-p2[1])

        def dot(v1, v2):
            return v1[0]*v2[0] + v1[1]*v2[1]

        count = 0

        # Thumb: tip (4) further from pinky base (17) than joint (3)
        if dist_sq(pt(4), pt(17)) > dist_sq(pt(3), pt(17)):
            count += 1

        # Four fingers: extended if fingertip-to-PIP vector points away from palm
        for mcp, pip, tip in [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]:
            if dot(sub(pt(mcp), pt(0)), sub(pt(tip), pt(pip))) > 0:
                count += 1

        return count

    def get_hand_center(self, lm_list: list) -> tuple:
        if not lm_list:
            return (0, 0)
        return (
            sum(p[1] for p in lm_list) // len(lm_list),
            sum(p[2] for p in lm_list) // len(lm_list),
        )
