"""
Sketch (annotation) mode for Project Vertex v4.

Modal — only active when explicitly engaged (voice "annotate" or D key).
While active:
  - 1-finger pinch  → draw cyan stroke
  - Fist (SCALE mode) → erase strokes near fist position
  - 2-hand gestures (orbit/zoom) pass through to the normal handler
  - HUD turns amber; "ANNOTATING" badge shown

Strokes are stored in normalized screen coordinates [0,1]×[0,1] on the
parent Hologram, so they travel with the hologram in the scene save/load.
"""
from gesture_engine import GestureMode


class SketchController:
    ERASE_RADIUS = 0.06   # normalized screen units

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        self._current_stroke: list[tuple[float, float]] = []
        self._was_pinching   = False

    def update(self, hand_data: dict, active_hologram):
        """Called every frame while sketch mode is active."""
        if active_hologram is None:
            return

        hands    = hand_data.get("both_hands", [])
        cam_w, cam_h = hand_data.get("cam_size", (640, 480))

        if not hands:
            if self._current_stroke:
                self._finish_stroke(active_hologram)
            return

        # Single hand drives drawing/erasing; 2-hand gestures pass through
        h = hands[0]

        # Fist (0 fingers) = erase
        if h.get("finger_count", 1) == 0:
            cx, cy = h["center"]
            self._erase_near(active_hologram,
                             cx / cam_w, cy / cam_h,
                             self.ERASE_RADIUS)
            if self._current_stroke:
                self._finish_stroke(active_hologram)
            self._was_pinching = False
            return

        # 1-finger pinch = draw
        if h["pinched"]:
            ix, iy = h["index_tip"]
            nx, ny = ix / cam_w, iy / cam_h
            # Deduplicate very close points
            if (not self._current_stroke or
                    _dist(self._current_stroke[-1], (nx, ny)) > 0.004):
                self._current_stroke.append((nx, ny))
        elif self._was_pinching and self._current_stroke:
            self._finish_stroke(active_hologram)

        self._was_pinching = h["pinched"]

    def finish_current(self, active_hologram):
        """Flush in-progress stroke when exiting sketch mode."""
        if self._current_stroke and active_hologram:
            self._finish_stroke(active_hologram)

    def get_preview(self) -> list[tuple[float, float]]:
        """Return the in-progress (not yet committed) stroke for live preview."""
        return list(self._current_stroke)

    def clear_strokes(self, active_hologram):
        if active_hologram:
            active_hologram.strokes.clear()
        self._current_stroke.clear()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _finish_stroke(self, hologram):
        from scene import Stroke
        if len(self._current_stroke) >= 2:
            hologram.strokes.append(
                Stroke(points=list(self._current_stroke),
                       color=(0.235, 0.827, 1.0),
                       width=2.0)
            )
        self._current_stroke = []

    def _erase_near(self, hologram, nx, ny, radius):
        hologram.strokes = [
            s for s in hologram.strokes
            if all(_dist(p, (nx, ny)) >= radius for p in s.points)
        ]


def _dist(p1, p2) -> float:
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5
