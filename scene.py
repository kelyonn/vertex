"""
Scene management for Project Vertex v4.
Manages a collection of Hologram objects: layout, focus, materialize/dismiss animations.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional

try:
    from OpenGL.GL import glDeleteLists
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stroke — a 3D annotation attached to a hologram
# ---------------------------------------------------------------------------

@dataclass
class Stroke:
    """A single annotation stroke stored in normalized screen coordinates."""
    points: list        # list of (nx, ny) in [0,1] x [0,1]
    color:  tuple = (0.235, 0.827, 1.0)   # JARVIS cyan by default
    width:  float = 2.0


# ---------------------------------------------------------------------------
# Hologram
# ---------------------------------------------------------------------------

@dataclass
class Hologram:
    """One holographic object in the scene."""
    name:         str
    rot_x:        float = 0.0
    rot_y:        float = 0.0
    target_rot_x: float = 0.0
    target_rot_y: float = 0.0
    scale:        float = 1.0
    pan_x:        float = 0.0   # camera-space horizontal offset (smoothed)
    pan_y:        float = 0.0   # camera-space vertical offset   (smoothed)
    target_pan_x: float = 0.0   # desired pan X — Scene.update() lerps pan toward this
    target_pan_y: float = 0.0   # desired pan Y
    is_active:    bool  = False
    gl_list_solid: int  = 0   # display list ID for solid pass  (0 = not compiled)
    gl_list_wire:  int  = 0   # display list ID for wireframe pass
    strokes:      list  = field(default_factory=list)   # list[Stroke]
    materialize_t:float = 0.0   # 0 = particles, 1 = fully materialized
    orbit_angle:  float = 0.0   # angle in the orbit ring (inactive only)
    dismiss_t:    float = -1.0  # ≥0 = dismissal in progress (counts to 1)

    @property
    def is_loaded(self) -> bool:
        return self.materialize_t >= 1.0

    @property
    def is_dismissing(self) -> bool:
        return self.dismiss_t >= 0.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    MAX_HOLOGRAMS  = 4
    ORBIT_RADIUS   = 3.2
    ORBIT_SPEED    = 0.18   # radians / second for inactive orbit
    MAT_SPEED      = 1 / 1.5    # materialize: 1.5 s
    DISMISS_SPEED  = 1 / 0.8    # dismiss: 0.8 s

    # Canonical name lookup (maps voice/alias → canonical name)
    NAME_ALIASES: dict[str, str] = {
        "arc reactor":   "arc_reactor",
        "reactor":       "arc_reactor",
        "arc":           "arc_reactor",
        "iron man":      "arc_reactor",
        "ironman":       "arc_reactor",
        "helix":         "dna_helix",
        "dna":           "dna_helix",
        "double helix":  "dna_helix",
        "spiral":        "dna_helix",
        "geodesic":      "geodesic",
        "buckyball":     "geodesic",
        "fullerene":     "geodesic",
        "soccer ball":   "geodesic",
        "soccer":        "geodesic",
        "cube":          "cube",
        "box":           "cube",
        "sphere":        "sphere",
        "ball":          "sphere",
        "orb":           "sphere",
        "torus":         "torus",
        "donut":         "torus",
        "ring":          "torus",
        "icosahedron":   "icosahedron",
        "ico":           "icosahedron",
        "icosa":         "icosahedron",
    }

    BUILT_IN_NAMES = [
        "arc_reactor", "dna_helix", "geodesic",
        "cube", "sphere", "torus", "icosahedron",
    ]

    def __init__(self):
        self.holograms: list[Hologram] = []

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def resolve_name(self, raw: str) -> str:
        cleaned = raw.lower().strip()
        return self.NAME_ALIASES.get(cleaned, cleaned.replace(" ", "_"))

    def add(self, name: str) -> Optional[Hologram]:
        name = self.resolve_name(name)
        if len([h for h in self.holograms if not h.is_dismissing]) >= self.MAX_HOLOGRAMS:
            print(f">> SCENE: max {self.MAX_HOLOGRAMS} holograms reached")
            return None
        if any(h.name == name and not h.is_dismissing for h in self.holograms):
            # Already present — just focus it
            self.focus(name)
            return None
        first = not any(not h.is_dismissing for h in self.holograms)
        h = Hologram(
            name=name,
            is_active=first,
            materialize_t=0.0,
            orbit_angle=random.uniform(0, 2 * math.pi),
        )
        self.holograms.append(h)
        self._relayout()
        return h

    def remove(self, name: str):
        name = self.resolve_name(name)
        for h in self.holograms:
            if h.name == name and not h.is_dismissing:
                h.dismiss_t = 0.0
                h.is_active = False
                break
        self._ensure_active()

    def focus(self, name: str):
        name = self.resolve_name(name)
        found = False
        for h in self.holograms:
            if h.name == name and not h.is_dismissing:
                h.is_active = True
                found = True
            else:
                h.is_active = False
        if not found:
            self._ensure_active()
        self._relayout()

    def cycle_focus(self, direction: int = 1):
        live = [h for h in self.holograms if not h.is_dismissing]
        if not live:
            return
        idx = next((i for i, h in enumerate(live) if h.is_active), 0)
        new_idx = (idx + direction) % len(live)
        for h in self.holograms:
            h.is_active = False
        live[new_idx].is_active = True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def active(self) -> Optional[Hologram]:
        return next((h for h in self.holograms if h.is_active and not h.is_dismissing), None)

    def live(self) -> list[Hologram]:
        return [h for h in self.holograms if not h.is_dismissing]

    # ------------------------------------------------------------------
    # Update tick
    # ------------------------------------------------------------------

    def update(self, dt: float):
        dead = []
        for h in self.holograms:
            if h.is_dismissing:
                h.dismiss_t = min(1.0, h.dismiss_t + dt * self.DISMISS_SPEED)
                if h.dismiss_t >= 1.0:
                    dead.append(h)
            elif h.materialize_t < 1.0:
                h.materialize_t = min(1.0, h.materialize_t + dt * self.MAT_SPEED)

            # Orbit inactive holograms
            if not h.is_active and not h.is_dismissing:
                h.orbit_angle += dt * self.ORBIT_SPEED

            # Smooth rotation and pan
            smooth = 0.12
            h.rot_x += (h.target_rot_x - h.rot_x) * smooth
            h.rot_y += (h.target_rot_y - h.rot_y) * smooth
            h.pan_x += (h.target_pan_x - h.pan_x) * smooth
            h.pan_y += (h.target_pan_y - h.pan_y) * smooth

        for h in dead:
            # Free GPU display lists if this was a custom (OBJ) mesh
            if _GL_AVAILABLE:
                if h.gl_list_solid:
                    glDeleteLists(h.gl_list_solid, 1)
                if h.gl_list_wire:
                    glDeleteLists(h.gl_list_wire, 1)
            self.holograms.remove(h)
        if dead:
            self._ensure_active()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {
            "holograms": [
                {
                    "name":      h.name,
                    "rot_x":     round(h.rot_x, 2),
                    "rot_y":     round(h.rot_y, 2),
                    "scale":     round(h.scale, 2),
                    "is_active": h.is_active,
                    "strokes":   [
                        {"points": s.points,
                         "color":  list(s.color),
                         "width":  s.width}
                        for s in h.strokes
                    ],
                }
                for h in self.live()
            ]
        }

    def load_state(self, data: dict):
        self.holograms.clear()
        for hd in data.get("holograms", []):
            h = Hologram(
                name=hd["name"],
                rot_x=hd.get("rot_x", 0.0),
                rot_y=hd.get("rot_y", 0.0),
                scale=hd.get("scale", 1.0),
                is_active=hd.get("is_active", False),
                materialize_t=1.0,
            )
            h.target_rot_x = h.rot_x
            h.target_rot_y = h.rot_y
            h.target_pan_x = h.pan_x
            h.target_pan_y = h.pan_y
            for sd in hd.get("strokes", []):
                h.strokes.append(Stroke(
                    points=sd["points"],
                    color=tuple(sd.get("color", [0.235, 0.827, 1.0])),
                    width=sd.get("width", 2.0),
                ))
            self.holograms.append(h)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_active(self):
        live = self.live()
        if live and not any(h.is_active for h in live):
            live[0].is_active = True

    def _relayout(self):
        """Evenly space inactive holograms in the orbit ring."""
        inactive = [h for h in self.live() if not h.is_active]
        n = len(inactive)
        if n == 0:
            return
        for i, h in enumerate(inactive):
            h.orbit_angle = i * (2 * math.pi / n)
