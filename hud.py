"""
JARVIS-style HUD for Project Vertex v4.
Drawn entirely with OpenGL 2D primitives and text blits.

Layout:
  TL corner → "VERTEX // v4.0" + active mode icon
  TR corner → FPS / vertex count / hologram name
  BL corner → rotation + scale + zoom readouts
  BR corner → hand status (finger count, mode name)
  Top-centre → transient toast notification (fades after 3 s)
  Bottom-centre → hologram deck dots (one per live hologram)
  Full-screen overlays:
    • Listening: cyan vignette + "LISTENING" pulse
    • Sketch:    amber tint + "ANNOTATING" badge
"""
import math
import time
import pygame
from OpenGL.GL import *
from gesture_engine import GestureMode, MODE_META

# ── JARVIS colour constants ────────────────────────────────────────────
CYAN    = (0.235, 0.827, 1.000)
AMBER   = (1.000, 0.651, 0.188)
WHITE   = (0.900, 0.960, 1.000)
DIM     = (0.350, 0.500, 0.600)
BLACK   = (0.008, 0.016, 0.035)

# RGB 0-255 variants for text rendering
_C_CYAN  = (  0, 210, 255)
_C_AMBER = (255, 166,  48)
_C_WHITE = (220, 240, 255)
_C_DIM   = ( 90, 140, 160)
_C_GREEN = ( 60, 220, 120)

# ── Text-rendering cache ───────────────────────────────────────────────
_FONTS: dict[str, pygame.font.Font] = {}
_TEX_CACHE: dict[tuple, tuple]      = {}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    key = f"{size}_{bold}"
    if key not in _FONTS:
        pygame.font.init()
        name = pygame.font.match_font("couriernew,courier,liberationmono,monospace")
        _FONTS[key] = pygame.font.Font(name, size)
        if bold:
            _FONTS[key].set_bold(True)
    return _FONTS[key]


def _blit(text: str, x: float, y: float, size: int,
          color=(220, 240, 255), bold: bool = False):
    """Render text at (x, y) in screen-space via a cached GL texture."""
    key = (text, size, color, bold)
    if key not in _TEX_CACHE:
        surf = _font(size, bold).render(text, True, color)
        tw, th = surf.get_size()
        try:
            data = pygame.image.tobytes(surf, "RGBA", True)
        except AttributeError:
            data = pygame.image.tostring(surf, "RGBA", True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        _TEX_CACHE[key] = (tex, tw, th)

    tex, tw, th = _TEX_CACHE[key]
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x,    y)
    glTexCoord2f(1, 1); glVertex2f(x+tw, y)
    glTexCoord2f(1, 0); glVertex2f(x+tw, y+th)
    glTexCoord2f(0, 0); glVertex2f(x,    y+th)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def _bracket(x, y, w, h, sz=14, r=1.0, g=1.0, b=1.0, a=0.7):
    """Draw L-brackets at all 4 corners of the rect (x,y,w,h)."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(r, g, b, a)
    glBegin(GL_LINES)
    # TL
    glVertex2f(x,    y);    glVertex2f(x+sz, y)
    glVertex2f(x,    y);    glVertex2f(x,    y+sz)
    # TR
    glVertex2f(x+w,  y);    glVertex2f(x+w-sz, y)
    glVertex2f(x+w,  y);    glVertex2f(x+w,    y+sz)
    # BL
    glVertex2f(x,    y+h);  glVertex2f(x+sz,   y+h)
    glVertex2f(x,    y+h);  glVertex2f(x,      y+h-sz)
    # BR
    glVertex2f(x+w,  y+h);  glVertex2f(x+w-sz, y+h)
    glVertex2f(x+w,  y+h);  glVertex2f(x+w,    y+h-sz)
    glEnd()
    glDisable(GL_BLEND)


def _quad(x, y, w, h, r, g, b, a):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(r, g, b, a)
    glBegin(GL_QUADS)
    glVertex2f(x,   y);   glVertex2f(x+w, y)
    glVertex2f(x+w, y+h); glVertex2f(x,   y+h)
    glEnd()
    glDisable(GL_BLEND)


# ── HUD class ──────────────────────────────────────────────────────────

class HUD:
    BRACKET_SZ = 18   # corner bracket arm length

    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        self._toast_text = ""
        self._toast_time = 0.0
        self._TOAST_LIFE = 3.0

    def show_toast(self, text: str):
        self._toast_text = text
        self._toast_time = time.time()

    # ── Main draw ──────────────────────────────────────────────────────

    def draw(self, scene, app_state: dict, fps: int):
        """Draw all HUD elements. Must be called inside a 2D ortho pass."""
        t           = time.time()
        is_sketch   = app_state.get("is_sketch_mode", False)
        is_listen   = app_state.get("is_listening", False)
        current_mode= app_state.get("current_mode", GestureMode.NONE)
        hand_data   = app_state.get("hand_data", {})
        zoom        = app_state.get("zoom_level", -5.0)
        wireframe   = app_state.get("wireframe", False)
        auto_rot    = app_state.get("auto_rotate", False)
        active      = scene.active()
        live        = scene.live()

        # ── Overlays first (behind everything else) ──
        if is_listen:
            self._draw_listening_overlay(t)
        if is_sketch:
            self._draw_sketch_overlay()

        # ── Screen corner brackets ──
        cr, cg, cb = CYAN if not is_sketch else AMBER
        _bracket(4, 4, self.W-8, self.H-8,
                 sz=self.BRACKET_SZ, r=cr, g=cg, b=cb, a=0.55)

        # ── Top-left: title + mode ──
        self._draw_top_left(current_mode, is_sketch, is_listen, wireframe, auto_rot)

        # ── Top-right: FPS / vertex count / name ──
        self._draw_top_right(fps, active, live)

        # ── Bottom-left: rotation + scale + zoom ──
        if active:
            self._draw_bottom_left(active, zoom)

        # ── Bottom-right: hand status ──
        self._draw_bottom_right(hand_data, current_mode)

        # ── Bottom-centre: hologram deck ──
        self._draw_deck(live, active)

        # ── Centre-top: toast ──
        self._draw_toast(t)

    # ── Sub-elements ──────────────────────────────────────────────────

    def _draw_top_left(self, mode, is_sketch, is_listen, wireframe, auto_rot):
        pad = 20
        _blit("PROJECT  VERTEX  //  v4.0", pad, pad, 14, _C_CYAN, bold=True)

        if is_listen:
            label = "LISTENING..."
            col   = _C_CYAN
        elif is_sketch:
            label = "ANNOTATING"
            col   = _C_AMBER
        else:
            label, col = MODE_META[mode][0], _C_CYAN

        _blit(f"[{label}]", pad, pad + 22, 13, col)

        flags = []
        if wireframe: flags.append("WIRE")
        if auto_rot:  flags.append("AUTO-ROT")
        if flags:
            _blit("  ".join(flags), pad, pad + 44, 11, _C_DIM)

    def _draw_top_right(self, fps: int, active, live: list):
        pad   = 20
        lines = [
            (f"FPS  {fps:3d}", _C_CYAN),
        ]
        if active:
            v, f = _shape_info(active.name)
            lines += [
                (f"VERTS  {v:,}", _C_DIM),
                (active.name.upper().replace("_", " "), _C_WHITE),
            ]
        if len(live) > 1:
            lines.append((f"HOLOGRAMS  {len(live)}", _C_DIM))

        for i, (txt, col) in enumerate(lines):
            tw = len(txt) * 8 + 10
            _blit(txt, self.W - tw - pad, pad + i * 20, 13, col)

    def _draw_bottom_left(self, active, zoom: float):
        pad  = 20
        y    = self.H - 120
        rows = [
            (f"ROT   X{active.rot_x:+.0f}°  Y{active.rot_y:+.0f}°", _C_DIM),
            (f"SCALE {active.scale:.2f}x", _C_DIM),
            (f"ZOOM  {zoom:.2f}", _C_DIM),
        ]
        for i, (txt, col) in enumerate(rows):
            _blit(txt, pad, y + i * 20, 13, col)

        # Mini rotation compass
        self._draw_compass(pad + 80, y - 40, active.rot_y)

    def _draw_compass(self, cx: float, cy: float, rot_y: float):
        r = 18
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(*DIM, 0.5)
        glBegin(GL_LINE_LOOP)
        for i in range(24):
            a = 2*math.pi*i/24
            glVertex2f(cx + r*math.cos(a), cy + r*math.sin(a))
        glEnd()
        # North needle
        angle = math.radians(-rot_y)
        glColor4f(*CYAN, 0.9)
        glBegin(GL_LINES)
        glVertex2f(cx, cy)
        glVertex2f(cx + r*0.8*math.sin(angle), cy - r*0.8*math.cos(angle))
        glEnd()
        glDisable(GL_BLEND)
        _blit("N", cx - 4, cy - r - 12, 9, _C_DIM)

    def _draw_bottom_right(self, hand_data: dict, current_mode: GestureMode):
        pad   = 20
        hands = hand_data.get("both_hands", [])
        if not hands:
            _blit("NO HAND DETECTED", self.W - 210, self.H - 40, 11, _C_DIM)
            return

        y = self.H - 20 - len(hands) * 36
        for h in hands:
            label  = h["label"].upper()
            fingers= h.get("finger_count", "?")
            mode_n = MODE_META[h["mode"]][0]
            pinched= "  [PINCH]" if h.get("pinched") else ""
            txt    = f"{label}  {fingers}F  {mode_n}{pinched}"
            tw     = len(txt) * 8 + 10
            _blit(txt, self.W - tw - pad, y, 12, _C_CYAN)

            # Finger pip dots
            pip_x = self.W - pad - 140
            for fi in range(5):
                on = fi < (h.get("finger_count") or 0)
                r, g, b = (CYAN if on else DIM)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glColor4f(r, g, b, 0.9 if on else 0.3)
                glBegin(GL_TRIANGLE_FAN)
                px = pip_x + fi * 14
                py = y + 22
                glVertex2f(px, py)
                for j in range(9):
                    a = 2*math.pi*j/8
                    glVertex2f(px + 4*math.cos(a), py + 4*math.sin(a))
                glEnd()
                glDisable(GL_BLEND)

            y += 36

    def _draw_deck(self, live: list, active):
        if len(live) <= 1:
            return
        n     = len(live)
        total = n * 22
        start = (self.W - total) / 2
        cy    = self.H - 22

        for i, h in enumerate(live):
            is_act = h.is_active
            cx     = start + i * 22 + 6
            r      = 6 if is_act else 4
            cr, cg, cb = (CYAN if is_act else DIM)
            a      = 1.0 if is_act else 0.4
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glColor4f(cr, cg, cb, a)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for j in range(13):
                ang = 2*math.pi*j/12
                glVertex2f(cx + r*math.cos(ang), cy + r*math.sin(ang))
            glEnd()
            glDisable(GL_BLEND)
            # Label
            name_short = h.name[:4].upper()
            _blit(name_short, cx - 12, cy + 8, 9, _C_CYAN if is_act else _C_DIM)

    def _draw_toast(self, now: float):
        if not self._toast_text:
            return
        age   = now - self._toast_time
        if age > self._TOAST_LIFE:
            return
        alpha = 1.0 - (age / self._TOAST_LIFE) ** 2
        tw    = len(self._toast_text) * 9 + 20
        x     = (self.W - tw) / 2
        y     = 54
        _quad(x - 6, y - 4, tw + 12, 28, 0, 0, 0, 0.5 * alpha)
        col = tuple(int(c * 255 * alpha) for c in CYAN)
        _blit(self._toast_text.upper(), x, y, 14, col, bold=True)

    def _draw_listening_overlay(self, t: float):
        # Subtle cyan vignette at screen edges
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        glColor4f(*CYAN, 0.04 * pulse)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self.W, 0)
        glVertex2f(self.W, self.H); glVertex2f(0, self.H)
        glEnd()
        glDisable(GL_BLEND)

        # "LISTENING..." badge
        txt = "LISTENING..."
        tw  = len(txt) * 10 + 16
        x   = (self.W - tw) / 2
        y   = self.H // 2 - 20
        _quad(x - 4, y - 4, tw + 8, 32, 0, 0.1, 0.2, 0.7)
        col = tuple(int(c * 255) for c in CYAN)
        _blit(txt, x, y, 16, col, bold=True)

    def _draw_sketch_overlay(self):
        # Barely-visible amber tint over the whole screen
        _quad(0, 0, self.W, self.H, *AMBER, 0.025)

        # "ANNOTATING" badge — top-centre
        txt = "[ ANNOTATING ]"
        tw  = len(txt) * 10 + 16
        x   = (self.W - tw) / 2
        y   = 8
        _quad(x - 4, y - 2, tw + 8, 28, 0, 0, 0, 0.6)
        _blit(txt, x, y + 3, 14, _C_AMBER, bold=True)


def _shape_info(name: str) -> tuple[int, int]:
    return {
        "cube":        (8,   6),
        "sphere":      (441, 400),
        "torus":       (576, 552),
        "icosahedron": (12,  20),
        "arc_reactor": (500, 480),
        "dna_helix":   (180, 160),
        "geodesic":    (42,  80),
    }.get(name, (0, 0))
