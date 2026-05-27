"""
OpenGL Renderer for Project Vertex v4 — JARVIS Hologram Workbench.

Visual approach:
  • Deep-void background (#020810)
  • Multi-pass fake bloom: fill (additive 8% alpha) + 3 wireframe passes
    with decreasing line-width and increasing opacity → additive stacking
    creates natural glow at intersections — no FBOs required
  • Animated scanlines (2D pass) at 5% opacity drifting downward
  • Tactical grid floor fading at distance
  • Inactive holograms orbit at half-scale, no bloom
  • Fingertip glow trails + targeting reticle (ambient, always-on)
  • Stylized sensor PiP: desaturate → cyan tint → scanlines → brackets
"""
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
import math
import time
import cv2
from collections import deque

from shapes import ShapeRenderer
from gesture_engine import GestureMode
from hud import HUD, _blit, _bracket, _quad
from scene import Scene

# ── JARVIS colour constants ────────────────────────────────────────────
_BG    = (0.008, 0.016, 0.035, 1.0)   # deep void
_CYAN  = (0.235, 0.827, 1.000)         # #3CD3FF
_AMBER = (1.000, 0.651, 0.188)         # #FFA630
_DIM   = (0.055, 0.280, 0.420)         # inactive hologram colour
_GRID  = (0.040, 0.180, 0.280)         # grid line colour

_TRAIL_LEN = 14   # fingertip trail frames


class Renderer:
    def __init__(self, config):
        self.config = config
        self.width  = config.get("display", "width")
        self.height = config.get("display", "height")

        self.shape_renderer = ShapeRenderer(wireframe=False)
        self.hud = HUD(self.width, self.height)

        self.fps         = 0
        self._frame_cnt  = 0
        self._last_fps_t = time.time()

        # PiP texture
        self._pip_tex: int = 0

        # Fingertip trail buffers: "hand_id_tip_index" → deque[(sx, sy)]
        self._trails: dict[str, deque] = {}

        self._init_gl()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST);  glDepthFunc(GL_LEQUAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*_BG)
        # Lighting (kept minimal for holographic look — mostly emissive)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 3, 2, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.1, 0.1, 0.12, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.5, 0.5, 0.55, 1])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def setup_camera(self):
        fov  = self.config.get("display", "fov")  or 45
        near = self.config.get("display", "near") or 0.1
        far  = self.config.get("display", "far")  or 50.0
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(fov, self.width / self.height, near, far)
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render_frame(self, screen, scene: Scene, app_state: dict):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        zoom       = app_state.get("zoom_level", -5.0)
        t          = app_state.get("t", time.time())
        wf         = app_state.get("wireframe", False)
        show_lines = app_state.get("show_lines", False)
        hand_data  = app_state.get("hand_data", {})
        cam_size   = hand_data.get("cam_size", (640, 480))

        glTranslatef(0, 0, zoom)

        # ── 3-D scene ────────────────────────────────────────────────
        if app_state.get("show_grid", True):
            self._draw_tactical_grid()
        if app_state.get("show_axes", False):
            self._draw_axes()

        for h in scene.holograms:
            self._draw_hologram(h, t, wf, show_lines)

        # ── 2-D overlay pass ──────────────────────────────────────────
        self._enter_2d()

        # Scanlines
        self._draw_scanlines(t)

        # Webcam sensor PiP
        webcam = app_state.get("webcam_frame")
        if app_state.get("pip_visible", True) and webcam is not None:
            self._render_pip(webcam)

        # Ambient fingertip trails disabled

        # Sketch stroke overlay
        active = scene.active()
        if active and active.strokes:
            self._draw_strokes(active.strokes)

        # Sketch preview (in-progress stroke)
        preview = app_state.get("sketch_preview", [])
        if preview:
            self._draw_stroke_preview(preview)

        # HUD
        if app_state.get("show_hud", True):
            self.hud.draw(scene, app_state, self.fps)

        self._exit_2d()

        # FPS counter
        self._frame_cnt += 1
        now = time.time()
        if now - self._last_fps_t >= 1.0:
            self.fps      = self._frame_cnt
            self._frame_cnt = 0
            self._last_fps_t = now

    # ------------------------------------------------------------------
    # Hologram rendering — JARVIS multi-pass
    # ------------------------------------------------------------------

    def _draw_hologram(self, h, t: float, user_wf: bool, show_lines: bool = False):
        is_active = h.is_active
        is_dismiss = h.is_dismissing

        mat   = h.materialize_t
        dism  = h.dismiss_t if is_dismiss else 0.0
        prog  = mat * (1.0 - dism)   # combined 0→1→0 lifecycle alpha

        if prog <= 0.01:
            return

        # Colour
        if is_active:
            cr, cg, cb = _CYAN
        else:
            cr, cg, cb = _DIM

        alpha_mul = prog ** 1.4

        glPushMatrix()

        if not is_active:
            ox = math.cos(h.orbit_angle) * Scene.ORBIT_RADIUS
            oz = math.sin(h.orbit_angle) * Scene.ORBIT_RADIUS
            glTranslatef(ox, 0, oz)
        else:
            # Apply pan offset for active hologram
            glTranslatef(h.pan_x, h.pan_y, 0)

        glRotatef(h.rot_x, 1, 0, 0)
        glRotatef(h.rot_y, 0, 1, 0)

        base_scale = h.scale * (1.0 if is_active else 0.45)
        # During materialize: scale in from 0
        if mat < 1.0:
            base_scale *= (0.2 + 0.8 * mat)
        glScalef(base_scale, base_scale, base_scale)

        # Flickering wireframe during early materialize/dismiss
        flicker_only = (mat < 0.5) or (is_dismiss and dism > 0.5)

        custom_mesh     = getattr(h, "custom_mesh", None)
        has_orig_colour = getattr(h, "has_original_color", False)
        gl_solid        = getattr(h, "gl_list_solid", 0)
        gl_wire         = getattr(h, "gl_list_wire",  0)
        self._jarvis_passes(h.name, t, cr, cg, cb, alpha_mul,
                            user_wf or flicker_only, is_active,
                            custom_mesh=custom_mesh,
                            has_orig_colour=has_orig_colour,
                            gl_solid=gl_solid, gl_wire=gl_wire,
                            show_lines=show_lines)
        glPopMatrix()

    def _jarvis_passes(self, name: str, t: float,
                       r: float, g: float, b: float,
                       alpha_mul: float, user_wf: bool, bright: bool,
                       custom_mesh=None, has_orig_colour: bool = False,
                       gl_solid: int = 0, gl_wire: int = 0,
                       show_lines: bool = False):
        """Multi-pass JARVIS rendering with additive-blended glow."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # additive blend = glow

        def _draw(wf):
            self.shape_renderer.draw_shape(name, t=t, wireframe=wf,
                                           custom_mesh=custom_mesh,
                                           solid_list=gl_solid,
                                           wire_list=gl_wire)

        # Pass 0: translucent fill / original colours
        if not user_wf:
            if has_orig_colour and custom_mesh:
                # Render with per-face original MTL colours at full opacity
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(1, 1, 1, 0.85 * alpha_mul)   # colour comes from face_colours
                _draw(False)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            else:
                glColor4f(r, g, b, 0.07 * alpha_mul)
                _draw(False)

        # Wireframe glow passes — run in wireframe mode, or when lines overlay is on
        if user_wf or show_lines:
            # Pass 1: outer glow (thick, faint)
            glLineWidth(5.0)
            glColor4f(r, g, b, 0.05 * alpha_mul)
            _draw(True)

            # Pass 2: mid glow
            glLineWidth(2.5)
            glow_factor = 1.4 if bright else 0.9
            glColor4f(r, g, b, 0.18 * alpha_mul * glow_factor)
            _draw(True)

            # Pass 3: crisp line
            glLineWidth(1.0)
            glColor4f(r, g, b, 0.90 * alpha_mul)
            _draw(True)

        glLineWidth(1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Grid & axes
    # ------------------------------------------------------------------

    def _draw_tactical_grid(self):
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        size, step = 8, 0.6
        yr = -1.5   # grid sits just below the hologram

        gr, gg, gb = _GRID
        for i in range(-size, size+1):
            dist   = abs(i) / size
            alpha  = max(0, (1.0 - dist*dist) * 0.45)
            glColor4f(gr, gg, gb, alpha)
            glBegin(GL_LINES)
            glVertex3f(i*step, yr, -size*step)
            glVertex3f(i*step, yr,  size*step)
            glEnd()
            glBegin(GL_LINES)
            glVertex3f(-size*step, yr, i*step)
            glVertex3f( size*step, yr, i*step)
            glEnd()

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_axes(self):
        glDisable(GL_LIGHTING); glLineWidth(2.0)
        glBegin(GL_LINES)
        for col, end in [((0.8,0.1,0.1),(2,0,0)),
                          ((0.1,0.8,0.1),(0,2,0)),
                          ((0.1,0.1,0.8),(0,0,2))]:
            glColor3fv(col); glVertex3f(0,0,0); glVertex3fv(end)
        glEnd()
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Webcam PiP
    # ------------------------------------------------------------------

    def _render_pip(self, bgr_frame):
        h_fr, w_fr = bgr_frame.shape[:2]
        pip_w = int(self.width * 0.19)
        pip_h = int(pip_w * h_fr / w_fr)
        x     = self.width  - pip_w - 14
        y     = self.height - pip_h - 14

        # Raw colour feed — no tinting
        rgb = cv2.cvtColor(cv2.resize(bgr_frame, (pip_w, pip_h)), cv2.COLOR_BGR2RGB)

        if self._pip_tex == 0:
            self._pip_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._pip_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pip_w, pip_h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.tobytes())

        # Background fill
        _quad(x-2, y-2, pip_w+4, pip_h+4, 0, 0, 0, 0.8)

        # Textured quad
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(x,       y)
        glTexCoord2f(1,0); glVertex2f(x+pip_w, y)
        glTexCoord2f(1,1); glVertex2f(x+pip_w, y+pip_h)
        glTexCoord2f(0,1); glVertex2f(x,       y+pip_h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Corner brackets around PiP
        cr, cg, cb = _CYAN
        _bracket(x-2, y-2, pip_w+4, pip_h+4, sz=8, r=cr, g=cg, b=cb, a=0.8)

        # "SENSOR FEED" label
        _blit("SENSOR FEED", x+4, y+4, 10, (0, 200, 240))

    # ------------------------------------------------------------------
    # Scanlines (2D)
    # ------------------------------------------------------------------

    def _draw_scanlines(self, t: float):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0, 0, 0, 0.035)
        offset = int(t * 18) % 4
        glBegin(GL_LINES)
        for y in range(offset, self.height, 4):
            glVertex2f(0, y); glVertex2f(self.width, y)
        glEnd()
        glDisable(GL_BLEND)

    # ------------------------------------------------------------------
    # Fingertip glow trails + targeting reticle
    # ------------------------------------------------------------------

    def _draw_trails(self, hand_data: dict, cam_size: tuple):
        if not hand_data:
            return
        cam_w, cam_h = cam_size
        sx = self.width  / cam_w
        sy = self.height / cam_h

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        active_keys = set()
        cr, cg, cb = _CYAN

        for hand in hand_data.get("both_hands", []):
            hid  = hand["id"]
            tips = hand.get("fingertips", [])

            for ti, (tx, ty) in enumerate(tips):
                key   = f"{hid}_{ti}"
                trail = self._trails.setdefault(key, deque(maxlen=_TRAIL_LEN))
                trail.append((tx * sx, ty * sy))
                active_keys.add(key)

                # Draw fading trail dots
                pts = list(trail)
                n   = len(pts)
                for i, (px, py) in enumerate(pts):
                    age    = i / max(1, n-1)
                    alpha  = age ** 1.8 * 0.7
                    radius = 2 + age * 5
                    glColor4f(cr, cg, cb, alpha * 0.35)
                    self._filled_circle_2d(px, py, radius, 8)

            # Targeting reticle at hand center
            cx, cy = hand["center"]
            rx, ry = cx * sx, cy * sy
            self._draw_reticle(rx, ry)

        # Stale trails: clear quietly
        for k in list(self._trails):
            if k not in active_keys:
                self._trails[k].clear()

        glDisable(GL_BLEND)

    def _draw_reticle(self, rx: float, ry: float):
        sz, inner = 22, 7
        cr, cg, cb = _CYAN
        glColor4f(cr, cg, cb, 0.75)
        glBegin(GL_LINES)
        # Left arm
        glVertex2f(rx-sz, ry); glVertex2f(rx-inner, ry)
        # Right arm
        glVertex2f(rx+inner, ry); glVertex2f(rx+sz, ry)
        # Top arm
        glVertex2f(rx, ry-sz); glVertex2f(rx, ry-inner)
        # Bottom arm
        glVertex2f(rx, ry+inner); glVertex2f(rx, ry+sz)
        glEnd()
        # Small centre dot
        glColor4f(cr, cg, cb, 0.5)
        self._filled_circle_2d(rx, ry, 2.5, 8)

    def _filled_circle_2d(self, cx: float, cy: float, r: float, segs: int = 8):
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(segs+1):
            a = 2*math.pi*i/segs
            glVertex2f(cx + r*math.cos(a), cy + r*math.sin(a))
        glEnd()

    # ------------------------------------------------------------------
    # Sketch stroke rendering
    # ------------------------------------------------------------------

    def _draw_strokes(self, strokes: list):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            self._draw_stroke_lines(stroke.points, stroke.color, stroke.width)
        glDisable(GL_BLEND)

    def _draw_stroke_preview(self, points: list):
        if len(points) < 2:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        self._draw_stroke_lines(points, _AMBER, 2.0)
        glDisable(GL_BLEND)

    def _draw_stroke_lines(self, points, color, width: float):
        cr, cg, cb = color
        # Outer glow
        glLineWidth(width + 2.5)
        glColor4f(cr, cg, cb, 0.25)
        glBegin(GL_LINE_STRIP)
        for nx, ny in points:
            glVertex2f(nx * self.width, ny * self.height)
        glEnd()
        # Core line
        glLineWidth(width)
        glColor4f(cr, cg, cb, 0.92)
        glBegin(GL_LINE_STRIP)
        for nx, ny in points:
            glVertex2f(nx * self.width, ny * self.height)
        glEnd()
        glLineWidth(1.0)

    # ------------------------------------------------------------------
    # 2-D pass helpers
    # ------------------------------------------------------------------

    def _enter_2d(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)

    def _exit_2d(self):
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
