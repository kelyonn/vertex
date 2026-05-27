"""
Project Vertex v4 — JARVIS Hologram Workbench
Main application entry-point.
"""
import argparse
import math
import os
import sys
import time
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, DROPFILE

from config   import Config
from scene    import Scene
from renderer import Renderer
from vision   import HandSensor
from gesture_engine import GestureMode
from sketch   import SketchController
from voice    import VoiceController, VOICE_ENABLED
from utils    import save_blueprint, load_blueprint, screenshot, export_obj, clamp, lerp, load_obj


# ── CLI ────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Project Vertex v4 — JARVIS Hologram Workbench")
    p.add_argument("--model", type=str, default=None,
                   help="Load a .obj model on startup")
    p.add_argument("--demo",  action="store_true",
                   help="Demo mode: start with arc_reactor + dna_helix")
    return p.parse_args()


# ── Application ────────────────────────────────────────────────────────

class VertexApp:
    def __init__(self, args):
        self.config = Config()

        # ── Window ──────────────────────────────────────────────────
        pygame.init()
        self.W = self.config.get("display", "width")  or 1200
        self.H = self.config.get("display", "height") or  900
        self.screen = pygame.display.set_mode(
            (self.W, self.H), DOUBLEBUF | OPENGL | DROPFILE)
        pygame.display.set_caption("PROJECT VERTEX  \\  v4.0  \\  JARVIS HOLOGRAM WORKBENCH")

        # ── Renderer + HUD ──────────────────────────────────────────
        self.renderer = Renderer(self.config)
        self.renderer.setup_camera()

        # ── Scene ───────────────────────────────────────────────────
        self.scene = Scene()
        if args.demo:
            self.scene.add("arc_reactor")
            self.scene.add("dna_helix")
        elif args.model and os.path.isfile(args.model):
            self._load_obj_file(args.model)
        else:
            default = self.config.get("scene", "default") or "arc_reactor"
            self.scene.add(default)

        # ── Webcam + hand sensor ─────────────────────────────────────
        self.cap    = None
        self._init_webcam()
        dc = self.config.get("hand_sensor", "detection_confidence") or 0.7
        tc = self.config.get("hand_sensor", "tracking_confidence")  or 0.7
        mp = self.config.get("hand_sensor", "min_palm_px")          or 35
        self.sensor = HandSensor(dc, tc, min_palm_px=mp)

        # ── Voice ───────────────────────────────────────────────────
        self.voice = VoiceController(self.config)

        # ── Sketch ──────────────────────────────────────────────────
        self.sketch = SketchController(self.W, self.H)

        # ── App state ───────────────────────────────────────────────
        self.zoom_level     = self.config.get("camera", "initial_z") or -5.0
        self.zoom_min       = self.config.get("camera", "zoom_min")  or -15.0
        self.zoom_max       = self.config.get("camera", "zoom_max")  or -2.0
        self.auto_rotate    = False
        self.show_grid      = self.config.get("rendering", "show_grid")
        self.show_axes      = self.config.get("rendering", "show_axes")
        self.show_hud       = self.config.get("rendering", "show_hud")
        self.pip_visible    = True
        self.is_sketch_mode = False
        self.wireframe      = self.config.get("rendering", "wireframe") or False
        self.show_lines     = False   # overlay cyan glow lines on top of solid fill

        # ── Gesture tracking ─────────────────────────────────────────
        # _pinch_anchor: None | (hand_id, (ax, ay), (anchor_pan_x, anchor_pan_y))
        # Set on first pinch frame; absolute position derived from it every frame.
        self._pinch_anchor  = None
        self._orbit_last    = None
        self._zoom_base_d   = None
        self._zoom_base_lvl = self.zoom_level
        self._current_mode  = GestureMode.NONE
        self.rot_sens       = self.config.get("controls", "rotation_sensitivity") or 0.5
        self.smoothing      = self.config.get("controls", "smoothing_factor")     or 0.12

        # ── OBJ background load queue ────────────────────────────────
        self._obj_queue: list = []

        # ── Out-of-range toast throttle ──────────────────────────────
        self._oor_last_toast: float = 0.0

        # ── SPACEBAR PTT state ──────────────────────────────────────
        self._space_held = False

        self.running = True
        self._print_banner()

    # ── Init helpers ──────────────────────────────────────────────────

    def _init_webcam(self):
        for idx in range(3):
            cap = __import__("cv2").VideoCapture(idx)
            if cap.isOpened():
                cap.set(3, 640); cap.set(4, 480); cap.set(5, 30)
                self.cap = cap
                print(f">> CAM: opened index {idx}")
                return
        print(">> CAM: not found — hand tracking disabled")

    def _print_banner(self):
        print(">> PROJECT VERTEX v4.0  — JARVIS HOLOGRAM WORKBENCH  ONLINE")
        print(">> ──────────────────────────────────────────────────────────")
        print(">>  KEYBOARD:  1-7 shape  W wire  L lines  R auto-rot  H hud")
        print(">>             V cam  S save  B load  P screenshot  DEL delete  ESC quit")
        print(">>             [ / ] cycle holograms  D sketch mode")
        print(">>             SPACE  push-to-talk voice command")
        print(">>  GESTURES:  1f pinch+drag = ORBIT   1f drag = MOVE")
        print(">>             fist hold 1s = RESET   2 hands = ZOOM")
        print(">>  SCALE:     +/= to grow   - to shrink")
        if VOICE_ENABLED:
            print(">>  VOICE:    SPACE to talk  |  'add helix'  'annotate'  'reset'")
        else:
            print(">>  VOICE:    disabled (install faster-whisper + sounddevice)")

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            dt = clock.tick(60) / 1000.0

            self._handle_events()
            self._flush_obj_queue()
            self._process_voice()
            self.scene.update(dt)

            # Webcam + gestures
            webcam_frame = None
            hand_data    = {}
            if self.cap:
                ok, frame = self.cap.read()
                if ok:
                    webcam_frame, hand_data = self.sensor.process_frame(frame)
                    if self.is_sketch_mode:
                        self.sketch.update(hand_data, self.scene.active())
                    else:
                        self._process_gestures(hand_data)

            # Auto-rotate
            if self.auto_rotate and (active := self.scene.active()):
                active.target_rot_y += 0.35

            # Build sketch preview
            sketch_preview = self.sketch.get_preview() if self.is_sketch_mode else []

            self.renderer.render_frame(
                screen=self.screen,
                scene=self.scene,
                app_state={
                    "zoom_level":    self.zoom_level,
                    "show_grid":     self.show_grid,
                    "show_axes":     self.show_axes,
                    "show_hud":      self.show_hud,
                    "pip_visible":   self.pip_visible,
                    "webcam_frame":  webcam_frame,
                    "current_mode":  self._current_mode,
                    "is_sketch_mode":self.is_sketch_mode,
                    "is_listening":  self.voice.is_listening,
                    "wireframe":     self.wireframe,
                    "show_lines":    self.show_lines,
                    "auto_rotate":   self.auto_rotate,
                    "hand_data":     hand_data,
                    "sketch_preview":sketch_preview,
                    "t":             time.time(),
                },
            )
            pygame.display.flip()

        self._cleanup()

    # ── Event handling ────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._on_key_down(event.key)

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE and self._space_held:
                    self.voice.ptt_stop()
                    self._space_held = False

            elif event.type == DROPFILE:
                path = event.file if hasattr(event, "file") else getattr(event, "filename", None)
                if path and path.lower().endswith(".obj"):
                    self._load_obj_file(path)

    def _on_key_down(self, key):
        # SPACEBAR — push-to-talk
        if key == pygame.K_SPACE:
            if not self._space_held:
                self.voice.ptt_start()
                self._space_held = True
            return

        # Shape keys 1-7
        from shapes import ShapeRenderer
        shape_map = {getattr(pygame, f"K_{i+1}"): name
                     for i, name in enumerate(ShapeRenderer.SHAPE_NAMES)}
        if key in shape_map:
            result = self.scene.add(shape_map[key])
            msg = f"ADDED: {shape_map[key].upper()}" if result else f"FOCUS: {shape_map[key].upper()}"
            self.renderer.hud.show_toast(msg)
            return

        # Sketch exit keys handled first
        if self.is_sketch_mode and key in (pygame.K_ESCAPE, pygame.K_d):
            self._exit_sketch()
            return

        actions = {
            pygame.K_w:       self._toggle_wireframe,
            pygame.K_r:       self._toggle_auto_rotate,
            pygame.K_h:       self._toggle_hud,
            pygame.K_v:       lambda: setattr(self, "pip_visible", not self.pip_visible),
            pygame.K_g:       lambda: setattr(self, "show_grid",   not self.show_grid),
            pygame.K_a:       lambda: setattr(self, "show_axes",   not self.show_axes),
            pygame.K_d:       self._enter_sketch,
            pygame.K_s:       lambda: save_blueprint(self.scene),
            pygame.K_b:       lambda: (load_blueprint(self.scene, "blueprint.json"),
                                       self.renderer.hud.show_toast("BLUEPRINT LOADED")),
            pygame.K_l:       self._toggle_lines,
            pygame.K_p:       self._do_screenshot,
            pygame.K_e:       self._do_export,
            pygame.K_EQUALS:       lambda: self._scale_active(+0.1),
            pygame.K_PLUS:         lambda: self._scale_active(+0.1),
            pygame.K_MINUS:        lambda: self._scale_active(-0.1),
            pygame.K_DELETE:       self._delete_active,
            pygame.K_BACKSPACE:    self._delete_active,
            pygame.K_LEFTBRACKET:  lambda: self.scene.cycle_focus(-1),
            pygame.K_RIGHTBRACKET: lambda: self.scene.cycle_focus(+1),
            pygame.K_ESCAPE:  lambda: setattr(self, "running", False),
        }
        if fn := actions.get(key):
            fn()

    # ── Gesture processing ────────────────────────────────────────────

    def _maybe_show_oor_toast(self):
        """Show OUT OF RANGE at most once every 2 seconds."""
        now = time.time()
        if now - self._oor_last_toast > 2.0:
            self.renderer.hud.show_toast("HAND OUT OF RANGE")
            self._oor_last_toast = now

    def _process_gestures(self, data: dict):
        raw_hands = data.get("both_hands", [])
        # Discard hands that are too far from the camera (palm too small in frame)
        hands = [h for h in raw_hands if not h.get("too_far")]
        if raw_hands and not hands:
            self._maybe_show_oor_toast()
        active = self.scene.active()

        # Two-hand zoom
        if len(hands) == 2 and data.get("zoom_distance"):
            self._current_mode = GestureMode.ZOOM
            dist = data["zoom_distance"]
            if self._zoom_base_d is None:
                self._zoom_base_d   = dist
                self._zoom_base_lvl = self.zoom_level
            if self._zoom_base_d > 1:
                ratio           = dist / self._zoom_base_d
                target_z        = self._zoom_base_lvl / ratio
                self.zoom_level = lerp(self.zoom_level,
                                       clamp(target_z, self.zoom_min, self.zoom_max),
                                       self.smoothing * 2)
        else:
            self._zoom_base_d = None

        # Fist hold → reset (zero targets only; lerp animates smoothly to zero)
        if data.get("fist_reset") and active:
            active.target_rot_x = active.target_rot_y = 0.0
            active.target_pan_x = active.target_pan_y = 0.0
            self.zoom_level   = self.config.get("camera", "initial_z") or -5.0
            active.scale      = 1.0
            self._pinch_anchor = None
            self._orbit_last   = None
            self.renderer.hud.show_toast("VIEW RESET")

        if not hands:
            self._current_mode = GestureMode.NONE
            return

        h       = hands[0]
        mode    = h["mode"]
        pinched = h["pinched"]

        # When pinching, finger count drops to ~0 → mode becomes NONE.
        # Treat an active pinch as ORBIT so pan tracking is not killed.
        effective_mode = GestureMode.ORBIT if pinched else mode
        if len(hands) < 2:
            self._current_mode = effective_mode

        if not active or effective_mode != GestureMode.ORBIT:
            self._pinch_anchor = None
            self._orbit_last   = None
            return

        def _pinch_pt(hand):
            ix, iy = hand["index_tip"]
            tx, ty = hand["thumb_tip"]
            return ((ix + tx) // 2, (iy + ty) // 2)

        cam_h = data.get("cam_size", (640, 480))[1]
        fov   = self.config.get("display", "fov") or 45
        # World units per camera pixel at current depth (frustum geometry, exact).
        px_to_world = abs(self.zoom_level) * 2 * math.tan(math.radians(fov / 2)) / cam_h

        DEAD_ZONE = 3

        if pinched:
            # ── Pinch + drag → ORBIT (rotate) ──────────────────────────────────
            # Track the pinch midpoint so "grab-and-spin" feels natural.
            self._pinch_anchor = None
            cx_p, cy_p = _pinch_pt(h)
            if self._orbit_last:
                dx = cx_p - self._orbit_last[0]
                dy = cy_p - self._orbit_last[1]
                if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
                    active.target_rot_y += dx * self.rot_sens
                    active.target_rot_x += dy * self.rot_sens
            self._orbit_last = (cx_p, cy_p)

        else:
            # ── 1-finger drag (no pinch) → MOVE hologram (anchor-and-track) ────
            self._orbit_last = None

            if self._pinch_anchor is not None:
                anchor_hid, (ax, ay), (apx, apy) = self._pinch_anchor
                tracked = next((hh for hh in hands if hh["id"] == anchor_hid), None)
                if tracked is None:
                    return                          # brief dropout — keep anchor
                h = tracked
            else:
                self._pinch_anchor = (
                    h["id"],
                    (h["center"][0], h["center"][1]),
                    (active.target_pan_x, active.target_pan_y),
                )
                return

            cx, cy = h["center"]
            ax, ay = self._pinch_anchor[1]
            apx, apy = self._pinch_anchor[2]
            active.target_pan_x = clamp(apx + (cx - ax) * px_to_world, -5.0, 5.0)
            active.target_pan_y = clamp(apy - (cy - ay) * px_to_world, -4.0, 4.0)

    # ── Voice command dispatch ────────────────────────────────────────

    def _process_voice(self):
        while True:
            cmd = self.voice.poll_command()
            if cmd is None:
                break
            self._handle_voice_command(cmd["command"], cmd.get("args", []))

    def _handle_voice_command(self, command: str, args: list):
        toast = None
        if command == "add" and args:
            self.scene.add(args[0])
            toast = f"MATERIALIZED: {args[0].upper()}"
        elif command == "remove" and args:
            self.scene.remove(args[0])
            toast = f"DISMISSED: {args[0].upper()}"
        elif command == "focus" and args:
            self.scene.focus(args[0])
            toast = f"FOCUS: {args[0].upper()}"
        elif command == "annotate":
            self._enter_sketch()
            toast = "ANNOTATION MODE"
        elif command == "done":
            self._exit_sketch()
            toast = "ANNOTATION CLOSED"
        elif command == "wireframe":
            self._toggle_wireframe()
        elif command == "solid":
            self.wireframe = False
            toast = "SOLID MODE"
        elif command == "reset":
            if active := self.scene.active():
                active.target_rot_x = active.target_rot_y = 0.0
                self.zoom_level = self.config.get("camera", "initial_z") or -5.0
            toast = "VIEW RESET"
        elif command == "rotate":
            self.auto_rotate = True
            toast = "AUTO-ROTATE ON"
        elif command == "stop":
            self.auto_rotate = False
            toast = "AUTO-ROTATE OFF"
        elif command == "screenshot":
            self._do_screenshot()
        elif command == "save":
            save_blueprint(self.scene)
            toast = "BLUEPRINT SAVED"
        elif command == "load":
            load_blueprint(self.scene)
            toast = "BLUEPRINT LOADED"
        elif command == "help":
            self._print_banner()

        if toast:
            self.renderer.hud.show_toast(toast)

    # ── Sketch mode ───────────────────────────────────────────────────

    def _enter_sketch(self):
        if not self.scene.active():
            self.renderer.hud.show_toast("NO ACTIVE HOLOGRAM")
            return
        self.is_sketch_mode = True
        self.renderer.hud.show_toast("ANNOTATION MODE ENGAGED")

    def _exit_sketch(self):
        self.sketch.finish_current(self.scene.active())
        self.is_sketch_mode = False
        self.renderer.hud.show_toast("ANNOTATION SAVED")

    # ── Toggle helpers ────────────────────────────────────────────────

    def _delete_active(self):
        if active := self.scene.active():
            self.scene.remove(active.name)
            self.renderer.hud.show_toast(f"DISMISSED: {active.name.upper()}")

    def _scale_active(self, delta: float):
        if active := self.scene.active():
            active.scale = clamp(active.scale + delta, 0.1, 5.0)

    def _toggle_wireframe(self):
        self.wireframe = not self.wireframe
        self.renderer.hud.show_toast(f"WIREFRAME {'ON' if self.wireframe else 'OFF'}")

    def _toggle_lines(self):
        self.show_lines = not self.show_lines
        self.renderer.hud.show_toast(f"LINES {'ON' if self.show_lines else 'OFF'}")

    def _toggle_auto_rotate(self):
        self.auto_rotate = not self.auto_rotate
        self.renderer.hud.show_toast(f"AUTO-ROTATE {'ON' if self.auto_rotate else 'OFF'}")

    def _toggle_hud(self):
        self.show_hud = not self.show_hud

    def _do_screenshot(self):
        if screenshot():
            self.renderer.hud.show_toast("SCREENSHOT SAVED")

    def _do_export(self):
        if active := self.scene.active():
            export_obj(active.name, size=active.scale)
            self.renderer.hud.show_toast(f"EXPORTED: {active.name.upper()}.OBJ")

    # ── OBJ drag-drop / CLI load ──────────────────────────────────────

    def _load_obj_file(self, path: str):
        """Kick off background OBJ load so the app stays responsive."""
        import threading
        name = os.path.splitext(os.path.basename(path))[0].lower().replace(" ", "_")
        self.renderer.hud.show_toast(f"LOADING: {name.upper()}…")

        def _worker():
            result = load_obj(path)
            if result is None:
                self.renderer.hud.show_toast(f"LOAD FAILED: {name.upper()}")
                return
            verts, faces, edges, face_colours = result
            # Post result back to main thread via a thread-safe queue
            self._obj_queue.append((name, verts, faces, edges, face_colours))

        threading.Thread(target=_worker, daemon=True).start()

    def _flush_obj_queue(self):
        """Called each frame on the GL thread — picks up completed background OBJ loads."""
        while self._obj_queue:
            name, verts, faces, edges, face_colours = self._obj_queue.pop(0)
            self.scene.add(name)
            if h := next((hh for hh in self.scene.holograms if hh.name == name), None):
                h.custom_mesh        = (verts, faces, edges, face_colours)
                h.has_original_color = bool(face_colours)
                # Compile display lists on the GL thread — one-time cost,
                # makes every subsequent frame a single glCallList per pass.
                solid_id, wire_id = self.renderer.shape_renderer.compile_mesh_lists(
                    verts, faces, edges, face_colours)
                h.gl_list_solid = solid_id
                h.gl_list_wire  = wire_id
            self.renderer.hud.show_toast(f"LOADED: {name.upper()}")

    # ── Cleanup ───────────────────────────────────────────────────────

    def _cleanup(self):
        self.voice.stop()
        if self.cap:
            self.cap.release()
        self.config.save()
        pygame.quit()
        print(">> SHUTDOWN COMPLETE")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    try:
        app = VertexApp(args)
        app.run()
    except KeyboardInterrupt:
        print("\n>> Interrupted.")
    except Exception as e:
        import traceback
        print(f">> FATAL: {e}")
        traceback.print_exc()
    finally:
        import contextlib
        with contextlib.suppress(Exception):
            pygame.quit()
        with contextlib.suppress(Exception):
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
