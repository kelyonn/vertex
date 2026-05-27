# Project Vertex v4.0

A gesture- and voice-driven 3D hologram workbench that runs entirely on a webcam. No special hardware. No depth sensor. Point your hand at your laptop and control three-dimensional objects in real time.

---

## Overview

Project Vertex uses MediaPipe hand tracking to translate physical hand gestures into 3D interactions — orbit, move, zoom, scale — applied to OpenGL-rendered holograms displayed in a JARVIS-style interface. Voice commands (push-to-talk via faster-whisper) handle scene management. Arbitrary `.obj` files can be dragged and dropped onto the window and manipulated immediately.

**Stack:** Python 3.11, Pygame, PyOpenGL, MediaPipe, faster-whisper

---

## Quick start

```bash
git clone https://github.com/yourusername/vertex
cd vertex
./run.sh          # macOS / Linux
run.bat           # Windows
```

On first run `run.sh` will create a virtualenv, install all dependencies, download the MediaPipe hand-landmarker model (~4 MB), and launch the application. The Whisper voice model (~75 MB) downloads in the background on first voice activation.

**Demo mode — starts with Arc Reactor and DNA Helix pre-loaded:**

```bash
./demo.sh
```

**Load a model at startup:**

```bash
./run.sh --model path/to/model.obj
```

---

## Gestures

Gestures target the active hologram only (full-brightness cyan, centred on screen).

| Hand shape | Action |
|---|---|
| 1 finger + pinch + drag | Orbit / rotate |
| 1 finger open drag | Move position |
| Both hands spread / close | Zoom in / out |
| Fist held for 1 second | Reset rotation, position, zoom, scale |

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1` – `7` | Add shape: Arc Reactor, DNA Helix, Geodesic, Cube, Sphere, Torus, Icosahedron |
| `[` / `]` | Cycle active hologram |
| `+` / `=` | Scale up |
| `-` | Scale down |
| `W` | Toggle wireframe (cyan lines only, no fill) |
| `L` | Toggle line overlay on top of solid fill |
| `D` | Enter / exit sketch annotation mode |
| `R` | Toggle auto-rotate |
| `V` | Toggle webcam sensor PiP |
| `G` | Toggle grid floor |
| `H` | Toggle HUD |
| `S` | Save blueprint |
| `B` | Load blueprint |
| `P` | Screenshot |
| `E` | Export active hologram as `.obj` |
| `DEL` / `BACKSPACE` | Delete active hologram |
| `SPACE` | Push-to-talk voice command |
| `ESC` | Quit |

---

## Voice commands

Hold `SPACE`, speak, release.

| Command | Effect |
|---|---|
| `add the helix` / `show me the arc reactor` | Materialise a hologram |
| `remove the cube` / `dismiss figure 3` | Dismiss and dissolve |
| `focus on the geodesic` / `select cube` | Switch active hologram |
| `annotate` | Enter sketch mode |
| `done` | Exit sketch mode |
| `wireframe` / `solid` | Toggle wireframe look |
| `reset` | Reset rotation and zoom |
| `rotate` / `stop` | Toggle auto-spin |
| `screenshot` | Save PNG to current directory |
| `save` / `load` | Save / restore blueprint JSON |

---

## Shapes

| Key | Shape | Notes |
|---|---|---|
| `1` | Arc Reactor | Animated — three counter-rotating rings, pulsing core |
| `2` | DNA Helix | Animated — spinning double helix with rungs |
| `3` | Geodesic | Frequency-2 geodesic sphere |
| `4` | Cube | Primitive |
| `5` | Sphere | Primitive |
| `6` | Torus | Primitive |
| `7` | Icosahedron | Primitive |

---

## Loading custom `.obj` models

Drag any `.obj` file onto the running application window. The model loads in the background — the app stays live during parsing. Once loaded, a "LOADED" notification appears and the model is added to the scene as a normal hologram, with all gestures available.

Files with an accompanying `.mtl` material file will have their diffuse colours preserved. Press `W` to switch to JARVIS wireframe style; press `L` to toggle the cyan line overlay on top of the solid-colour view.

Large meshes are automatically decimated to a maximum of 80,000 faces for real-time performance. Geometry is compiled into OpenGL display lists on load, so rendering is GPU-cached and frame-rate-independent of mesh complexity after the initial compile.

**Example models** are provided in the `examples/` directory:

| File | Description |
|---|---|
| `torus_knot.obj` | Trefoil (2,3) torus knot |
| `mobius.obj` | Möbius strip |
| `crystal.obj` | Hexagonal gem |
| `saturn.obj` | Sphere with planetary ring |
| `saddle.obj` | Hyperbolic paraboloid |
| `rocket.obj` | Low-poly rocket with fins |
| `sphere_hires.obj` | High-density sphere |
| `helix_tower.obj` | Twin helix tower |

---

## Sketch annotation mode

1. Say `annotate` or press `D` — HUD accent colour shifts to amber
2. Pinch to draw glowing cyan strokes anchored to the active hologram
3. Make a fist to erase nearby strokes
4. Orbit and zoom still work with two hands while annotating
5. Say `done` or press `D` / `ESC` to exit — strokes are saved with the hologram

---

## Multi-hologram workspace

- Default scene: one Arc Reactor, centred, full brightness
- Add holograms by voice or keyboard (`1`–`7`)
- Maximum of 4 holograms in the scene simultaneously
- Inactive holograms orbit at half scale in a ring around the active one
- Focus any hologram with voice or `[` / `]`
- All gestures act exclusively on the active hologram

---

## Hand distance filtering

Hands that are too far from the camera (palm occupying less than ~35 pixels on the sensor feed) are ignored — no gesture fires, and the sensor PiP displays a red `FAR` marker. The threshold is adjustable in `config.json` under `hand_sensor.min_palm_px` without any code change.

---

## Architecture

```
main.py ──┬── scene.py          Scene graph, Hologram state, animation ticks
          ├── renderer.py        OpenGL pipeline, multi-pass additive glow, display-list mesh cache
          ├── hud.py             JARVIS HUD — brackets, readouts, toasts, mode indicators
          ├── vision.py          MediaPipe hand sensor, palm-distance filter, gesture data
          ├── gesture_engine.py  Debounced gesture state machine
          ├── sketch.py          Modal pinch-annotation controller
          ├── voice.py           faster-whisper PTT, fuzzy grammar dispatch
          ├── shapes.py          Primitives, animated hero shapes, OBJ display-list renderer
          └── utils.py           OBJ parser (with MTL colours, decimation), blueprint I/O
```

---

## Requirements

- Python 3.11 or later
- Webcam (built-in or USB)
- macOS, Linux, or Windows

All Python dependencies are installed automatically by `run.sh` / `run.bat`. Voice features require `faster-whisper` and `sounddevice`, which are included in `requirements.txt`.

---

## License

MIT — see [LICENSE](LICENSE)
