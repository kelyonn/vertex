"""
Utility functions for Project Vertex v4.
Blueprint save/load, screenshot, OBJ export, OBJ import, math helpers.
"""
import json
import math
import os
from datetime import datetime
import pygame


# ---------------------------------------------------------------------------
# Blueprint (save / load scene state)
# ---------------------------------------------------------------------------

def save_blueprint(scene, filename="blueprint.json"):
    """Save the full scene state (holograms + strokes) to JSON."""
    data = {
        "project_name": "VERTEX_V4",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **scene.save_state(),
    }
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f">> SAVED: {filename}")
        return True
    except Exception as e:
        print(f">> SAVE ERROR: {e}")
        return False


def load_blueprint(scene, filename="blueprint.json"):
    """Load scene state from JSON into the existing scene object."""
    if not os.path.exists(filename):
        print(f">> No blueprint at: {filename}")
        return False
    try:
        with open(filename) as f:
            data = json.load(f)
        scene.load_state(data)
        ts = data.get("timestamp", "unknown")
        print(f">> LOADED: {filename} (saved {ts})")
        return True
    except Exception as e:
        print(f">> LOAD ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def screenshot(directory=".") -> str:
    """Save the current OpenGL framebuffer as a timestamped PNG."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"vertex_{ts}.png")
    try:
        from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
        w, h = pygame.display.get_surface().get_size()
        raw  = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        surf = pygame.image.fromstring(raw, (w, h), "RGB")
        surf = pygame.transform.flip(surf, False, True)
        pygame.image.save(surf, path)
        print(f">> SCREENSHOT: {path}")
    except Exception as e:
        print(f">> SCREENSHOT ERROR: {e}")
        return ""
    return path


# ---------------------------------------------------------------------------
# OBJ Import
# ---------------------------------------------------------------------------

# Maximum faces to render at full detail. Larger meshes are decimated.
_MAX_FACES = 80_000


def _parse_mtl(mtl_path: str) -> dict[str, tuple[float, float, float]]:
    """Parse a .mtl file and return {material_name: (r, g, b)} diffuse colours."""
    colours: dict[str, tuple[float, float, float]] = {}
    current = None
    try:
        with open(mtl_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "newmtl" and len(parts) > 1:
                    current = parts[1]
                elif parts[0] == "Kd" and current and len(parts) >= 4:
                    try:
                        colours[current] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    except ValueError:
                        pass
    except OSError:
        pass
    return colours


def parse_obj_content(text: str, base_dir: str = "") -> tuple[list, list, list]:
    """
    Pure-Python Wavefront OBJ parser.
    Returns (vertices, faces, face_colours) where:
      vertices:     list of (x, y, z) floats
      faces:        list of (i, j, k) int tuples (0-based, triangulated)
      face_colours: list of (r, g, b) per face, or [] if no materials found
    Handles n-gons via fan triangulation.
    """
    verts: list[tuple] = []
    faces: list[tuple] = []
    face_colours: list[tuple] = []

    # MTL support
    mtl_colours: dict[str, tuple] = {}
    current_colour: tuple | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tok = parts[0]

        if tok == "v" and len(parts) >= 4:
            try:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                pass

        elif tok == "mtllib" and len(parts) > 1 and base_dir:
            mtl_path = os.path.join(base_dir, parts[1])
            mtl_colours = _parse_mtl(mtl_path)

        elif tok == "usemtl" and len(parts) > 1:
            current_colour = mtl_colours.get(parts[1])

        elif tok == "f" and len(parts) >= 4:
            raw_idx = [p.split("/")[0] for p in parts[1:]]
            try:
                idx = [int(i) - 1 for i in raw_idx]
                # Resolve negative (relative) indices
                n = len(verts)
                idx = [i if i >= 0 else n + i + 1 for i in idx]
                # Fan-triangulate n-gons
                new_tris = len(idx) - 2
                for j in range(new_tris):
                    faces.append((idx[0], idx[j + 1], idx[j + 2]))
                    face_colours.append(current_colour)
            except (ValueError, IndexError):
                pass

    return verts, faces, face_colours


def _decimate(faces: list, face_colours: list, max_faces: int) -> tuple[list, list]:
    """Uniformly subsample faces to stay under max_faces."""
    n = len(faces)
    if n <= max_faces:
        return faces, face_colours
    step = n / max_faces
    kept_f, kept_c = [], []
    i = 0.0
    while int(i) < n and len(kept_f) < max_faces:
        idx = int(i)
        kept_f.append(faces[idx])
        kept_c.append(face_colours[idx] if face_colours else None)
        i += step
    print(f">> OBJ: decimated {n:,} → {len(kept_f):,} faces for real-time rendering")
    return kept_f, kept_c


def load_obj(path: str) -> tuple | None:
    """
    Load a .obj file. Returns (verts, faces, edges, face_colours) or None.
      verts:        list of (x,y,z) normalised to unit cube
      faces:        list of (i,j,k) 0-based triangles
      edges:        frozenset of sorted (a,b) pairs — pre-computed for wireframe
      face_colours: list of (r,g,b) per face, or [] if no materials
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f">> OBJ LOAD ERROR {path}: {e}")
        return None

    base_dir = os.path.dirname(os.path.abspath(path))
    verts, faces, face_colours = parse_obj_content(text, base_dir=base_dir)

    if not verts or not faces:
        print(f">> OBJ: no geometry in {os.path.basename(path)}")
        return None

    # Decimate large meshes
    faces, face_colours = _decimate(faces, face_colours, _MAX_FACES)

    # Normalise to unit cube centred at origin
    xs, ys, zs = zip(*verts)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) or 1.0
    sc = 2.0 / span
    verts = [((x-cx)*sc, (y-cy)*sc, (z-cz)*sc) for x, y, z in verts]

    # Pre-compute unique edges once — never recomputed at render time
    edges: list[tuple] = list({
        tuple(sorted((tri[i], tri[(i+1) % 3])))
        for tri in faces
        for i in range(3)
    })

    has_colours = any(c is not None for c in face_colours)
    colours_out = face_colours if has_colours else []

    print(f">> OBJ loaded: {len(verts):,} verts  {len(faces):,} faces  "
          f"{len(edges):,} edges  colours={'yes' if has_colours else 'no'}")
    return verts, faces, edges, colours_out


# ---------------------------------------------------------------------------
# OBJ Export (procedural shapes)
# ---------------------------------------------------------------------------

def _build_sphere(r=1.0, slices=24, stacks=24):
    verts = []
    for j in range(stacks+1):
        phi = math.pi*j/stacks
        for i in range(slices):
            th = 2*math.pi*i/slices
            verts.append((r*math.sin(phi)*math.cos(th),
                           r*math.cos(phi),
                           r*math.sin(phi)*math.sin(th)))
    tris = []
    for j in range(stacks):
        for i in range(slices):
            a = j*slices+i+1; b = j*slices+(i+1)%slices+1
            c = (j+1)*slices+i+1; d = (j+1)*slices+(i+1)%slices+1
            tris += [(a,b,d),(a,d,c)]
    return verts, tris


def _build_cube(s=1.0):
    v = [( s,-s,-s),( s, s,-s),(-s, s,-s),(-s,-s,-s),
         ( s,-s, s),( s, s, s),(-s,-s, s),(-s, s, s)]
    f = [(1,2,3,4),(5,8,7,6),(1,5,6,2),(3,7,8,4),(1,4,8,5),(2,6,7,3)]
    t = []
    for a,b,c,d in f:
        t += [(a,b,c),(a,c,d)]
    return v, t


def export_obj(shape_type: str, size: float = 1.0, filename: str = "export.obj"):
    builder = {"sphere": _build_sphere, "cube": _build_cube}.get(shape_type)
    if builder is None:
        print(f">> OBJ export: shape '{shape_type}' not exportable, using cube")
        builder = _build_cube
    try:
        verts, tris = builder(size)
        with open(filename, "w") as f:
            f.write(f"# Project Vertex v4 Export\n# Shape: {shape_type}\n")
            f.write(f"# Date: {datetime.now().isoformat(timespec='seconds')}\no {shape_type}\n\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for t in tris:
                f.write(f"f {t[0]} {t[1]} {t[2]}\n")
        print(f">> EXPORTED: {filename} ({len(verts)} verts, {len(tris)} faces)")
        return True
    except Exception as e:
        print(f">> EXPORT ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lerp(a, b, t):
    return a + (b - a) * t
