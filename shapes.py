"""
3D Shape definitions for Project Vertex v4.

Primitives (4): cube, sphere, torus, icosahedron
Heroes    (3): arc_reactor, dna_helix, geodesic

All draw_* functions respect self.wireframe unless wireframe= is passed explicitly.
Animated shapes (arc_reactor, dna_helix) accept a t= time parameter.
"""
from OpenGL.GL import *
import math


class ShapeRenderer:
    def __init__(self, wireframe: bool = False):
        self.wireframe = wireframe

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    @staticmethod
    def _normalize(v):
        l = math.sqrt(v[0]**2+v[1]**2+v[2]**2)
        return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-10 else (0.0, 1.0, 0.0)

    @classmethod
    def _face_normal(cls, verts, idx):
        v0, v1, v2 = verts[idx[0]], verts[idx[1]], verts[idx[2]]
        a = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
        b = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
        return cls._normalize(cls._cross(a, b))

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def draw_cube(self, size=1.0, wireframe=None):
        wf = self.wireframe if wireframe is None else wireframe
        s  = size
        verts = [
            ( s,-s,-s),( s, s,-s),(-s, s,-s),(-s,-s,-s),
            ( s,-s, s),( s, s, s),(-s,-s, s),(-s, s, s),
        ]
        faces = [
            ((0,1,2,3),(0,0,-1)),((4,7,6,5),(0,0,1)),
            ((0,4,5,1),(1,0,0)), ((2,6,7,3),(-1,0,0)),
            ((0,3,7,4),(0,-1,0)),((1,5,6,2),(0,1,0)),
        ]
        edges = [(0,1),(0,3),(0,4),(2,1),(2,3),(2,7),(6,3),(6,4),(6,7),(5,1),(5,4),(5,7)]
        if wf:
            glBegin(GL_LINES)
            for a,b in edges:
                glVertex3fv(verts[a]); glVertex3fv(verts[b])
            glEnd()
        else:
            glBegin(GL_QUADS)
            for idxs, n in faces:
                glNormal3fv(n)
                for v in idxs: glVertex3fv(verts[v])
            glEnd()

    def draw_sphere(self, radius=1.0, slices=20, stacks=20, wireframe=None):
        wf = self.wireframe if wireframe is None else wireframe
        if wf:
            glBegin(GL_LINES)
            for i in range(slices):
                a1 = 2*math.pi*i/slices; a2 = 2*math.pi*(i+1)/slices
                for j in range(stacks):
                    p1 = math.pi*j/stacks; p2 = math.pi*(j+1)/stacks
                    x1=radius*math.sin(p1)*math.cos(a1); y1=radius*math.cos(p1); z1=radius*math.sin(p1)*math.sin(a1)
                    x2=radius*math.sin(p2)*math.cos(a1); y2=radius*math.cos(p2); z2=radius*math.sin(p2)*math.sin(a1)
                    x3=radius*math.sin(p1)*math.cos(a2); y3=radius*math.cos(p1); z3=radius*math.sin(p1)*math.sin(a2)
                    glVertex3f(x1,y1,z1); glVertex3f(x2,y2,z2)
                    glVertex3f(x1,y1,z1); glVertex3f(x3,y3,z3)
            glEnd()
        else:
            for i in range(stacks):
                glBegin(GL_QUAD_STRIP)
                for j in range(slices+1):
                    for k in range(2):
                        phi=math.pi*(i+k)/stacks; theta=2*math.pi*j/slices
                        x=math.sin(phi)*math.cos(theta); y=math.cos(phi); z=math.sin(phi)*math.sin(theta)
                        glNormal3f(x,y,z); glVertex3f(radius*x,radius*y,radius*z)
                glEnd()

    def draw_torus(self, inner_radius=0.3, outer_radius=1.0, segments=24, rings=24, wireframe=None):
        wf = self.wireframe if wireframe is None else wireframe
        ir, or_ = inner_radius, outer_radius
        if wf:
            glBegin(GL_LINES)
            for i in range(rings):
                for j in range(segments):
                    def _tp(u, v):
                        r = or_ + ir*math.cos(v)
                        return (r*math.cos(u), ir*math.sin(v), r*math.sin(u))
                    u0=2*math.pi*i/rings;   u1=2*math.pi*(i+1)/rings
                    v0=2*math.pi*j/segments;v1=2*math.pi*(j+1)/segments
                    glVertex3fv(_tp(u0,v0)); glVertex3fv(_tp(u1,v0))
                    glVertex3fv(_tp(u0,v0)); glVertex3fv(_tp(u0,v1))
            glEnd()
        else:
            for i in range(rings):
                glBegin(GL_QUAD_STRIP)
                for j in range(segments+1):
                    for k in range(2):
                        u=2*math.pi*(i+k)/rings; v=2*math.pi*j/segments
                        cv=math.cos(v); sv=math.sin(v); r=or_+ir*cv
                        glNormal3f(cv*math.cos(u), sv, cv*math.sin(u))
                        glVertex3f(r*math.cos(u), ir*sv, r*math.sin(u))
                glEnd()

    def draw_icosahedron(self, size=1.0, wireframe=None):
        wf  = self.wireframe if wireframe is None else wireframe
        phi = (1+math.sqrt(5))/2
        raw = [
            (-1,phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),
            (0,-1,phi),(0,1,phi),(0,-1,-phi),(0,1,-phi),
            (phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1),
        ]
        sc = size / math.sqrt(1+phi**2)
        verts = [(x*sc,y*sc,z*sc) for x,y,z in raw]
        faces = [
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
        ]
        edges = set()
        for f in faces:
            for i in range(3): edges.add(tuple(sorted((f[i],f[(i+1)%3]))))
        if wf:
            glBegin(GL_LINES)
            for a,b in edges: glVertex3fv(verts[a]); glVertex3fv(verts[b])
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for f in faces:
                glNormal3fv(self._face_normal(verts, f))
                for v in f: glVertex3fv(verts[v])
            glEnd()

    # ------------------------------------------------------------------
    # Hero shapes
    # ------------------------------------------------------------------

    def draw_arc_reactor(self, size=1.0, t=0.0, wireframe=None):
        """Arc Reactor: 3 counter-rotating rings + pulsing core + 6 radial coils."""
        wf = self.wireframe if wireframe is None else wireframe

        # --- Outer ring (stationary, slight forward tilt) ---
        glPushMatrix()
        glRotatef(12, 1, 0, 0)
        self.draw_torus(inner_radius=size*0.048, outer_radius=size*0.92,
                        segments=36, rings=20, wireframe=wf)
        glPopMatrix()

        # --- Middle ring (45° tilt, slow rotation) ---
        glPushMatrix()
        glRotatef(45, 1, 0, 0)
        glRotatef(math.degrees(t * 0.45), 0, 0, 1)
        self.draw_torus(inner_radius=size*0.038, outer_radius=size*0.62,
                        segments=28, rings=16, wireframe=wf)
        glPopMatrix()

        # --- Inner ring (vertical, counter-rotation) ---
        glPushMatrix()
        glRotatef(90, 1, 0, 0)
        glRotatef(math.degrees(-t * 0.72), 0, 0, 1)
        self.draw_torus(inner_radius=size*0.028, outer_radius=size*0.33,
                        segments=22, rings=12, wireframe=wf)
        glPopMatrix()

        # --- 6 radial coil spokes ---
        glBegin(GL_LINES)
        for i in range(6):
            angle = math.radians(i * 60 + math.degrees(t * 0.15))
            x1 = math.cos(angle)*size*0.14; z1 = math.sin(angle)*size*0.14
            x2 = math.cos(angle)*size*0.56; z2 = math.sin(angle)*size*0.56
            glVertex3f(x1, 0, z1); glVertex3f(x2, 0, z2)
        glEnd()

        # --- Pulsing core ---
        pulse = 0.5 + 0.5 * math.sin(t * 3.5)
        core_r = size * (0.10 + 0.025 * pulse)
        if wf:
            glBegin(GL_LINE_LOOP)
            for i in range(20):
                a = 2*math.pi*i/20
                glVertex3f(core_r*math.cos(a), 0, core_r*math.sin(a))
            glEnd()
        else:
            glBegin(GL_TRIANGLE_FAN)
            glNormal3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            for i in range(21):
                a = 2*math.pi*i/20
                glVertex3f(core_r*math.cos(a), 0, core_r*math.sin(a))
            glEnd()

        # --- Center emitter sphere ---
        self.draw_sphere(radius=size*0.065, slices=10, stacks=10, wireframe=wf)

    def draw_dna_helix(self, size=1.0, t=0.0, wireframe=None):
        """Animated double helix with rungs."""
        turns      = 3
        height     = size * 2.4
        strand_r   = size * 0.48
        n_pts      = 80
        n_rungs    = turns * 5
        rung_step  = n_pts // n_rungs

        strand1, strand2 = [], []
        spin = t * 0.6  # animation speed

        for i in range(n_pts + 1):
            frac  = i / n_pts
            y     = height * (frac - 0.5)
            theta = 2*math.pi * turns * frac + spin
            x1    = strand_r * math.cos(theta)
            z1    = strand_r * math.sin(theta)
            x2    = strand_r * math.cos(theta + math.pi)
            z2    = strand_r * math.sin(theta + math.pi)
            strand1.append((x1, y, z1))
            strand2.append((x2, y, z2))

        # Draw strand 1
        glBegin(GL_LINE_STRIP)
        for p in strand1: glVertex3fv(p)
        glEnd()

        # Draw strand 2
        glBegin(GL_LINE_STRIP)
        for p in strand2: glVertex3fv(p)
        glEnd()

        # Rungs
        for i in range(0, n_pts + 1, rung_step):
            if i < len(strand1):
                glBegin(GL_LINES)
                glVertex3fv(strand1[i]); glVertex3fv(strand2[i])
                glEnd()

        # Node dots at rung attachment points (both strands)
        if not wireframe:
            for i in range(0, n_pts + 1, rung_step):
                if i < len(strand1):
                    for p in (strand1[i], strand2[i]):
                        self.draw_sphere(radius=size*0.05, slices=6, stacks=6, wireframe=False)

    def draw_geodesic(self, size=1.0, t=0.0, wireframe=None):
        """Frequency-2 geodesic sphere (subdivided icosahedron projected to sphere)."""
        wf = self.wireframe if wireframe is None else wireframe

        phi = (1+math.sqrt(5))/2
        raw_ico = [
            (-1,phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),
            (0,-1,phi),(0,1,phi),(0,-1,-phi),(0,1,-phi),
            (phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1),
        ]
        def sph(x,y,z):
            l = math.sqrt(x*x+y*y+z*z)
            return (x/l*size, y/l*size, z/l*size)

        verts = [sph(*v) for v in raw_ico]
        faces_ico = [
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
        ]

        # Subdivide each triangle once and project midpoints to sphere
        mid_cache: dict[tuple, int] = {}

        def midpoint(a, b):
            key = (min(a,b), max(a,b))
            if key not in mid_cache:
                ax,ay,az = verts[a]; bx,by,bz = verts[b]
                mid_cache[key] = len(verts)
                verts.append(sph((ax+bx)/2, (ay+by)/2, (az+bz)/2))
            return mid_cache[key]

        sub_faces = []
        for f in faces_ico:
            a, b, c = f
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            sub_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]

        edges = set()
        for f in sub_faces:
            for i in range(3):
                edges.add(tuple(sorted((f[i], f[(i+1)%3]))))

        if wf:
            glBegin(GL_LINES)
            for a, b in edges:
                glVertex3fv(verts[a]); glVertex3fv(verts[b])
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for f in sub_faces:
                cx = (verts[f[0]][0]+verts[f[1]][0]+verts[f[2]][0])/3
                cy = (verts[f[0]][1]+verts[f[1]][1]+verts[f[2]][1])/3
                cz = (verts[f[0]][2]+verts[f[1]][2]+verts[f[2]][2])/3
                l  = math.sqrt(cx*cx+cy*cy+cz*cz)
                if l > 0: glNormal3f(cx/l,cy/l,cz/l)
                for vi in f: glVertex3fv(verts[vi])
            glEnd()

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    SHAPE_NAMES = [
        "arc_reactor", "dna_helix", "geodesic",
        "cube", "sphere", "torus", "icosahedron",
    ]

    # Keyboard shortcuts (1-7)
    SHAPE_KEYS = {str(i+1): name for i, name in enumerate(SHAPE_NAMES)}

    def compile_mesh_lists(self, verts: list, faces: list,
                           edges: list, face_colours: list) -> tuple[int, int]:
        """
        Compile solid + wireframe display lists from pre-loaded mesh data.
        Must be called from the main GL thread (inside _flush_obj_queue).
        Returns (solid_id, wire_id) — integer GL list names.
        """
        n = len(verts)

        solid_id = int(glGenLists(1))
        glNewList(solid_id, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for fi, tri in enumerate(faces):
            if all(i < n for i in tri):
                col = face_colours[fi] if face_colours and fi < len(face_colours) else None
                if col is not None:
                    glColor3f(*col)
                glNormal3fv(self._face_normal(verts, tri))
                for i in tri:
                    glVertex3fv(verts[i])
        glEnd()
        glEndList()

        wire_id = int(glGenLists(1))
        glNewList(wire_id, GL_COMPILE)
        glBegin(GL_LINES)
        for a, b in edges:
            if a < n and b < n:
                glVertex3fv(verts[a])
                glVertex3fv(verts[b])
        glEnd()
        glEndList()

        return solid_id, wire_id

    def draw_custom_mesh(self, verts: list, faces: list,
                         edges: list | None = None,
                         face_colours: list | None = None,
                         wireframe=None,
                         solid_list: int = 0,
                         wire_list:  int = 0):
        """
        Render an OBJ-loaded mesh.
        Fast path: glCallList when list IDs are compiled (1 GL call).
        Fallback:  immediate mode (used when no GL context exists, e.g. tests).
        """
        wf = self.wireframe if wireframe is None else wireframe

        # Fast path — single GL call
        list_id = wire_list if wf else solid_list
        if list_id:
            glCallList(list_id)
            return

        # Immediate-mode fallback (no compiled list yet)
        n = len(verts)
        if wf:
            if edges is None:
                edge_set: set = set()
                for tri in faces:
                    for i in range(3):
                        edge_set.add(tuple(sorted((tri[i], tri[(i+1) % 3]))))
                edges = list(edge_set)
            glBegin(GL_LINES)
            for a, b in edges:
                if a < n and b < n:
                    glVertex3fv(verts[a])
                    glVertex3fv(verts[b])
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for fi, tri in enumerate(faces):
                if all(i < n for i in tri):
                    col = face_colours[fi] if face_colours and fi < len(face_colours) else None
                    if col is not None:
                        glColor3f(*col)
                    glNormal3fv(self._face_normal(verts, tri))
                    for i in tri:
                        glVertex3fv(verts[i])
            glEnd()

    def draw_shape(self, shape_type: str, size: float = 1.0, t: float = 0.0,
                   wireframe=None, custom_mesh=None,
                   solid_list: int = 0, wire_list: int = 0):
        """Draw a shape by name. wireframe= overrides self.wireframe if set.
        custom_mesh may be (verts, faces) or (verts, faces, edges, face_colours).
        solid_list / wire_list are pre-compiled GL display list IDs."""
        orig_wf = self.wireframe
        if wireframe is not None:
            self.wireframe = wireframe
        try:
            if custom_mesh is not None:
                verts   = custom_mesh[0]
                faces   = custom_mesh[1]
                edges   = custom_mesh[2] if len(custom_mesh) > 2 else None
                colours = custom_mesh[3] if len(custom_mesh) > 3 else None
                self.draw_custom_mesh(verts, faces, edges, colours,
                                      solid_list=solid_list, wire_list=wire_list)
                return
            {
                "cube":        lambda: self.draw_cube(size),
                "sphere":      lambda: self.draw_sphere(size),
                "torus":       lambda: self.draw_torus(size*0.3, size),
                "icosahedron": lambda: self.draw_icosahedron(size),
                "arc_reactor": lambda: self.draw_arc_reactor(size, t),
                "dna_helix":   lambda: self.draw_dna_helix(size, t),
                "geodesic":    lambda: self.draw_geodesic(size, t),
            }.get(shape_type, lambda: None)()  # unknown names render nothing, not a cube
        finally:
            self.wireframe = orig_wf

    def get_shape_info(self, shape_type: str) -> tuple[int, int]:
        """Approximate (vertices, faces) counts for the HUD."""
        return {
            "cube":        (8,   6),
            "sphere":      (441, 400),
            "torus":       (576, 552),
            "icosahedron": (12,  20),
            "arc_reactor": (500, 480),
            "dna_helix":   (180, 160),
            "geodesic":    (42,  80),
        }.get(shape_type, (0, 0))
