"""
Self-contained 2D wave engine for in-browser execution under Pyodide.

Browser bundle for static S3/CloudFront deploy. Canonical package code lives in
``waveform_2d.main`` and ``waveform_2d.live_sim`` — update both when changing
physics (see ``webapp/pyodide/README.md``).

Physics: explicit (leapfrog) FDTD (Langtangen & Linge). Rendering uses fixed-scale
256-entry LUTs for RippleGL-style consistent colouring.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core FDTD update (book eq. 117 interior / eq. 118 first step)
# ---------------------------------------------------------------------------


def advance(u, u_n, u_nm1, Cx2, Cy2, dt2, f_a, *, V=None, B=0.0, dt=0.0,
            step1=False, boundary="neumann"):
    Cx2 = np.asarray(Cx2, dtype=float)
    Cy2 = np.asarray(Cy2, dtype=float)

    if boundary == "neumann":
        p = np.pad(u_n, 1, mode="reflect")
        u_xx = p[:-2, 1:-1] - 2.0 * u_n + p[2:, 1:-1]
        u_yy = p[1:-1, :-2] - 2.0 * u_n + p[1:-1, 2:]
        sl = (slice(None), slice(None))
    else:
        c = u_n[1:-1, 1:-1]
        u_xx = u_n[:-2, 1:-1] - 2.0 * c + u_n[2:, 1:-1]
        u_yy = u_n[1:-1, :-2] - 2.0 * c + u_n[1:-1, 2:]
        sl = (slice(1, -1), slice(1, -1))

    def _crop(a):
        return a[sl] if np.ndim(a) else a

    Cx2s, Cy2s, Bs = _crop(Cx2), _crop(Cy2), _crop(B)
    f_s = _crop(f_a) if f_a is not None else 0.0
    stencil = Cx2s * u_xx + Cy2s * u_yy

    if step1:
        new = u_n[sl] + 0.5 * stencil + 0.5 * dt2 * f_s
        if V is not None:
            new = new + dt * _crop(V) * (1.0 - Bs)
    else:
        new = (2.0 * u_n[sl] - (1.0 - Bs) * u_nm1[sl] + stencil + dt2 * f_s) / (1.0 + Bs)

    u[:] = 0.0
    u[sl] = new
    return u


class WaveField:
    def __init__(self, height, width, *, c=1.0, dx=1.0, dy=1.0, dt=None,
                 b=0.0, boundary="neumann"):
        self.Ny, self.Nx = height, width
        self.dx, self.dy = dx, dy
        self.boundary = boundary
        c_arr = np.broadcast_to(np.asarray(c, dtype=float), (self.Ny, self.Nx)).copy()
        cmax = float(c_arr.max())
        cfl = 1.0 / (cmax * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
        self.dt = float(dt) if dt is not None else 0.9 * cfl
        self.c = c_arr
        self._set_courant()
        self.B = 0.5 * np.asarray(b, dtype=float) * self.dt if np.ndim(b) else 0.5 * b * self.dt
        self.u = np.zeros((self.Ny, self.Nx))
        self.u_n = np.zeros((self.Ny, self.Nx))
        self.u_nm1 = np.zeros((self.Ny, self.Nx))
        self.n = 0

    def _set_courant(self):
        self.Cx2 = (self.c * self.dt / self.dx) ** 2
        self.Cy2 = (self.c * self.dt / self.dy) ** 2
        self.dt2 = self.dt**2

    def set_velocity(self, c_arr, *, recalc_dt=True):
        """Update the velocity map and, by default, shrink/grow ``dt`` for CFL stability."""
        self.c[:] = c_arr
        if recalc_dt:
            cmax = float(self.c.max())
            cfl = 1.0 / (cmax * np.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
            self.dt = 0.9 * cfl
        self._set_courant()

    @property
    def field(self):
        return self.u_n

    def step(self, f_a=None):
        advance(self.u, self.u_n, self.u_nm1, self.Cx2, self.Cy2, self.dt2, f_a,
                B=self.B, dt=self.dt, step1=(self.n == 0), boundary=self.boundary)
        self.u_nm1, self.u_n, self.u = self.u_n, self.u, self.u_nm1
        self.n += 1


def apply_border_damping(field, border, b_max=0.32):
    Ny, Nx = field.Ny, field.Nx
    b = np.zeros((Ny, Nx))
    ramp = np.linspace(1.0, 0.0, border) ** 2
    for layer in range(border):
        val = b_max * ramp[layer]
        b[layer, :] = np.maximum(b[layer, :], val)
        b[Ny - 1 - layer, :] = np.maximum(b[Ny - 1 - layer, :], val)
        b[:, layer] = np.maximum(b[:, layer], val)
        b[:, Nx - 1 - layer] = np.maximum(b[:, Nx - 1 - layer], val)
    field.B = 0.5 * b * field.dt


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

LUT_SIZE = 256
C_WATER = 1.0
C_STEEL = 4.0


def _interp_lut(stops):
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float)
    grid = np.linspace(0.0, 1.0, LUT_SIZE)
    lut = np.empty((LUT_SIZE, 3))
    for ch in range(3):
        lut[:, ch] = np.interp(grid, xs, cols[:, ch])
    return np.clip(lut, 0, 255).astype(np.uint8)


_LUTS = {
    "ripple": _interp_lut([(0.00, (6, 10, 38)), (0.22, (18, 64, 140)),
                           (0.50, (14, 102, 150)), (0.74, (118, 198, 226)),
                           (1.00, (255, 255, 255))]),
    "rwb": _interp_lut([(0.00, (33, 80, 170)), (0.50, (245, 245, 245)),
                        (1.00, (200, 40, 40))]),
    "gray": _interp_lut([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]),
    "ocean": _interp_lut([(0.00, (2, 6, 20)), (0.35, (8, 60, 90)),
                          (0.5, (10, 90, 110)), (0.7, (40, 170, 160)),
                          (1.0, (220, 255, 230))]),
    "magma": _interp_lut([(0.0, (0, 0, 4)), (0.25, (80, 18, 90)),
                          (0.5, (182, 55, 90)), (0.75, (252, 137, 97)),
                          (1.0, (252, 253, 191))]),
}


def colormaps():
    return list(_LUTS.keys())


# ---------------------------------------------------------------------------
# Geometry tools (drawable structures)
# ---------------------------------------------------------------------------

# wall_*  -> reflecting Dirichlet barriers (u = 0)
# steel_* -> slower steel annulus / ring (heterogeneous velocity)
STRUCT_KINDS = (
    "wall_block", "wall_circle", "steel_ring", "steel_pipe",
)


def _ix(sim, xf):
    return int(np.clip(xf, 0.0, 1.0) * (sim.width - 1))


def _iy(sim, yf):
    return int(np.clip(yf, 0.0, 1.0) * (sim.height - 1))


def _radius_px(sim, rf):
    return max(2.0, rf * min(sim.width, sim.height) * 0.5)


def _disk_mask(sim, cx_f, cy_f, r_f):
    cx, cy = _ix(sim, cx_f), _iy(sim, cy_f)
    r = _radius_px(sim, r_f)
    yy, xx = np.ogrid[:sim.height, :sim.width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _annulus_mask(sim, cx_f, cy_f, r_outer_f, r_inner_f):
    outer = _disk_mask(sim, cx_f, cy_f, r_outer_f)
    inner = _disk_mask(sim, cx_f, cy_f, r_inner_f)
    return outer & ~inner


def _rect_mask(sim, x0_f, y0_f, x1_f, y1_f):
    x0, x1 = sorted((_ix(sim, x0_f), _ix(sim, x1_f)))
    y0, y1 = sorted((_iy(sim, y0_f), _iy(sim, y1_f)))
    m = np.zeros((sim.height, sim.width), dtype=bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def _struct_contains(sim, s, x_f, y_f):
    kind = s["kind"]
    if kind == "wall_block":
        x0, x1 = sorted((s["x0"], s["x1"]))
        y0, y1 = sorted((s["y0"], s["y1"]))
        return x0 <= x_f <= x1 and y0 <= y_f <= y1
    if kind in ("wall_circle", "steel_ring", "steel_pipe"):
        cx, cy = s["cx"], s["cy"]
        r = s.get("r", s.get("r_outer", 0.1))
        dx = (x_f - cx) * sim.width
        dy = (y_f - cy) * sim.height
        return dx * dx + dy * dy <= (_radius_px(sim, r) ** 2)
    return False


SCENES = ("tank", "double_slit", "pipe", "blank")


class LiveSimulation:
    def __init__(self, width=180, height=180):
        self.width = int(width)
        self.height = int(height)
        self.lut_name = "ripple"
        self.lut = _LUTS[self.lut_name]
        self.scale = 0.4
        self.frequency = 1.0
        self.absorbing = True
        self.paused = False
        self.base_period = 26.0
        self.oscillators = []
        self.structures = []
        self.wall_mask = None
        self.media_mask = None
        self._preset_wall = None   # scene-fixed walls (e.g. double-slit barrier)
        self.field = None
        self.scene = "tank"
        self._rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        self._rgba[..., 3] = 255
        self.set_scene("tank")

    def _new_field(self, c=1.0):
        f = WaveField(self.height, self.width, c=c, boundary="neumann")
        if self.absorbing:
            apply_border_damping(f, border=24, b_max=0.32)
        return f

    def _rebuild_geometry(self):
        """Bake ``self.structures`` into velocity map and wall mask."""
        H, W = self.height, self.width
        c = np.full((H, W), C_WATER)
        wall = np.zeros((H, W), dtype=bool)
        steel = np.zeros((H, W), dtype=bool)

        for s in self.structures:
            kind = s["kind"]
            if kind == "wall_block":
                wall |= _rect_mask(self, s["x0"], s["y0"], s["x1"], s["y1"])
            elif kind == "wall_circle":
                wall |= _disk_mask(self, s["cx"], s["cy"], s["r"])
            elif kind in ("steel_ring", "steel_pipe"):
                steel |= _annulus_mask(self, s["cx"], s["cy"], s["r_outer"], s["r_inner"])
            elif kind == "hollow_circle":  # alias: reflecting outer + steel ring
                steel |= _annulus_mask(self, s["cx"], s["cy"], s["r_outer"], s["r_inner"])

        c[steel] = C_STEEL

        if self._preset_wall is not None:
            wall |= self._preset_wall

        if self.field is None:
            self.field = self._new_field(c=c)
        else:
            self.field.set_velocity(c, recalc_dt=True)
            if self.absorbing:
                apply_border_damping(self.field, border=24, b_max=0.32)

        self.wall_mask = wall if wall.any() else None
        self.media_mask = steel if steel.any() else None
        self._enforce_walls()

    def _enforce_walls(self):
        """Clamp all displacement levels to zero on reflecting wall cells."""
        if self.wall_mask is None or self.field is None:
            return
        self.field.u[self.wall_mask] = 0.0
        self.field.u_n[self.wall_mask] = 0.0
        self.field.u_nm1[self.wall_mask] = 0.0

    def _is_wall(self, iy, ix):
        return self.wall_mask is not None and self.wall_mask[iy, ix]

    def set_scene(self, name):
        if name not in SCENES:
            name = "tank"
        self.scene = name
        self.oscillators = []
        self.structures = []
        self.wall_mask = None
        self.media_mask = None
        self._preset_wall = None
        H, W = self.height, self.width

        if name == "tank":
            self.field = self._new_field(c=C_WATER)
            self.add_oscillator(0.5, 0.32)

        elif name == "blank":
            self.field = self._new_field(c=C_WATER)

        elif name == "double_slit":
            self.field = self._new_field(c=C_WATER)
            wall = np.zeros((H, W), dtype=bool)
            bx = int(W * 0.42)
            wall[:, bx:bx + 3] = True
            gap = max(4, int(H * 0.03))
            c1, c2 = int(H * 0.40), int(H * 0.60)
            wall[c1 - gap:c1 + gap, bx:bx + 3] = False
            wall[c2 - gap:c2 + gap, bx:bx + 3] = False
            self._preset_wall = wall
            self.wall_mask = wall
            self._enforce_walls()
            self.frequency = 1.3
            self.add_oscillator(0.22, 0.5)

        elif name == "pipe":
            # Hollow circular pipe cross-section: water bore + steel annulus.
            self.field = self._new_field(c=C_WATER)
            self.structures = [{
                "kind": "steel_pipe",
                "cx": 0.5, "cy": 0.5,
                "r_outer": 0.72,   # outer radius (fraction of half-grid)
                "r_inner": 0.55,   # inner bore radius -> wall thickness ~17% of half-grid
            }]
            self._rebuild_geometry()
            self.frequency = 1.1
            self.add_oscillator(0.5, 0.5, aperture=max(2, int(H * 0.04)))

    # -- drawable structures --
    def add_wall_block(self, x0, y0, x1, y1):
        self.structures.append({
            "kind": "wall_block",
            "x0": float(min(x0, x1)), "y0": float(min(y0, y1)),
            "x1": float(max(x0, x1)), "y1": float(max(y0, y1)),
        })
        self._rebuild_geometry()

    def add_wall_circle(self, cx, cy, r):
        self.structures.append({"kind": "wall_circle", "cx": float(cx), "cy": float(cy), "r": float(r)})
        self._rebuild_geometry()

    def add_steel_ring(self, cx, cy, r_outer, r_inner):
        """Hollow circle / pipe wall: water inside and outside, steel in the ring."""
        ro, ri = float(r_outer), float(r_inner)
        if ri >= ro:
            ri = ro * 0.75
        self.structures.append({
            "kind": "steel_ring", "cx": float(cx), "cy": float(cy),
            "r_outer": ro, "r_inner": ri,
        })
        self._rebuild_geometry()

    def add_steel_pipe(self, cx, cy, r_outer, wall_frac=0.18):
        """Shortcut: pipe with wall thickness as fraction of outer radius."""
        ro = float(r_outer)
        ri = ro * (1.0 - float(wall_frac))
        self.add_steel_ring(cx, cy, ro, ri)

    def clear_structures(self):
        self.structures = []
        self._rebuild_geometry()

    def erase_at(self, x_frac, y_frac):
        """Remove the topmost structure under (x_frac, y_frac)."""
        for i in range(len(self.structures) - 1, -1, -1):
            if _struct_contains(self, self.structures[i], x_frac, y_frac):
                self.structures.pop(i)
                self._rebuild_geometry()
                return True
        return False

    # -- sources --
    def add_oscillator(self, x_frac, y_frac, aperture=0, amp=1.0):
        ix = _ix(self, x_frac)
        iy = _iy(self, y_frac)
        if self._is_wall(iy, ix) and aperture == 0:
            return
        self.oscillators.append({"ix": ix, "iy": iy, "amp": amp, "aperture": aperture})

    def add_drip(self, x_frac, y_frac, amp=1.5):
        if self.field is None:
            return
        ix, iy = _ix(self, x_frac), _iy(self, y_frac)
        if self._is_wall(iy, ix):
            return
        r = 3
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        bump = amp * np.exp(-(xx**2 + yy**2) / 2.0)
        y0, y1 = max(0, iy - r), min(self.height, iy + r + 1)
        x0, x1 = max(0, ix - r), min(self.width, ix + r + 1)
        sub = bump[(y0 - (iy - r)):(y0 - (iy - r)) + (y1 - y0),
                   (x0 - (ix - r)):(x0 - (ix - r)) + (x1 - x0)].copy()
        if self.wall_mask is not None:
            sub[self.wall_mask[y0:y1, x0:x1]] = 0.0
        self.field.u_n[y0:y1, x0:x1] += sub
        self.field.u_nm1[y0:y1, x0:x1] += sub

    def clear_sources(self):
        self.oscillators = []

    def set_param(self, key, value):
        if key == "frequency":
            self.frequency = float(value)
        elif key == "brightness":
            self.scale = float(np.clip(value, 0.02, 2.0))
        elif key == "colormap":
            self.lut_name = value if value in _LUTS else "ripple"
            self.lut = _LUTS[self.lut_name]
        elif key == "absorbing":
            self.absorbing = bool(value)
            self.set_scene(self.scene)
        elif key == "paused":
            self.paused = bool(value)

    def reset(self):
        self.set_scene(self.scene)

    def step(self, n_steps=1):
        if self.field is None or self.paused:
            return
        f = self.field
        period = max(4.0, self.base_period / max(0.05, self.frequency))
        omega = 2.0 * np.pi / period
        for _ in range(int(n_steps)):
            f.step()
            t = f.n
            for osc in self.oscillators:
                val = osc["amp"] * np.sin(omega * t)
                ap = osc["aperture"]
                if ap > 0:
                    y0, y1 = max(0, osc["iy"] - ap), min(self.height, osc["iy"] + ap + 1)
                    if self.wall_mask is not None:
                        strip = f.u_n[y0:y1, osc["ix"]].copy()
                        mask = self.wall_mask[y0:y1, osc["ix"]]
                        strip[~mask] = val
                        f.u_n[y0:y1, osc["ix"]] = strip
                    else:
                        f.u_n[y0:y1, osc["ix"]] = val
                elif not self._is_wall(osc["iy"], osc["ix"]):
                    f.u_n[osc["iy"], osc["ix"]] = val
            self._enforce_walls()

    def render(self):
        scale = self.scale if self.scale > 1e-9 else 1e-9
        raw = self.field.field / scale
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        x = np.clip(raw, -1.0, 1.0)
        idx = ((x + 1.0) * 0.5 * (LUT_SIZE - 1)).astype(np.int32)
        rgb = self.lut[idx].astype(np.float32)
        if self.media_mask is not None:
            tint = np.array([150, 150, 165], dtype=np.float32)
            m = self.media_mask[..., None]
            rgb = np.where(m, 0.70 * rgb + 0.30 * tint, rgb)
        if self.wall_mask is not None:
            rgb[self.wall_mask] = np.array([70, 66, 52], dtype=np.float32)
        self._rgba[..., :3] = rgb.astype(np.uint8)
        return self._rgba.tobytes()
