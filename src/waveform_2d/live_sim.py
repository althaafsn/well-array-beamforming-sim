"""
Interactive live simulation: scenes, drawable geometry, sources, and rendering.

Used by the FastAPI WebSocket server (``webapp.server``) and mirrored in the
browser-only Pyodide bundle (``webapp/pyodide/wave_engine.py``).
"""

from __future__ import annotations

import numpy as np

from .main import WaveField, apply_border_damping
from .webapp.colormap import LUT_SIZE, Renderer

C_WATER = 1.0
C_STEEL = 4.0

SCENES = ("tank", "double_slit", "pipe", "blank")
STRUCT_KINDS = ("wall_block", "wall_circle", "steel_ring", "steel_pipe")


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
    return _disk_mask(sim, cx_f, cy_f, r_outer_f) & ~_disk_mask(sim, cx_f, cy_f, r_inner_f)


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


class LiveSimulation:
    """Ripple-tank style driver with drawable structures and fixed-scale rendering."""

    def __init__(self, width: int = 180, height: int = 180) -> None:
        self.width = int(width)
        self.height = int(height)
        self.renderer = Renderer("ripple", scale=0.4)
        self.frequency = 1.0
        self.absorbing = True
        self.paused = False
        self.base_period = 26.0
        self.oscillators: list[dict] = []
        self.structures: list[dict] = []
        self.wall_mask = None
        self.media_mask = None
        self._preset_wall = None
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

    def _rebuild_geometry(self) -> None:
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
            elif kind in ("steel_ring", "steel_pipe", "hollow_circle"):
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

    def _enforce_walls(self) -> None:
        if self.wall_mask is None or self.field is None:
            return
        self.field.u[self.wall_mask] = 0.0
        self.field.u_n[self.wall_mask] = 0.0
        self.field.u_nm1[self.wall_mask] = 0.0

    def _is_wall(self, iy: int, ix: int) -> bool:
        return self.wall_mask is not None and bool(self.wall_mask[iy, ix])

    def set_scene(self, name: str) -> None:
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
            self.field = self._new_field(c=C_WATER)
            self.structures = [{
                "kind": "steel_pipe", "cx": 0.5, "cy": 0.5,
                "r_outer": 0.72, "r_inner": 0.55,
            }]
            self._rebuild_geometry()
            self.frequency = 1.1
            self.add_oscillator(0.5, 0.5, aperture=max(2, int(H * 0.04)))

    def add_wall_block(self, x0, y0, x1, y1) -> None:
        self.structures.append({
            "kind": "wall_block",
            "x0": float(min(x0, x1)), "y0": float(min(y0, y1)),
            "x1": float(max(x0, x1)), "y1": float(max(y0, y1)),
        })
        self._rebuild_geometry()

    def add_wall_circle(self, cx, cy, r) -> None:
        self.structures.append({"kind": "wall_circle", "cx": float(cx), "cy": float(cy), "r": float(r)})
        self._rebuild_geometry()

    def add_steel_ring(self, cx, cy, r_outer, r_inner) -> None:
        ro, ri = float(r_outer), float(r_inner)
        if ri >= ro:
            ri = ro * 0.75
        self.structures.append({
            "kind": "steel_ring", "cx": float(cx), "cy": float(cy),
            "r_outer": ro, "r_inner": ri,
        })
        self._rebuild_geometry()

    def add_steel_pipe(self, cx, cy, r_outer, wall_frac=0.18) -> None:
        self.add_steel_ring(cx, cy, r_outer, float(r_outer) * (1.0 - float(wall_frac)))

    def clear_structures(self) -> None:
        self.structures = []
        self._rebuild_geometry()

    def erase_at(self, x_frac, y_frac) -> bool:
        for i in range(len(self.structures) - 1, -1, -1):
            if _struct_contains(self, self.structures[i], x_frac, y_frac):
                self.structures.pop(i)
                self._rebuild_geometry()
                return True
        return False

    def add_oscillator(self, x_frac, y_frac, aperture=0, amp=1.0) -> None:
        ix, iy = _ix(self, x_frac), _iy(self, y_frac)
        if self._is_wall(iy, ix) and aperture == 0:
            return
        self.oscillators.append({"ix": ix, "iy": iy, "amp": amp, "aperture": aperture})

    def add_drip(self, x_frac, y_frac, amp=1.5) -> None:
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

    def clear_sources(self) -> None:
        self.oscillators = []

    def set_param(self, key, value) -> None:
        if key == "frequency":
            self.frequency = float(value)
        elif key == "brightness":
            self.renderer.scale = float(np.clip(value, 0.02, 2.0))
        elif key == "colormap":
            self.renderer.set_lut(str(value))
        elif key == "absorbing":
            self.absorbing = bool(value)
            self.set_scene(self.scene)
        elif key == "paused":
            self.paused = bool(value)

    def reset(self) -> None:
        self.set_scene(self.scene)

    def step(self, n_steps=1) -> None:
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
                        strip[~self.wall_mask[y0:y1, osc["ix"]]] = val
                        f.u_n[y0:y1, osc["ix"]] = strip
                    else:
                        f.u_n[y0:y1, osc["ix"]] = val
                elif not self._is_wall(osc["iy"], osc["ix"]):
                    f.u_n[osc["iy"], osc["ix"]] = val
            self._enforce_walls()

    def render(self) -> bytes:
        scale = self.renderer.scale if self.renderer.scale > 1e-9 else 1e-9
        raw = np.nan_to_num(self.field.field / scale, nan=0.0, posinf=1.0, neginf=-1.0)
        x = np.clip(raw, -1.0, 1.0)
        idx = ((x + 1.0) * 0.5 * (LUT_SIZE - 1)).astype(np.int32)
        rgb = self.renderer.lut[idx].astype(np.float32)
        if self.media_mask is not None:
            tint = np.array([150, 150, 165], dtype=np.float32)
            m = self.media_mask[..., None]
            rgb = np.where(m, 0.70 * rgb + 0.30 * tint, rgb)
        if self.wall_mask is not None:
            rgb[self.wall_mask] = np.array([70, 66, 52], dtype=np.float32)
        self._rgba[..., :3] = rgb.astype(np.uint8)
        return self._rgba.tobytes()
