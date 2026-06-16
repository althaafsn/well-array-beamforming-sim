"""
Consistent colour mapping for the wave field, in the spirit of Falstad's
RippleGL ripple-tank viewer.

The key idea behind "consistent coloring" is that the field is mapped through a
*fixed* amplitude scale and a *fixed* 256-entry lookup table (LUT), so a given
displacement always renders as the same colour from frame to frame (no per-frame
auto-scaling). Crests saturate toward white, troughs toward the deep colour, and
the calm medium sits in the middle.
"""

from __future__ import annotations

import numpy as np

LUT_SIZE = 256


def _interp_lut(stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Build an (LUT_SIZE, 3) uint8 table by linearly interpolating colour stops."""
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float)
    grid = np.linspace(0.0, 1.0, LUT_SIZE)
    lut = np.empty((LUT_SIZE, 3))
    for ch in range(3):
        lut[:, ch] = np.interp(grid, xs, cols[:, ch])
    return np.clip(lut, 0, 255).astype(np.uint8)


# RippleGL-like watery scheme: deep-blue troughs, calm navy zero, white crests.
_RIPPLE = _interp_lut([
    (0.00, (6, 10, 38)),
    (0.22, (18, 64, 140)),
    (0.50, (14, 102, 150)),
    (0.74, (118, 198, 226)),
    (1.00, (255, 255, 255)),
])

# Symmetric diverging (classic scientific) blue-white-red.
_RWB = _interp_lut([
    (0.00, (33, 80, 170)),
    (0.50, (245, 245, 245)),
    (1.00, (200, 40, 40)),
])

_GRAY = _interp_lut([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])


def _mpl_lut(name: str) -> np.ndarray:
    """Sample a matplotlib colormap into our LUT (matplotlib is already a dep)."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap(name, LUT_SIZE)
    rgba = (np.asarray([cmap(i) for i in range(LUT_SIZE)]) * 255).astype(np.uint8)
    return rgba[:, :3]


_BUILTINS = {"ripple": _RIPPLE, "rwb": _RWB, "gray": _GRAY}
_CACHE: dict[str, np.ndarray] = {}


def get_lut(name: str) -> np.ndarray:
    name = (name or "ripple").lower()
    if name in _BUILTINS:
        return _BUILTINS[name]
    if name not in _CACHE:
        try:
            _CACHE[name] = _mpl_lut(name)
        except Exception:
            _CACHE[name] = _RIPPLE
    return _CACHE[name]


def available() -> list[str]:
    return ["ripple", "rwb", "gray", "viridis", "twilight", "magma", "RdBu_r"]


class Renderer:
    """Maps a float field to RGBA bytes with a fixed scale, LUT and media overlay."""

    def __init__(self, lut_name: str = "ripple", scale: float = 0.4) -> None:
        self.set_lut(lut_name)
        self.scale = float(scale)
        self.alpha = np.uint8(255)

    def set_lut(self, name: str) -> None:
        self.lut_name = name
        self.lut = get_lut(name)

    def render(
        self,
        field: np.ndarray,
        *,
        wall_mask: np.ndarray | None = None,
        wall_rgb: tuple[int, int, int] = (70, 66, 52),
        media_mask: np.ndarray | None = None,
        media_rgb: tuple[int, int, int] = (150, 150, 165),
        media_alpha: float = 0.30,
    ) -> bytes:
        """Return C-contiguous RGBA bytes (shape Ny x Nx x 4)."""
        scale = self.scale if self.scale > 1e-9 else 1e-9
        x = np.clip(field / scale, -1.0, 1.0)
        idx = ((x + 1.0) * 0.5 * (LUT_SIZE - 1)).astype(np.int32)
        rgb = self.lut[idx].astype(np.float32)  # (Ny, Nx, 3)

        if media_mask is not None and media_mask.any():
            tint = np.asarray(media_rgb, dtype=np.float32)
            m = media_mask[..., None]
            rgb = np.where(m, (1.0 - media_alpha) * rgb + media_alpha * tint, rgb)

        if wall_mask is not None and wall_mask.any():
            rgb[wall_mask] = np.asarray(wall_rgb, dtype=np.float32)

        Ny, Nx = field.shape
        out = np.empty((Ny, Nx, 4), dtype=np.uint8)
        out[..., :3] = rgb.astype(np.uint8)
        out[..., 3] = self.alpha
        return out.tobytes()
