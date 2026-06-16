"""
2D finite-difference wave engine.

Implements the explicit (leapfrog) scheme from
    Langtangen & Linge, "Finite Difference Methods for Wave Equations" (2016),
solving

    rho * u_tt + b * u_t = div(q grad u) + f,        q = c^2 (variable velocity),

on a rectangular grid. Equation/section references below point into that book:

    - Interior update            : eq. (117)            -> advance()
    - Special first step (u^1)    : eq. (118)            -> advance(step1=True)
    - Variable coefficient q=c^2  : eq. (112), Sec. 7    -> per-cell Cx2/Cy2 arrays
    - Neumann reflecting walls    : Sec. 6 (ghost cells) -> boundary="neumann"
    - Damping (sponge / ABC)      : Sec. 7.8             -> b field; see also the
      Lorin et al. review of absorbing boundary conditions / PML
    - Stability (CFL)             : Sec. 10.5            -> dt <= 1/(c*sqrt(sum 1/h^2))
    - Verification (quadratic)    : Sec. 12.3            -> test_quadratic()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core update -- book eq. (117) interior, eq. (118) first step
# ---------------------------------------------------------------------------


def advance(
    u: np.ndarray,
    u_n: np.ndarray,
    u_nm1: np.ndarray,
    Cx2,
    Cy2,
    dt2: float,
    f_a: np.ndarray | None,
    *,
    V=None,
    B=0.0,
    dt: float = 0.0,
    step1: bool = False,
    boundary: str = "dirichlet",
) -> np.ndarray:
    """
    Advance one time level, writing the new field into ``u`` and returning it.

    ``Cx2``/``Cy2`` are ``(c*dt/dx)**2`` / ``(c*dt/dy)**2``; either may be a
    scalar (constant velocity) or a full-grid array (variable velocity, eq. 112).
    ``B = b*dt/2`` is the discrete damping coefficient (scalar or array).
    ``boundary`` is ``"dirichlet"`` (u=0) or ``"neumann"`` (zero-gradient walls).
    """
    Cx2 = np.asarray(Cx2, dtype=float)
    Cy2 = np.asarray(Cy2, dtype=float)

    if boundary == "neumann":
        # Ghost cells via mirror padding => u_{-1}=u_{1}, i.e. du/dn = 0 (Sec. 6.6).
        p = np.pad(u_n, 1, mode="reflect")
        u_xx = p[:-2, 1:-1] - 2.0 * u_n + p[2:, 1:-1]
        u_yy = p[1:-1, :-2] - 2.0 * u_n + p[1:-1, 2:]
        sl: tuple = (slice(None), slice(None))
    elif boundary == "dirichlet":
        c = u_n[1:-1, 1:-1]
        u_xx = u_n[:-2, 1:-1] - 2.0 * c + u_n[2:, 1:-1]
        u_yy = u_n[1:-1, :-2] - 2.0 * c + u_n[1:-1, 2:]
        sl = (slice(1, -1), slice(1, -1))
    else:
        raise ValueError(f"unknown boundary: {boundary!r}")

    def _crop(a):
        return a[sl] if np.ndim(a) else a

    Cx2s, Cy2s, Bs = _crop(Cx2), _crop(Cy2), _crop(B)
    f_s = _crop(f_a) if f_a is not None else 0.0

    stencil = Cx2s * u_xx + Cy2s * u_yy

    if step1:
        # First step combines the scheme (n=0) with u_t = V  (eq. 118).
        new = u_n[sl] + 0.5 * stencil + 0.5 * dt2 * f_s
        if V is not None:
            new = new + dt * _crop(V) * (1.0 - Bs)
    else:
        new = (2.0 * u_n[sl] - (1.0 - Bs) * u_nm1[sl] + stencil + dt2 * f_s) / (1.0 + Bs)

    u[:] = 0.0  # zeroes the Dirichlet border; harmless full overwrite for Neumann
    u[sl] = new
    return u


# ---------------------------------------------------------------------------
# Book-style solver with three-array reference switching (Sec. 12.2)
# ---------------------------------------------------------------------------


def solver(
    I,
    V,
    f,
    c,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    dt: float,
    T: float,
    *,
    b=0.0,
    boundary: str = "dirichlet",
    user_action=None,
):
    """
    Solve the 2D wave equation on (0,Lx) x (0,Ly) for t in (0,T].

    ``I``, ``V``, ``f`` and ``c`` may be callables (evaluated on the vectorized
    mesh ``xv``, ``yv``) or constants/arrays. Returns ``(u, x, y, t)`` with ``u``
    the field at the final time level (note the switching caveat in Sec. 12.1).
    """
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    Nt = int(round(T / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)
    xv, yv = x[:, None], y[None, :]
    shape = (Nx + 1, Ny + 1)

    cc = c(xv, yv) if callable(c) else c
    Cx2 = (np.asarray(cc, float) * dt / dx) ** 2
    Cy2 = (np.asarray(cc, float) * dt / dy) ** 2
    dt2 = dt * dt
    B = 0.5 * np.asarray(b, float) * dt if np.ndim(b) else 0.5 * b * dt

    u = np.zeros(shape)
    u_n = np.zeros(shape)
    u_nm1 = np.zeros(shape)

    u_n[:] = I(xv, yv) if callable(I) else I
    if callable(V):
        Va = V(xv, yv)
    elif V is None:
        Va = np.zeros(shape)
    else:
        Va = V

    if user_action is not None:
        user_action(u_n, x, xv, y, yv, t, 0)

    f_a = f(xv, yv, t[0]) if callable(f) else (np.zeros(shape) if f is None else f)
    advance(u, u_n, u_nm1, Cx2, Cy2, dt2, f_a, V=Va, B=B, dt=dt, step1=True, boundary=boundary)
    u_nm1, u_n, u = u_n, u, u_nm1
    if user_action is not None:
        user_action(u_n, x, xv, y, yv, t, 1)

    for n in range(1, Nt):
        f_a = f(xv, yv, t[n]) if callable(f) else (np.zeros(shape) if f is None else f)
        advance(u, u_n, u_nm1, Cx2, Cy2, dt2, f_a, B=B, boundary=boundary)
        u_nm1, u_n, u = u_n, u, u_nm1
        if user_action is not None:
            user_action(u_n, x, xv, y, yv, t, n + 1)

    return u_n, x, y, t


# ---------------------------------------------------------------------------
# Stateful field for interactive / pulse-echo use
# ---------------------------------------------------------------------------


class WaveField:
    """
    Mutable 2D wave state advanced one leapfrog step at a time.

    Holds three displacement levels (``u_nm1``, ``u_n``, ``u``) plus the
    precomputed Courant arrays. ``c`` is the wave speed (scalar or full-grid
    array for heterogeneous media, e.g. fluid lumen vs. steel wall). ``dt``
    defaults to 90% of the CFL limit (Sec. 10.5).
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        c=1.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float | None = None,
        b=0.0,
        boundary: str = "neumann",
    ) -> None:
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

    def _set_courant(self) -> None:
        self.Cx2 = (self.c * self.dt / self.dx) ** 2
        self.Cy2 = (self.c * self.dt / self.dy) ** 2
        self.dt2 = self.dt**2

    def set_velocity(self, c_arr, *, recalc_dt: bool = True) -> None:
        """Update velocity map; optionally recompute ``dt`` for CFL stability."""
        self.c[:] = c_arr
        if recalc_dt:
            cmax = float(self.c.max())
            cfl = 1.0 / (cmax * np.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
            self.dt = 0.9 * cfl
        self._set_courant()

    def reset(self) -> None:
        self.u[:] = 0.0
        self.u_n[:] = 0.0
        self.u_nm1[:] = 0.0
        self.n = 0

    @property
    def field(self) -> np.ndarray:
        """Current displacement field (the most recently computed level)."""
        return self.u_n

    def step(self, f_a: np.ndarray | None = None) -> None:
        """Advance one time step, applying source term ``f_a`` (optional)."""
        advance(
            self.u, self.u_n, self.u_nm1,
            self.Cx2, self.Cy2, self.dt2, f_a,
            B=self.B, dt=self.dt, step1=(self.n == 0), boundary=self.boundary,
        )
        self.u_nm1, self.u_n, self.u = self.u_n, self.u, self.u_nm1
        self.n += 1


# ---------------------------------------------------------------------------
# Source injection / probe readout
# ---------------------------------------------------------------------------


def inject_point(field: WaveField, iy: int, ix: int, value: float) -> None:
    """Set a single-cell initial displacement (impulse) at (iy, ix)."""
    field.u_n[iy, ix] += float(value)


def read_probe(field: WaveField, iy: int, ix: int) -> float:
    """Read the displacement at one cell -- a simulated receiver sample."""
    return float(field.u_n[iy, ix])


def apply_border_damping(field: WaveField, border: int, b_max: float = 0.5) -> None:
    """
    Build a graded damping (sponge) layer of width ``border`` cells so outgoing
    waves are absorbed near the edges -- a simple absorbing boundary condition
    (cf. Lorin et al. review; book Sec. 7.8). Updates ``field.B`` in place.
    """
    Ny, Nx = field.Ny, field.Nx
    b = np.zeros((Ny, Nx))
    ramp = np.linspace(1.0, 0.0, border) ** 2  # strongest at the outer edge
    for layer in range(border):
        val = b_max * ramp[layer]
        b[layer, :] = np.maximum(b[layer, :], val)
        b[Ny - 1 - layer, :] = np.maximum(b[Ny - 1 - layer, :], val)
        b[:, layer] = np.maximum(b[:, layer], val)
        b[:, Nx - 1 - layer] = np.maximum(b[:, Nx - 1 - layer], val)
    field.B = 0.5 * b * field.dt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rasterise_line(iy1: int, ix1: int, iy2: int, ix2: int) -> list[tuple[int, int]]:
    """
    Bresenham line rasterisation. Returns all cells on the segment including
    both endpoints, in order from (iy1, ix1) to (iy2, ix2). Useful for drawing
    pipe walls / reflectors into a velocity map.
    """
    cells: list[tuple[int, int]] = []
    dy = abs(iy2 - iy1)
    dx = abs(ix2 - ix1)
    sy = 1 if iy2 > iy1 else -1
    sx = 1 if ix2 > ix1 else -1
    iy, ix = iy1, ix1
    if dx >= dy:
        err = dx // 2
        while ix != ix2:
            cells.append((iy, ix))
            err -= dy
            if err < 0:
                iy += sy
                err += dx
            ix += sx
    else:
        err = dy // 2
        while iy != iy2:
            cells.append((iy, ix))
            err -= dx
            if err < 0:
                ix += sx
                err += dy
            iy += sy
    cells.append((iy2, ix2))
    return cells


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def plot_field(field: WaveField, path: str | None = None, vlim: float | None = None):
    """Render the field with matplotlib; save to ``path`` if given, else show."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    v = vlim if vlim is not None else float(np.abs(field.field).max() or 1.0)
    im = ax.imshow(field.field, cmap="RdBu_r", vmin=-v, vmax=v, origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"u  (step {field.n})")
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Verification -- exact quadratic solution (book Sec. 12.3, eq. 119)
# ---------------------------------------------------------------------------


def test_quadratic(Nx: int = 12, Ny: int = 14, tol: float = 1e-11) -> float:
    """
    The solution u_e = x(Lx-x) y(Ly-y) (1 + t/2) satisfies BOTH the PDE and the
    discrete equations, so the solver must reproduce it to machine precision.
    Returns the max error and asserts it is below ``tol``.
    """
    Lx, Ly, c = 3.0, 2.0, 1.5

    def u_exact(x, y, t):
        return x * (Lx - x) * y * (Ly - y) * (1 + 0.5 * t)

    def I(x, y):
        return u_exact(x, y, 0)

    def V(x, y):
        return 0.5 * x * (Lx - x) * y * (Ly - y)

    def f(x, y, t):
        return 2 * c**2 * (1 + 0.5 * t) * (x * (Lx - x) + y * (Ly - y))

    errors: list[float] = []

    def check(u, x, xv, y, yv, t, n):
        errors.append(float(np.abs(u - u_exact(xv, yv, t[n])).max()))

    dt = 0.9 / (c * np.sqrt(1 / (Lx / Nx) ** 2 + 1 / (Ly / Ny) ** 2))
    solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T=18 * dt,
           boundary="dirichlet", user_action=check)

    max_err = max(errors)
    assert max_err < tol, f"quadratic verification failed: max_err={max_err:.3e}"
    return max_err


# ---------------------------------------------------------------------------
# Prototype demo -- single point source at (0, 0)
# ---------------------------------------------------------------------------


def main() -> None:
    err = test_quadratic()
    print(f"[verify] quadratic-solution max error = {err:.2e}  -> PASS")

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "waveform_2d_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    field = WaveField(height=200, width=200, c=1.0, boundary="neumann")
    inject_point(field, iy=0, ix=0, value=1.0)  # single point source at the origin

    n_steps = 240
    saved = []
    for k in range(n_steps + 1):
        if k % 30 == 0:
            p = out_dir / f"frame_{k:04d}.png"
            plot_field(field, path=str(p))  # per-frame auto color scale
            saved.append(p)
        field.step()

    print(f"[demo] point source at (0,0): {len(saved)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
