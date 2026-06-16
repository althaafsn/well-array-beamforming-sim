"""
Acoustic pulse-echo engine for transducer / well-pipe NDT.

Builds on the finite-difference wave theory in
    Langtangen & Linge, "Finite Difference Methods for Wave Equations" (2016),
but uses a first-order velocity-stress (pressure-velocity) staggered FDTD, which
admits a true Perfectly Matched Layer. The PML follows the complex-coordinate
stretching / quadratic absorbing-profile construction reviewed in
    Lorin et al., "A friendly review of absorbing boundary conditions and
    perfectly matched layers ...",
realized here as a split-field (Berenger) PML: p = px + py with a graded
absorbing function sigma(nu) = sigma0 * (depth/L)^2 in each layer.

Governing acoustic system (variable density rho, bulk modulus kappa = rho c^2):

    dvx/dt = -(1/rho) dp/dx
    dvy/dt = -(1/rho) dp/dy
    dp/dt  = -kappa (dvx/dx + dvy/dy) + s

Split PML adds sigma_x to the x-operators and sigma_y to the y-operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

from .main import _rasterise_line


# ---------------------------------------------------------------------------
# Materials (SI units): wave speed c [m/s], density rho [kg/m^3]
# ---------------------------------------------------------------------------

WATER = (1480.0, 1000.0)
STEEL = (5900.0, 7850.0)


# ---------------------------------------------------------------------------
# Source time functions (center-frequency controlled)
# ---------------------------------------------------------------------------


def ricker(t: np.ndarray, fc: float, t0: float | None = None) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet with center frequency ``fc``."""
    if t0 is None:
        t0 = 1.0 / fc
    a = (np.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def tone_burst(t: np.ndarray, fc: float, n_cycles: int = 3, t0: float | None = None) -> np.ndarray:
    """Gaussian-windowed ``n_cycles``-cycle sinusoid at center frequency ``fc``."""
    duration = n_cycles / fc
    if t0 is None:
        t0 = 0.5 * duration
    sigma = duration / 5.0
    window = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return np.sin(2.0 * np.pi * fc * (t - t0)) * window


# ---------------------------------------------------------------------------
# Pipe-wall material map (water lumen + steel wall + corrosion pit)
# ---------------------------------------------------------------------------


@dataclass
class PipeModel:
    c: np.ndarray            # wave speed field (Ny, Nx)
    rho: np.ndarray          # density field (Ny, Nx)
    dx: float
    wall_col: np.ndarray     # inner-wall column index per row
    td_row0: int             # transducer/probe aperture start row
    td_row1: int             # aperture end row (exclusive)
    td_col: int              # transducer/probe column
    meta: dict = _field(default_factory=dict)


def build_pipe_model(
    Ny: int,
    Nx: int,
    dx: float,
    *,
    gap_mm: float = 12.0,
    wall_mm: float = 10.0,
    td_offset_mm: float = 3.0,
    aperture_mm: float = 3.0,
    corrosion: bool = False,
    pit_depth_mm: float = 3.0,
    pit_radius_mm: float = 1.5,
) -> PipeModel:
    """
    Construct a 2D cross-section: water bore on the left, a steel wall, with an
    optional semicircular corrosion pit carved into the inner wall face directly
    ahead of the transducer. The inner-wall surface is rasterised with the
    Bresenham helper so the metal/water interface is a clean integer contour.
    """
    mm = 1e-3
    c = np.full((Ny, Nx), WATER[0])
    rho = np.full((Ny, Nx), WATER[1])

    base_col = int(round((gap_mm * mm) / dx))
    y_center = Ny // 2
    pit_r = int(round((pit_radius_mm * mm) / dx))
    pit_d = int(round((pit_depth_mm * mm) / dx))

    # Inner-wall column per row (larger column = wall recedes / metal loss).
    wall_col = np.full(Ny, base_col, dtype=int)
    if corrosion:
        for i in range(Ny):
            dy = i - y_center
            if abs(dy) <= pit_r:
                recede = int(round(pit_d * np.sqrt(max(0.0, 1.0 - (dy / pit_r) ** 2))))
                wall_col[i] = base_col + recede

    # Rasterise the inner-wall surface contour, then fill steel to its right.
    surface = np.zeros((Ny, Nx), dtype=bool)
    for i in range(Ny - 1):
        for (yy, xx) in _rasterise_line(i, int(wall_col[i]), i + 1, int(wall_col[i + 1])):
            if 0 <= yy < Ny and 0 <= xx < Nx:
                surface[yy, xx] = True
    cols = np.arange(Nx)[None, :]
    steel_mask = cols >= wall_col[:, None]
    steel_mask |= surface

    c[steel_mask] = STEEL[0]
    rho[steel_mask] = STEEL[1]

    td_col = int(round((td_offset_mm * mm) / dx))
    half_ap = int(round((aperture_mm * mm) / dx)) // 2
    td_row0, td_row1 = y_center - half_ap, y_center + half_ap + 1

    return PipeModel(
        c=c, rho=rho, dx=dx, wall_col=wall_col,
        td_row0=td_row0, td_row1=td_row1, td_col=td_col,
        meta={"base_col": base_col, "y_center": y_center, "pit_depth_mm": pit_depth_mm if corrosion else 0.0},
    )


# ---------------------------------------------------------------------------
# Split-field PML acoustic solver
# ---------------------------------------------------------------------------


class AcousticPML2D:
    """Velocity-stress acoustic FDTD with a split-field (Berenger) PML."""

    def __init__(
        self,
        c: np.ndarray,
        rho: np.ndarray,
        dx: float,
        *,
        pml_cells: int = 20,
        cfl: float = 0.5,
        R0: float = 1e-6,
        m: int = 2,
    ) -> None:
        self.c = np.asarray(c, float)
        self.rho = np.asarray(rho, float)
        self.Ny, self.Nx = self.c.shape
        self.dx = self.dy = dx
        self.kappa = self.rho * self.c**2

        cmax = float(self.c.max())
        self.dt = cfl * dx / (cmax * np.sqrt(2.0))

        # Quadratic absorbing profile sigma(nu) = coeff * (depth/L)^2 (per paper),
        # scaled by the local wave speed so both water and steel are matched.
        L = pml_cells
        coeff = (m + 1) * np.log(1.0 / R0) / (2.0 * L * dx)  # units 1/m
        Kx = np.zeros(self.Nx)
        Ky = np.zeros(self.Ny)
        for k in range(L):
            frac = (L - k) / L
            val = coeff * frac**2
            Kx[k] = Kx[self.Nx - 1 - k] = val
            Ky[k] = Ky[self.Ny - 1 - k] = val
        self.sigx = Kx[None, :] * self.c   # 1/s
        self.sigy = Ky[:, None] * self.c
        self.pml_cells = L

        self.p = np.zeros((self.Ny, self.Nx))
        self.px = np.zeros((self.Ny, self.Nx))
        self.py = np.zeros((self.Ny, self.Nx))
        self.vx = np.zeros((self.Ny, self.Nx - 1))
        self.vy = np.zeros((self.Ny - 1, self.Nx))
        self.n = 0

        # Precompute face quantities and PML update factors.
        dt = self.dt
        self._rho_x = 0.5 * (self.rho[:, 1:] + self.rho[:, :-1])
        self._rho_y = 0.5 * (self.rho[1:, :] + self.rho[:-1, :])
        sx_f = 0.5 * (self.sigx[:, 1:] + self.sigx[:, :-1])
        sy_f = 0.5 * (self.sigy[1:, :] + self.sigy[:-1, :])
        self._vx_a = (1 - 0.5 * sx_f * dt) / (1 + 0.5 * sx_f * dt)
        self._vx_b = (dt / self._rho_x) / (1 + 0.5 * sx_f * dt)
        self._vy_a = (1 - 0.5 * sy_f * dt) / (1 + 0.5 * sy_f * dt)
        self._vy_b = (dt / self._rho_y) / (1 + 0.5 * sy_f * dt)
        self._px_a = (1 - 0.5 * self.sigx * dt) / (1 + 0.5 * self.sigx * dt)
        self._px_b = (dt * self.kappa) / (1 + 0.5 * self.sigx * dt)
        self._py_a = (1 - 0.5 * self.sigy * dt) / (1 + 0.5 * self.sigy * dt)
        self._py_b = (dt * self.kappa) / (1 + 0.5 * self.sigy * dt)

    def step(self, source_value: float = 0.0, src_rows: slice | None = None, src_col: int | None = None) -> None:
        # 1) velocity from pressure gradient (rigid v=0 at the outer domain faces)
        self.vx = self._vx_a * self.vx - self._vx_b * (self.p[:, 1:] - self.p[:, :-1]) / self.dx
        self.vy = self._vy_a * self.vy - self._vy_b * (self.p[1:, :] - self.p[:-1, :]) / self.dy

        # 2) split pressure from velocity divergence
        vxf = np.pad(self.vx, ((0, 0), (1, 1)))
        vyf = np.pad(self.vy, ((1, 1), (0, 0)))
        dvx_dx = (vxf[:, 1:] - vxf[:, :-1]) / self.dx
        dvy_dy = (vyf[1:, :] - vyf[:-1, :]) / self.dy
        self.px = self._px_a * self.px - self._px_b * dvx_dx
        self.py = self._py_a * self.py - self._py_b * dvy_dy

        # 3) soft pressure source (split equally between the two halves)
        if source_value and src_rows is not None and src_col is not None:
            self.px[src_rows, src_col] += 0.5 * source_value
            self.py[src_rows, src_col] += 0.5 * source_value

        self.p = self.px + self.py
        self.n += 1


# ---------------------------------------------------------------------------
# Envelope + time-of-flight
# ---------------------------------------------------------------------------


def envelope(x: np.ndarray) -> np.ndarray:
    """Analytic-signal envelope via the FFT Hilbert transform (no scipy)."""
    n = x.size
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


# ---------------------------------------------------------------------------
# Pulse-echo run
# ---------------------------------------------------------------------------


def run_pulse_echo(model: PipeModel, *, fc: float = 1e6, n_steps: int = 3000,
                   pml_cells: int = 20, snapshots=(0,)):
    sim = AcousticPML2D(model.c, model.rho, model.dx, pml_cells=pml_cells)
    dt = sim.dt
    t = np.arange(n_steps) * dt
    src = tone_burst(t, fc, n_cycles=3)

    rows = slice(model.td_row0, model.td_row1)
    ascan = np.zeros(n_steps)
    frames: dict[int, np.ndarray] = {}
    for k in range(n_steps):
        sim.step(source_value=float(src[k]) * 1e6, src_rows=rows, src_col=model.td_col)
        ascan[k] = float(sim.p[rows, model.td_col].mean())
        if k in snapshots:
            frames[k] = sim.p.copy()
    return t, ascan, dt, frames, sim


def infer_distance(t: np.ndarray, ascan: np.ndarray, fc: float, c_water: float = WATER[0]):
    """
    Detect the first wall echo after the emission window and infer the wall
    distance *from the transducer*: d = c_water * (t_echo - t0) / 2, where t0 is
    the tone-burst emission center.
    """
    env = envelope(ascan)
    t0 = 0.5 * 3.0 / fc                 # tone_burst center (n_cycles=3)
    emit_end = t0 + 3.0 / fc + 1.0e-6   # skip the burst + 1 us guard
    mask = t > emit_end
    if not mask.any():
        return None, env, None
    idx = int(np.argmax(env * mask))
    tof = t[idx] - t0
    return c_water * tof / 2.0, env, t[idx]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[2] / "outputs" / "waveform_2d_pml"
    out.mkdir(parents=True, exist_ok=True)

    dx = 1e-4          # 0.1 mm
    Ny, Nx = 200, 260
    fc = 1e6           # 1 MHz
    gap_mm = 12.0

    # ---- PML absorption test (homogeneous water) ----
    c0 = np.full((Ny, Nx), WATER[0])
    r0 = np.full((Ny, Nx), WATER[1])
    sim = AcousticPML2D(c0, r0, dx, pml_cells=20)
    tt = np.arange(1200) * sim.dt
    s = tone_burst(tt, fc, n_cycles=3)
    peak = 0.0
    for k in range(1200):
        sim.step(source_value=float(s[k]) * 1e6, src_rows=slice(Ny // 2, Ny // 2 + 1), src_col=Nx // 2)
        peak = max(peak, float(np.abs(sim.p).max()))
    interior = sim.p[25:-25, 25:-25]
    resid = float(np.abs(interior).max()) / peak
    print(f"[pml] residual interior amplitude after pulse exits = {resid*100:.3f}% of peak")

    # ---- Pulse-echo: pristine vs corroded wall ----
    n_steps = 3500
    snaps = (400, 1500, 2700)
    results = {}
    for label, corr in (("pristine", False), ("corroded", True)):
        model = build_pipe_model(Ny, Nx, dx, gap_mm=gap_mm, td_offset_mm=3.0, corrosion=corr,
                                 pit_depth_mm=3.0, pit_radius_mm=1.5, aperture_mm=3.0)
        t, ascan, dt, frames, simm = run_pulse_echo(
            model, fc=fc, n_steps=n_steps, snapshots=snaps)
        dist, env, t_echo = infer_distance(t, ascan, fc)
        results[label] = (t, ascan, env, dist, model, frames)
        geom_mm = (model.wall_col[model.meta["y_center"]] - model.td_col) * dx * 1e3
        print(f"[echo:{label}] echo @ {t_echo*1e6:.2f} us -> inferred wall {dist*1e3:.2f} mm "
              f"from transducer  (geometric {geom_mm:.2f} mm)")

        # velocity map + snapshots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(model.c, cmap="viridis", origin="upper")
        axes[0].axvline(model.td_col, color="r", lw=1)
        axes[0].set_title(f"{label}: c map (water/steel)")
        for ax, kk in zip(axes[1:], snaps):
            v = np.abs(frames[kk]).max() or 1.0
            ax.imshow(frames[kk], cmap="RdBu_r", vmin=-v, vmax=v, origin="upper")
            ax.set_title(f"p @ step {kk} ({kk*dt*1e6:.1f} us)")
        fig.savefig(out / f"field_{label}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ---- A-scan comparison ----
    fig, ax = plt.subplots(figsize=(9, 4))
    norm = max(results[l][2].max() for l in results)
    for label in ("pristine", "corroded"):
        t, ascan, env, dist, model, _ = results[label]
        ax.plot(t * 1e6, env / norm, label=f"{label}: wall {dist*1e3:.1f} mm")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("echo envelope (norm.)")
    ax.set_title(f"Pulse-echo A-scan @ {fc/1e6:.0f} MHz, nominal gap {gap_mm:.0f} mm")
    ax.legend()
    fig.savefig(out / "ascan.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] outputs -> {out}")


if __name__ == "__main__":
    main()
