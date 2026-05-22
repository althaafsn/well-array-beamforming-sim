from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from well_array_sim.internal import load_internal_scenario
from well_array_sim.internal.axial_scan import AxialScanResult, simulate_axial_scan
from well_array_sim.internal.figure_layers import FigureLayers
from well_array_sim.internal.corrosion.engine import CorrosionSnapshot
from well_array_sim.internal.gui_views import (
    MODE_AXIAL,
    MODE_CORROSION,
    MODE_SINGLE,
    VIEWS_AXIAL,
    VIEWS_CORROSION,
    VIEWS_SINGLE,
    build_figure_for_view,
    view_supports_overlays,
)
from well_array_sim.internal.pulse_echo_result import PulseEchoResult
from well_array_sim.internal.ray_forward import simulate_pulse_echo_2d
from well_array_sim.internal.scenario import DEFAULT_STEER_ANGLE_DEG, InternalScenario

DEFAULT_SCENARIO = Path("scenarios/internal_pipe_default.yaml")
DEFAULT_OUTPUT_PREFIX = "gui_demo"
OUTPUTS_DIR = Path("outputs")
OVERLAY_DEBOUNCE_MS = 200
DEFAULT_PREVIEW_ANGLE_STEP = 5.0
UI_QUEUE_POLL_MS = 50

MODE_HINTS = {
    MODE_SINGLE: "One ultrasound pulse at one azimuth — echo waveform and blind range estimate.",
    MODE_AXIAL: "Rotate 360° at each pipe station — wall radius map (angular SAFT when configured).",
    MODE_CORROSION: "Wall thickness loss over time — simulation ground truth, not UT inference.",
}


@dataclass(frozen=True)
class _RunJob:
    scenario_path: Path
    mode: str
    angle_deg: float
    angle_step_deg: float
    corrosion_year_yr: float


@dataclass(frozen=True)
class _RunOutcome:
    scenario: InternalScenario
    mode: str
    pulse_echo_result: PulseEchoResult | None
    axial_result: AxialScanResult | None
    corrosion_snapshot: CorrosionSnapshot | None
    default_time_us: float
    status: str


class PipeSimGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Internal Pipe Pulse-Echo Sim")
        self.root.minsize(900, 600)

        self.scenario = None
        self.pulse_echo_result = None
        self.axial_result = None
        self.corrosion_snapshot = None
        self.current_figure: Figure | None = None
        self._figure_layers: FigureLayers | None = None
        self._layer_cache_key: tuple | None = None
        self._running = False
        self._overlay_after_id: str | None = None
        self._ui_queue: queue.Queue[object] = queue.Queue()
        self._worker_threads: list[threading.Thread] = []

        self._build_controls()
        self._build_canvas()
        self._on_mode_change()
        self._set_status("Ready. Pick a simulation type and click Run.")
        self.root.after(UI_QUEUE_POLL_MS, self._poll_ui_queue)

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            panel,
            text="Rotating UT tool inside a fluid-filled pipe",
            wraplength=220,
            font=("", 9, "italic"),
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(panel, text="Scenario").grid(row=1, column=0, sticky=tk.W)
        self.scenario_var = tk.StringVar(value=str(DEFAULT_SCENARIO))
        ttk.Entry(panel, textvariable=self.scenario_var, width=32).grid(row=2, column=0, sticky=tk.EW, pady=(0, 8))

        ttk.Label(panel, text="Simulation").grid(row=3, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value=MODE_SINGLE)
        mode_frame = ttk.Frame(panel)
        mode_frame.grid(row=4, column=0, sticky=tk.W, pady=(0, 4))
        for text, value in (
            ("One shot", MODE_SINGLE),
            ("Pipe sweep (360°)", MODE_AXIAL),
            ("Corrosion (ground truth)", MODE_CORROSION),
        ):
            ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self._on_mode_change,
            ).pack(anchor=tk.W)

        self.mode_hint_var = tk.StringVar(value=MODE_HINTS[MODE_SINGLE])
        ttk.Label(
            panel,
            textvariable=self.mode_hint_var,
            wraplength=220,
            foreground="#555555",
        ).grid(row=5, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(panel, text="Angle (deg)").grid(row=6, column=0, sticky=tk.W)
        self.angle_var = tk.DoubleVar(value=DEFAULT_STEER_ANGLE_DEG)
        self.angle_spin = ttk.Spinbox(panel, from_=0.0, to=359.0, textvariable=self.angle_var, width=10)
        self.angle_spin.grid(row=7, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(panel, text="Angle step (preview deg)").grid(row=8, column=0, sticky=tk.W)
        self.step_var = tk.DoubleVar(value=DEFAULT_PREVIEW_ANGLE_STEP)
        self.step_spin = ttk.Spinbox(panel, from_=0.5, to=45.0, increment=0.5, textvariable=self.step_var, width=10)
        self.step_spin.grid(row=9, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(panel, text="Corrosion year").grid(row=10, column=0, sticky=tk.W)
        self.corrosion_year_var = tk.DoubleVar(value=0.0)
        self.corrosion_year_spin = ttk.Spinbox(
            panel,
            from_=0.0,
            to=50.0,
            increment=0.5,
            textvariable=self.corrosion_year_var,
            width=10,
            state="disabled",
        )
        self.corrosion_year_spin.grid(row=11, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(panel, text="View").grid(row=12, column=0, sticky=tk.W)
        self.view_var = tk.StringVar(value=VIEWS_SINGLE[0])
        self.view_combo = ttk.Combobox(panel, textvariable=self.view_var, state="readonly", width=18)
        self.view_combo.grid(row=13, column=0, sticky=tk.EW, pady=(0, 8))
        self.view_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_view_change())

        ttk.Label(panel, text="Packet time (µs)").grid(row=14, column=0, sticky=tk.W)
        self.time_us_var = tk.DoubleVar(value=80.0)
        self.time_spin = ttk.Spinbox(
            panel,
            from_=0.0,
            to=500.0,
            increment=1.0,
            textvariable=self.time_us_var,
            width=10,
            command=self._on_time_change,
        )
        self.time_spin.grid(row=15, column=0, sticky=tk.W, pady=(0, 8))
        self.time_spin.bind("<Return>", lambda _e: self._on_time_change())
        self.time_spin.bind("<FocusOut>", lambda _e: self._on_time_change())

        self.show_inferred_var = tk.BooleanVar(value=True)
        self.show_gt_var = tk.BooleanVar(value=True)
        self.inferred_check = ttk.Checkbutton(
            panel,
            text="Show inferred",
            variable=self.show_inferred_var,
            command=self._schedule_overlay_redraw,
        )
        self.inferred_check.grid(row=16, column=0, sticky=tk.W)
        self.gt_check = ttk.Checkbutton(
            panel,
            text="Show ground truth",
            variable=self.show_gt_var,
            command=self._schedule_overlay_redraw,
        )
        self.gt_check.grid(row=17, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(panel, text="Output prefix").grid(row=18, column=0, sticky=tk.W)
        self.prefix_var = tk.StringVar(value=DEFAULT_OUTPUT_PREFIX)
        ttk.Entry(panel, textvariable=self.prefix_var, width=20).grid(row=19, column=0, sticky=tk.EW, pady=(0, 8))

        btn_frame = ttk.Frame(panel)
        btn_frame.grid(row=20, column=0, sticky=tk.EW, pady=(0, 8))
        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Save PNG", command=self._on_save).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.status_var, wraplength=220).grid(row=21, column=0, sticky=tk.W)
        panel.columnconfigure(0, weight=1)

    def _build_canvas(self) -> None:
        self.canvas_frame = ttk.Frame(self.root, padding=8)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas: FigureCanvasTkAgg | None = None

    def _on_mode_change(self) -> None:
        mode = self.mode_var.get()
        self.mode_hint_var.set(MODE_HINTS.get(mode, ""))
        if mode == MODE_SINGLE:
            self.view_combo["values"] = VIEWS_SINGLE
            self.view_var.set(VIEWS_SINGLE[0])
            self.angle_spin.state(["!disabled"])
            self.step_spin.state(["disabled"])
            self._set_corrosion_year_state(enabled=False)
        elif mode == MODE_AXIAL:
            self.view_combo["values"] = VIEWS_AXIAL
            self.view_var.set(VIEWS_AXIAL[0])
            self.angle_spin.state(["disabled"])
            self.step_spin.state(["!disabled"])
            self._set_corrosion_year_state(enabled=self.scenario is not None and self.scenario.has_corrosion())
        else:
            self.view_combo["values"] = VIEWS_CORROSION
            self.view_var.set(VIEWS_CORROSION[0])
            self.angle_spin.state(["disabled"])
            self.step_spin.state(["disabled"])
            self._set_corrosion_year_state(enabled=True)
        self._update_time_control_state()
        self._clear_stale_results()
        self._on_view_change()

    def _set_corrosion_year_state(self, *, enabled: bool) -> None:
        if enabled:
            self.corrosion_year_spin.state(["!disabled"])
        else:
            self.corrosion_year_spin.state(["disabled"])

    def _update_time_control_state(self) -> None:
        if self.mode_var.get() == MODE_SINGLE and self.view_var.get() == "Packet scene":
            self.time_spin.state(["!disabled"])
        else:
            self.time_spin.state(["disabled"])

    def _on_time_change(self) -> None:
        if self.pulse_echo_result is not None and self.view_var.get() == "Packet scene":
            self._layer_cache_key = None
            self._redraw_from_cache()

    def _clear_display_widgets(self) -> None:
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def _clear_stale_results(self) -> None:
        self.pulse_echo_result = None
        self.axial_result = None
        self.corrosion_snapshot = None
        self._figure_layers = None
        self._layer_cache_key = None
        self._clear_display_widgets()

    def _on_view_change(self) -> None:
        self._update_time_control_state()
        overlays = view_supports_overlays(self.view_var.get())
        state = ["!disabled"] if overlays else ["disabled"]
        self.inferred_check.state(state)
        self.gt_check.state(state)
        self._figure_layers = None
        self._layer_cache_key = None
        self._redraw_from_cache()

    def _schedule_overlay_redraw(self) -> None:
        if self._overlay_after_id is not None:
            self.root.after_cancel(self._overlay_after_id)
        self._overlay_after_id = self.root.after(OVERLAY_DEBOUNCE_MS, self._redraw_overlays_only)

    def _packet_time_s(self) -> float:
        return float(self.time_us_var.get()) * 1e-6

    def _result_cache_key(self) -> tuple | None:
        if self.scenario is None:
            return None
        mode = self.mode_var.get()
        view = self.view_var.get()
        if mode == MODE_SINGLE and self.pulse_echo_result is not None:
            key: tuple = (mode, view, id(self.scenario), id(self.pulse_echo_result))
            if view == "Packet scene":
                key = key + (self._packet_time_s(),)
            return key
        if mode == MODE_AXIAL and self.axial_result is not None:
            return (mode, view, id(self.scenario), id(self.axial_result))
        if mode == MODE_CORROSION and self.corrosion_snapshot is not None:
            return (mode, view, id(self.scenario), id(self.corrosion_snapshot))
        return None

    def _has_cached_result(self) -> bool:
        return self._result_cache_key() is not None

    def _redraw_overlays_only(self) -> None:
        self._overlay_after_id = None
        if not self._has_cached_result():
            return
        cache_key = self._result_cache_key()
        if (
            self.view_var.get() != "Packet scene"
            and self._figure_layers is not None
            and self._layer_cache_key == cache_key
            and self._figure_layers.supports_overlay_toggle
            and self.canvas is not None
        ):
            self._figure_layers.apply_overlays(
                show_inferred=self.show_inferred_var.get(),
                show_ground_truth=self.show_gt_var.get(),
            )
            self.canvas.draw_idle()
            return
        self._redraw_from_cache()

    def _redraw_from_cache(self) -> None:
        if not self._has_cached_result():
            return
        view = self.view_var.get()
        try:
            layers = FigureLayers(fig=Figure())
            fig = build_figure_for_view(
                mode=self.mode_var.get(),
                view=view,
                scenario=self.scenario,
                pulse_echo_result=self.pulse_echo_result,
                axial_result=self.axial_result,
                corrosion_snapshot=self.corrosion_snapshot,
                show_inferred=self.show_inferred_var.get(),
                show_ground_truth=self.show_gt_var.get(),
                layers=layers,
                packet_time_s=self._packet_time_s(),
            )
            self._figure_layers = layers if layers.supports_overlay_toggle and view != "Packet scene" else None
            self._layer_cache_key = self._result_cache_key()
            self._show_matplotlib(fig)
        except Exception as exc:
            self._set_status(f"Redraw error: {exc}")

    def _show_matplotlib(self, fig: Figure) -> None:
        old_canvas = self.canvas
        old_figure = self.current_figure
        self.canvas = None

        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()
        if old_figure is not None and old_figure is not fig:
            plt.close(old_figure)

        self.current_figure = fig
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _post_to_ui(self, callback) -> None:
        """Schedule work on the Tk main thread (safe from background workers)."""
        self._ui_queue.put(callback)

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception as exc:
                self._set_status(f"UI error: {exc}")
        self.root.after(UI_QUEUE_POLL_MS, self._poll_ui_queue)

    def _snapshot_run_job(self) -> _RunJob:
        return _RunJob(
            scenario_path=Path(self.scenario_var.get()),
            mode=self.mode_var.get(),
            angle_deg=float(self.angle_var.get()),
            angle_step_deg=float(self.step_var.get()),
            corrosion_year_yr=float(self.corrosion_year_var.get()),
        )

    @staticmethod
    def _wall_profile_for_job(scenario: InternalScenario, job: _RunJob):
        if scenario.has_corrosion() and job.mode in (MODE_SINGLE, MODE_AXIAL):
            return scenario.effective_wall_profile(job.corrosion_year_yr)
        return scenario.wall_profile

    @staticmethod
    def _execute_run_job(job: _RunJob) -> _RunOutcome:
        scenario = load_internal_scenario(job.scenario_path)
        if job.mode == MODE_CORROSION:
            if not scenario.has_corrosion():
                raise ValueError("Scenario YAML must include a corrosion: block")
            engine = scenario.build_corrosion_engine()
            engine.run_to(job.corrosion_year_yr)
            snap = engine.snapshot()
            return _RunOutcome(
                scenario=scenario,
                mode=job.mode,
                pulse_echo_result=None,
                axial_result=None,
                corrosion_snapshot=snap,
                default_time_us=80.0,
                status=(
                    f"T={snap.time_yr:g} yr | pits={snap.n_pits} | "
                    f"uniform ML={snap.uniform_loss_m*1000:.2f} mm | "
                    f"min t_rem={snap.cloud.remaining_thickness_m.min()*1000:.2f} mm"
                ),
            )

        wall_profile = PipeSimGui._wall_profile_for_job(scenario, job)
        if job.mode == MODE_SINGLE:
            result = simulate_pulse_echo_2d(
                scenario.pipe,
                scenario.fluid,
                scenario.steel,
                scenario.transducer,
                scenario.timing,
                math.radians(job.angle_deg),
                z_m=0.0,
                wall_profile=wall_profile,
                echo=scenario.echo,
                inference=scenario.inference,
            )
            one_way_us = result.ground_truth_distance_m / result.fluid_vp * 1e6
            status = (
                f"θ={job.angle_deg:.1f}° | inferred={result.inferred_distance_m * 1000:.1f} mm | "
                f"true={result.ground_truth_distance_m * 1000:.1f} mm"
            )
            return _RunOutcome(
                scenario=scenario,
                mode=job.mode,
                pulse_echo_result=result,
                axial_result=None,
                corrosion_snapshot=None,
                default_time_us=one_way_us,
                status=status,
            )

        axial_result = simulate_axial_scan(
            scenario.pipe,
            scenario.fluid,
            scenario.steel,
            scenario.transducer,
            scenario.timing,
            scenario.z_stations(),
            angle_step_deg=job.angle_step_deg,
            z_step_m=float(scenario.scan.get("z_step_m", 0.01)),
            wall_profile=wall_profile,
            echo=scenario.echo,
            inference=scenario.inference,
        )
        status = (
            f"{len(axial_result.z_stations_m)} z stations | "
            f"{len(axial_result.angles_deg)} angles @ {job.angle_step_deg:g}° (ray + inference) | "
            f"mean inferred={axial_result.inferred_distance_m.mean() * 1000:.1f} mm"
        )
        return _RunOutcome(
            scenario=scenario,
            mode=job.mode,
            pulse_echo_result=None,
            axial_result=axial_result,
            corrosion_snapshot=None,
            default_time_us=80.0,
            status=status,
        )

    def _apply_run_outcome(self, outcome: _RunOutcome) -> None:
        self.scenario = outcome.scenario
        if outcome.scenario.has_corrosion():
            self._set_corrosion_year_state(enabled=True)
        if outcome.mode == MODE_SINGLE:
            self.pulse_echo_result = outcome.pulse_echo_result
            self.axial_result = None
            self.corrosion_snapshot = None
            self.time_us_var.set(round(outcome.default_time_us, 1))
        elif outcome.mode == MODE_AXIAL:
            self.pulse_echo_result = None
            self.axial_result = outcome.axial_result
            self.corrosion_snapshot = None
        else:
            self.pulse_echo_result = None
            self.axial_result = None
            self.corrosion_snapshot = outcome.corrosion_snapshot
        self._figure_layers = None
        self._layer_cache_key = None
        self._running = False
        self.run_btn.state(["!disabled"])
        self._update_time_control_state()
        self._redraw_from_cache()
        self._set_status(outcome.status)

    def _apply_run_failure(self, exc: Exception) -> None:
        self._running = False
        self.run_btn.state(["!disabled"])
        self._set_status(f"Run failed: {exc}")

    def _on_run(self) -> None:
        if self._running:
            return
        self._running = True
        self.run_btn.state(["disabled"])
        self._set_status("Running simulation...")
        job = self._snapshot_run_job()

        def worker() -> None:
            try:
                outcome = self._execute_run_job(job)
                self._post_to_ui(lambda: self._apply_run_outcome(outcome))
            except Exception as exc:
                # Bind exc as a default argument so the exception object
                # is captured now (the except-block local is cleared
                # after the block to avoid reference cycles).
                self._post_to_ui(lambda exc=exc: self._apply_run_failure(exc))

        t = threading.Thread(target=worker, daemon=False)
        # track worker threads so we can join them on shutdown
        self._worker_threads.append(t)
        t.start()

    def shutdown(self, *, join_timeout: float = 0.5) -> None:
        """Gracefully shutdown GUI: join workers, process UI queue, and destroy root.

        Must be called on the main thread.
        """
        # Prevent further runs
        self._running = False
        # Join worker threads briefly
        for t in list(self._worker_threads):
            try:
                t.join(join_timeout)
            except Exception:
                pass
        # Drain and execute any pending UI callbacks so they run while Tk is alive
        while True:
            try:
                cb = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cb()
            except Exception:
                pass
        # Clear widgets and destroy root
        try:
            self._clear_display_widgets()
        except Exception:
            pass
        try:
            # root.destroy must be called on main thread where mainloop runs
            self.root.destroy()
        except Exception:
            pass

    def _on_save(self) -> None:
        if self.current_figure is None:
            self._set_status("Nothing to save. Run a simulation first.")
            return
        prefix = self.prefix_var.get().strip() or DEFAULT_OUTPUT_PREFIX
        view_slug = self.view_var.get().lower().replace(" ", "_")
        out_path = OUTPUTS_DIR / f"{prefix}_{view_slug}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_figure.savefig(out_path, dpi=150)
        self._set_status(f"Saved: {out_path}")


def main() -> None:
    root = tk.Tk()
    gui = PipeSimGui(root)
    # Ensure our shutdown runs when window is closed
    root.protocol("WM_DELETE_WINDOW", gui.shutdown)
    try:
        root.mainloop()
    finally:
        try:
            gui.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
