from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from well_array_sim.internal.corrosion.config import CorrosionConfig, parse_corrosion_config
from well_array_sim.internal.corrosion.bridge import wall_profile_from_point_cloud
from well_array_sim.internal.corrosion.engine import CorrosionEngine
from well_array_sim.internal.pipe import BoreFluid, Pipe2D, Pipe3D, SteelWall
from well_array_sim.internal.transducer import PointTransducer
from well_array_sim.internal.wall_profile import (
    EchoConfig,
    WallProfile,
    build_wall_profile,
    parse_echo_config,
)

from well_array_sim.internal.axial_scan import z_stations_m

DEFAULT_STEER_ANGLE_DEG = 45.0
DEFAULT_ENGINE = "ray"
DEFAULT_CENTER_FREQ_HZ = 250_000.0
DEFAULT_BANDWIDTH = 0.7
DEFAULT_Z_STEP_M = 0.01
DEFAULT_ANGLE_STEP_DEG = 1.0


@dataclass(frozen=True)
class InferenceConfig:
    mode: str = "saft"
    r_min_m: float = 0.07
    r_max_m: float = 0.14
    r_step_m: float = 0.0005


@dataclass(frozen=True)
class InternalScenario:
    pipe: Pipe2D
    fluid: BoreFluid
    steel: SteelWall
    transducer: PointTransducer
    timing: dict[str, float]
    raw: dict[str, Any]
    wall_profile: WallProfile | None = None
    echo: EchoConfig | None = None
    inference: InferenceConfig | None = None
    corrosion: CorrosionConfig | None = None
    _corrosion_profile_cache: dict[float, WallProfile] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def engine(self) -> str:
        return str(self.raw.get("engine", DEFAULT_ENGINE))

    @property
    def pipe_3d(self) -> Pipe3D:
        length_m = self.raw.get("pipe", {}).get("length_m")
        return Pipe3D.from_profile(self.pipe, length_m=length_m)

    @property
    def scan(self) -> dict:
        return self.raw.get("scan", {})

    @property
    def z_step_m(self) -> float:
        return float(self.scan.get("z_step_m", DEFAULT_Z_STEP_M))

    @property
    def angle_step_deg(self) -> float:
        return float(self.scan.get("angle_step_deg", DEFAULT_ANGLE_STEP_DEG))

    def z_stations(self):
        cfg = self.scan
        z_end = cfg.get("z_end_m")
        return z_stations_m(
            length_m=self.pipe_3d.length_m,
            z_step_m=self.z_step_m,
            z_start_m=float(cfg.get("z_start_m", 0.0)),
            z_end_m=None if z_end is None else float(z_end),
        )

    def has_corrosion(self) -> bool:
        return self.corrosion is not None

    def build_corrosion_engine(self) -> CorrosionEngine:
        if self.corrosion is None:
            raise ValueError("Scenario has no corrosion block")
        return CorrosionEngine.from_pipe3d(self.pipe_3d, self.corrosion)

    def wall_profile_at_year(self, year_yr: float) -> WallProfile:
        """Evolve corrosion to year_yr and return WallProfile for acoustics."""
        if self.corrosion is None:
            raise ValueError("Scenario has no corrosion block")
        yr = float(year_yr)
        if yr in self._corrosion_profile_cache:
            return self._corrosion_profile_cache[yr]
        engine = self.build_corrosion_engine()
        engine.run_to(yr)
        profile = wall_profile_from_point_cloud(engine.cloud, self.pipe)
        self._corrosion_profile_cache[yr] = profile
        return profile

    def effective_wall_profile(self, corrosion_year_yr: float | None = None) -> WallProfile | None:
        """Wall profile for simulation: corrosion at year, else static YAML profile."""
        if corrosion_year_yr is not None and self.corrosion is not None:
            return self.wall_profile_at_year(corrosion_year_yr)
        return self.wall_profile


def load_scenario_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_transducer(data: dict[str, Any]) -> PointTransducer:
    td = data.get("transducer")
    if isinstance(td, dict):
        return PointTransducer(
            center_freq_hz=float(td.get("center_freq_hz", DEFAULT_CENTER_FREQ_HZ)),
            bandwidth=float(td.get("bandwidth", DEFAULT_BANDWIDTH)),
            position_xy=(
                float(td.get("x_m", 0.0)),
                float(td.get("y_m", 0.0)),
            ),
        )
    return PointTransducer(
        center_freq_hz=DEFAULT_CENTER_FREQ_HZ,
        bandwidth=DEFAULT_BANDWIDTH,
    )


def parse_inference_config(data: dict[str, Any], *, nominal_inner_radius_m: float) -> InferenceConfig:
    cfg = data.get("inference")
    if not isinstance(cfg, dict):
        return InferenceConfig(
            r_min_m=max(0.01, nominal_inner_radius_m * 0.65),
            r_max_m=nominal_inner_radius_m * 1.35,
            r_step_m=0.0005,
        )
    return InferenceConfig(
        mode=str(cfg.get("mode", "saft")),
        r_min_m=float(cfg.get("r_min_m", max(0.01, nominal_inner_radius_m * 0.65))),
        r_max_m=float(cfg.get("r_max_m", nominal_inner_radius_m * 1.35)),
        r_step_m=float(cfg.get("r_step_m", 0.0005)),
    )


def parse_internal_scenario(data: dict[str, Any]) -> InternalScenario:
    pipe_cfg = data["pipe"]
    fluid_cfg = data["medium"]["bore_fluid"]
    steel_cfg = data.get("materials", {}).get("steel", {"rho": 7850, "vp": 5778})
    pipe = Pipe2D(
        inner_radius_m=float(pipe_cfg["inner_radius_m"]),
        wall_thickness_m=float(pipe_cfg["wall_thickness_m"]),
    )
    fluid = BoreFluid(rho=float(fluid_cfg["rho"]), vp=float(fluid_cfg["vp"]))
    steel = SteelWall(rho=float(steel_cfg["rho"]), vp=float(steel_cfg["vp"]))
    transducer = parse_transducer(data)
    timing = {k: float(v) for k, v in data["timing"].items()}
    z_stations = z_stations_m(
        length_m=float(pipe_cfg.get("length_m", 0.4)),
        z_step_m=float(data.get("scan", {}).get("z_step_m", 0.01)),
        z_start_m=float(data.get("scan", {}).get("z_start_m", 0.0)),
        z_end_m=None
        if data.get("scan", {}).get("z_end_m") is None
        else float(data["scan"]["z_end_m"]),
    )
    wall_profile = build_wall_profile(
        data,
        z_stations=z_stations,
        nominal_inner_radius_m=pipe.inner_radius_m,
    )
    echo = parse_echo_config(data)
    inference = parse_inference_config(data, nominal_inner_radius_m=pipe.inner_radius_m)
    corrosion = parse_corrosion_config(data)
    return InternalScenario(
        pipe=pipe,
        fluid=fluid,
        steel=steel,
        transducer=transducer,
        timing=timing,
        raw=data,
        wall_profile=wall_profile,
        echo=echo,
        inference=inference,
        corrosion=corrosion,
    )


def load_internal_scenario(path: Path | str) -> InternalScenario:
    return parse_internal_scenario(load_scenario_yaml(path))
