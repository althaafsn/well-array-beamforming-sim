"""
Lagrangian wave packets for 2D ray acoustics (φ = 0 slice).

Each TX time sample launches one packet with the full RF sample weight (Gaussian
envelope × carrier). Packets superpose coherently at the receiver into a delayed
copy of the tone-burst transmit pulse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from well_array_sim.core.pulse import gaussian_envelope


@dataclass(frozen=True)
class WavePacket:
    """One time-sample of the outgoing pulse traveling along a ray."""

    launch_time_s: float
    amplitude: float
    envelope_amplitude: float
    direction_xy: np.ndarray
    wall_distance_m: float
    one_way_time_s: float
    round_trip_time_s: float

    @property
    def arrival_time_s(self) -> float:
        return self.launch_time_s + self.round_trip_time_s


@dataclass(frozen=True)
class PacketTrajectory:
    """Per-packet data for time-step visualization."""

    launch_time_s: np.ndarray
    wall_distance_m: np.ndarray
    one_way_time_s: np.ndarray
    direction_xy: np.ndarray
    amplitude: np.ndarray
    envelope_amplitude: np.ndarray


def beam_direction_xy(theta_rad: float) -> np.ndarray:
    return np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=float)


def spawn_packets_from_pulse(
    time_s: np.ndarray,
    p_tx: np.ndarray,
    *,
    theta_rad: float,
    wall_distance_m: float,
    fluid_vp: float,
    center_freq_hz: float,
    bandwidth: float,
    tp0_s: float = 8e-6,
) -> list[WavePacket]:
    """Launch one packet per TX sample; weight follows the Gaussian envelope."""
    direction = beam_direction_xy(theta_rad)
    one_way_s = wall_distance_m / fluid_vp
    round_trip_s = 2.0 * one_way_s
    envelope = gaussian_envelope(
        time_s,
        f0_hz=center_freq_hz,
        bandwidth=bandwidth,
        tp0_s=tp0_s,
    )
    return [
        WavePacket(
            launch_time_s=float(t),
            amplitude=float(a),
            envelope_amplitude=float(env),
            direction_xy=direction,
            wall_distance_m=wall_distance_m,
            one_way_time_s=one_way_s,
            round_trip_time_s=round_trip_s,
        )
        for t, a, env in zip(time_s, p_tx, envelope)
    ]


def trajectory_from_packets(packets: list[WavePacket]) -> PacketTrajectory:
    if not packets:
        return PacketTrajectory(
            launch_time_s=np.empty(0),
            wall_distance_m=np.empty(0),
            one_way_time_s=np.empty(0),
            direction_xy=np.empty((0, 2)),
            amplitude=np.empty(0),
            envelope_amplitude=np.empty(0),
        )
    return PacketTrajectory(
        launch_time_s=np.array([p.launch_time_s for p in packets], dtype=float),
        wall_distance_m=np.array([p.wall_distance_m for p in packets], dtype=float),
        one_way_time_s=np.array([p.one_way_time_s for p in packets], dtype=float),
        direction_xy=np.vstack([p.direction_xy for p in packets]),
        amplitude=np.array([p.amplitude for p in packets], dtype=float),
        envelope_amplitude=np.array([p.envelope_amplitude for p in packets], dtype=float),
    )


def _arrival_index(time_s: np.ndarray, arrival_time_s: float) -> int:
    dt_s = float(time_s[1] - time_s[0]) if len(time_s) > 1 else 1e-6
    t0_s = float(time_s[0]) if len(time_s) else 0.0
    return int(round((arrival_time_s - t0_s) / dt_s))


def synthesize_received_trace(
    time_s: np.ndarray,
    packets: list[WavePacket],
    *,
    reflection_coeff: float,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    """
    Sum packet arrivals into p_rx(t).

    Each packet deposits its signed RF weight at its arrival time, producing a
    delayed, scaled copy of the transmit tone burst (oscillating echo).
    """
    if not packets:
        return np.zeros(len(time_s), dtype=float)
    p_rx = np.zeros(len(time_s), dtype=float)
    scale = reflection_coeff * amplitude_scale

    for packet in packets:
        idx = _arrival_index(time_s, packet.arrival_time_s)
        if 0 <= idx < len(p_rx):
            p_rx[idx] += scale * packet.amplitude

    return p_rx


def add_trace_noise(
    trace: np.ndarray,
    *,
    snr_db: float | None,
    rng: np.random.Generator | None,
    reference_peak: float | None = None,
) -> np.ndarray:
    if snr_db is None or rng is None:
        return trace
    signal_peak = reference_peak if reference_peak is not None else float(np.max(np.abs(trace)))
    if signal_peak <= 0:
        return trace
    noise_rms = signal_peak / (10.0 ** (snr_db / 20.0))
    return trace + rng.normal(0.0, noise_rms, size=trace.shape)


def packet_positions_at_time(
    trajectory: PacketTrajectory,
    t_s: float,
    *,
    fluid_vp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Positions of active packets at time t_s.

    Returns (x, y, envelope amplitudes) for packets in flight (outbound or return).
    """
    xs: list[float] = []
    ys: list[float] = []
    amps: list[float] = []

    for i in range(len(trajectory.launch_time_s)):
        t0 = float(trajectory.launch_time_s[i])
        if t_s < t0:
            continue
        s = float(trajectory.wall_distance_m[i])
        direction = trajectory.direction_xy[i]
        dir_u = direction / max(np.linalg.norm(direction), 1e-12)
        wall_pt = s * dir_u
        one_way = s / fluid_vp
        return_time = t0 + 2.0 * one_way
        if t_s > return_time:
            continue

        dt = t_s - t0
        if dt <= one_way:
            dist = fluid_vp * dt
            pos = dist * dir_u
        else:
            dist_back = fluid_vp * (dt - one_way)
            pos = wall_pt - dist_back * dir_u

        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        amps.append(float(trajectory.envelope_amplitude[i]))

    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.array(xs), np.array(ys), np.array(amps)
