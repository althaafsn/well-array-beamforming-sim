from __future__ import annotations

import numpy as np

from well_array_sim.internal.geometry3d import (
    axial_scan_pointcloud_3d,
    beam_line_segment,
    beam_pointcloud_2d,
    beam_pointcloud_3d,
    circle_pointcloud_2d,
    cylinder_pointcloud,
    cylinder_surface,
)
from well_array_sim.internal.pipe import Pipe2D, Pipe3D, default_display_length_m


def test_pipe3d_wraps_2d_radii() -> None:
    pipe2d = Pipe2D(inner_radius_m=0.109, wall_thickness_m=0.013)
    pipe3d = Pipe3D(profile=pipe2d, length_m=0.4)
    assert pipe3d.inner_radius_m == pipe2d.inner_radius_m
    assert pipe3d.outer_radius_m == pipe2d.outer_radius_m
    assert pipe3d.length_m == 0.4
    assert pipe3d.z_center_m == 0.2


def test_default_display_length_scales_with_radius() -> None:
    r = 0.109
    assert default_display_length_m(r) >= 2.0 * r


def test_cylinder_surface_shapes() -> None:
    pipe3d = Pipe3D.from_profile(Pipe2D(0.1, 0.01), length_m=0.3)
    x, y, z = cylinder_surface(pipe3d, radius_m=pipe3d.inner_radius_m, n_theta=16, n_z=5)
    assert x.shape == y.shape == z.shape == (5, 16)
    assert np.allclose(x[:, 0] ** 2 + y[:, 0] ** 2, 0.1**2, atol=1e-9)
    assert np.allclose(z[0, :], 0.0)
    assert np.allclose(z[-1, :], 0.3)


def test_beam_line_is_axially_invariant() -> None:
    seg = beam_line_segment(
        beam_dir_xy=np.array([1.0, 0.0]),
        wall_distance_m=0.1,
        z0=0.15,
        z1=0.25,
    )
    assert seg.shape == (2, 3)
    assert np.isclose(seg[0, 0], 0.0) and np.isclose(seg[1, 0], 0.1)
    assert np.allclose(seg[:, 1], 0.0)
    assert np.allclose(seg[:, 2], [0.15, 0.25])


def test_cylinder_pointcloud_shape() -> None:
    pipe3d = Pipe3D.from_profile(Pipe2D(0.1, 0.01), length_m=0.3)
    pts = cylinder_pointcloud(pipe3d, radius_m=0.1, n_theta=8, n_z=4)
    assert pts.shape == (32, 3)


def test_beam_pointcloud_2d_endpoints() -> None:
    pts = beam_pointcloud_2d(beam_dir_xy=np.array([1.0, 0.0]), wall_distance_m=0.1, n_points=5)
    assert pts.shape == (5, 2)
    assert np.allclose(pts[0], [0.0, 0.0])
    assert np.isclose(pts[-1, 0], 0.1)


def test_axial_scan_pointcloud_3d_shape() -> None:
    z_vals = np.array([0.0, 0.1, 0.2])
    angles_deg = np.array([0.0, 90.0, 180.0])
    radius_m = np.full((3, 3), 0.109)
    pts = axial_scan_pointcloud_3d(z_vals, angles_deg, radius_m)
    assert pts.shape == (9, 3)
    assert np.allclose(pts[0], [0.109, 0.0, 0.0], atol=1e-9)
    assert np.isclose(pts[1, 2], 0.0)
    assert np.isclose(pts[3, 2], 0.1)
