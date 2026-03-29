"""Tests for EvacuationGeodesic -- optimal evacuation paths (§4.4).

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-29
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdsafe.core.evacuation_geodesic import EvacuationGeodesic


class TestDistanceMap:
    """Distance map computation via multi-source Dijkstra."""

    def test_exit_has_zero_distance(self) -> None:
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((10, 10))
        exits = [np.array([5.0, 5.0])]
        dist = geo.compute_distance_map(density, exits)
        assert dist[5, 5] == 0.0

    def test_distance_increases_from_exit(self) -> None:
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((20, 20))
        exits = [np.array([10.0, 10.0])]
        dist = geo.compute_distance_map(density, exits)
        # Distance at (0,0) should be > distance at (5,5)
        assert dist[0, 0] > dist[5, 5]
        assert dist[5, 5] > dist[10, 10]

    def test_high_density_blocks_path(self) -> None:
        """A wall of critical density should make cells impassable."""
        geo = EvacuationGeodesic(dx_m=1.0, rho_critical=6.0)
        density = np.zeros((10, 20))
        # Wall of density 7.0 at x=10
        density[:, 10] = 7.0
        exits = [np.array([15.0, 5.0])]
        dist = geo.compute_distance_map(density, exits)
        # Cells on the far side of the wall should be unreachable
        assert dist[5, 5] == np.inf

    def test_obstacle_mask_blocks_path(self) -> None:
        """An obstacle mask should make cells impassable."""
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((10, 20))
        obstacle = np.zeros((10, 20), dtype=bool)
        obstacle[:, 10] = True  # wall at x=10
        exits = [np.array([15.0, 5.0])]
        dist = geo.compute_distance_map(density, exits, obstacle_mask=obstacle)
        # Cells behind the wall should be unreachable
        assert dist[5, 5] == np.inf

    def test_multiple_exits(self) -> None:
        """With two exits, each cell maps to the nearest exit."""
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((10, 20))
        exits = [np.array([0.0, 5.0]), np.array([19.0, 5.0])]
        dist = geo.compute_distance_map(density, exits)
        # Center cell should have lower distance than a far corner
        d_center = dist[5, 10]  # y=5, x=10
        d_corner = dist[0, 10]  # y=0, x=10 (far from both exits in y)
        assert d_center < d_corner, (
            f"Center ({d_center:.2f}) should be closer than corner ({d_corner:.2f})"
        )


class TestFindPath:
    """Path finding from start to nearest exit."""

    def test_straight_path_no_obstacles(self) -> None:
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((10, 20))
        exits = [np.array([19.0, 5.0])]
        result = geo.find_path(np.array([0.0, 5.0]), density, exits)

        assert result.evac_feasible
        assert result.travel_time_s > 0
        assert result.path_length_m > 0
        # Path should move toward the exit (increasing x)
        assert result.path[-1, 0] > result.path[0, 0]

    def test_path_avoids_dense_zone(self) -> None:
        """Path should route around a dense zone, not through it."""
        geo = EvacuationGeodesic(dx_m=0.5, rho_critical=6.0)
        density = np.zeros((20, 40))
        # Dense zone blocking direct path at x=10, y=5..15
        density[5:15, 18:22] = 5.5  # high but not critical

        exits = [np.array([19.0, 5.0])]
        start = np.array([0.5, 5.0])

        result = geo.find_path(start, density, exits)
        assert result.evac_feasible
        assert result.travel_time_s > 0

    def test_infeasible_path(self) -> None:
        """When exit is completely blocked by critical density, path is infeasible."""
        geo = EvacuationGeodesic(dx_m=1.0, rho_critical=6.0)
        density = np.zeros((10, 20))
        # Complete wall of critical density
        density[:, 10] = 7.0
        exits = [np.array([15.0, 5.0])]
        result = geo.find_path(np.array([5.0, 5.0]), density, exits)
        assert not result.evac_feasible
        assert result.travel_time_s == np.inf

    def test_path_length_reasonable(self) -> None:
        """Path length should be close to straight-line distance in free space."""
        geo = EvacuationGeodesic(dx_m=0.5)
        density = np.zeros((20, 40))
        exits = [np.array([19.0, 5.0])]
        start = np.array([0.5, 5.0])
        result = geo.find_path(start, density, exits)

        straight_line = np.linalg.norm(start - exits[0])
        # Path should be at most 50% longer than straight line
        assert result.path_length_m < straight_line * 1.5, (
            f"Path ({result.path_length_m:.1f}m) much longer than "
            f"straight line ({straight_line:.1f}m)"
        )


class TestEvacuationPhysics:
    """Physical consistency of evacuation results."""

    def test_higher_density_longer_travel_time(self) -> None:
        """Higher density → slower walking → longer travel time."""
        geo = EvacuationGeodesic(dx_m=1.0)
        exits = [np.array([19.0, 5.0])]
        start = np.array([0.0, 5.0])

        r_free = geo.find_path(start, np.zeros((10, 20)), exits)
        r_dense = geo.find_path(start, np.full((10, 20), 3.0), exits)

        assert r_dense.travel_time_s > r_free.travel_time_s, (
            f"Dense ({r_dense.travel_time_s:.1f}s) should take longer "
            f"than free ({r_free.travel_time_s:.1f}s)"
        )

    def test_travel_time_scales_with_distance(self) -> None:
        """Doubling distance should roughly double travel time."""
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((10, 40))

        r_near = geo.find_path(np.array([10.0, 5.0]), density,
                               [np.array([19.0, 5.0])])
        r_far = geo.find_path(np.array([0.0, 5.0]), density,
                              [np.array([19.0, 5.0])])

        ratio = r_far.travel_time_s / r_near.travel_time_s
        assert 1.5 < ratio < 3.0, (
            f"Travel time ratio {ratio:.2f} not proportional to distance"
        )

    def test_bottleneck_density_at_start(self) -> None:
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.full((10, 20), 2.5)
        exits = [np.array([19.0, 5.0])]
        result = geo.find_path(np.array([5.0, 5.0]), density, exits)
        assert result.bottleneck_rho == pytest.approx(2.5)

    def test_distance_map_shape(self) -> None:
        geo = EvacuationGeodesic(dx_m=1.0)
        density = np.zeros((15, 25))
        exits = [np.array([10.0, 7.0])]
        dist = geo.compute_distance_map(density, exits)
        assert dist.shape == (15, 25)
        assert dist.dtype == np.float64
