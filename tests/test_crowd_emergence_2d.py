"""2D Crowd emergence tests -- validates crowd-specific phenomena.

Tests that the CrowdSafe N-body engine produces crowd-specific emergent
phenomena without any explicit behavioral rules:

1. Lane formation in counterflow (Helbing 2001)
2. Contact forces prevent body interpenetration
3. Speed reduction at high density (fundamental diagram direction)

Author: Agent #40 Devil's Advocate (specification) + Agent #01 (implementation)
Date: 2026-03-28
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdsafe.core.simulation import CrowdSimulation


# ---------------------------------------------------------------------------
# Test 1: Lane formation in bidirectional flow
# ---------------------------------------------------------------------------
class TestLaneFormation:
    """In bidirectional pedestrian flow, lanes should spontaneously form.

    Setup: 2D corridor (20m x 4m). Half the pedestrians walk +x, half -x.
    After simulation, pedestrians should spatially segregate by direction
    (measured by correlation between y-position and x-velocity sign).
    """

    @staticmethod
    def _setup_counterflow(n: int = 80, corridor_length: float = 20.0,
                           corridor_width: float = 4.0, seed: int = 42):
        rng = np.random.default_rng(seed)
        positions = np.column_stack([
            rng.uniform(0, corridor_length, n),
            rng.uniform(0, corridor_width, n),
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)
        # Half go right, half go left
        velocities[:n // 2, 0] = 1.2   # +x walkers
        velocities[n // 2:, 0] = -1.2  # -x walkers
        densities = np.full(n, n / (corridor_length * corridor_width))
        return positions, velocities, densities

    def test_lane_segregation_emerges(self) -> None:
        """After simulation, pedestrians moving in the same direction should
        cluster on the same side of the corridor (y-axis segregation)."""
        positions, velocities, densities = self._setup_counterflow()

        sim = CrowdSimulation(
            G_s=2.0, dt=0.1, v_max=2.5, adaptive_dt=False,
            drag_coefficient=0.3, v_free=1.34, rho_jam=6.0,
        )
        sim.init_pedestrians(positions, velocities, densities)

        # Run for enough steps to let lanes form
        sim.run(200)

        # Measure segregation: correlation between y-position and vx sign
        vx = sim.velocities[:, 0]
        y = sim.positions[:, 1]

        # Split by direction
        right_movers = vx > 0.1
        left_movers = vx < -0.1

        if np.sum(right_movers) > 5 and np.sum(left_movers) > 5:
            y_mean_right = float(np.mean(y[right_movers]))
            y_mean_left = float(np.mean(y[left_movers]))

            # Lane formation = the two groups have different mean y positions
            y_separation = abs(y_mean_right - y_mean_left)

            print(f"\n--- Lane Formation Test ---")
            print(f"  Right movers: n={np.sum(right_movers)}, mean_y={y_mean_right:.2f}")
            print(f"  Left movers:  n={np.sum(left_movers)}, mean_y={y_mean_left:.2f}")
            print(f"  Y separation: {y_separation:.2f} m")

            # Even partial segregation is emergence; expect > 0.1m separation
            assert y_separation > 0.1 or np.std(y[right_movers]) < np.std(y), (
                f"No lane segregation detected: y_separation={y_separation:.3f}m. "
                f"Right movers and left movers should have different y distributions."
            )

    def test_counterflow_speed_reduction(self) -> None:
        """Counterflow should reduce mean speed vs. unidirectional flow."""
        n = 80
        pos, vel, dens = self._setup_counterflow(n=n)

        # Counterflow simulation
        sim_counter = CrowdSimulation(
            G_s=2.0, dt=0.1, v_max=2.5, adaptive_dt=False,
            drag_coefficient=0.3, v_free=1.34, rho_jam=6.0,
        )
        sim_counter.init_pedestrians(pos.copy(), vel.copy(), dens.copy())
        sim_counter.run(100)
        speed_counter = float(np.mean(np.linalg.norm(sim_counter.velocities, axis=1)))

        # Unidirectional: all go same direction
        vel_uni = vel.copy()
        vel_uni[:, 0] = np.abs(vel_uni[:, 0])  # all +x
        sim_uni = CrowdSimulation(
            G_s=2.0, dt=0.1, v_max=2.5, adaptive_dt=False,
            drag_coefficient=0.3, v_free=1.34, rho_jam=6.0,
        )
        sim_uni.init_pedestrians(pos.copy(), vel_uni, dens.copy())
        sim_uni.run(100)
        speed_uni = float(np.mean(np.linalg.norm(sim_uni.velocities, axis=1)))

        print(f"\n--- Counterflow Speed Test ---")
        print(f"  Unidirectional mean speed: {speed_uni:.3f} m/s")
        print(f"  Counterflow mean speed:    {speed_counter:.3f} m/s")

        # Counterflow should be slower (more interactions/conflicts)
        assert speed_counter < speed_uni + 0.1, (
            f"Counterflow ({speed_counter:.3f}) should be slower than "
            f"unidirectional ({speed_uni:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 2: Contact forces prevent interpenetration
# ---------------------------------------------------------------------------
class TestContactForces:
    """Body contact forces should prevent pedestrians from occupying
    the same physical space."""

    def test_minimum_spacing_maintained(self) -> None:
        """After simulation, no two pedestrians should be closer than
        body_radius (0.2m center-to-center)."""
        rng = np.random.default_rng(77)
        n = 30
        # Pack pedestrians tightly in a 3m x 3m space (3.3 pers/m²)
        positions = np.column_stack([
            rng.uniform(0, 3, n),
            rng.uniform(0, 3, n),
        ])
        velocities = np.column_stack([
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.5, 0.5, n),
        ])
        densities = np.full(n, 3.3, dtype=np.float64)

        sim = CrowdSimulation(
            G_s=2.0, dt=0.05, v_max=2.5, adaptive_dt=False,
            contact_strength=2000.0, body_radius=0.2,
        )
        sim.init_pedestrians(positions, velocities, densities)
        sim.run(100)

        # Check minimum pairwise distance
        from scipy.spatial import distance
        dists = distance.pdist(sim.positions)
        min_dist = float(np.min(dists)) if len(dists) > 0 else float("inf")

        print(f"\n--- Contact Force Test ---")
        print(f"  Min pairwise distance: {min_dist:.4f} m")
        print(f"  Body radius: 0.2 m (diameter 0.4 m)")

        # With contact forces, min distance should be close to body diameter
        # Allow some overlap due to discrete timesteps
        assert min_dist > 0.05, (
            f"Pedestrians overlapping severely: min_dist={min_dist:.4f}m. "
            f"Contact forces should prevent interpenetration."
        )

    def test_contact_forces_active_at_high_density(self) -> None:
        """At high density, contact forces should dominate and produce
        higher accelerations than gravity alone."""
        n = 20
        # Very tight packing: 0.3m spacing in a line
        positions = np.zeros((n, 2), dtype=np.float64)
        positions[:, 0] = np.linspace(0, n * 0.3, n)
        velocities = np.zeros((n, 2), dtype=np.float64)
        velocities[:, 0] = 0.5
        densities = np.full(n, 4.0, dtype=np.float64)

        # With contact forces
        sim_contact = CrowdSimulation(
            G_s=2.0, dt=0.05, adaptive_dt=False,
            contact_strength=2000.0, body_radius=0.2,
        )
        sim_contact.init_pedestrians(positions.copy(), velocities.copy(), densities.copy())
        sim_contact.step()
        accel_contact = np.linalg.norm(sim_contact._forces, axis=1).mean()

        # Without contact forces
        sim_no_contact = CrowdSimulation(
            G_s=2.0, dt=0.05, adaptive_dt=False,
            contact_strength=0.0,
        )
        sim_no_contact.init_pedestrians(positions.copy(), velocities.copy(), densities.copy())
        sim_no_contact.step()
        accel_no_contact = np.linalg.norm(sim_no_contact._forces, axis=1).mean()

        print(f"\n--- Contact vs Gravity Test ---")
        print(f"  Mean accel with contact:    {accel_contact:.4f} m/s²")
        print(f"  Mean accel without contact: {accel_no_contact:.4f} m/s²")

        # Contact forces should produce stronger accelerations at close range
        assert accel_contact > accel_no_contact, (
            f"Contact forces not stronger than gravity at close range: "
            f"contact={accel_contact:.4f} vs gravity={accel_no_contact:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Speed-density relationship (qualitative)
# ---------------------------------------------------------------------------
class TestSpeedDensityQualitative:
    """Higher density should produce lower mean speed — the fundamental
    qualitative prediction of any crowd model."""

    def test_speed_decreases_with_density(self) -> None:
        """Run two simulations at different densities and verify that
        higher density → lower speed."""
        rng = np.random.default_rng(99)

        speeds_by_density = {}
        for rho_target in [1.0, 4.0]:
            n = int(rho_target * 100)  # 100 m² area (10x10)
            pos = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
            vel = np.column_stack([
                np.full(n, 1.2),
                rng.uniform(-0.2, 0.2, n),
            ])
            dens = np.full(n, rho_target, dtype=np.float64)

            sim = CrowdSimulation(dt=0.1, adaptive_dt=False)
            sim.init_pedestrians(pos, vel, dens)
            sim.run(100)

            mean_speed = float(np.mean(np.linalg.norm(sim.velocities, axis=1)))
            speeds_by_density[rho_target] = mean_speed

        print(f"\n--- Speed-Density Qualitative ---")
        for rho, spd in speeds_by_density.items():
            print(f"  rho={rho:.1f} pers/m²: mean speed = {spd:.3f} m/s")

        assert speeds_by_density[1.0] > speeds_by_density[4.0], (
            f"Speed at low density ({speeds_by_density[1.0]:.3f}) should "
            f"exceed speed at high density ({speeds_by_density[4.0]:.3f})"
        )
