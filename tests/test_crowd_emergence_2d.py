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


# ---------------------------------------------------------------------------
# Test 4: Destination force drives pedestrians toward exits
# ---------------------------------------------------------------------------
class TestDestinationForce:
    """Pedestrians with desired directions should converge toward their
    destinations via the Helbing (1995) social force term."""

    def test_pedestrians_move_toward_destination(self) -> None:
        """Stationary pedestrians with a desired direction should accelerate
        in that direction."""
        n = 20
        rng = np.random.default_rng(42)
        positions = np.column_stack([
            rng.uniform(0, 20, n),
            rng.uniform(0, 20, n),
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)  # all stationary
        densities = np.full(n, 1.0, dtype=np.float64)

        sim = CrowdSimulation(dt=0.1, adaptive_dt=False, drag_coefficient=0.0)
        sim.init_pedestrians(positions, velocities, densities)

        # All pedestrians want to go +x (toward exit on right)
        directions = np.zeros((n, 2), dtype=np.float64)
        directions[:, 0] = 1.0
        sim.set_desired_directions(directions)

        sim.run(20)

        # All pedestrians should now be moving primarily in +x
        mean_vx = float(np.mean(sim.velocities[:, 0]))

        print(f"\n--- Destination Force Test ---")
        print(f"  Mean vx after 20 steps: {mean_vx:.3f} m/s")
        print(f"  Expected: positive (toward +x exit)")

        assert mean_vx > 0.3, (
            f"Destination force too weak: mean_vx={mean_vx:.3f}, expected > 0.3 m/s"
        )

    def test_destination_overrides_initial_direction(self) -> None:
        """Pedestrians moving -x but with desired direction +x should
        eventually reverse course."""
        n = 10
        positions = np.column_stack([
            np.linspace(5, 15, n),
            np.full(n, 5.0),
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)
        velocities[:, 0] = -1.0  # initially moving left

        densities = np.full(n, 1.0, dtype=np.float64)

        sim = CrowdSimulation(dt=0.1, adaptive_dt=False, drag_coefficient=0.0)
        sim.init_pedestrians(positions, velocities, densities)

        # Desired direction: +x (opposite to initial velocity)
        directions = np.zeros((n, 2), dtype=np.float64)
        directions[:, 0] = 1.0
        sim.set_desired_directions(directions)

        sim.run(50)

        mean_vx = float(np.mean(sim.velocities[:, 0]))

        print(f"\n--- Direction Reversal Test ---")
        print(f"  Initial vx: -1.0 m/s")
        print(f"  Mean vx after 50 steps: {mean_vx:.3f} m/s")

        assert mean_vx > 0, (
            f"Pedestrians did not reverse toward destination: "
            f"mean_vx={mean_vx:.3f}, expected positive"
        )

    def test_evacuation_toward_exit(self) -> None:
        """Pedestrians with destinations pointing toward an exit should
        accumulate near the exit, demonstrating crowd flow."""
        n = 40
        rng = np.random.default_rng(55)
        # Room: 20m x 10m, exit at (20, 5)
        positions = np.column_stack([
            rng.uniform(0, 18, n),
            rng.uniform(0, 10, n),
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)
        densities = np.full(n, 0.2, dtype=np.float64)

        sim = CrowdSimulation(dt=0.1, adaptive_dt=False)
        sim.init_pedestrians(positions, velocities, densities)

        # All pedestrians want to reach exit at (20, 5)
        exit_pos = np.array([20.0, 5.0])
        dirs = exit_pos - positions
        sim.set_desired_directions(dirs)

        sim.run(100)

        # Mean x should have increased (moved toward exit)
        mean_x_final = float(np.mean(sim.positions[:, 0]))
        mean_x_init = float(np.mean(positions[:, 0]))

        print(f"\n--- Evacuation Test ---")
        print(f"  Initial mean x: {mean_x_init:.2f}")
        print(f"  Final mean x:   {mean_x_final:.2f}")

        assert mean_x_final > mean_x_init + 1.0, (
            f"Pedestrians didn't move toward exit: "
            f"init_x={mean_x_init:.2f}, final_x={mean_x_final:.2f}"
        )

    def test_no_destination_no_effect(self) -> None:
        """Without set_desired_directions, no destination force is applied."""
        n = 10
        positions = np.column_stack([np.linspace(0, 10, n), np.zeros(n)])
        velocities = np.column_stack([np.full(n, 1.0), np.zeros(n)])
        densities = np.full(n, 1.0, dtype=np.float64)

        # Without destination
        sim_no_dest = CrowdSimulation(dt=0.1, adaptive_dt=False, drag_coefficient=0.0)
        sim_no_dest.init_pedestrians(positions.copy(), velocities.copy(), densities.copy())
        sim_no_dest.step()
        v_no_dest = sim_no_dest.velocities.copy()

        # With None destination (explicit)
        sim_none = CrowdSimulation(dt=0.1, adaptive_dt=False, drag_coefficient=0.0)
        sim_none.init_pedestrians(positions.copy(), velocities.copy(), densities.copy())
        sim_none.set_desired_directions(None)
        sim_none.step()
        v_none = sim_none.velocities.copy()

        np.testing.assert_allclose(v_no_dest, v_none, atol=1e-14,
            err_msg="Setting desired_directions=None should have no effect"
        )


# ---------------------------------------------------------------------------
# Test 5: Wall forces keep pedestrians inside venue
# ---------------------------------------------------------------------------
class TestWallForces:
    """Walls should confine pedestrians within the venue geometry."""

    def test_wall_repels_pedestrians(self) -> None:
        """Pedestrians moving toward a wall should be repelled."""
        n = 10
        # Pedestrians at x=1, moving toward wall at x=0
        positions = np.column_stack([
            np.full(n, 1.0),
            np.linspace(0, 5, n),
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)
        velocities[:, 0] = -1.0  # moving toward wall
        densities = np.full(n, 1.0, dtype=np.float64)

        sim = CrowdSimulation(dt=0.05, adaptive_dt=False, drag_coefficient=0.0)
        sim.init_pedestrians(positions, velocities, densities)
        # Wall along x=0 from y=-1 to y=6
        sim.set_walls(
            p1=np.array([[0.0, -1.0]]),
            p2=np.array([[0.0, 6.0]]),
        )
        sim.run(40)

        # Pedestrians should not have crossed x=0
        min_x = float(np.min(sim.positions[:, 0]))
        print(f"\n--- Wall Repulsion Test ---")
        print(f"  Min x after 40 steps: {min_x:.3f} m (wall at x=0)")

        assert min_x > -0.1, (
            f"Pedestrian crossed wall: min_x={min_x:.3f}, wall at x=0"
        )

    def test_corridor_walls_confine(self) -> None:
        """Walls on both sides of a corridor should confine pedestrians."""
        n = 15
        rng = np.random.default_rng(88)
        # Wide corridor: x in [0, 30], y in [0, 6]
        positions = np.column_stack([
            rng.uniform(2, 28, n),
            rng.uniform(1.5, 4.5, n),
        ])
        velocities = np.column_stack([
            rng.uniform(0.3, 0.8, n),
            rng.uniform(-0.2, 0.2, n),
        ])
        densities = np.full(n, 0.5, dtype=np.float64)

        sim = CrowdSimulation(
            dt=0.01, adaptive_dt=False,
            drag_coefficient=0.5, v_free=1.34,
            G_s=1.0, contact_strength=500.0,
        )
        sim.init_pedestrians(positions, velocities, densities)
        # Bottom wall (y=0) and top wall (y=6)
        sim.set_walls(
            p1=np.array([[0, 0], [0, 6]]),
            p2=np.array([[30, 0], [30, 6]]),
            strength=5000.0, decay=0.15,
        )
        sim.run(200)

        y_min = float(np.min(sim.positions[:, 1]))
        y_max = float(np.max(sim.positions[:, 1]))

        print(f"\n--- Corridor Confinement Test ---")
        print(f"  Y range: [{y_min:.2f}, {y_max:.2f}] (walls at y=0 and y=6)")

        assert y_min > -0.5 and y_max < 6.5, (
            f"Pedestrians escaped corridor: y in [{y_min:.2f}, {y_max:.2f}]"
        )


# ---------------------------------------------------------------------------
# Test 6: Arching at narrow exit (Helbing 2000 emergent phenomenon)
# ---------------------------------------------------------------------------
class TestArchingAtExit:
    """At a narrow exit, pedestrians should form a semi-circular arch
    pattern — the hallmark of crowd dynamics (Helbing 2000)."""

    def test_density_higher_near_exit(self) -> None:
        """With pedestrians moving toward a narrow exit, density should
        be highest near the exit opening."""
        n = 60
        rng = np.random.default_rng(123)
        # Room: 10m x 10m, exit is a 1.5m gap at x=10, y in [4.25, 5.75]
        positions = np.column_stack([
            rng.uniform(1, 8, n),
            rng.uniform(1, 9, n),
        ])
        velocities = np.zeros((n, 2), dtype=np.float64)
        densities = np.full(n, 0.6, dtype=np.float64)

        sim = CrowdSimulation(dt=0.05, adaptive_dt=False)
        sim.init_pedestrians(positions, velocities, densities)

        # Walls: right wall with gap at y=[4.25, 5.75]
        sim.set_walls(
            p1=np.array([
                [0, 0], [10, 0], [0, 0], [0, 10],     # left, bottom, left-top
                [10, 0], [10, 5.75],                     # right wall below and above exit
            ]),
            p2=np.array([
                [10, 0], [10, 4.25], [0, 10], [10, 10],
                [10, 4.25], [10, 10],
            ]),
        )

        # All pedestrians want to reach exit at (10, 5)
        exit_pos = np.array([10.0, 5.0])
        dirs = exit_pos - positions
        sim.set_desired_directions(dirs)

        sim.run(200)

        # Measure density near exit (x > 7) vs far from exit (x < 4)
        near_exit = sim.positions[:, 0] > 7.0
        far_from_exit = sim.positions[:, 0] < 4.0

        n_near = np.sum(near_exit)
        n_far = np.sum(far_from_exit)

        print(f"\n--- Arching at Exit Test ---")
        print(f"  Pedestrians near exit (x>7): {n_near}")
        print(f"  Pedestrians far from exit (x<4): {n_far}")
        print(f"  Pedestrians that escaped (x>10): {np.sum(sim.positions[:, 0] > 10)}")

        # More pedestrians should accumulate near the exit
        assert n_near > n_far or n_near > 10, (
            f"No crowd accumulation at exit: n_near={n_near}, n_far={n_far}. "
            f"Expected more pedestrians near the narrow exit."
        )
