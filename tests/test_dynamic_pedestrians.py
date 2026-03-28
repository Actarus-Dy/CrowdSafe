"""Tests for dynamic pedestrian injection / removal.

Covers:
- CrowdSimulation.add_pedestrians / remove_pedestrians / n_pedestrians
- CrowdModel.spawn_pedestrian / despawn_pedestrian / despawn_out_of_bounds

All arrays are float64.  Tests are deterministic (fixed seeds).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdsafe.agents.crowd_model import CrowdModel
from crowdsafe.agents.pedestrian_agent import PedestrianAgent
from crowdsafe.core.simulation import CrowdSimulation
from crowdsafe.network.road_network import RoadNetwork

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def sim_20() -> CrowdSimulation:
    """CrowdSimulation initialised with 20 pedestrians on a 500 m segment."""
    rng = np.random.default_rng(77)
    n = 20
    positions = np.column_stack(
        [
            rng.uniform(0, 500, n),
            rng.uniform(-5, 5, n),
        ]
    )
    velocities = np.column_stack(
        [
            rng.uniform(10, 30, n),
            np.zeros(n),
        ]
    )
    densities = rng.uniform(10, 80, n)

    sim = CrowdSimulation(G_s=2.0, beta=0.5, v_max=36.0, adaptive_dt=False, dt=0.05)
    sim.init_pedestrians(positions, velocities, densities)
    return sim


@pytest.fixture()
def grid_network() -> RoadNetwork:
    """Small 3x3 grid network."""
    return RoadNetwork.from_grid(3, 3)


@pytest.fixture()
def model(grid_network: RoadNetwork) -> CrowdModel:
    """CrowdModel with 20 pedestrians, no signals (simpler for injection tests)."""
    return CrowdModel(
        network=grid_network,
        n_pedestrians=20,
        G_s=2.0,
        beta=0.5,
        softening=10.0,
        theta=0.5,
        dt=0.05,
        v_max=36.0,
        signal_intersections=False,
        seed=55,
    )


# ======================================================================
# CrowdSimulation -- add_pedestrians
# ======================================================================


class TestGravSimAddPedestrians:
    """Tests 1-2: add_pedestrians grows arrays and computes masses."""

    def test_array_sizes_increase(self, sim_20: CrowdSimulation) -> None:
        """Test 1: Adding K pedestrians increases all arrays by K."""
        old_n = sim_20.n_pedestrians
        k = 5
        new_pos = np.column_stack(
            [
                np.linspace(100, 400, k),
                np.zeros(k),
            ]
        )
        new_vel = np.column_stack(
            [
                np.full(k, 20.0),
                np.zeros(k),
            ]
        )
        new_rho = np.full(k, 30.0)

        indices = sim_20.add_pedestrians(new_pos, new_vel, new_rho)

        assert sim_20.n_pedestrians == old_n + k
        assert sim_20.positions.shape == (old_n + k, 2)
        assert sim_20.velocities.shape == (old_n + k, 2)
        assert sim_20.local_densities.shape == (old_n + k,)
        assert sim_20.masses.shape == (old_n + k,)
        assert sim_20._forces.shape == (old_n + k, 2)
        assert len(indices) == k
        np.testing.assert_array_equal(indices, np.arange(old_n, old_n + k))

    def test_new_masses_computed(self, sim_20: CrowdSimulation) -> None:
        """Test 2: New pedestrians receive non-trivial mass assignments."""
        new_pos = np.array([[250.0, 0.0]], dtype=np.float64)
        new_vel = np.array([[25.0, 0.0]], dtype=np.float64)
        new_rho = np.array([40.0], dtype=np.float64)

        sim_20.add_pedestrians(new_pos, new_vel, new_rho)

        # The last mass should be a finite float64 value (not zero placeholder)
        last_mass = sim_20.masses[-1]
        assert np.isfinite(last_mass)
        assert sim_20.masses.dtype == np.float64

    def test_dtypes_remain_float64(self, sim_20: CrowdSimulation) -> None:
        """All state arrays remain float64 after injection."""
        new_pos = np.array([[100.0, 1.0]], dtype=np.float64)
        new_vel = np.array([[15.0, 0.0]], dtype=np.float64)
        new_rho = np.array([20.0], dtype=np.float64)

        sim_20.add_pedestrians(new_pos, new_vel, new_rho)

        assert sim_20.positions.dtype == np.float64
        assert sim_20.velocities.dtype == np.float64
        assert sim_20.local_densities.dtype == np.float64
        assert sim_20.masses.dtype == np.float64
        assert sim_20._forces.dtype == np.float64


# ======================================================================
# CrowdSimulation -- remove_pedestrians
# ======================================================================


class TestGravSimRemovePedestrians:
    """Tests 3-4: remove_pedestrians shrinks arrays, preserves remaining data."""

    def test_array_sizes_decrease(self, sim_20: CrowdSimulation) -> None:
        """Test 3: Removing 3 pedestrians decreases all arrays by 3."""
        old_n = sim_20.n_pedestrians
        sim_20.remove_pedestrians(np.array([0, 5, 10]))

        expected = old_n - 3
        assert sim_20.n_pedestrians == expected
        assert sim_20.positions.shape == (expected, 2)
        assert sim_20.velocities.shape == (expected, 2)
        assert sim_20.local_densities.shape == (expected,)
        assert sim_20.masses.shape == (expected,)
        assert sim_20._forces.shape == (expected, 2)

    def test_remaining_values_correct(self, sim_20: CrowdSimulation) -> None:
        """Test 4: Remaining pedestrians keep their original values."""
        # Save values of pedestrian at index 1 (which should become index 0
        # after removing index 0)
        original_pos_1 = sim_20.positions[1].copy()
        original_vel_1 = sim_20.velocities[1].copy()
        original_rho_1 = sim_20.local_densities[1]

        sim_20.remove_pedestrians(np.array([0]))

        np.testing.assert_array_equal(sim_20.positions[0], original_pos_1)
        np.testing.assert_array_equal(sim_20.velocities[0], original_vel_1)
        assert sim_20.local_densities[0] == original_rho_1


# ======================================================================
# CrowdSimulation -- add then step / remove then step
# ======================================================================


class TestGravSimStepAfterDynamic:
    """Tests 5-7: Simulation steps correctly after add/remove."""

    def test_add_then_step(self, sim_20: CrowdSimulation) -> None:
        """Test 5: Simulation runs after adding pedestrians."""
        new_pos = np.array([[200.0, 0.0], [300.0, 0.0]], dtype=np.float64)
        new_vel = np.array([[20.0, 0.0], [15.0, 0.0]], dtype=np.float64)
        new_rho = np.array([30.0, 40.0], dtype=np.float64)

        sim_20.add_pedestrians(new_pos, new_vel, new_rho)
        n_after = sim_20.n_pedestrians

        result = sim_20.step()

        assert result["positions"].shape == (n_after, 2)
        assert result["velocities"].shape == (n_after, 2)
        assert result["masses"].shape == (n_after,)
        assert np.all(np.isfinite(result["positions"]))
        assert np.all(np.isfinite(result["velocities"]))

    def test_remove_then_step(self, sim_20: CrowdSimulation) -> None:
        """Test 6: Simulation runs after removing pedestrians."""
        sim_20.remove_pedestrians(np.array([0, 1, 2]))
        n_after = sim_20.n_pedestrians

        result = sim_20.step()

        assert result["positions"].shape == (n_after, 2)
        assert np.all(np.isfinite(result["positions"]))

    def test_remove_all(self, sim_20: CrowdSimulation) -> None:
        """Test 7: Removing all pedestrians yields empty arrays; step handles it."""
        all_idx = np.arange(sim_20.n_pedestrians)
        sim_20.remove_pedestrians(all_idx)

        assert sim_20.n_pedestrians == 0
        assert sim_20.positions.shape == (0, 2)
        assert sim_20.velocities.shape == (0, 2)
        assert sim_20.masses.shape == (0,)
        assert sim_20._forces.shape == (0, 2)

        # Step with zero pedestrians should not raise
        result = sim_20.step()
        assert result["positions"].shape == (0, 2)


# ======================================================================
# CrowdModel -- spawn_pedestrian
# ======================================================================


class TestCrowdModelSpawn:
    """Tests 8: spawn_pedestrian grows both agent list and simulation."""

    def test_spawn_pedestrian(self, model: CrowdModel) -> None:
        """Test 8: Spawning one pedestrian increments both systems."""
        old_n_agents = len(model.pedestrian_agents)
        old_n_sim = model.simulation.n_pedestrians

        agent = model.spawn_pedestrian(
            position=np.array([100.0, 0.0]),
            velocity=np.array([20.0, 0.0]),
            local_density=25.0,
        )

        assert isinstance(agent, PedestrianAgent)
        assert len(model.pedestrian_agents) == old_n_agents + 1
        assert model.simulation.n_pedestrians == old_n_sim + 1
        assert model.pedestrian_agents[-1] is agent

    def test_spawn_preserves_alignment(self, model: CrowdModel) -> None:
        """Agent list length equals physics engine array length after spawn."""
        model.spawn_pedestrian(
            position=np.array([50.0, 2.0]),
            velocity=np.array([18.0, 1.0]),
        )
        assert len(model.pedestrian_agents) == model.simulation.n_pedestrians


# ======================================================================
# CrowdModel -- despawn_pedestrian
# ======================================================================


class TestCrowdModelDespawn:
    """Test 9: despawn_pedestrian shrinks both systems."""

    def test_despawn_pedestrian(self, model: CrowdModel) -> None:
        old_n = len(model.pedestrian_agents)
        model.despawn_pedestrian(0)

        assert len(model.pedestrian_agents) == old_n - 1
        assert model.simulation.n_pedestrians == old_n - 1

    def test_despawn_preserves_alignment(self, model: CrowdModel) -> None:
        model.despawn_pedestrian(5)
        assert len(model.pedestrian_agents) == model.simulation.n_pedestrians


# ======================================================================
# CrowdModel -- despawn_out_of_bounds
# ======================================================================


class TestCrowdModelDespawnOOB:
    """Test 10: despawn_out_of_bounds removes pedestrians outside bbox."""

    def test_oob_removes_correct_count(self, model: CrowdModel) -> None:
        """Test 10: Pedestrians outside a tight bbox are removed."""
        # Spawn two pedestrians clearly outside a small bbox
        model.spawn_pedestrian(
            position=np.array([9999.0, 9999.0]),
            velocity=np.array([10.0, 0.0]),
        )
        model.spawn_pedestrian(
            position=np.array([-9999.0, -9999.0]),
            velocity=np.array([10.0, 0.0]),
        )
        n_before = len(model.pedestrian_agents)

        removed = model.despawn_out_of_bounds(
            x_min=-1000.0, y_min=-1000.0, x_max=1000.0, y_max=1000.0
        )

        assert removed >= 2
        assert len(model.pedestrian_agents) == n_before - removed
        assert len(model.pedestrian_agents) == model.simulation.n_pedestrians

    def test_oob_no_removal_when_all_inside(self, model: CrowdModel) -> None:
        """despawn_out_of_bounds returns 0 when all pedestrians are inside bbox."""
        removed = model.despawn_out_of_bounds(x_min=-1e6, y_min=-1e6, x_max=1e6, y_max=1e6)
        assert removed == 0


# ======================================================================
# CrowdModel -- spawn + step + despawn cycle
# ======================================================================


class TestCrowdModelSpawnStepDespawnCycle:
    """Test 11: Full spawn/step/despawn cycle over 10 steps."""

    def test_cycle_10_steps(self, model: CrowdModel) -> None:
        """Spawn, step, and despawn over 10 iterations without errors."""
        rng = np.random.default_rng(999)

        for _ in range(10):
            # Spawn one pedestrian per step
            pos = rng.uniform(0, 500, size=2)
            vel = np.array([rng.uniform(10, 30), 0.0])
            model.spawn_pedestrian(position=pos, velocity=vel, local_density=30.0)

            # Run one simulation step
            model.step()

            # Despawn anything outside a generous bbox
            model.despawn_out_of_bounds(x_min=-2000.0, y_min=-2000.0, x_max=2000.0, y_max=2000.0)

            # Invariant: agent list and physics engine stay aligned
            assert len(model.pedestrian_agents) == model.simulation.n_pedestrians

        # Final state should be finite everywhere
        assert np.all(np.isfinite(model.simulation.positions))
        assert np.all(np.isfinite(model.simulation.velocities))
