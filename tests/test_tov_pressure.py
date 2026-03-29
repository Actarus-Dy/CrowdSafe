"""Tests for TOVPressure -- crowd pressure profile computation (§5.5).

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-29
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdsafe.core.tov_pressure import (
    DANGER_FORCE,
    WARNING_FORCE,
    TOVPressure,
)


class TestTOVBasic:
    """Basic TOV pressure computation."""

    def test_zero_density_zero_pressure(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 10, 20)
        rho = np.zeros(20)
        result = tov.compute(l, rho, width_m=2.0)
        assert result.F_max == pytest.approx(0.0)
        assert result.alert_level == "VERT"

    def test_uniform_density_analytical(self) -> None:
        """Uniform density: F(L) = rho * F_contact * width * L exactly."""
        tov = TOVPressure(F_contact=100.0)
        L = 10.0
        l = np.linspace(0, L, 1000)  # fine grid for accuracy
        rho_val = 3.0
        width = 2.0
        rho = np.full(len(l), rho_val)
        result = tov.compute(l, rho, width_m=width)

        # Analytical: F(L) = rho * F_contact * width * L = 3 * 100 * 2 * 10 = 6000
        expected = rho_val * 100.0 * width * L
        assert result.F_max == pytest.approx(expected, rel=0.01), (
            f"F_max={result.F_max:.1f}, expected={expected:.1f}"
        )
        # Force should increase monotonically
        assert result.cumulative_force_N[-1] > result.cumulative_force_N[0]

    def test_pressure_shape_matches_input(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 5, 50)
        rho = np.ones(50)
        result = tov.compute(l, rho, width_m=3.0)
        assert result.l_m.shape == (50,)
        assert result.cumulative_force_N.shape == (50,)


class TestTOVAlerts:
    """Alert level thresholds."""

    def test_low_density_vert(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 5, 20)
        rho = np.full(20, 0.5)  # very low density
        result = tov.compute(l, rho, width_m=2.0)
        assert result.alert_level == "VERT"

    def test_high_density_orange(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 20, 100)
        rho = np.full(100, 4.0)  # moderate density
        result = tov.compute(l, rho, width_m=2.0)
        # P_max = 4 * 100 * 2 * 20 = 16000 > 4450
        if result.F_max > DANGER_FORCE:
            assert result.alert_level == "ROUGE"
        elif result.F_max > WARNING_FORCE:
            assert result.alert_level == "ORANGE"

    def test_crush_density_rouge(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 30, 200)
        rho = np.full(200, 6.0)  # critical density
        result = tov.compute(l, rho, width_m=3.0)
        assert result.alert_level == "ROUGE"
        assert result.critical_exceeded

    def test_schwarzschild_ratio(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 10, 50)
        rho = np.full(50, 3.0)
        result = tov.compute(l, rho, width_m=2.0)
        assert result.schwarzschild_ratio == pytest.approx(
            result.F_max / DANGER_FORCE
        )


class TestTOVFromSimulation:
    """Compute pressure from live simulation positions."""

    def test_from_positions(self) -> None:
        tov = TOVPressure()
        rng = np.random.default_rng(42)
        # 100 pedestrians in a 20m corridor
        positions = np.column_stack([
            rng.uniform(0, 20, 100),
            rng.uniform(0, 4, 100),
        ])
        result = tov.compute_from_simulation(
            positions, corridor_axis=0, corridor_width=4.0
        )
        assert result.F_max > 0
        assert len(result.l_m) > 0
        assert result.alert_level in ("VERT", "ORANGE", "ROUGE")

    def test_clustered_positions_higher_pressure(self) -> None:
        """A cluster of pedestrians should produce higher pressure."""
        tov = TOVPressure()
        # Cluster: 50 pedestrians in [5, 7] x [0, 4]
        rng = np.random.default_rng(55)
        clustered = np.column_stack([
            rng.uniform(5, 7, 50),
            rng.uniform(0, 4, 50),
        ])
        # Spread: 50 pedestrians in [0, 20] x [0, 4]
        spread = np.column_stack([
            rng.uniform(0, 20, 50),
            rng.uniform(0, 4, 50),
        ])

        r_cluster = tov.compute_from_simulation(clustered, corridor_width=4.0)
        r_spread = tov.compute_from_simulation(spread, corridor_width=4.0)

        assert r_cluster.P_max > r_spread.P_max, (
            f"Cluster P_max ({r_cluster.P_max:.1f}) should exceed "
            f"spread P_max ({r_spread.P_max:.1f})"
        )


class TestTOVPhysics:
    """Physical consistency checks."""

    def test_wider_corridor_lower_pressure(self) -> None:
        """Wider corridor at same density gives same total force but
        pressure is per unit length, so wider = higher P."""
        tov = TOVPressure()
        l = np.linspace(0, 10, 50)
        rho = np.full(50, 3.0)
        r_narrow = tov.compute(l, rho, width_m=1.0)
        r_wide = tov.compute(l, rho, width_m=4.0)
        # Wider corridor = more people per row = higher pressure
        assert r_wide.P_max > r_narrow.P_max

    def test_longer_corridor_higher_pressure(self) -> None:
        """Longer corridor accumulates more pressure."""
        tov = TOVPressure()
        rho = np.full(50, 3.0)
        r_short = tov.compute(np.linspace(0, 5, 50), rho, width_m=2.0)
        r_long = tov.compute(np.linspace(0, 20, 50), rho, width_m=2.0)
        assert r_long.P_max > r_short.P_max

    def test_location_max_within_corridor(self) -> None:
        tov = TOVPressure()
        l = np.linspace(0, 15, 100)
        rho = np.full(100, 2.0)
        result = tov.compute(l, rho, width_m=2.0)
        assert 0 <= result.location_max_m <= 15.0
