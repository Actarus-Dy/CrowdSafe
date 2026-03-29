"""Tests for integrated safety monitoring (check_safety).

Validates that CrowdSimulation.check_safety() correctly combines
critical density, TOV pressure, and evacuation geodesics into a
coherent safety report.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-29
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdsafe.core.simulation import CrowdSimulation


class TestCheckSafety:
    """Integrated safety check returns coherent reports."""

    @staticmethod
    def _make_sim(n: int = 50, spread: float = 20.0, seed: int = 42):
        rng = np.random.default_rng(seed)
        pos = np.column_stack([rng.uniform(0, spread, n), rng.uniform(0, spread, n)])
        vel = np.column_stack([rng.uniform(0.5, 1.2, n), rng.uniform(-0.2, 0.2, n)])
        dens = np.full(n, n / (spread * spread))
        sim = CrowdSimulation(adaptive_dt=False, dt=0.1)
        sim.init_pedestrians(pos, vel, dens)
        sim.step()
        return sim

    def test_returns_required_keys(self) -> None:
        sim = self._make_sim()
        report = sim.check_safety()
        assert "density_alert" in report
        assert "max_density" in report
        assert "tov_profile" in report
        assert "tov_alert" in report
        assert "overall_alert" in report

    def test_overall_alert_is_string(self) -> None:
        sim = self._make_sim()
        report = sim.check_safety()
        assert report["overall_alert"] in ("VERT", "JAUNE", "ORANGE", "ROUGE", "CRITIQUE")

    def test_tov_profile_is_present(self) -> None:
        sim = self._make_sim()
        report = sim.check_safety(corridor_width=20.0)
        assert report["tov_profile"].F_max >= 0

    def test_with_exits_provides_evacuation(self) -> None:
        sim = self._make_sim(n=20, spread=10.0)
        exits = [np.array([10.0, 5.0])]
        report = sim.check_safety(exits=exits, dx_m=1.0)
        assert report["evacuation_distance_map"] is not None
        assert report["evacuation_max_time"] is not None

    def test_without_exits_no_evacuation(self) -> None:
        sim = self._make_sim()
        report = sim.check_safety()
        assert report["evacuation_distance_map"] is None

    def test_high_density_elevates_alert(self) -> None:
        """Pack pedestrians tightly → max density should be elevated."""
        rng = np.random.default_rng(42)
        n = 80
        # Pack in 4x4m area → 5 pers/m²
        pos = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 4, n)])
        vel = np.zeros((n, 2), dtype=np.float64)
        dens = np.full(n, 5.0)
        sim = CrowdSimulation(
            adaptive_dt=False, dt=0.1,
            contact_strength=0.0, density_radius=3.0,
        )
        sim.init_pedestrians(pos, vel, dens)
        # Don't step — check density at init
        report = sim.check_safety()
        assert report["max_density"] > 1.0, (
            f"Expected elevated density, got {report['max_density']}"
        )

    def test_low_density_vert(self) -> None:
        sim = self._make_sim(n=10, spread=50.0)  # 0.004 pers/m²
        report = sim.check_safety()
        assert report["density_alert"].value == "VERT"

    def test_overall_takes_worst(self) -> None:
        """Overall alert should be the worst of density and TOV."""
        sim = self._make_sim()
        report = sim.check_safety()
        alert_order = {"VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3, "CRITIQUE": 4}
        overall_rank = alert_order[report["overall_alert"]]
        density_rank = alert_order[report["density_alert"].value]
        tov_rank = alert_order[report["tov_alert"]]
        assert overall_rank >= max(density_rank, tov_rank)
