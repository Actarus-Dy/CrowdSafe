"""Tests for the scientific validation module.

Uses quick mode for fast CI execution (~10s).

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-23
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdsafe.validation.emergence import (
    gini_coefficient,
    run_emergence_analysis,
)
from crowdsafe.validation.fundamental_diagram import (
    run_fd_sweep,
    weidmann_speed,
)
from crowdsafe.validation.report import run_validation_suite

# ---------------------------------------------------------------------------
# Fundamental Diagram
# ---------------------------------------------------------------------------


class TestWeidmannSpeed:
    def test_free_flow(self) -> None:
        assert weidmann_speed(0.0) == pytest.approx(1.34)

    def test_jam(self) -> None:
        assert weidmann_speed(6.0) == pytest.approx(0.0)

    def test_half_density_slower_than_linear(self) -> None:
        # Weidmann is concave: v(rho_jam/2) < v_free/2
        v_half = weidmann_speed(3.0)
        assert 0.0 < v_half < 1.34 / 2, (
            f"Weidmann at rho=3.0 should be less than linear half: got {v_half}"
        )

    def test_over_jam(self) -> None:
        assert weidmann_speed(8.0) == 0.0

    def test_monotonically_decreasing(self) -> None:
        densities = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.9]
        speeds = [weidmann_speed(rho) for rho in densities]
        for i in range(len(speeds) - 1):
            assert speeds[i] > speeds[i + 1], (
                f"Weidmann not monotonic: v({densities[i]})={speeds[i]:.3f} "
                f">= v({densities[i+1]})={speeds[i+1]:.3f}"
            )


class TestFDSweep:
    def test_sweep_returns_expected_keys(self) -> None:
        result = run_fd_sweep(
            densities=[1.0, 3.0, 5.0],
            n_steps=50,
            warmup_steps=20,
            seed=42,
        )
        assert "r_squared" in result
        assert "rmse" in result
        assert "densities" in result
        assert "measured_speeds" in result
        assert len(result["densities"]) == 3

    def test_sweep_r_squared_above_threshold(self) -> None:
        """With Weidmann drag (G_s=0), R² should exceed 0.95."""
        result = run_fd_sweep(
            densities=[0.5, 1.0, 2.0, 3.0, 4.0],
            n_steps=100,
            warmup_steps=50,
            seed=42,
        )
        assert result["r_squared"] > 0.95, (
            f"R²={result['r_squared']:.4f} too low — Weidmann drag model "
            f"should converge to theoretical speed-density curve"
        )

    def test_speed_decreases_with_density(self) -> None:
        """Mean speed should generally decrease with density."""
        result = run_fd_sweep(
            densities=[0.5, 2.5, 5.0],
            n_steps=100,
            warmup_steps=30,
            seed=42,
        )
        speeds = result["measured_speeds"]
        assert speeds[0] > speeds[-1], (
            f"Speed at rho={result['densities'][0]} ({speeds[0]:.2f}) should exceed "
            f"speed at rho={result['densities'][-1]} ({speeds[-1]:.2f})"
        )


# ---------------------------------------------------------------------------
# Emergence
# ---------------------------------------------------------------------------


class TestGiniCoefficient:
    def test_equal_values(self) -> None:
        assert gini_coefficient(np.array([10.0, 10.0, 10.0])) == pytest.approx(0.0)

    def test_maximal_inequality(self) -> None:
        g = gini_coefficient(np.array([0.0, 0.0, 100.0]))
        assert g > 0.5

    def test_empty(self) -> None:
        assert gini_coefficient(np.array([])) == 0.0


class TestEmergenceAnalysis:
    def test_returns_expected_structure(self) -> None:
        result = run_emergence_analysis(n_steps=50, seed=42)
        assert "gravity_on" in result
        assert "gravity_off" in result
        assert "emergence_score" in result
        assert isinstance(result["emergence_score"], float)

    def test_gravity_on_has_metrics(self) -> None:
        result = run_emergence_analysis(n_steps=50, seed=42)
        g = result["gravity_on"]
        for key in (
            "upstream_deceleration_ms",
            "variance_ratio",
            "gini_initial",
            "gini_final",
            "wave_speed_ms",
        ):
            assert key in g, f"Missing key {key}"


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_quick_report_generates(self) -> None:
        """Quick validation should complete without errors."""
        report = run_validation_suite(quick=True, seed=42)
        assert "overall_verdict" in report
        assert report["mode"] == "quick"
        assert report["fundamental_diagram"]["verdict"] in ("PASS", "FAIL")
        assert report["emergence"]["verdict"] in ("PASS", "FAIL")

    def test_quick_report_deterministic(self) -> None:
        """Same seed should produce identical results across all fields."""
        r1 = run_validation_suite(quick=True, seed=42)
        r2 = run_validation_suite(quick=True, seed=42)
        # Aggregates
        assert r1["fundamental_diagram"]["r_squared"] == r2["fundamental_diagram"]["r_squared"]
        assert r1["emergence"]["score"] == r2["emergence"]["score"]
        assert r1["overall_verdict"] == r2["overall_verdict"]
        # Detailed data points
        for d1, d2 in zip(r1["fundamental_diagram"]["data"], r2["fundamental_diagram"]["data"]):
            assert d1["measured_speed"] == d2["measured_speed"]
        # Emergence detail
        for key in r1["emergence"]["gravity_on"]:
            assert r1["emergence"]["gravity_on"][key] == r2["emergence"]["gravity_on"][key]
