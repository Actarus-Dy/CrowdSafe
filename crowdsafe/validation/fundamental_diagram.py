"""Fundamental Diagram validation -- Weidmann speed-density fit.

Runs a systematic density sweep in a corridor and compares the
emergent mean speed at each density against the Weidmann (1993) model:

    v(rho) = v_free * (1 - exp(-1.913 * (1/rho - 1/rho_jam)))

Reports R², RMSE, and per-density data points for plotting.

Reference: Weidmann, U. (1993). Transporttechnik der Fussgänger.

Author: Agent #23 Scientific Validation Tester
Date: 2026-03-28
"""

from __future__ import annotations

import math

import numpy as np

from crowdsafe.core.simulation import CrowdSimulation


def weidmann_speed(rho: float, v_free: float = 1.34, rho_jam: float = 6.0) -> float:
    """Theoretical Weidmann (1993) equilibrium walking speed.

    v(rho) = v_free * (1 - exp(-1.913 * (1/rho - 1/rho_jam)))
    """
    if rho <= 0.01:
        return v_free
    if rho >= rho_jam:
        return 0.0
    exponent = -1.913 * (1.0 / rho - 1.0 / rho_jam)
    return v_free * max(0.0, 1.0 - math.exp(exponent))


# Backward compatibility alias
def greenshields_speed(rho: float, v_free: float = 1.34, rho_jam: float = 6.0) -> float:
    """Deprecated alias for weidmann_speed."""
    return weidmann_speed(rho, v_free, rho_jam)


def run_fd_sweep(
    densities: list[float] | None = None,
    G_s: float = 0.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    v_free: float = 1.34,
    rho_jam: float = 6.0,
    corridor_length: float = 50.0,
    corridor_width: float = 4.0,
    n_steps: int = 400,
    warmup_steps: int = 200,
    seed: int = 42,
) -> dict:
    """Run a density sweep and measure emergent speed-density relationship.

    Parameters
    ----------
    densities : list[float], optional
        Densities in pers/m² to test. Default: 0.5 to 5.0 in steps of 0.5.
    n_steps : int
        Total simulation steps per density point.
    warmup_steps : int
        Steps to discard before measuring (let transients settle).

    Returns
    -------
    dict with keys:
        - densities: list[float] — tested densities [pers/m²]
        - measured_speeds: list[float] — mean speed at each density [m/s]
        - theoretical_speeds: list[float] — Weidmann prediction [m/s]
        - r_squared: float — R² goodness of fit
        - rmse: float — root mean squared error [m/s]
        - data_points: list[dict] — per-density detail
    """
    if densities is None:
        densities = [0.5 * i for i in range(1, 11)]  # 0.5 to 5.0

    rng = np.random.default_rng(seed)
    results = []

    for rho in densities:
        # Number of pedestrians from density and corridor area
        area = corridor_length * corridor_width
        n_ped = max(2, int(rho * area))

        # Initial conditions: random positions in corridor
        positions = np.zeros((n_ped, 2), dtype=np.float64)
        positions[:, 0] = rng.uniform(0, corridor_length, n_ped)
        positions[:, 1] = rng.uniform(0, corridor_width, n_ped)

        # Initial speed: v_free for ALL densities (NOT v_eq!) to avoid
        # circular validation. The model must converge to the correct
        # equilibrium from arbitrary initial conditions.
        speeds = np.full(n_ped, v_free, dtype=np.float64)
        speeds += rng.normal(0, 0.1, n_ped)  # small noise for symmetry breaking
        speeds = np.clip(speeds, 0.1, v_free)
        velocities = np.zeros((n_ped, 2), dtype=np.float64)
        velocities[:, 0] = speeds

        local_densities = np.full(n_ped, float(rho), dtype=np.float64)

        sim = CrowdSimulation(
            G_s=G_s,
            beta=beta,
            softening=0.5,
            dt=0.5,
            v_max=v_free + 0.5,
            adaptive_dt=False,
            drag_coefficient=gamma,
            v_free=v_free,
            rho_jam=rho_jam,
            contact_strength=0.0,  # disable contact forces for clean FD
            use_gpu=False,
        )
        sim.init_pedestrians(positions, velocities, local_densities)

        # Run warmup — override local_densities each step to maintain
        # the target density (avoids edge effects in small corridors).
        target_densities = np.full(n_ped, float(rho), dtype=np.float64)
        for _ in range(warmup_steps):
            sim.step()
            sim.local_densities = target_densities.copy()

        # Measure over remaining steps
        speed_samples = []
        for _ in range(n_steps - warmup_steps):
            sim.step()
            sim.local_densities = target_densities.copy()
            mean_spd = float(np.mean(np.linalg.norm(sim.velocities, axis=1)))
            speed_samples.append(mean_spd)

        measured = float(np.mean(speed_samples))
        theoretical = weidmann_speed(rho, v_free, rho_jam)

        results.append(
            {
                "density": rho,
                "n_pedestrians": n_ped,
                "measured_speed": measured,
                "theoretical_speed": theoretical,
                "speed_std": float(np.std(speed_samples)),
            }
        )

    # Compute fit metrics
    measured_arr = np.array([r["measured_speed"] for r in results])
    theoretical_arr = np.array([r["theoretical_speed"] for r in results])

    ss_res = np.sum((measured_arr - theoretical_arr) ** 2)
    ss_tot = np.sum((measured_arr - np.mean(measured_arr)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((measured_arr - theoretical_arr) ** 2)))

    return {
        "densities": [r["density"] for r in results],
        "measured_speeds": [r["measured_speed"] for r in results],
        "theoretical_speeds": [r["theoretical_speed"] for r in results],
        "r_squared": r_squared,
        "rmse": rmse,
        "data_points": results,
        "parameters": {
            "G_s": G_s,
            "beta": beta,
            "gamma": gamma,
            "v_free": v_free,
            "rho_jam": rho_jam,
        },
    }
