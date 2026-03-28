"""Benchmark GravCore pipeline: 1k, 5k, 10k pedestrians.

Standalone benchmark script (not a test).  Measures wall-clock time for
N simulation steps at various population sizes.

Usage
-----
    python -m benchmarks.bench_gravcore

    # or directly:
    python benchmarks/bench_gravcore.py

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import sys
import time

import numpy as np

from crowdsafe.core.simulation import CrowdSimulation


def benchmark(n_pedestrians: int, n_steps: int = 10) -> dict:
    """Run *n_steps* of the CrowdSafe pipeline with *n_pedestrians*.

    Parameters
    ----------
    n_pedestrians : int
        Number of simulated pedestrians.
    n_steps : int
        Number of simulation steps to execute.

    Returns
    -------
    dict
        Timing results with keys ``n_pedestrians``, ``n_steps``,
        ``total_s``, ``ms_per_step``.
    """
    rng = np.random.default_rng(42)

    # Create random pedestrians on a 2 km x 0.02 km road (4 lanes)
    positions = np.column_stack([
        rng.uniform(0, 2000, n_pedestrians),
        rng.uniform(-10, 10, n_pedestrians),
    ])
    velocities = np.column_stack([
        rng.uniform(15, 35, n_pedestrians),   # ~54-126 km/h
        np.zeros(n_pedestrians),
    ])
    densities = rng.uniform(10, 80, n_pedestrians)

    sim = CrowdSimulation(G_s=5.0, beta=0.5)
    sim.init_pedestrians(positions, velocities, densities)

    start = time.perf_counter()
    for _ in range(n_steps):
        sim.step()
    elapsed = time.perf_counter() - start

    return {
        "n_pedestrians": n_pedestrians,
        "n_steps": n_steps,
        "total_s": elapsed,
        "ms_per_step": elapsed / n_steps * 1000,
    }


def main() -> None:
    """Run benchmarks for 1k, 5k, and 10k pedestrians and print results."""
    print("GravCore Pipeline Benchmark")
    print("=" * 50)
    print(f"{'N':>8}  {'ms/step':>10}  {'total (s)':>10}  {'steps':>6}")
    print("-" * 50)

    for n in [1000, 5000, 10000]:
        result = benchmark(n)
        print(
            f"{result['n_pedestrians']:>8}  "
            f"{result['ms_per_step']:>10.1f}  "
            f"{result['total_s']:>10.2f}  "
            f"{result['n_steps']:>6}"
        )

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
