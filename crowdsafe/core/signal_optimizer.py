"""SignalOptimizer -- time-integrated potential optimizer for traffic signals.

Replaces the instantaneous-proxy optimizer in ``potential_field.py`` with a
proper temporal approach that evaluates different green-phase timings over a
finite time horizon.

Algorithm
---------
For each candidate ``green_ns`` value (swept 15..90 s in 5 s increments):

1. Linearly extrapolate pedestrian positions forward in time over the horizon.
2. At each evaluation timestep, determine the current signal phase (NS green,
   amber, or EW green) and place red-light obstacle masses accordingly.
3. Compute the gravitational potential at the intersection center from the
   extrapolated pedestrians *and* the red-light obstacles.
4. Accumulate the potential integral: ``Phi_total += Phi(t) * dt_eval``.

The candidate timing with the **highest** (least negative) integrated
potential is selected -- less negative means less congestion pressure.

Sign convention (matches ``potential_field.py``):
    - Positive mass (slow pedestrian) -> Phi < 0 -> congestion well (bad)
    - Negative mass (fast pedestrian) -> Phi > 0 -> fluid zone (good)
    - Goal: **maximize** Phi_integral (minimize congestion)

Complexity
----------
O(T * C * N) where T = horizon / dt_eval evaluation points, C = 16 candidate
timings, and N = number of nearby pedestrians.  For typical parameters
(T=24, C=16, N~100) this is about 38 400 potential evaluations -- well within
real-time budgets.

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crowdsafe.core.potential_field import compute_potential_field

__all__ = [
    "estimate_phi_integral",
    "optimize_signal_timing",
]


def estimate_phi_integral(
    pedestrian_positions: npt.NDArray[np.float64],
    pedestrian_velocities: npt.NDArray[np.float64],
    pedestrian_masses: npt.NDArray[np.float64],
    intersection_pos: npt.NDArray[np.float64],
    green_ns: float,
    green_ew: float,
    red_light_mass: float = 50.0,
    radius: float = 200.0,
    G_s: float = 5.0,
    horizon_s: float = 120.0,
    dt_eval: float = 5.0,
) -> float:
    """Estimate the time-integrated potential at an intersection over a horizon.

    Uses linear extrapolation of pedestrian trajectories (not a full simulation)
    to estimate future positions, then computes the gravitational potential at
    each evaluation timestep accounting for which phase is green/red.

    Parameters
    ----------
    pedestrian_positions : ndarray, shape (N, 2), dtype float64
        Current (x, y) positions of all pedestrians.
    pedestrian_velocities : ndarray, shape (N, 2), dtype float64
        Current (vx, vy) velocities of all pedestrians.
    pedestrian_masses : ndarray, shape (N,), dtype float64
        Signed gravitational mass of each pedestrian.
    intersection_pos : ndarray, shape (2,), dtype float64
        Center of the intersection.
    green_ns : float
        Green time for NS direction in seconds.
    green_ew : float
        Green time for EW direction in seconds.
    red_light_mass : float, default 50.0
        Gravitational mass injected for each red-light obstacle.
    radius : float, default 200.0
        Only pedestrians within this distance (meters) are considered.
    G_s : float, default 5.0
        Social gravitational constant.
    horizon_s : float, default 120.0
        Time horizon for the integration in seconds (default 2 minutes).
    dt_eval : float, default 5.0
        Evaluation timestep in seconds (coarser than simulation dt for speed).

    Returns
    -------
    float
        Total integrated potential (Phi * dt summed over the horizon).
        Higher (less negative) values indicate less congestion.
        Returns 0.0 if no pedestrians are within the radius.

    Notes
    -----
    The cycle structure is: ``green_ns`` -> 10 s amber -> ``green_ew`` -> wrap.
    During amber, all four stop-line positions carry red-light obstacles.

    Linear extrapolation is a first-order approximation; it does not account
    for pedestrian interactions or braking at the red light.  This is intentional:
    the optimizer needs a *fast* proxy, not a full simulation.
    """
    pedestrian_positions = np.asarray(pedestrian_positions, dtype=np.float64)
    pedestrian_velocities = np.asarray(pedestrian_velocities, dtype=np.float64)
    pedestrian_masses = np.asarray(pedestrian_masses, dtype=np.float64)
    intersection_pos = np.asarray(intersection_pos, dtype=np.float64)

    # Select pedestrians within radius of the intersection
    if len(pedestrian_masses) == 0:
        return 0.0

    dist = np.linalg.norm(pedestrian_positions - intersection_pos[np.newaxis, :], axis=1)
    mask = dist <= radius

    if not mask.any():
        return 0.0

    pos = pedestrian_positions[mask].copy()  # (K, 2)
    vel = pedestrian_velocities[mask].copy()  # (K, 2)
    masses = pedestrian_masses[mask].copy()  # (K,)

    # Amber / clearance duration is fixed at 10 s
    amber_s = 10.0
    cycle = green_ns + green_ew + amber_s

    n_eval = int(horizon_s / dt_eval)
    phi_total = 0.0

    # Classify pedestrians as NS or EW based on dominant velocity component.
    # NS pedestrians: |vy| > |vx|; EW pedestrians: |vx| >= |vy|.
    is_ns = np.abs(vel[:, 1]) > np.abs(vel[:, 0])  # (K,)
    is_ew = ~is_ns

    # Stop-line distance from intersection center
    stop_dist = 15.0

    # Precompute stop-line positions for clamping
    # EW stop lines at x = +/- stop_dist; NS stop lines at y = +/- stop_dist
    ew_stop_plus = intersection_pos[0] + stop_dist
    ew_stop_minus = intersection_pos[0] - stop_dist
    ns_stop_plus = intersection_pos[1] + stop_dist
    ns_stop_minus = intersection_pos[1] - stop_dist

    for step in range(n_eval):
        t = step * dt_eval

        # Linearly extrapolate pedestrian positions
        future_pos = pos + vel * t  # (K, 2)

        # Determine current signal phase within the cycle
        t_in_cycle = t % cycle

        if t_in_cycle < green_ns:
            # NS green -> EW red
            phase = "ns_green"
            obstacles_pos = np.array(
                [
                    intersection_pos + np.array([15.0, 0.0]),
                    intersection_pos - np.array([15.0, 0.0]),
                ],
                dtype=np.float64,
            )
            obstacles_mass = np.array([red_light_mass, red_light_mass], dtype=np.float64)
        elif t_in_cycle < green_ns + amber_s:
            # Amber: all blocked
            phase = "amber"
            obstacles_pos = np.array(
                [
                    intersection_pos + np.array([15.0, 0.0]),
                    intersection_pos - np.array([15.0, 0.0]),
                    intersection_pos + np.array([0.0, 15.0]),
                    intersection_pos - np.array([0.0, 15.0]),
                ],
                dtype=np.float64,
            )
            obstacles_mass = np.array([red_light_mass] * 4, dtype=np.float64)
        else:
            # EW green -> NS red
            phase = "ew_green"
            obstacles_pos = np.array(
                [
                    intersection_pos + np.array([0.0, 15.0]),
                    intersection_pos - np.array([0.0, 15.0]),
                ],
                dtype=np.float64,
            )
            obstacles_mass = np.array([red_light_mass, red_light_mass], dtype=np.float64)

        # Clamp pedestrian positions at stop lines when their direction is red.
        # This models the fact that pedestrians queue at a red light rather than
        # passing through it.  Without clamping, linear extrapolation causes
        # pedestrians to fly past the intersection, making the timing irrelevant.
        clamped_pos = future_pos.copy()

        ew_red = (phase == "ns_green") or (phase == "amber")
        ns_red = (phase == "ew_green") or (phase == "amber")

        if ew_red:
            # EW pedestrians approaching from +x should stop at +stop_dist
            approaching_plus = is_ew & (vel[:, 0] < 0)
            overshoot = approaching_plus & (clamped_pos[:, 0] < ew_stop_plus)
            clamped_pos[overshoot, 0] = ew_stop_plus

            # EW pedestrians approaching from -x should stop at -stop_dist
            approaching_minus = is_ew & (vel[:, 0] > 0)
            overshoot = approaching_minus & (clamped_pos[:, 0] > ew_stop_minus)
            clamped_pos[overshoot, 0] = ew_stop_minus

        if ns_red:
            # NS pedestrians approaching from +y should stop at +stop_dist
            approaching_plus = is_ns & (vel[:, 1] < 0)
            overshoot = approaching_plus & (clamped_pos[:, 1] < ns_stop_plus)
            clamped_pos[overshoot, 1] = ns_stop_plus

            # NS pedestrians approaching from -y should stop at -stop_dist
            approaching_minus = is_ns & (vel[:, 1] > 0)
            overshoot = approaching_minus & (clamped_pos[:, 1] > ns_stop_minus)
            clamped_pos[overshoot, 1] = ns_stop_minus

        # Evaluate the potential experienced by each pedestrian due to the
        # red-light obstacles ONLY.  This measures how much the current
        # signal phase is blocking each pedestrian.  Pedestrians near a red
        # light feel a strong negative potential well.
        phi_at_pedestrians = compute_potential_field(
            obstacles_pos, obstacles_mass, clamped_pos, G_s=G_s
        )  # (K,) -- potential at each pedestrian due to obstacles

        # Weight by pedestrian absolute mass so that congested (heavy) pedestrians
        # contribute more to the cost.
        weighted_phi = np.sum(phi_at_pedestrians * np.abs(masses))

        phi_total += weighted_phi * dt_eval

    return float(phi_total)


def optimize_signal_timing(
    pedestrian_positions: npt.NDArray[np.float64],
    pedestrian_velocities: npt.NDArray[np.float64],
    pedestrian_masses: npt.NDArray[np.float64],
    intersection_pos: npt.NDArray[np.float64],
    radius: float = 200.0,
    G_s: float = 5.0,
    horizon_s: float = 120.0,
    red_light_mass: float = 50.0,
) -> dict:
    """Find optimal green timing by sweeping ``green_ns`` from 15 to 90 s.

    For each candidate timing, :func:`estimate_phi_integral` is called to
    evaluate the time-integrated potential.  The timing with the highest
    (least negative) integral is selected -- higher means less congestion.

    Parameters
    ----------
    pedestrian_positions : ndarray, shape (N, 2), dtype float64
        Current (x, y) positions of all pedestrians.
    pedestrian_velocities : ndarray, shape (N, 2), dtype float64
        Current (vx, vy) velocities of all pedestrians.
    pedestrian_masses : ndarray, shape (N,), dtype float64
        Signed gravitational mass of each pedestrian.
    intersection_pos : ndarray, shape (2,), dtype float64
        Center of the intersection.
    radius : float, default 200.0
        Only pedestrians within this distance (meters) are considered.
    G_s : float, default 5.0
        Social gravitational constant.
    horizon_s : float, default 120.0
        Time horizon for integration in seconds.
    red_light_mass : float, default 50.0
        Gravitational mass for red-light obstacles.

    Returns
    -------
    dict
        Optimization result with keys:

        - ``'green_ns'`` : float -- optimal NS green duration (seconds).
        - ``'green_ew'`` : float -- corresponding EW green duration.
        - ``'cycle_s'``  : float -- total cycle length (green_ns + green_ew + 10).
        - ``'phi_integral'`` : float -- best integrated potential value.
        - ``'phi_fixed_60'`` : float -- reference integral for fixed 60/50 timing.
        - ``'improvement_pct'`` : float -- percent improvement over fixed timing.

    Notes
    -----
    The sweep evaluates 16 candidate timings (15, 20, ..., 90).  For each,
    ``green_ew = max(10, 120 - green_ns - 10)`` ensures a minimum EW green
    of 10 s.  The 10 s clearance (amber) is always included.

    Improvement percentage is computed as::

        improvement_pct = (phi_best - phi_fixed) / |phi_fixed| * 100

    A positive value means the optimized timing has less congestion than the
    fixed 60/50 baseline.
    """
    pedestrian_positions = np.asarray(pedestrian_positions, dtype=np.float64)
    pedestrian_velocities = np.asarray(pedestrian_velocities, dtype=np.float64)
    pedestrian_masses = np.asarray(pedestrian_masses, dtype=np.float64)
    intersection_pos = np.asarray(intersection_pos, dtype=np.float64)

    best = None

    for green_ns in range(15, 91, 5):
        green_ew = max(10, 120 - green_ns - 10)

        phi = estimate_phi_integral(
            pedestrian_positions,
            pedestrian_velocities,
            pedestrian_masses,
            intersection_pos,
            float(green_ns),
            float(green_ew),
            red_light_mass=red_light_mass,
            radius=radius,
            G_s=G_s,
            horizon_s=horizon_s,
        )

        if best is None or phi > best["phi"]:
            best = {
                "green_ns": green_ns,
                "green_ew": green_ew,
                "phi": phi,
            }

    # Reference: fixed 60/50 timing
    phi_fixed = estimate_phi_integral(
        pedestrian_positions,
        pedestrian_velocities,
        pedestrian_masses,
        intersection_pos,
        60.0,
        50.0,
        red_light_mass=red_light_mass,
        radius=radius,
        G_s=G_s,
        horizon_s=horizon_s,
    )

    if phi_fixed != 0.0:
        improvement = (best["phi"] - phi_fixed) / abs(phi_fixed) * 100.0
    else:
        improvement = 0.0

    return {
        "green_ns": float(best["green_ns"]),
        "green_ew": float(best["green_ew"]),
        "cycle_s": float(best["green_ns"] + best["green_ew"] + 10),
        "phi_integral": float(best["phi"]),
        "phi_fixed_60": float(phi_fixed),
        "improvement_pct": float(improvement),
    }
