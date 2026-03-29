"""CrowdSimulation -- full CrowdSafe pipeline integrating all core modules.

Connects :class:`MassAssigner`, :class:`ForceEngine`, the leapfrog integrator,
and the potential-field evaluator into a single coherent simulation loop.

Pipeline per step
-----------------
1. Compute segment mean speed from current velocities.
2. Assign signed gravitational masses via :class:`MassAssigner`.
3. Compute forces via :meth:`ForceEngine.compute_all` (Barnes-Hut O(N log N)).
4. Integrate positions and velocities via :func:`leapfrog_step` (KDK).
5. Optionally recompute an adaptive timestep for the next step.

All arithmetic is float64.  No Python loops on the hot path -- the only loops
live inside the Barnes-Hut tree traversal (which is inherently recursive).

Author: Agent #01 Python Scientific Developer
Date: 2026-03-22
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crowdsafe.core.critical_density import AlertLevel, CriticalDensityMonitor
from crowdsafe.core.force_engine import ForceEngine
from crowdsafe.core.force_engine_gpu import GPU_AVAILABLE, ForceEngineGPU
from crowdsafe.core.force_engine_numba import (
    NUMBA_AVAILABLE,
    ForceEngineBHNumba,
    ForceEngineNumba,
)
from crowdsafe.core.integrator import adaptive_dt, leapfrog_step
from crowdsafe.core.mass_assigner import MassAssigner
from crowdsafe.core.potential_field import compute_potential_field

__all__ = ["CrowdSimulation"]

# Floor for |m_i| when converting force -> acceleration, to avoid instability
# for near-zero mass particles.
_MASS_FLOOR: float = 0.01


class CrowdSimulation:
    """Full CrowdSafe simulation pipeline.

    Parameters
    ----------
    G_s : float, default 2.0
        Social gravitational constant (calibrated for crowd dynamics).
    beta : float, default 1.0
        Mass-assignment exponent (linear for pedestrians).
    softening : float, default 0.5
        Force softening length in meters (personal space radius).
    rho_scale : float, default 2.0
        Reference density for mass normalisation [pers/m²].
    theta : float, default 0.5
        Barnes-Hut opening-angle parameter.
    dt : float, default 0.5
        Base integration timestep in seconds.
    v_max : float, default 2.5
        Maximum pedestrian speed in m/s (running).
    adaptive_dt : bool, default True
        If True, recompute the timestep after each step using the CFL
        condition.  Otherwise use the fixed ``dt``.
    drag_coefficient : float, default 0.5
        Weidmann drag coefficient (gamma). When > 0, adds a drag
        enrichment term: ``a_drag = gamma * (v_eq(rho) - |v|) * direction``.
        Set to 0.0 for pure gravity (no drag).
    v_free : float, default 1.34
        Free-flow walking speed in m/s (4.8 km/h, Weidmann model).
    rho_jam : float, default 6.0
        Critical crowd density in pers/m² (Schwarzschild threshold).
    density_radius : float, default 5.0
        Neighborhood radius in meters for local density calculation.
    use_gpu : bool or None, default None
        If True, use CuPy GPU-accelerated force engine. If False, use CPU.
        If None (default), auto-detect: use GPU if CuPy is available.

    Attributes
    ----------
    positions : ndarray, shape (N, 2), dtype float64
        Current pedestrian positions.
    velocities : ndarray, shape (N, 2), dtype float64
        Current pedestrian velocities.
    masses : ndarray, shape (N,), dtype float64
        Most recently assigned gravitational masses.
    local_densities : ndarray, shape (N,), dtype float64
        Local crowd density at each pedestrian [pers/m²].
    step_count : int
        Number of completed simulation steps.
    """

    def __init__(
        self,
        G_s: float = 2.0,
        beta: float = 1.0,
        softening: float = 0.5,
        rho_scale: float = 2.0,
        theta: float = 0.5,
        dt: float = 0.5,
        v_max: float = 2.5,
        adaptive_dt: bool = True,
        drag_coefficient: float = 0.5,
        v_free: float = 1.34,
        rho_jam: float = 6.0,
        density_radius: float = 5.0,
        body_radius: float = 0.2,
        contact_strength: float = 2000.0,
        contact_range: float = 0.08,
        use_gpu: bool | None = None,
    ) -> None:
        self.G_s: float = float(G_s)
        self.theta: float = float(theta)
        self.dt: float = float(dt)
        self.v_max: float = float(v_max)
        self.use_adaptive_dt: bool = adaptive_dt

        # Drag enrichment parameters (Weidmann equilibrium speed model).
        # When drag_coefficient > 0, an additional acceleration term is applied:
        #   a_drag_i = gamma * (v_eq(rho_i) - |v_i|) * direction_i
        # where v_eq(rho) = v_free * (1 - exp(-1.913 * (1/rho - 1/rho_jam)))
        # This represents self-propulsion vs crowd friction.  Gravity still
        # provides inter-pedestrian social force interactions.
        self._drag_coefficient: float = float(drag_coefficient)
        self._v_free: float = float(v_free)
        self._rho_jam: float = float(rho_jam)
        self._density_radius: float = float(density_radius)

        # Body contact force parameters (Helbing et al. 2000).
        # Exponential repulsion at close range: F = A * exp((2*r - d) / B) * n_ij
        # where r = body_radius, d = center-to-center distance, B = contact_range.
        self._body_radius: float = float(body_radius)
        self._contact_strength: float = float(contact_strength)
        self._contact_range: float = float(contact_range)

        # GPU auto-detection
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        self.use_gpu: bool = use_gpu and GPU_AVAILABLE

        # Sub-modules — engine auto-selection: GPU > Numba > Python
        # Numba naive O(N²) is fastest for N < ~2000 due to JIT + N3L.
        # Numba BH O(N log N) wins for larger N. GPU wins for N < max_n.
        self._mass_assigner = MassAssigner(beta=beta, rho_scale=rho_scale)
        if self.use_gpu:
            self._force_engine = ForceEngineGPU(G_s=G_s, softening=softening)
        elif NUMBA_AVAILABLE:
            self._force_engine = ForceEngineNumba(G_s=G_s, softening=softening)
            self._force_engine_bh = ForceEngineBHNumba(G_s=G_s, softening=softening)
        else:
            self._force_engine = ForceEngine(G_s=G_s, softening=softening)

        # Destination force relaxation time (Helbing 1995).
        # F_dest = (v_desired - v_i) / tau
        # where v_desired = v_free * desired_direction_i
        self._tau: float = 0.5  # relaxation time [s]

        # State arrays -- set by init_pedestrians
        self.positions: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.velocities: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self.local_densities: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.masses: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

        # Per-pedestrian desired direction (unit vector toward destination).
        # Shape (N, 2). If None, no destination force is applied and
        # the drag enrichment provides the baseline speed regulation.
        self.desired_directions: npt.NDArray[np.float64] | None = None

        # Internal force cache (accelerations) for leapfrog continuity
        self._forces: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)

        # Static point obstacles (e.g. columns, barriers).
        # Convention: negative mass = repulsive (repels all pedestrians).
        #             positive mass = attractive (herding, curiosity).
        self._obstacle_positions: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self._obstacle_masses: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

        # Wall segments: line-based repulsive boundaries.
        # Each wall is defined by two endpoints (p1, p2).
        # Repulsive force: F = A_wall * exp(-d_perp / B_wall) * normal_away
        self._wall_p1: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self._wall_p2: npt.NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
        self._wall_strength: float = 3000.0  # repulsion strength [N]
        self._wall_range: float = 0.1  # decay length [m]

        # Bookkeeping
        self.step_count: int = 0
        self._mean_speed: float = 0.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_pedestrians(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        local_densities: np.ndarray,
    ) -> None:
        """Set initial conditions for the pedestrian population.

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            Initial (x, y) positions of all pedestrians.
        velocities : ndarray, shape (N, 2), dtype float64
            Initial (vx, vy) velocities of all pedestrians.
        local_densities : ndarray, shape (N,), dtype float64
            Local crowd density at each pedestrian position [pers/m²].
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.local_densities = np.asarray(local_densities, dtype=np.float64)

        n = len(self.local_densities)
        if self.positions.shape != (n, 2):
            raise ValueError(
                f"positions shape {self.positions.shape} incompatible with "
                f"{n} densities; expected ({n}, 2)"
            )
        if self.velocities.shape != (n, 2):
            raise ValueError(
                f"velocities shape {self.velocities.shape} incompatible with "
                f"{n} densities; expected ({n}, 2)"
            )

        # Compute initial masses and forces so that the first leapfrog step
        # has a valid force cache.
        self._mean_speed = self._compute_mean_speed()
        self.masses = self._mass_assigner.assign(
            self._speeds(), self._mean_speed, self.local_densities
        )
        self._forces = self._compute_accelerations(self.positions)
        self.step_count = 0

    def set_desired_directions(
        self,
        directions: np.ndarray,
    ) -> None:
        """Set per-pedestrian desired direction vectors.

        When set, a destination force is applied at each step:
            F_dest_i = (v_desired_i - v_i) / tau
        where v_desired_i = v_free * directions[i].

        Parameters
        ----------
        directions : ndarray, shape (N, 2), dtype float64
            Unit direction vectors toward each pedestrian's destination.
            Will be normalized to unit length.  Pass None to disable.
        """
        if directions is None:
            self.desired_directions = None
            return
        directions = np.asarray(directions, dtype=np.float64)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        self.desired_directions = directions / norms

    # ------------------------------------------------------------------
    # State cloning (for prediction without mutating the live sim)
    # ------------------------------------------------------------------
    def clone(self) -> CrowdSimulation:
        """Create an independent deep copy of this simulation.

        The clone shares no mutable state with the original -- modifying
        one does not affect the other.  This is used for prediction:
        clone the live simulation, run the clone forward T seconds, and
        read off the predicted state.

        Returns
        -------
        CrowdSimulation
            A new simulation with identical configuration and state.
        """
        c = CrowdSimulation(
            G_s=self.G_s,
            beta=self._mass_assigner.beta,
            softening=self._force_engine.epsilon,
            rho_scale=self._mass_assigner.rho_scale,
            theta=self.theta,
            dt=self.dt,
            v_max=self.v_max,
            adaptive_dt=self.use_adaptive_dt,
            drag_coefficient=self._drag_coefficient,
            v_free=self._v_free,
            rho_jam=self._rho_jam,
            density_radius=self._density_radius,
            body_radius=self._body_radius,
            contact_strength=self._contact_strength,
            contact_range=self._contact_range,
            use_gpu=self.use_gpu,
        )
        # Deep-copy all state arrays
        c.positions = self.positions.copy()
        c.velocities = self.velocities.copy()
        c.local_densities = self.local_densities.copy()
        c.masses = self.masses.copy()
        c._forces = self._forces.copy()
        c._obstacle_positions = self._obstacle_positions.copy()
        c._obstacle_masses = self._obstacle_masses.copy()
        c._wall_p1 = self._wall_p1.copy()
        c._wall_p2 = self._wall_p2.copy()
        c._wall_strength = self._wall_strength
        c._wall_range = self._wall_range
        c.step_count = self.step_count
        c._mean_speed = self._mean_speed
        if self.desired_directions is not None:
            c.desired_directions = self.desired_directions.copy()
        return c

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, horizon_s: float) -> dict:
        """Run a cloned simulation forward and return the predicted state.

        Parameters
        ----------
        horizon_s : float
            Prediction horizon in seconds (e.g. 900 for T+15min).

        Returns
        -------
        dict
            Keys: ``'positions'``, ``'velocities'``, ``'masses'``,
            ``'mean_speed'``, ``'step_count'``, ``'horizon_s'``,
            ``'n_steps_run'``.
        """
        if self.n_pedestrians == 0:
            return {
                "positions": np.empty((0, 2), dtype=np.float64),
                "velocities": np.empty((0, 2), dtype=np.float64),
                "masses": np.empty(0, dtype=np.float64),
                "mean_speed": 0.0,
                "step_count": self.step_count,
                "horizon_s": 0.0,
                "n_steps_run": 0,
            }

        clone = self.clone()
        return clone.run_until(horizon_s)

    def run_until(self, horizon_s: float) -> dict:
        """Run THIS simulation forward for *horizon_s* seconds (in-place).

        Unlike :meth:`predict`, this does NOT clone -- it mutates ``self``.
        Use on a clone obtained via :meth:`clone` to avoid modifying the
        live simulation.

        Parameters
        ----------
        horizon_s : float
            Time horizon in seconds.

        Returns
        -------
        dict
            Same keys as :meth:`predict`.
        """
        elapsed = 0.0
        n_steps = 0
        max_steps = max(1, int(horizon_s / 0.005))
        while elapsed < horizon_s and n_steps < max_steps:
            dt_before = self.dt
            self.step()
            elapsed += dt_before
            n_steps += 1

        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "masses": self.masses.copy(),
            "mean_speed": float(np.mean(np.linalg.norm(self.velocities, axis=1)))
            if self.n_pedestrians > 0
            else 0.0,
            "step_count": self.step_count,
            "horizon_s": elapsed,
            "n_steps_run": n_steps,
        }

    # ------------------------------------------------------------------
    # Dynamic pedestrian injection / removal
    # ------------------------------------------------------------------
    def add_pedestrians(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        local_densities: np.ndarray,
    ) -> npt.NDArray[np.intp]:
        """Add K pedestrians to the running simulation.

        Parameters
        ----------
        positions : array_like, shape (K, 2)
            Positions of the new pedestrians.
        velocities : array_like, shape (K, 2)
            Velocities of the new pedestrians.
        local_densities : array_like, shape (K,)
            Local crowd density at each new pedestrian [pers/m²].

        Returns
        -------
        ndarray, shape (K,), dtype intp
            Indices assigned to the new pedestrians in the state arrays.
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
        local_densities = np.asarray(local_densities, dtype=np.float64).ravel()

        k = len(positions)
        old_n = len(self.positions)

        self.positions = np.vstack([self.positions, positions])
        self.velocities = np.vstack([self.velocities, velocities])
        self.local_densities = np.concatenate([self.local_densities, local_densities])

        # Compute masses for new pedestrians using current mean speed
        new_masses = self._mass_assigner.assign(
            np.linalg.norm(velocities, axis=1),
            self._mean_speed,
            local_densities,
        )
        self.masses = np.concatenate([self.masses, new_masses])

        # Extend forces array (zero initial force for new pedestrians)
        self._forces = np.vstack([self._forces, np.zeros((k, 2), dtype=np.float64)])

        return np.arange(old_n, old_n + k)

    def remove_pedestrians(self, indices: np.ndarray) -> None:
        """Remove pedestrians at given indices from the simulation.

        Parameters
        ----------
        indices : array_like, dtype intp
            Indices of pedestrians to remove.  Must be valid indices into
            the current state arrays.
        """
        indices = np.asarray(indices, dtype=np.intp)
        mask = np.ones(len(self.positions), dtype=bool)
        mask[indices] = False

        self.positions = self.positions[mask]
        self.velocities = self.velocities[mask]
        self.local_densities = self.local_densities[mask]
        self.masses = self.masses[mask]
        self._forces = self._forces[mask]

    @property
    def n_pedestrians(self) -> int:
        """Current number of pedestrians in the simulation."""
        return len(self.positions)

    # ------------------------------------------------------------------
    # Obstacle management (red-light masses, barriers, etc.)
    # ------------------------------------------------------------------
    def set_obstacles(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> None:
        """Set static obstacle positions and masses for force computation.

        Obstacles participate in force calculation (they exert forces on
        pedestrians) but are NOT integrated -- their positions do not change.

        Parameters
        ----------
        positions : array-like, shape (K, 2), dtype float64
            (x, y) positions of the K obstacles.
        masses : array-like, shape (K,), dtype float64
            Signed gravitational mass for each obstacle.

        Raises
        ------
        ValueError
            If *positions* and *masses* have incompatible shapes.
        """
        self._obstacle_positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        self._obstacle_masses = np.asarray(masses, dtype=np.float64).ravel()

        k = len(self._obstacle_masses)
        if self._obstacle_positions.shape != (k, 2):
            raise ValueError(
                f"obstacle positions shape {self._obstacle_positions.shape} "
                f"incompatible with {k} masses; expected ({k}, 2)"
            )

    def clear_obstacles(self) -> None:
        """Remove all obstacles from the simulation."""
        self._obstacle_positions = np.empty((0, 2), dtype=np.float64)
        self._obstacle_masses = np.empty(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Wall management (line-segment boundaries)
    # ------------------------------------------------------------------
    def set_walls(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        strength: float = 3000.0,
        decay: float = 0.1,
    ) -> None:
        """Set wall segments as repulsive line boundaries.

        Each wall is defined by two endpoints. Pedestrians within range
        experience exponential repulsion perpendicular to the wall.

        Parameters
        ----------
        p1 : array-like, shape (W, 2)
            Start points of W wall segments.
        p2 : array-like, shape (W, 2)
            End points of W wall segments.
        strength : float, default 3000.0
            Repulsion strength in Newtons.
        decay : float, default 0.1
            Exponential decay length in meters.
        """
        self._wall_p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
        self._wall_p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
        self._wall_strength = float(strength)
        self._wall_range = float(decay)

    def clear_walls(self) -> None:
        """Remove all walls from the simulation."""
        self._wall_p1 = np.empty((0, 2), dtype=np.float64)
        self._wall_p2 = np.empty((0, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self) -> dict:
        """Execute one simulation step.

        Returns
        -------
        dict
            Step results with keys:

            - ``'positions'``: (N, 2) ndarray
            - ``'velocities'``: (N, 2) ndarray
            - ``'masses'``: (N,) ndarray
            - ``'mean_speed'``: float
            - ``'dt_used'``: float
            - ``'step_count'``: int
        """
        dt = self.dt

        # Leapfrog KDK integration.  The force_fn callback recomputes
        # masses at the new positions (using the current mean speed) and
        # converts the resulting forces to accelerations.
        pos_new, vel_new, forces_new = leapfrog_step(
            self.positions,
            self.velocities,
            self._forces,
            dt,
            force_fn=self._compute_accelerations,
            v_max=self.v_max,
        )

        # Commit new state
        self.positions = pos_new
        self.velocities = vel_new
        self._forces = forces_new

        # Update local densities from current positions (once per step)
        self.local_densities = self._compute_local_densities(self.positions)

        # Recompute mean speed and masses at the committed state
        self._mean_speed = self._compute_mean_speed()
        self.masses = self._mass_assigner.assign(
            self._speeds(), self._mean_speed, self.local_densities
        )

        self.step_count += 1

        # Adaptive timestep for the next step
        if self.use_adaptive_dt:
            self.dt = adaptive_dt(self.positions, self.velocities)

        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "masses": self.masses.copy(),
            "mean_speed": self._mean_speed,
            "dt_used": dt,
            "step_count": self.step_count,
        }

    # ------------------------------------------------------------------
    # Safety monitoring (TOV + Geodesics + Critical Density)
    # ------------------------------------------------------------------
    def check_safety(
        self,
        corridor_axis: int = 0,
        corridor_width: float = 4.0,
        exits: list | None = None,
        density_grid_shape: tuple[int, int] | None = None,
        dx_m: float = 0.5,
    ) -> dict:
        """Run a comprehensive safety check on the current simulation state.

        Integrates TOV pressure, critical density, and evacuation geodesics
        into a single safety report.

        Parameters
        ----------
        corridor_axis : int, default 0
            Axis along which to compute TOV pressure (0=x, 1=y).
        corridor_width : float, default 4.0
            Width of corridor for TOV computation [m].
        exits : list of ndarray, optional
            Exit positions for geodesic computation. If None, geodesics
            are skipped.
        density_grid_shape : tuple (ny, nx), optional
            Grid shape for geodesic density map. If None, auto-computed.
        dx_m : float, default 0.5
            Grid cell size for geodesic computation [m].

        Returns
        -------
        dict with keys:
            - 'density_alert': AlertLevel (highest density alert)
            - 'max_density': float (peak local density)
            - 'tov_profile': PressureProfile (corridor force profile)
            - 'tov_alert': str ('VERT', 'ORANGE', 'ROUGE')
            - 'evacuation': EvacuationResult or None (if exits provided)
            - 'overall_alert': str (worst alert across all checks)
        """
        from crowdsafe.core.tov_pressure import TOVPressure

        result: dict = {}

        # 1. Critical density check
        monitor = CriticalDensityMonitor()
        density_alert = monitor.check_point_densities(self.local_densities)
        result["density_alert"] = density_alert
        result["max_density"] = float(np.max(self.local_densities)) \
            if len(self.local_densities) > 0 else 0.0

        # 2. TOV pressure profile
        tov = TOVPressure()
        tov_profile = tov.compute_from_simulation(
            self.positions,
            corridor_axis=corridor_axis,
            corridor_width=corridor_width,
        )
        result["tov_profile"] = tov_profile
        result["tov_alert"] = tov_profile.alert_level

        # 3. Evacuation geodesics (optional)
        if exits is not None and len(exits) > 0:
            from crowdsafe.core.evacuation_geodesic import EvacuationGeodesic

            geo = EvacuationGeodesic(v_max=self._v_free, dx_m=dx_m)

            # Build density grid from positions
            if density_grid_shape is None:
                x_range = float(np.ptp(self.positions[:, 0])) + 2 * dx_m
                y_range = float(np.ptp(self.positions[:, 1])) + 2 * dx_m
                nx_grid = max(10, int(x_range / dx_m))
                ny_grid = max(10, int(y_range / dx_m))
            else:
                ny_grid, nx_grid = density_grid_shape

            x_min = float(np.min(self.positions[:, 0])) - dx_m
            y_min = float(np.min(self.positions[:, 1])) - dx_m

            density_grid = np.zeros((ny_grid, nx_grid), dtype=np.float64)
            for pos in self.positions:
                ix = min(max(round((pos[0] - x_min) / dx_m), 0), nx_grid - 1)
                iy = min(max(round((pos[1] - y_min) / dx_m), 0), ny_grid - 1)
                density_grid[iy, ix] += 1.0 / (dx_m * dx_m)

            # Adjust exit coordinates relative to grid origin
            adjusted_exits = [
                np.array([ex[0] - x_min, ex[1] - y_min]) for ex in exits
            ]

            # Compute distance map from exits
            dist_map = geo.compute_distance_map(density_grid, adjusted_exits)
            result["evacuation_distance_map"] = dist_map
            result["evacuation_max_time"] = float(np.min([
                dist_map[
                    min(max(round((pos[1] - y_min) / dx_m), 0), ny_grid - 1),
                    min(max(round((pos[0] - x_min) / dx_m), 0), nx_grid - 1),
                ]
                for pos in self.positions
            ])) if len(self.positions) > 0 else 0.0
        else:
            result["evacuation_distance_map"] = None
            result["evacuation_max_time"] = None

        # 4. Overall alert (worst of all checks)
        alerts = [density_alert.value, tov_profile.alert_level]
        alert_order = {"VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3, "CRITIQUE": 4}
        result["overall_alert"] = max(alerts, key=lambda a: alert_order.get(a, 0))

        return result

    # ------------------------------------------------------------------
    # Multi-step runner
    # ------------------------------------------------------------------
    def run(self, n_steps: int) -> list[dict]:
        """Run *n_steps* simulation steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to execute.

        Returns
        -------
        list[dict]
            List of length *n_steps*, each element being the dict returned
            by :meth:`step`.
        """
        return [self.step() for _ in range(n_steps)]

    # ------------------------------------------------------------------
    # Potential field
    # ------------------------------------------------------------------
    def get_potential_field(self, grid_centers: np.ndarray) -> npt.NDArray[np.float64]:
        """Compute the gravitational potential field at the current state.

        Parameters
        ----------
        grid_centers : ndarray, shape (M, 2), dtype float64
            Evaluation points where the potential is computed.

        Returns
        -------
        ndarray, shape (M,), dtype float64
            Scalar potential at each grid point.
        """
        return compute_potential_field(self.positions, self.masses, grid_centers, G_s=self.G_s)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_local_densities(
        self, positions: np.ndarray, radius: float | None = None
    ) -> npt.NDArray[np.float64]:
        """Compute local crowd density for each pedestrian.

        Counts pedestrians within a Euclidean radius and converts to
        pers/m² using circular area.  Uses ``scipy.spatial.cKDTree``
        for O(N log N) vectorized neighbor counting.

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            Current pedestrian positions.
        radius : float or None
            Neighborhood radius in meters. If None, uses
            ``self._density_radius`` (default 5.0 m).

        Returns
        -------
        densities : ndarray, shape (N,), dtype float64
            Local crowd density at each pedestrian [pers/m²].
        """
        from scipy.spatial import cKDTree

        if radius is None:
            radius = self._density_radius

        n = len(positions)
        if n <= 1:
            area = np.pi * radius**2
            return np.full(n, 1.0 / area, dtype=np.float64)

        tree = cKDTree(positions)
        counts = tree.query_ball_point(positions, r=radius, return_length=True)
        counts = np.asarray(counts, dtype=np.float64)

        # Convert count in radius to pers/m² (2D area-based density)
        area = np.pi * radius**2
        densities = counts / area

        return densities

    def _compute_wall_forces(
        self,
        positions: npt.NDArray[np.float64],
        abs_masses: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute repulsive forces from wall segments.

        For each pedestrian, finds the nearest point on each wall segment
        and applies exponential repulsion: F = A * exp(-d / B) * normal.

        Returns accelerations (N, 2).
        """
        n = len(positions)
        wall_accel = np.zeros((n, 2), dtype=np.float64)
        cutoff = 5 * self._wall_range + self._body_radius  # interaction range

        for k in range(len(self._wall_p1)):
            p1 = self._wall_p1[k]
            p2 = self._wall_p2[k]
            seg = p2 - p1
            seg_len_sq = np.dot(seg, seg)
            if seg_len_sq < 1e-12:
                continue

            # Project each pedestrian onto the wall segment
            # t = clamp(dot(pos - p1, seg) / |seg|², 0, 1)
            ap = positions - p1  # (N, 2)
            t = np.clip(np.dot(ap, seg) / seg_len_sq, 0.0, 1.0)  # (N,)

            # Nearest point on segment
            nearest = p1 + t[:, np.newaxis] * seg  # (N, 2)
            diff = positions - nearest  # (N, 2) away from wall
            dist = np.linalg.norm(diff, axis=1)  # (N,)

            # Only apply force within cutoff
            active = dist < cutoff
            if not np.any(active):
                continue

            dist_safe = np.maximum(dist[active], 1e-6)
            normal = diff[active] / dist_safe[:, np.newaxis]

            # Exponential repulsion
            overlap = self._body_radius - dist[active]
            force_mag = self._wall_strength * np.exp(
                np.clip(overlap / self._wall_range, -10, 10)
            )
            force_vec = force_mag[:, np.newaxis] * normal

            # Use fixed pedestrian mass (80 kg) for force-to-acceleration
            # conversion. Wall forces are physical (Newtons), not gravitational.
            pedestrian_mass_kg = 80.0
            idx = np.where(active)[0]
            wall_accel[idx] += force_vec / pedestrian_mass_kg

        return wall_accel

    def _compute_contact_forces(
        self,
        positions: npt.NDArray[np.float64],
        abs_masses: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute short-range body contact forces (Helbing et al. 2000).

        Exponential repulsion when center-to-center distance d < 2*body_radius:
            F = A * exp((2r - d) / B) * unit_normal_away

        Uses cKDTree for O(N log N) neighbor search within contact range.

        Returns accelerations (N, 2), not forces.
        """
        from scipy.spatial import cKDTree

        n = len(positions)
        contact_accel = np.zeros((n, 2), dtype=np.float64)

        # Only check pairs within interaction range (2*body_radius + 3*contact_range)
        cutoff = 2 * self._body_radius + 3 * self._contact_range
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

        if len(pairs) == 0:
            return contact_accel

        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        dx = positions[idx_j] - positions[idx_i]  # (P, 2)
        dist = np.linalg.norm(dx, axis=1)  # (P,)
        dist_safe = np.maximum(dist, 1e-6)

        # Overlap: 2*r - d (positive when bodies overlap)
        overlap = 2 * self._body_radius - dist  # (P,)

        # Only compute force where there is overlap or near-overlap
        active = overlap > -3 * self._contact_range
        if not np.any(active):
            return contact_accel

        # Force magnitude: A * exp(overlap / B)
        force_mag = self._contact_strength * np.exp(
            np.clip(overlap[active] / self._contact_range, -10, 10)
        )

        # Unit normal from j to i (repulsion pushes i away from j)
        normal = -dx[active] / dist_safe[active, np.newaxis]  # (A, 2)
        force_vec = force_mag[:, np.newaxis] * normal  # (A, 2)

        # Convert to acceleration: a = F / m
        ai = idx_i[active]
        aj = idx_j[active]
        accel_i = force_vec / abs_masses[ai, np.newaxis]
        accel_j = -force_vec / abs_masses[aj, np.newaxis]

        # Accumulate (Newton's 3rd law)
        np.add.at(contact_accel, ai, accel_i)
        np.add.at(contact_accel, aj, accel_j)

        return contact_accel

    def _speeds(self) -> npt.NDArray[np.float64]:
        """Return speed magnitudes for all pedestrians.  Shape (N,)."""
        return np.linalg.norm(self.velocities, axis=1)

    def _compute_mean_speed(self) -> float:
        """Return the population mean speed."""
        if self.velocities.shape[0] == 0:
            return 0.0
        return float(np.mean(self._speeds()))

    def _compute_accelerations(
        self,
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Force callback for the leapfrog integrator.

        1. Recompute masses at *positions* using the current mean speed.
        2. Compute forces via Barnes-Hut.
        3. Convert forces to accelerations: a_i = F_i / max(|m_i|, 0.01).

        Parameters
        ----------
        positions : ndarray, shape (N, 2), dtype float64
            Particle positions (may differ from ``self.positions`` during
            the leapfrog drift sub-step).
        velocities : ndarray, shape (N, 2), dtype float64, optional
            Particle velocities to use for mass assignment and drag.
            When called from the leapfrog integrator this is ``v_half``
            (the half-kick velocities), ensuring symplecticity.
            If *None*, falls back to ``self.velocities`` (used during
            init_pedestrians bootstrap).

        Returns
        -------
        ndarray, shape (N, 2), dtype float64
            Acceleration vectors for all pedestrians.
        """
        # Use provided velocities (v_half from leapfrog) when available,
        # otherwise fall back to self.velocities (bootstrap / init).
        if velocities is None:
            velocities = self.velocities

        # Recompute masses at the (possibly drifted) positions.
        # We use the current mean speed -- it does not change within a step.
        speeds = np.linalg.norm(velocities, axis=1)
        masses = self._mass_assigner.assign(speeds, self._mean_speed, self.local_densities)

        n_pedestrians = len(masses)

        # Concatenate static obstacles (red-light masses, etc.) so they
        # participate in force computation.  Obstacles exert forces on
        # pedestrians but are NOT integrated -- we slice them off afterwards.
        if len(self._obstacle_masses) > 0:
            all_positions = np.vstack([positions, self._obstacle_positions])
            all_masses = np.concatenate([masses, self._obstacle_masses])
        else:
            all_positions = positions
            all_masses = masses

        # Force computation — auto-select engine by N:
        # Numba naive for N < 2000, Numba BH for N >= 2000 (if available)
        engine = self._force_engine
        if hasattr(self, "_force_engine_bh") and len(all_masses) >= 2000:
            engine = self._force_engine_bh
        all_forces = engine.compute_all(all_positions, all_masses, theta=self.theta)

        # Keep only the pedestrian forces; discard forces on obstacles.
        forces = all_forces[:n_pedestrians]

        # Convert force -> acceleration: a = F / |m|, with floor to avoid
        # division by zero for near-zero mass particles.
        abs_masses = np.maximum(np.abs(masses), _MASS_FLOOR)  # (N,)
        accelerations = forces / abs_masses[:, np.newaxis]  # (N, 2)

        # --- Body contact forces (Helbing et al. 2000) ---
        # Exponential repulsion when pedestrians overlap (d < 2*body_radius).
        # F_contact = A * exp((2r - d) / B) * unit_normal
        # This prevents interpenetration and models crush forces at high density.
        if self._contact_strength > 0 and n_pedestrians > 1:
            accelerations += self._compute_contact_forces(positions, abs_masses)

        # --- Wall repulsion forces ---
        if len(self._wall_p1) > 0:
            accelerations += self._compute_wall_forces(positions, abs_masses)

        # --- Destination force (Helbing 1995 social force model) ---
        # F_dest = (v_desired - v_i) / tau
        # where v_desired = v_free * desired_direction_i
        if self.desired_directions is not None and len(self.desired_directions) == n_pedestrians:
            v_desired = self._v_free * self.desired_directions  # (N, 2)
            accel_dest = (v_desired - velocities) / self._tau  # (N, 2)
            accelerations += accel_dest

        # --- Drag enrichment (Weidmann equilibrium speed model) ---
        # Self-propulsion vs crowd friction.
        # a_drag_i = gamma * (v_eq(rho_i) - |v_i|) * direction_i
        # When |v_i| > v_eq: deceleration.  When |v_i| < v_eq: acceleration.
        if self._drag_coefficient > 0:
            speed = np.linalg.norm(velocities, axis=1, keepdims=True)  # (N, 1)
            speed_scalar = speed.ravel()  # (N,)

            # Weidmann (1993) equilibrium speed from local density
            # v_eq(rho) = v_free * (1 - exp(-1.913 * (1/rho - 1/rho_jam)))
            # At rho→0: v_eq→v_free.  At rho=rho_jam: v_eq→0.
            rho_safe = np.maximum(self.local_densities, 0.01)  # avoid div/0
            exponent = -1.913 * (1.0 / rho_safe - 1.0 / self._rho_jam)
            v_eq = self._v_free * np.maximum(
                0.0, 1.0 - np.exp(exponent)
            )  # (N,)

            # Unit direction vector (along current velocity); fallback to +x
            # for stationary pedestrians to avoid zero-division.
            safe_speed = np.maximum(speed, 1e-6)
            direction = np.where(
                speed > 1e-6,
                velocities / safe_speed,
                np.array([[1.0, 0.0]], dtype=np.float64),
            )  # (N, 2)

            # Drag acceleration: scalar (v_eq - |v|) applied along direction
            drag_scalar = self._drag_coefficient * (v_eq - speed_scalar)  # (N,)
            accelerations += drag_scalar[:, np.newaxis] * direction  # (N, 2)

        return accelerations
