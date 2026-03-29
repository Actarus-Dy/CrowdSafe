"""TOVPressure -- crowd force profile along corridors (Janus §5.5).

Transposition of the Tolman-Oppenheimer-Volkoff equation from stellar
structure to crowd dynamics.  In a star, pressure supports against
gravitational collapse; in a crowd corridor, cumulative contact force
accumulates as density increases toward a bottleneck.

    Star  (§5.5):  dp/dr = -rho * g_eff * (1 + relativistic corrections)
    Crowd:         dF/dl = rho_crowd(l) * F_contact_per_person * width

The integrated quantity F(l) is a **cumulative contact force** [N]:
    F(l) = integral_0^l rho(s) * F_contact * width ds

This is NOT a pressure (N/m²). It represents the total force transmitted
through the crowd cross-section at position l. To convert to pressure,
divide by the corridor cross-section area: P = F / (width * body_depth).

Alert thresholds (cumulative force, calibrated from crowd crush literature):
    F > 2700 N  -> WARNING (compression noticeable, surveillance active)
    F > 4450 N  -> DANGER (thoracic compression risk -> evacuate)

These thresholds are derived from biomechanical estimates: a 4450 N force
distributed across a 0.3m × 1.5m torso cross-section yields ~10 kPa,
which approaches the threshold for respiratory compromise.

Reference
---------
Janus Civil C-14 CrowdSafe Technical Plan, Section 3.2.
Still, G.K. (2014). Introduction to Crowd Science.
Fruin, J. (1993). The causes and prevention of crowd disasters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["TOVPressure", "PressureProfile"]

# Cumulative force thresholds [N]
WARNING_FORCE: float = 2700.0
DANGER_FORCE: float = 4450.0

# Backward compatibility aliases
WARNING_PRESSURE = WARNING_FORCE
DANGER_PRESSURE = DANGER_FORCE


@dataclass(frozen=True, slots=True)
class PressureProfile:
    """Result of a TOV force computation along a corridor."""

    l_m: npt.NDArray[np.float64]
    cumulative_force_N: npt.NDArray[np.float64]
    F_max: float
    alert_level: str
    location_max_m: float
    schwarzschild_ratio: float
    critical_exceeded: bool

    @property
    def pressure_N_m(self) -> npt.NDArray[np.float64]:
        """Backward compatibility alias for cumulative_force_N."""
        return self.cumulative_force_N

    @property
    def P_max(self) -> float:
        """Backward compatibility alias for F_max."""
        return self.F_max


class TOVPressure:
    """Compute crowd pressure profiles along corridors.

    Parameters
    ----------
    F_contact : float, default 100.0
        Typical body contact force per person [N/pers].
    warning_threshold : float, default 2700.0
        Warning pressure threshold [N/m].
    danger_threshold : float, default 4450.0
        Danger pressure threshold [N/m] (thoracic compression).
    """

    def __init__(
        self,
        F_contact: float = 100.0,
        warning_threshold: float = WARNING_PRESSURE,
        danger_threshold: float = DANGER_PRESSURE,
    ) -> None:
        self.F_contact = float(F_contact)
        self.warning_threshold = float(warning_threshold)
        self.danger_threshold = float(danger_threshold)

    def compute(
        self,
        l: npt.NDArray[np.float64],
        rho_profile: npt.NDArray[np.float64],
        width_m: float,
    ) -> PressureProfile:
        """Compute cumulative pressure along a corridor.

        Parameters
        ----------
        l : ndarray, shape (M,)
            Position along corridor [m], monotonically increasing.
        rho_profile : ndarray, shape (M,)
            Crowd density at each position [pers/m²].
        width_m : float
            Corridor width [m].

        Returns
        -------
        PressureProfile
            Pressure profile with alert level and Schwarzschild ratio.
        """
        from scipy.integrate import cumulative_trapezoid

        l = np.asarray(l, dtype=np.float64)
        rho_profile = np.asarray(rho_profile, dtype=np.float64)
        width_m = float(width_m)

        # TOV §5.5: dF/dl = rho(l) * F_contact * width
        # Cumulative force accumulates from entrance toward bottleneck.
        dF_dl = rho_profile * self.F_contact * width_m
        force = cumulative_trapezoid(dF_dl, l, initial=0.0)

        F_max = float(np.max(np.abs(force)))
        idx_max = int(np.argmax(np.abs(force)))

        alert = "VERT"
        if F_max > self.danger_threshold:
            alert = "ROUGE"
        elif F_max > self.warning_threshold:
            alert = "ORANGE"

        return PressureProfile(
            l_m=l,
            cumulative_force_N=force,
            F_max=F_max,
            alert_level=alert,
            location_max_m=float(l[idx_max]),
            schwarzschild_ratio=F_max / self.danger_threshold,
            critical_exceeded=F_max > self.danger_threshold,
        )

    def compute_from_simulation(
        self,
        positions: npt.NDArray[np.float64],
        corridor_axis: int = 0,
        corridor_width: float = 4.0,
        n_bins: int = 50,
    ) -> PressureProfile:
        """Compute pressure profile from live simulation positions.

        Bins pedestrians along the corridor axis and computes the
        density profile, then integrates TOV pressure.

        Parameters
        ----------
        positions : ndarray, shape (N, 2)
            Pedestrian positions.
        corridor_axis : int, default 0
            Axis along corridor (0=x, 1=y).
        corridor_width : float, default 4.0
            Width of corridor perpendicular to axis [m].
        n_bins : int, default 50
            Number of bins along corridor.

        Returns
        -------
        PressureProfile
        """
        coords = positions[:, corridor_axis]
        l_min, l_max = float(np.min(coords)), float(np.max(coords))
        if l_max - l_min < 0.1:
            l = np.array([0.0, 1.0])
            return self.compute(l, np.array([0.0, 0.0]), corridor_width)

        bin_edges = np.linspace(l_min, l_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        counts, _ = np.histogram(coords, bins=bin_edges)
        # Density = count / (bin_width * corridor_width)
        rho_profile = counts.astype(np.float64) / (bin_width * corridor_width)

        return self.compute(bin_centers, rho_profile, corridor_width)
