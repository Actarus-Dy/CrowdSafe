"""CriticalDensityMonitor -- Schwarzschild threshold detection for crowd safety.

Transposition of the Schwarzschild radius (Janus model section 5.6) to crowd
dynamics.  The gravitational collapse threshold r_s = 2GM/c^2 maps to a
critical crowd density rho_c = 6 pers/m^2: beyond this threshold, individual
movement becomes impossible and the crowd behaves as a single compressive body.

Alert levels:
    rho < 2.0 pers/m^2  -> VERT   (free circulation)
    rho in [2.0, 4.0)   -> JAUNE  (constrained flow, monitoring)
    rho in [4.0, 6.0)   -> ORANGE (active surveillance, prepare intervention)
    rho >= 6.0           -> ROUGE  (Schwarzschild exceeded, immediate action)
    rho >= 8.0           -> CRITIQUE (crush pressure, emergency evacuation)

Reference
---------
Janus Civil C-14 CrowdSafe Technical Plan, Section 1.4 (Schwarzschild S5.6).
Fruin (1987) -- Pedestrian Planning and Design (level of service).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

__all__ = ["AlertLevel", "CriticalDensityMonitor", "DensityReport"]


class AlertLevel(Enum):
    """Crowd density alert levels, analogous to Schwarzschild thresholds."""

    VERT = "VERT"
    JAUNE = "JAUNE"
    ORANGE = "ORANGE"
    ROUGE = "ROUGE"
    CRITIQUE = "CRITIQUE"


# Density thresholds in pers/m^2
RHO_FREE: float = 2.0
RHO_CONSTRAINED: float = 4.0
RHO_CRITICAL: float = 6.0
RHO_CRUSH: float = 8.0


@dataclass(frozen=True, slots=True)
class DensityReport:
    """Result of a critical density check."""

    max_density: float
    mean_density: float
    alert_level: AlertLevel
    critical_area_m2: float
    danger_area_m2: float
    critical_mask: npt.NDArray[np.bool_]
    schwarzschild_ratio: float
    schwarzschild_exceeded: bool


class CriticalDensityMonitor:
    """Real-time crowd density monitor with Schwarzschild threshold detection.

    Parameters
    ----------
    rho_critical : float, default 6.0
        Critical density threshold in pers/m^2 (analogous to r_s).
    rho_crush : float, default 8.0
        Crush density threshold in pers/m^2 (emergency level).
    """

    def __init__(
        self,
        rho_critical: float = RHO_CRITICAL,
        rho_crush: float = RHO_CRUSH,
    ) -> None:
        self.rho_critical = float(rho_critical)
        self.rho_crush = float(rho_crush)

    def check(
        self,
        density_map: npt.NDArray[np.float64],
        dx_m: float = 0.5,
    ) -> DensityReport:
        """Evaluate a density grid and return an alert report.

        Parameters
        ----------
        density_map : ndarray, shape (ny, nx), dtype float64
            Crowd density at each grid cell [pers/m^2].
        dx_m : float, default 0.5
            Grid cell size in meters.

        Returns
        -------
        DensityReport
            Alert level, critical zones, and Schwarzschild ratio.
        """
        density_map = np.asarray(density_map, dtype=np.float64)
        area_pixel = dx_m**2

        critical_mask = density_map >= self.rho_critical
        danger_mask = density_map >= RHO_CONSTRAINED

        max_density = float(density_map.max()) if density_map.size > 0 else 0.0
        mean_density = float(density_map.mean()) if density_map.size > 0 else 0.0

        critical_area = float(np.sum(critical_mask) * area_pixel)
        danger_area = float(np.sum(danger_mask) * area_pixel)

        schwarzschild_ratio = max_density / self.rho_critical if self.rho_critical > 0 else 0.0

        alert_level = self._classify(max_density)

        return DensityReport(
            max_density=max_density,
            mean_density=mean_density,
            alert_level=alert_level,
            critical_area_m2=critical_area,
            danger_area_m2=danger_area,
            critical_mask=critical_mask,
            schwarzschild_ratio=schwarzschild_ratio,
            schwarzschild_exceeded=max_density >= self.rho_critical,
        )

    def check_point_densities(
        self,
        densities: npt.NDArray[np.float64],
    ) -> AlertLevel:
        """Quick check on per-agent density values.

        Parameters
        ----------
        densities : ndarray, shape (N,), dtype float64
            Local density at each pedestrian position [pers/m^2].

        Returns
        -------
        AlertLevel
            Highest alert level across all agents.
        """
        if len(densities) == 0:
            return AlertLevel.VERT
        max_rho = float(np.max(densities))
        return self._classify(max_rho)

    def _classify(self, rho: float) -> AlertLevel:
        """Map a density value to an alert level."""
        if rho >= self.rho_crush:
            return AlertLevel.CRITIQUE
        if rho >= self.rho_critical:
            return AlertLevel.ROUGE
        if rho >= RHO_CONSTRAINED:
            return AlertLevel.ORANGE
        if rho >= RHO_FREE:
            return AlertLevel.JAUNE
        return AlertLevel.VERT
