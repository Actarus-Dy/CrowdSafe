"""MassAssigner -- gravitational mass computation for CrowdSafe pedestrians.

Each pedestrian *i* is assigned a signed gravitational mass at every timestep
according to the deviation of its speed from the mean flow speed:

    m_i = sgn(v_mean - v_i) * |v_mean - v_i|^beta * rho(x_i) / rho_0

where
    v_i           -- instantaneous speed of pedestrian *i*  [m/s]
    v_mean        -- mean flow speed across the population [m/s]
    beta          -- exponent controlling nonlinearity (default 1.0)
    rho(x_i)      -- local crowd density at pedestrian position [pers/m²]
    rho_0         -- reference density scale (``rho_scale``) [pers/m²]

Classification thresholds (|m| compared to 0.1):
    * m >  +0.1  ->  "slow"    (positive mass, attractor, herding seed)
    * m <  -0.1  ->  "fast"    (negative mass, repulsor, fluid zone)
    * |m| <= 0.1 ->  "neutral"

Reference
---------
Janus Civil C-14 CrowdSafe Technical Plan, Section 1.2.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MassAssigner"]

# ---------------------------------------------------------------------------
# Classification threshold (absolute mass value)
# ---------------------------------------------------------------------------
_NEUTRAL_THRESHOLD: float = 0.1


class MassAssigner:
    """Compute and classify gravitational masses for a pedestrian population.

    Parameters
    ----------
    beta : float, default 1.0
        Exponent applied to the absolute speed deviation.  ``beta = 1``
        gives a linear relationship; ``beta > 1`` amplifies large
        deviations; ``0 < beta < 1`` compresses them.
    rho_scale : float, default 2.0
        Reference density ``rho_0`` used for normalisation [pers/m²].
        Must be strictly positive.

    Raises
    ------
    ValueError
        If *beta* is negative or *rho_scale* is non-positive.
    """

    def __init__(self, beta: float = 1.0, rho_scale: float = 2.0) -> None:
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if rho_scale <= 0.0:
            raise ValueError(f"rho_scale must be positive, got {rho_scale}")
        self.beta: float = float(beta)
        self.rho_scale: float = float(rho_scale)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def assign(
        self,
        speeds: np.ndarray,
        v_mean: float,
        local_densities: np.ndarray,
    ) -> np.ndarray:
        """Compute gravitational mass for every pedestrian.

        Parameters
        ----------
        speeds : np.ndarray, shape (N,)
            Instantaneous speed of each pedestrian [m/s].  Must be float64.
        v_mean : float
            Mean flow speed [m/s].
        local_densities : np.ndarray, shape (N,)
            Local crowd density at each pedestrian position [pers/m²].

        Returns
        -------
        np.ndarray, shape (N,), dtype float64
            Signed gravitational mass for each pedestrian.

        Notes
        -----
        Fully vectorized -- no Python-level loops.
        """
        speeds = np.asarray(speeds, dtype=np.float64)
        local_densities = np.asarray(local_densities, dtype=np.float64)
        v_mean = float(v_mean)

        # delta = v_mean - v_i  (positive when pedestrian is slower than mean)
        delta: np.ndarray = v_mean - speeds

        # signed power: sgn(delta) * |delta|^beta
        abs_delta: np.ndarray = np.abs(delta)
        signed_power: np.ndarray = np.sign(delta) * np.power(abs_delta, self.beta)

        # density normalisation
        masses: np.ndarray = signed_power * (local_densities / self.rho_scale)

        return masses

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify(self, masses: np.ndarray) -> np.ndarray:
        """Classify each mass as ``'slow'``, ``'fast'``, or ``'neutral'``.

        Parameters
        ----------
        masses : np.ndarray, shape (N,)
            Signed gravitational masses (output of :meth:`assign`).

        Returns
        -------
        np.ndarray, shape (N,), dtype '<U7'
            Label for each pedestrian.
        """
        masses = np.asarray(masses, dtype=np.float64)
        labels = np.full(masses.shape, "neutral", dtype="<U7")
        labels[masses > _NEUTRAL_THRESHOLD] = "slow"
        labels[masses < -_NEUTRAL_THRESHOLD] = "fast"
        return labels
