"""EvacuationGeodesic -- optimal evacuation paths via density-weighted Dijkstra.

Transposition of geodesic equations (Janus §4.4) to crowd evacuation.
In general relativity, light follows geodesics — the shortest path in
curved spacetime.  Here, pedestrians follow the fastest path through a
density field, where high density slows walking speed (Weidmann 1993).

    Metric (§4.5):   g_uv = 1/v²(rho) * delta_uv
    Walking speed:   v(rho) = v_max * max(0, 1 - rho/rho_c)
    Christoffel:     Gamma ~ grad(rho) -> path refraction toward low density

Algorithm: multi-source Dijkstra on a density grid, with cost = dx/v(rho).
Zones where rho >= rho_c are impassable (analogous to r < r_s).

Reference
---------
Janus Civil C-14 CrowdSafe Technical Plan, Section 3.4.
Weidmann, U. (1993). Transporttechnik der Fussgänger.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["EvacuationGeodesic", "EvacuationResult"]


@dataclass(frozen=True, slots=True)
class EvacuationResult:
    """Result of an evacuation geodesic computation."""

    path: npt.NDArray[np.float64]
    travel_time_s: float
    path_length_m: float
    bottleneck_rho: float
    evac_feasible: bool
    distance_map: npt.NDArray[np.float64]


class EvacuationGeodesic:
    """Compute optimal evacuation paths through a density field.

    Parameters
    ----------
    v_max : float, default 1.34
        Free-flow walking speed [m/s] (Weidmann).
    rho_critical : float, default 6.0
        Critical density [pers/m²] — impassable (Schwarzschild §5.6).
    dx_m : float, default 0.5
        Grid cell size [m].
    """

    def __init__(
        self,
        v_max: float = 1.34,
        rho_critical: float = 6.0,
        dx_m: float = 0.5,
    ) -> None:
        self.v_max = float(v_max)
        self.rho_critical = float(rho_critical)
        self.dx_m = float(dx_m)

    def compute_distance_map(
        self,
        density_map: npt.NDArray[np.float64],
        exits: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Compute travel-time distance map from all exits via Dijkstra.

        Parameters
        ----------
        density_map : ndarray, shape (ny, nx)
            Crowd density at each grid cell [pers/m²].
        exits : list of ndarray, shape (2,)
            Exit positions in world coordinates [m].

        Returns
        -------
        ndarray, shape (ny, nx)
            Travel time from each cell to the nearest exit [s].
            Impassable cells have value np.inf.
        """
        density_map = np.asarray(density_map, dtype=np.float64)
        ny, nx = density_map.shape
        dx = self.dx_m

        # Walking speed map: v(rho) = v_max * max(0, 1 - rho/rho_c)
        v_map = self.v_max * np.maximum(0.0, 1.0 - density_map / self.rho_critical)
        # Cost per cell: time to traverse = dx / v(rho)
        with np.errstate(divide="ignore"):
            cost_map = np.where(v_map > 0.01, dx / v_map, np.inf)

        # Multi-source Dijkstra from all exits
        dist = np.full((ny, nx), np.inf, dtype=np.float64)
        heap: list[tuple[float, int, int]] = []

        for ex in exits:
            ix = int(ex[0] / dx)
            iy = int(ex[1] / dx)
            if 0 <= ix < nx and 0 <= iy < ny:
                dist[iy, ix] = 0.0
                heapq.heappush(heap, (0.0, ix, iy))

        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (1, 1), (-1, 1), (1, -1)]

        while heap:
            d, ix, iy = heapq.heappop(heap)
            if d > dist[iy, ix]:
                continue
            for dix, diy in neighbors:
                nx_ = ix + dix
                ny_ = iy + diy
                if 0 <= nx_ < nx and 0 <= ny_ < ny:
                    step = dx * (math.sqrt(2) if abs(dix) + abs(diy) == 2 else 1.0)
                    nd = d + cost_map[ny_, nx_] * step / dx
                    if nd < dist[ny_, nx_]:
                        dist[ny_, nx_] = nd
                        heapq.heappush(heap, (nd, nx_, ny_))

        return dist

    def find_path(
        self,
        start: npt.NDArray[np.float64],
        density_map: npt.NDArray[np.float64],
        exits: list[npt.NDArray[np.float64]],
    ) -> EvacuationResult:
        """Find the optimal evacuation path from start to nearest exit.

        Parameters
        ----------
        start : ndarray, shape (2,)
            Start position in world coordinates [m].
        density_map : ndarray, shape (ny, nx)
            Crowd density at each grid cell [pers/m²].
        exits : list of ndarray, shape (2,)
            Exit positions in world coordinates [m].

        Returns
        -------
        EvacuationResult
            Path, travel time, and feasibility.
        """
        start = np.asarray(start, dtype=np.float64)
        density_map = np.asarray(density_map, dtype=np.float64)
        ny, nx = density_map.shape
        dx = self.dx_m

        dist = self.compute_distance_map(density_map, exits)

        # Trace path from start via gradient descent on distance map
        sx = min(max(int(start[0] / dx), 0), nx - 1)
        sy = min(max(int(start[1] / dx), 0), ny - 1)

        path = [start.copy()]
        x, y = sx, sy

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (1, 1), (-1, 1), (1, -1)]

        for _ in range(10000):
            neighbor_cells = [
                (x + dix, y + diy)
                for dix, diy in neighbors
                if 0 <= x + dix < nx and 0 <= y + diy < ny
            ]
            if not neighbor_cells:
                break
            best = min(neighbor_cells, key=lambda p: dist[p[1], p[0]])
            if dist[best[1], best[0]] >= dist[y, x]:
                break
            x, y = best
            path.append(np.array([x * dx, y * dx], dtype=np.float64))
            if dist[y, x] < 0.01:
                break

        travel_time = float(dist[sy, sx])
        path_arr = np.array(path, dtype=np.float64)
        path_length = float(np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))) \
            if len(path_arr) > 1 else 0.0

        bottleneck_rho = float(density_map[sy, sx]) if 0 <= sy < ny and 0 <= sx < nx else 0.0

        return EvacuationResult(
            path=path_arr,
            travel_time_s=travel_time,
            path_length_m=path_length,
            bottleneck_rho=bottleneck_rho,
            evac_feasible=travel_time < np.inf,
            distance_map=dist,
        )
