"""Wrapped bearing sensing and a single, diagnostic bearing-line LS estimator."""

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def sense_bearings(positions, source, noise_rad):
    delta = np.asarray(source) - np.asarray(positions)
    bearings = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) + noise_rad)
    # A colocated point source has undefined bearing; never use atan2(0, 0) as data.
    bearings[np.linalg.norm(delta, axis=1) <= 1e-9] = np.nan
    return bearings


@dataclass
class Estimate:
    position: np.ndarray | None
    status: str
    rank: int | None = None
    condition: float | None = None
    backward_rays: int = 0
    angular_rmse_rad: float | None = None

    @property
    def valid(self):
        return self.status == "valid"


def estimate_position(positions, bearings, max_condition=1e6):
    positions, bearings = np.asarray(positions, dtype=float), np.asarray(bearings, dtype=float)
    if positions.shape != (len(bearings), 2):
        raise ValueError("Expected one 2D sensor position per bearing")
    if max_condition <= 1 or not np.isfinite(max_condition):
        raise ValueError("Condition limit must be finite and greater than one")
    if len(bearings) < 2:
        return Estimate(None, "insufficient_measurements")
    if not np.isfinite(positions).all() or not np.isfinite(bearings).all():
        return Estimate(None, "nonfinite_measurements")
    bearings = wrap_angle(bearings)
    normal = np.column_stack((-np.sin(bearings), np.cos(bearings)))
    # Solve relative to the sensor centroid to avoid a large global-origin offset.
    origin = positions.mean(axis=0)
    rhs = np.sum(normal * (positions - origin), axis=1)
    try:
        relative, _, rank, singular = np.linalg.lstsq(normal, rhs, rcond=1e-12)
    except np.linalg.LinAlgError:
        return Estimate(None, "solver_failure")
    rank = int(rank)
    if rank < 2:
        return Estimate(None, "rank_deficient", rank)
    condition = float(singular[0] / singular[-1])
    if condition > max_condition:
        return Estimate(None, "ill_conditioned", rank, condition)
    position = relative + origin
    if not np.isfinite(position).all():
        return Estimate(None, "nonfinite_solution", rank, condition)
    delta = position - positions
    directions = np.column_stack((np.cos(bearings), np.sin(bearings)))
    # A forward ray needs positive range. Also reject a candidate at a sensor.
    backwards = int(np.count_nonzero(np.sum(delta * directions, axis=1) <= 1e-9))
    residual = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - bearings)
    angular_rmse = float(np.sqrt(np.mean(residual**2)))
    if backwards:
        return Estimate(None, "direction_inconsistent", rank, condition, backwards, angular_rmse)
    return Estimate(position, "valid", rank, condition, 0, angular_rmse)
