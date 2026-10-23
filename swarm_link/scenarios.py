"""Source-independent deployments and matched, explicitly seeded scenarios."""

from dataclasses import dataclass

import numpy as np


FORMATIONS = ("line", "polygon", "compact", "random")


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (11, 29, 47, 71, 101, 131, 173, 211)
    trials_per_seed: int = 125
    drone_counts: tuple[int, ...] = (3, 5, 8)
    communication_ranges: tuple[float, ...] = (200.0, 350.0, 500.0)
    width: float = 1000.0
    height: float = 1000.0
    station: tuple[float, float] = (500.0, 100.0)
    rms_radius: float = 100.0
    noise_std_deg: float = 2.0
    max_condition: float = 1e6
    bootstrap_samples: int = 2000
    benchmark: str = "randomized"
    deployment_centre: tuple[float, float] = (500.0, 350.0)
    source_radius: float = 75.0

    @property
    def benchmark_version(self):
        return f"{self.benchmark}-v1"

    @property
    def reference_distance(self):
        return float(np.hypot(self.width, self.height))

    @property
    def placement_margin(self):
        # A centred N-point cloud of RMS radius r has max radius <= sqrt(N-1)*r.
        return float(np.sqrt(max(self.drone_counts) - 1) * self.rms_radius)

    def validate(self):
        if self.benchmark not in ("randomized", "controlled"):
            raise ValueError("Benchmark must be randomized or controlled")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide distinct, non-negative seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be non-negative")
        if self.trials_per_seed < 1 or self.bootstrap_samples < 1:
            raise ValueError("Trial and bootstrap counts must be positive")
        if not self.drone_counts or any(n < 2 for n in self.drone_counts):
            raise ValueError("At least two drones are required")
        if len(set(self.drone_counts)) != len(self.drone_counts):
            raise ValueError("Drone counts must be distinct")
        if not self.communication_ranges or len(set(self.communication_ranges)) != len(self.communication_ranges):
            raise ValueError("Communication ranges must be nonempty and distinct")
        values = [self.width, self.height, self.rms_radius, self.max_condition,
                  self.noise_std_deg, *self.station, *self.communication_ranges]
        if not np.isfinite(values).all():
            raise ValueError("Configuration values must be finite")
        if self.rms_radius <= 0 or self.noise_std_deg < 0 or self.max_condition <= 1:
            raise ValueError("Invalid radius, noise or conditioning limit")
        if min(self.communication_ranges) <= 0:
            raise ValueError("Communication ranges must be positive")
        if min(self.width, self.height) <= 2 * self.placement_margin:
            raise ValueError("Map too small for the shared placement envelope")
        if not (0 <= self.station[0] <= self.width and 0 <= self.station[1] <= self.height):
            raise ValueError("Station must be inside the environment")
        if self.benchmark == "controlled":
            centre = np.asarray(self.deployment_centre)
            if centre.shape != (2,) or not np.isfinite(centre).all():
                raise ValueError("Controlled deployment centre must be a finite 2D point")
            if np.any(centre < self.placement_margin) or np.any(centre > np.array([self.width, self.height]) - self.placement_margin):
                raise ValueError("Controlled centre must fit the shared formation envelope")
            if not np.isfinite(self.source_radius) or self.source_radius <= 0:
                raise ValueError("Controlled source radius must be finite and positive")
            if np.any(centre < self.source_radius) or np.any(centre + self.source_radius > [self.width, self.height]):
                raise ValueError("Controlled source disk must fit inside the map")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    trial: int
    source: np.ndarray
    centre: np.ndarray
    orientation: float
    noise_rad: np.ndarray


def rng_for(seed, trial, stream, *substreams):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, trial, stream, *substreams])))


def make_scenario(config, seed, trial):
    # Separate streams keep source, centre, heading, noise and random placement independent.
    if config.benchmark == "controlled":
        centre = np.array(config.deployment_centre)
        source_rng = rng_for(seed, trial, 0)
        angle = source_rng.uniform(-np.pi, np.pi)
        radius = config.source_radius * np.sqrt(source_rng.uniform())
        source = centre + radius * np.array([np.cos(angle), np.sin(angle)])
    else:
        # Preserve randomized-v1 draws exactly, including the original stream IDs.
        source = rng_for(seed, trial, 0).uniform([0, 0], [config.width, config.height])
        margin = config.placement_margin
        centre = rng_for(seed, trial, 1).uniform(
            [margin, margin], [config.width - margin, config.height - margin])
    orientation = float(rng_for(seed, trial, 2).uniform(-np.pi, np.pi))
    noise = rng_for(seed, trial, 3).normal(
        0, np.deg2rad(config.noise_std_deg), max(config.drone_counts))
    prefix = "controlled-" if config.benchmark == "controlled" else ""
    return Scenario(f"{prefix}s{seed}-t{trial:04d}", seed, trial, source, centre, orientation, noise)


def formation_positions(name, n, centre, orientation, rms_radius, rng):
    """No source argument: all clouds share centroid and RMS radius, without clipping."""
    if n < 2 or rms_radius <= 0 or name not in FORMATIONS:
        raise ValueError("Unknown formation or invalid size/radius")
    if name == "line":
        offsets = np.column_stack((np.linspace(-1, 1, n), np.zeros(n)))
    elif name == "polygon":
        angles = np.arange(n) * 2 * np.pi / n
        offsets = np.column_stack((np.cos(angles), np.sin(angles)))
    elif name == "compact":
        # Closest lattice sites give a small filled grid; stable tie breaking is explicit.
        side = int(np.ceil(np.sqrt(n)))
        grid = np.arange(side) - (side - 1) / 2
        offsets = np.array([(x, y) for y in grid for x in grid])
        order = np.lexsort((offsets[:, 0], offsets[:, 1], np.sum(offsets**2, axis=1)))
        offsets = offsets[order[:n]]
    else:
        angles = rng.uniform(-np.pi, np.pi, n)
        radii = np.sqrt(rng.uniform(0, 1, n))
        offsets = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    offsets = offsets - offsets.mean(axis=0)
    offsets *= rms_radius / np.sqrt(np.mean(np.sum(offsets**2, axis=1)))
    rotation = np.array([[np.cos(orientation), -np.sin(orientation)],
                         [np.sin(orientation), np.cos(orientation)]])
    return offsets @ rotation.T + np.asarray(centre)
