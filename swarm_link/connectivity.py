"""Per-drone station reachability and drone-only components."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Network:
    adjacency: np.ndarray  # Drones first; ground station is the last vertex.
    components: list[list[int]]  # Drone-only connected components, excluding ground.
    component_id: np.ndarray
    hops: np.ndarray  # -1 means unreachable; direct station access has one hop.
    categories: list[str]

    @property
    def reachable(self):
        return self.hops >= 1

    @property
    def all_reachable(self):
        return bool(self.reachable.all())


def analyze_network(positions, station, communication_range):
    positions = np.asarray(positions, dtype=float)
    points = np.vstack((positions, station))
    if communication_range <= 0 or not np.isfinite(communication_range):
        raise ValueError("Communication range must be finite and positive")
    if not np.isfinite(points).all():
        raise ValueError("Network coordinates must be finite")
    n = len(positions)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    adjacency = distances <= communication_range
    np.fill_diagonal(adjacency, False)
    distances_to_station = np.full(n + 1, -1, dtype=int)
    distances_to_station[n] = 0
    queue = deque([n])
    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[node]):
            if distances_to_station[neighbor] == -1:
                distances_to_station[neighbor] = distances_to_station[node] + 1
                queue.append(neighbor)
    component_id = np.full(n, -1, dtype=int)
    components = []
    for start in range(n):
        if component_id[start] != -1:
            continue
        label = len(components)
        component_id[start] = label
        component = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            component.append(int(node))
            for neighbor in np.flatnonzero(adjacency[node, :n]):
                if component_id[neighbor] == -1:
                    component_id[neighbor] = label
                    queue.append(neighbor)
        components.append(sorted(component))
    hops = distances_to_station[:n]
    categories = ["direct" if hops[i] == 1 else "relay" if hops[i] > 1 else
                  "peer_only" if adjacency[i, :n].any() else "isolated" for i in range(n)]
    return Network(adjacency, components, component_id, hops, categories)
