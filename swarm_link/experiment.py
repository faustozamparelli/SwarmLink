"""Matched static/sequential experiments, explicit denominators and paired uncertainty."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .connectivity import Network, analyze_network, reachability_metrics
from .localization import Estimate, estimate_position, sense_bearings
from .scenarios import FORMATIONS, formation_positions, make_scenario, rng_for


def quality_score(error, all_reachable, valid, reference_distance):
    if reference_distance <= 0 or not np.isfinite(reference_distance):
        raise ValueError("Reference distance must be finite and positive")
    if not all_reachable or not valid:
        return 0.0
    if error is None or error < 0 or not np.isfinite(error):
        raise ValueError("A valid estimate requires finite non-negative error")
    return float(reference_distance / (reference_distance + error))


@dataclass
class Trial:
    record: dict
    drones: list[dict]
    positions: np.ndarray
    bearings: np.ndarray
    network: Network
    estimate: Estimate


def evaluate_trial(config, scenario, formation, positions, communication_range):
    network = analyze_network(positions, config.station, communication_range)
    noise = scenario.noise_rad[:len(positions)]
    bearings = sense_bearings(positions, scenario.source, noise)
    usable = network.reachable & np.isfinite(bearings)
    # This is the information boundary: unreachable bearings never enter the estimator.
    estimate = estimate_position(positions[usable], bearings[usable], config.max_condition)
    error = float(np.linalg.norm(estimate.position - scenario.source)) if estimate.valid else None
    case_id = f"{scenario.scenario_id}-n{len(positions)}-r{communication_range:g}"
    counts = Counter(network.categories)
    record = dict(
        scenario_id=scenario.scenario_id, case_id=case_id, seed=scenario.seed,
        trial=scenario.trial, formation=formation, n_drones=len(positions),
        communication_range_m=communication_range,
        source_x_m=float(scenario.source[0]), source_y_m=float(scenario.source[1]),
        centre_x_m=float(scenario.centre[0]), centre_y_m=float(scenario.centre[1]),
        orientation_rad=scenario.orientation,
        fully_connected=network.all_reachable,
        unreachable_drones=int(np.count_nonzero(~network.reachable)),
        drone_components=network.components,
        direct_drones=counts["direct"], relay_drones=counts["relay"],
        peer_only_drones=counts["peer_only"], isolated_drones=counts["isolated"],
        used_measurements=int(usable.sum()), missing_measurements=int((~np.isfinite(bearings)).sum()),
        estimate_valid=estimate.valid, estimate_status=estimate.status,
        estimate_x_m=float(estimate.position[0]) if estimate.valid else None,
        estimate_y_m=float(estimate.position[1]) if estimate.valid else None,
        rank=estimate.rank, condition=estimate.condition, backward_rays=estimate.backward_rays,
        angular_rmse_rad=estimate.angular_rmse_rad, error_m=error,
        quality=quality_score(error, network.all_reachable, estimate.valid, config.reference_distance),
    )
    drones = []
    for i, position in enumerate(positions):
        label = int(network.component_id[i])
        drones.append(dict(
            scenario_id=scenario.scenario_id, case_id=case_id, formation=formation,
            n_drones=len(positions), communication_range_m=communication_range,
            drone_id=i, x_m=float(position[0]), y_m=float(position[1]),
            noise_rad=float(noise[i]), bearing_rad=float(bearings[i]) if np.isfinite(bearings[i]) else None,
            connectivity=network.categories[i], station_reachable=bool(network.reachable[i]),
            station_hops=int(network.hops[i]) if network.reachable[i] else None,
            drone_component_id=label, drone_component_size=len(network.components[label]),
            used_by_estimator=bool(usable[i]),
        ))
    return Trial(record, drones, positions, bearings, network, estimate)


def run_experiments(config):
    config.validate()
    records, drone_records, scenes = [], [], []
    disconnected_case = None
    # The first seed/trial and largest N/range are selected before observing outcomes.
    for seed in config.seeds:
        for index in range(config.trials_per_seed):
            scenario = make_scenario(config, seed, index)
            for n in config.drone_counts:
                for name in FORMATIONS:
                    positions = formation_positions(name, n, scenario.centre, scenario.orientation,
                                                    config.rms_radius, rng_for(seed, index, 1000 + n))
                    for radius in config.communication_ranges:
                        trial = evaluate_trial(config, scenario, name, positions, radius)
                        records.append(trial.record)
                        drone_records.extend(trial.drones)
                        if seed == config.seeds[0] and index == 0 and n == max(config.drone_counts) and radius == max(config.communication_ranges):
                            scenes.append(trial)
                        if disconnected_case is None and not trial.network.all_reachable:
                            disconnected_case = (scenario, n, radius)
    # Diagnostic illustration only: keep all formations of the first failed case.
    # Every original trial remains in the statistical results, without selection.
    if disconnected_case is not None:
        scenario, n, radius = disconnected_case
        for name in FORMATIONS:
            positions = formation_positions(name, n, scenario.centre, scenario.orientation,
                                            config.rms_radius, rng_for(scenario.seed, scenario.trial, 1000 + n))
            scenes.append(evaluate_trial(config, scenario, name, positions, radius))
    return records, drone_records, scenes


def mean_ci(values, samples, seed=90210):
    """Percentile bootstrap of independent scenario blocks, never individual drones."""
    values = np.asarray(values, dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(samples)
    # Bound temporary memory even for longer runs.
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        means[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def scenario_quality(records):
    blocks = defaultdict(list)
    for row in records:
        blocks[row["scenario_id"]].append(row["quality"])
    return {key: float(np.mean(values)) for key, values in sorted(blocks.items())}


def quality_sensitivity(records, reference_distance, bootstrap_samples):
    """Rescore the same saved errors/eligibility; never alter primary Q or rerun sensors."""
    result = []
    for multiplier in (0.01, 0.1, 1.0, 10.0):
        distance = reference_distance * multiplier
        groups = defaultdict(list)
        for row in records:
            score = quality_score(row["error_m"], row["fully_connected"], row["estimate_valid"], distance)
            groups[row["formation"]].append(dict(scenario_id=row["scenario_id"], quality=score))
        rows = []
        for name in FORMATIONS:
            rows.append(dict(reference_multiplier=multiplier, reference_distance_m=distance,
                             formation=name, mean_quality=float(np.mean([row["quality"] for row in groups[name]])),
                             quality_ci95=mean_ci(list(scenario_quality(groups[name]).values()), bootstrap_samples)))
        result.extend(sorted(rows, key=lambda row: row["mean_quality"], reverse=True))
    return result


def summarize(records, bootstrap_samples):
    errors = [row["error_m"] for row in records if row["fully_connected"] and row["estimate_valid"]]
    total = len(records)
    connected = sum(row["fully_connected"] for row in records)
    failures = sum(not row["estimate_valid"] for row in records)
    connected_failures = sum(row["fully_connected"] and not row["estimate_valid"] for row in records)
    seeds = defaultdict(list)
    for row in records:
        seeds[row["seed"]].append(row["quality"])
    quality = [row["quality"] for row in records]
    drone_observations = sum(row["n_drones"] for row in records)
    category_totals = {category: sum(row[f"{category}_drones"] for row in records)
                       for category in ("direct", "relay", "peer_only", "isolated")}
    return dict(
        trials=total, fully_connected_trials=connected, connectivity_pct=100 * connected / total,
        drone_observations=drone_observations,
        drone_station_reachable_pct=100 * (category_totals["direct"] + category_totals["relay"]) / drone_observations,
        **{f"{category}_drone_pct": 100 * count / drone_observations for category, count in category_totals.items()},
        unreachable_drones=sum(row["unreachable_drones"] for row in records),
        mean_unreachable_drones=float(np.mean([row["unreachable_drones"] for row in records])),
        estimator_failures=failures, estimator_failure_pct=100 * failures / total,
        connected_estimator_failures=connected_failures,
        connected_estimator_failure_pct=100 * connected_failures / connected if connected else None,
        accuracy_trials=len(errors), excluded_disconnected_trials=total - connected,
        excluded_connected_invalid_trials=connected_failures,
        mean_error_m=float(np.mean(errors)) if errors else None,
        median_error_m=float(np.median(errors)) if errors else None,
        rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        mean_quality=float(np.mean(quality)),
        quality_std=float(np.std(quality, ddof=1)) if total > 1 else 0.0,
        quality_ci95=mean_ci(list(scenario_quality(records).values()), bootstrap_samples),
        per_seed_quality={str(seed): float(np.mean(values)) for seed, values in seeds.items()},
        failure_reasons=dict(sorted(Counter(row["estimate_status"] for row in records if not row["estimate_valid"]).items())),
    )


def build_summaries(records, drone_records, bootstrap_samples, reference_distance=None):
    by_formation = defaultdict(list)
    by_setting = defaultdict(list)
    for row in records:
        by_formation[row["formation"]].append(row)
        by_setting[(row["formation"], row["n_drones"], row["communication_range_m"])].append(row)
    overall = [{"formation": name, **summarize(by_formation[name], bootstrap_samples)} for name in FORMATIONS]
    overall.sort(key=lambda row: row["mean_quality"], reverse=True)
    settings = [dict(formation=name, n_drones=n, communication_range_m=radius,
                     **summarize(rows, bootstrap_samples))
                for (name, n, radius), rows in sorted(by_setting.items())]
    # Pair at the base scenario level; keep all sizes/ranges in each resampled block.
    quality = {name: scenario_quality(rows) for name, rows in by_formation.items()}
    pairwise = []
    for a, b in combinations(FORMATIONS, 2):
        if quality[a].keys() != quality[b].keys():
            raise ValueError("Formations are not paired on identical scenarios")
        delta = [quality[a][key] - quality[b][key] for key in quality[a]]
        low, high = mean_ci(delta, bootstrap_samples)
        pairwise.append(dict(formation_a=a, formation_b=b, mean_quality_difference=float(np.mean(delta)),
                             ci95_low=low, ci95_high=high,
                             evidence="inconclusive" if low <= 0 <= high else "a_higher" if low > 0 else "b_higher"))
    drone_groups = defaultdict(list)
    for row in drone_records:
        drone_groups[(row["formation"], row["n_drones"], row["communication_range_m"], row["drone_id"])].append(row)
    drones = []
    for (name, n, radius, drone_id), rows in sorted(drone_groups.items()):
        counts = Counter(row["connectivity"] for row in rows)
        drones.append(dict(formation=name, n_drones=n, communication_range_m=radius,
                           drone_id=drone_id, trials=len(rows),
                           station_reachable_pct=100 * sum(row["station_reachable"] for row in rows) / len(rows),
                           **{f"{category}_pct": 100 * counts[category] / len(rows)
                              for category in ("direct", "relay", "peer_only", "isolated")}))
    result = dict(overall=overall, by_setting=settings, pairwise=pairwise, per_drone=drones)
    if reference_distance is not None:
        result["quality_sensitivity"] = quality_sensitivity(records, reference_distance, bootstrap_samples)
    return result


CONTROLLERS = ("hold_still", "random_movement")


def controller_displacements(controller, n_drones, rng, max_displacement):
    """Open-loop controls receive only team size, an independent RNG and motion budget."""
    if controller == "hold_still":
        return np.zeros((n_drones, 2))
    if controller != "random_movement":
        raise ValueError("Unknown sequential controller")
    angles = rng.uniform(-np.pi, np.pi, n_drones)
    radii = max_displacement * np.sqrt(rng.uniform(0, 1, n_drones))
    return radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))


def move_positions(positions, displacements, bounds, max_displacement):
    """Cap-free validation, then truncate at map walls; connectivity never blocks a move."""
    positions, displacements = np.asarray(positions, float), np.asarray(displacements, float)
    if displacements.shape != positions.shape or not np.isfinite(displacements).all():
        raise ValueError("Expected one finite displacement per drone")
    if np.any(np.linalg.norm(displacements, axis=1) > max_displacement + 1e-10):
        raise ValueError("Requested movement exceeds the speed/time-step limit")
    return np.clip(positions + displacements, [0, 0], bounds)


def run_episode(config, scenario, initial_positions, communication_range, controller,
                movement_steps=60, step_seconds=2.0, max_speed=5.0):
    """One t=0 observation and T move/measure cycles; history contains delivered data only."""
    if not isinstance(movement_steps, int) or movement_steps < 0:
        raise ValueError("Movement steps must be a non-negative integer")
    if not np.isfinite([step_seconds, max_speed]).all() or step_seconds <= 0 or max_speed < 0:
        raise ValueError("Time step must be positive and speed non-negative, both finite")
    if controller not in CONTROLLERS:
        raise ValueError("Unknown sequential controller")
    positions = np.array(initial_positions, dtype=float, copy=True)
    bounds = np.array([config.width, config.height])
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2:
        raise ValueError("Expected at least two 2D drone positions")
    if not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions > bounds):
        raise ValueError("Initial positions must be finite and inside the map")
    n = len(positions)
    case_id = f"{scenario.scenario_id}-n{n}-r{communication_range:g}"
    identity = dict(scenario_id=scenario.scenario_id, case_id=case_id, controller=controller,
                    seed=scenario.seed, trial=scenario.trial, n_drones=n,
                    communication_range_m=communication_range)
    steps, drones, reachability = [], [], []
    history_positions, history_bearings = [], []
    travel = np.zeros(n)
    requested = np.zeros_like(positions)
    actual = np.zeros_like(positions)
    previous = None
    for step in range(movement_steps + 1):
        if step:
            requested = controller_displacements(controller, n, rng_for(scenario.seed, scenario.trial, 5000, step),
                                                  max_speed * step_seconds)
            updated = move_positions(positions, requested, bounds, max_speed * step_seconds)
            actual = updated - positions
            travel += np.linalg.norm(actual, axis=1)
            positions = updated
        noise = (scenario.noise_rad[:n] if step == 0 else
                 rng_for(scenario.seed, scenario.trial, 3000, step).normal(0, np.deg2rad(config.noise_std_deg), n))
        bearings = sense_bearings(positions, scenario.source, noise)
        network = analyze_network(positions, config.station, communication_range)
        delivered = network.reachable & np.isfinite(bearings)
        # These copies retain each measurement's acquisition pose, including after a later outage.
        history_positions.append(positions[delivered].copy())
        history_bearings.append(bearings[delivered].copy())
        estimate = estimate_position(np.concatenate(history_positions), np.concatenate(history_bearings), config.max_condition)
        error = float(np.linalg.norm(estimate.position - scenario.source)) if estimate.valid else None
        reachability.append(network.reachable.copy())
        steps.append(dict(**identity, step=step, time_s=step * step_seconds,
                          fully_connected=network.all_reachable, unreachable_drones=int((~network.reachable).sum()),
                          delivered_measurements=int(delivered.sum()),
                          cumulative_measurements=sum(len(values) for values in history_bearings),
                          estimate_valid=estimate.valid, estimate_status=estimate.status,
                          estimate_x_m=float(estimate.position[0]) if estimate.valid else None,
                          estimate_y_m=float(estimate.position[1]) if estimate.valid else None,
                          error_m=error, rank=estimate.rank, condition=estimate.condition,
                          backward_rays=estimate.backward_rays, angular_rmse_rad=estimate.angular_rmse_rad,
                          swarm_travel_m=float(travel.sum())))
        for i in range(n):
            component = int(network.component_id[i])
            drones.append(dict(**identity, step=step, time_s=step * step_seconds, drone_id=i,
                               x_m=float(positions[i, 0]), y_m=float(positions[i, 1]),
                               requested_dx_m=float(requested[i, 0]), requested_dy_m=float(requested[i, 1]),
                               actual_dx_m=float(actual[i, 0]), actual_dy_m=float(actual[i, 1]),
                               boundary_limited=bool(np.linalg.norm(actual[i] - requested[i]) > 1e-10),
                               travel_m=float(travel[i]), noise_rad=float(noise[i]),
                               bearing_rad=float(bearings[i]) if np.isfinite(bearings[i]) else None,
                               measurement_delivered=bool(delivered[i]), connectivity=network.categories[i],
                               station_reachable=bool(network.reachable[i]),
                               station_hops=int(network.hops[i]) if network.reachable[i] else None,
                               drone_component_id=component, drone_component_size=len(network.components[component]),
                               disconnection_event=bool(previous is not None and previous[i] and not network.reachable[i]),
                               reconnection_event=bool(previous is not None and not previous[i] and network.reachable[i])))
        previous = network.reachable.copy()
    reachable = np.array(reachability)
    drone_metrics = [dict(**identity, drone_id=i, travel_m=float(travel[i]),
                          **reachability_metrics(reachable[:, i], step_seconds)) for i in range(n)]
    episode = dict(**identity, movement_steps=movement_steps, duration_s=movement_steps * step_seconds,
                   source_x_m=float(scenario.source[0]), source_y_m=float(scenario.source[1]),
                   **reachability_metrics(reachable.all(axis=1), step_seconds),
                   estimate_valid=estimate.valid, estimate_status=estimate.status, error_m=error,
                   quality=quality_score(error, network.all_reachable, estimate.valid, config.reference_distance),
                   swarm_travel_m=float(travel.sum()), cumulative_measurements=steps[-1]["cumulative_measurements"],
                   generated_measurements=(movement_steps + 1) * n,
                   drone_disconnection_events=sum(row["disconnection_events"] for row in drone_metrics),
                   drone_reconnection_events=sum(row["reconnection_events"] for row in drone_metrics))
    return episode, steps, drones, drone_metrics


def run_sequential(config, initial_formations=("polygon",), movement_steps=60, step_seconds=2.0, max_speed=5.0):
    config.validate()
    if not initial_formations or len(set(initial_formations)) != len(initial_formations):
        raise ValueError("Initial formations must be nonempty and distinct")
    if any(name not in FORMATIONS for name in initial_formations):
        raise ValueError("Unknown initial formation")
    episodes, steps, drones, drone_episodes = [], [], [], []
    for seed in config.seeds:
        for index in range(config.trials_per_seed):
            scenario = make_scenario(config, seed, index)
            for n in config.drone_counts:
                for formation in initial_formations:
                    positions = formation_positions(formation, n, scenario.centre, scenario.orientation,
                                                    config.rms_radius, rng_for(seed, index, 1000 + n))
                    for radius in config.communication_ranges:
                        for controller in CONTROLLERS:
                            output = run_episode(config, scenario, positions, radius, controller,
                                                 movement_steps, step_seconds, max_speed)
                            for destination, rows in zip((episodes, steps, drones, drone_episodes), output):
                                if isinstance(rows, dict):
                                    rows = [rows]
                                for row in rows:
                                    row["initial_formation"] = formation
                                    row["case_id"] += f"-{formation}"
                                destination.extend(rows)
    return episodes, steps, drones, drone_episodes


def summarize_episodes(episodes, bootstrap_samples):
    errors = [row["error_m"] for row in episodes if row["terminal_connected"] and row["estimate_valid"]]
    total = len(episodes)
    terminal = sum(row["terminal_connected"] for row in episodes)
    return dict(
        episodes=total, terminal_connected_episodes=terminal,
        terminal_connectivity_pct=100 * terminal / total,
        continuous_connectivity_pct=100 * np.mean([row["continuously_connected"] for row in episodes]),
        mean_connected_step_fraction=float(np.mean([row["connected_step_fraction"] for row in episodes])),
        mean_swarm_outage_s=float(np.mean([row["outage_duration_s"] for row in episodes])),
        drone_disconnection_events=sum(row["drone_disconnection_events"] for row in episodes),
        drone_reconnection_events=sum(row["drone_reconnection_events"] for row in episodes),
        estimator_failure_pct=100 * sum(not row["estimate_valid"] for row in episodes) / total,
        terminal_connected_estimator_failure_pct=(100 * sum(row["terminal_connected"] and not row["estimate_valid"] for row in episodes) / terminal if terminal else None),
        accuracy_episodes=len(errors), excluded_terminal_disconnected=total - terminal,
        excluded_terminal_connected_invalid=terminal - len(errors),
        mean_error_m=float(np.mean(errors)) if errors else None,
        median_error_m=float(np.median(errors)) if errors else None,
        rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        mean_quality=float(np.mean([row["quality"] for row in episodes])),
        quality_ci95=mean_ci(list(scenario_quality(episodes).values()), bootstrap_samples),
        mean_swarm_travel_m=float(np.mean([row["swarm_travel_m"] for row in episodes])),
        mean_delivered_measurements=float(np.mean([row["cumulative_measurements"] for row in episodes])),
        failure_reasons=dict(Counter(row["estimate_status"] for row in episodes if not row["estimate_valid"])),
    )


def build_sequential_summaries(episodes, bootstrap_samples):
    by_controller, by_setting = defaultdict(list), defaultdict(list)
    for row in episodes:
        by_controller[row["controller"]].append(row)
        by_setting[(row["controller"], row["initial_formation"], row["n_drones"], row["communication_range_m"])].append(row)
    quality = {name: scenario_quality(rows) for name, rows in by_controller.items()}
    if quality[CONTROLLERS[0]].keys() != quality[CONTROLLERS[1]].keys():
        raise ValueError("Controllers must share identical scenario blocks")
    delta = [quality[CONTROLLERS[1]][key] - quality[CONTROLLERS[0]][key] for key in quality[CONTROLLERS[0]]]
    return dict(
        overall=[dict(controller=name, **summarize_episodes(by_controller[name], bootstrap_samples)) for name in CONTROLLERS],
        by_setting=[dict(controller=name, initial_formation=formation, n_drones=n, communication_range_m=radius,
                         **summarize_episodes(rows, bootstrap_samples))
                    for (name, formation, n, radius), rows in sorted(by_setting.items())],
        paired_quality_difference=dict(comparison="random_movement - hold_still", mean=float(np.mean(delta)),
                                       ci95=mean_ci(delta, bootstrap_samples)),
    )
