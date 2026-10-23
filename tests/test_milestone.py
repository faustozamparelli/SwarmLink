from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from swarm_link.connectivity import analyze_network, reachability_metrics
from swarm_link.experiment import (build_summaries, evaluate_trial, mean_ci,
                                   quality_score, run_experiments, summarize,
                                   run_episode, run_sequential, move_positions,
                                   controller_displacements, build_sequential_summaries, summarize_episodes)
from swarm_link.localization import estimate_position, sense_bearings, wrap_angle
from swarm_link.report import save_results, save_sequential_results
from swarm_link.scenarios import (Config, FORMATIONS, formation_positions,
                                  make_scenario, rng_for)


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0., 0.], [10., 0.], [0., 10.]])
        self.source = np.array([3., 4.])
        self.bearings = sense_bearings(self.positions, self.source, np.zeros(3))

    def test_noiseless_recovery(self):
        estimate = estimate_position(self.positions, self.bearings)
        self.assertTrue(estimate.valid)
        assert_allclose(estimate.position, self.source, atol=1e-12)
        self.assertLess(estimate.angular_rmse_rad, 1e-12)

    def test_angular_wraparound(self):
        delta = wrap_angle((-np.pi + 1e-5) - (np.pi - 1e-5))
        self.assertAlmostEqual(float(delta), 2e-5)
        shifted = self.bearings + np.array([2, -4, 6]) * np.pi
        assert_allclose(estimate_position(self.positions, shifted).position, self.source, atol=1e-12)

    def test_sensing_wraps_across_pi(self):
        bearing = sense_bearings(np.array([[0., 0.]]), [-1., 0.], np.array([.01]))
        self.assertAlmostEqual(float(bearing[0]), -np.pi + .01)

    def test_rank_deficient_geometry(self):
        estimate = estimate_position(np.array([[0., 0.], [1., 0.], [2., 0.]]), np.zeros(3))
        self.assertEqual(estimate.status, "rank_deficient")
        self.assertIsNone(estimate.position)

    def test_numerically_ill_conditioned_geometry(self):
        estimate = estimate_position(np.array([[0., 0.], [0., 1.]]), np.array([0., 1e-7]))
        self.assertEqual(estimate.status, "ill_conditioned")
        self.assertGreater(estimate.condition, 1e6)
        self.assertIsNone(estimate.position)

    def test_infinite_line_solution_behind_ray_is_invalid(self):
        bearings = self.bearings.copy()
        bearings[0] += np.pi
        estimate = estimate_position(self.positions, bearings)
        self.assertEqual(estimate.status, "direction_inconsistent")
        self.assertEqual(estimate.backward_rays, 1)
        self.assertIsNone(estimate.position)

    def test_insufficient_and_nonfinite_measurements(self):
        self.assertEqual(estimate_position(self.positions[:1], self.bearings[:1]).status, "insufficient_measurements")
        self.assertEqual(estimate_position(self.positions, [0, 1, np.nan]).status, "nonfinite_measurements")

    def test_colocated_source_has_no_defined_bearing(self):
        bearings = sense_bearings(self.positions, self.positions[0], np.zeros(3))
        self.assertTrue(np.isnan(bearings[0]))


class ConnectivityTests(unittest.TestCase):
    def test_multihop_relay(self):
        network = analyze_network([[1, 0], [2, 0], [3, 0]], [0, 0], 1.01)
        self.assertEqual(network.categories, ["direct", "relay", "relay"])
        assert_array_equal(network.hops, [1, 2, 3])
        self.assertTrue(network.all_reachable)
        self.assertEqual(network.components, [[0, 1, 2]])

    def test_disconnected_subnetwork_and_isolated_drone(self):
        network = analyze_network([[1, 0], [8, 0], [8.5, 0], [20, 0]], [0, 0], 1)
        self.assertEqual(network.categories, ["direct", "peer_only", "peer_only", "isolated"])
        self.assertEqual(network.components, [[0], [1, 2], [3]])
        assert_array_equal(network.reachable, [True, False, False, False])
        self.assertFalse(network.all_reachable)

    def test_drone_only_components_exclude_station_bridges(self):
        network = analyze_network([[-1, 0], [1, 0]], [0, 0], 1)
        self.assertEqual(network.components, [[0], [1]])
        self.assertEqual(network.categories, ["direct", "direct"])
        self.assertTrue(network.all_reachable)

    def test_peer_connected_swarm_without_station_fails(self):
        network = analyze_network([[10, 0], [11, 0], [12, 0]], [0, 0], 1.1)
        self.assertEqual(network.categories, ["peer_only"] * 3)
        self.assertFalse(network.all_reachable)

    def test_exact_range_boundary_is_connected(self):
        self.assertTrue(analyze_network([[3, 4]], [0, 0], 5).all_reachable)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(seeds=(7, 19), trials_per_seed=4, drone_counts=(3, 5),
                            communication_ranges=(200., 500.), bootstrap_samples=80)
        cls.records, cls.drones, cls.scenes = run_experiments(cls.config)

    def test_identical_seeds_reproduce_complete_results(self):
        records, drones, _ = run_experiments(self.config)
        self.assertEqual(records, self.records)
        self.assertEqual(drones, self.drones)

    def test_all_trials_retained_and_matched(self):
        self.assertEqual(len(self.records), 2 * 4 * 2 * 2 * 4)
        cases = {}
        for row in self.records:
            cases.setdefault(row["case_id"], []).append(row)
        for rows in cases.values():
            self.assertEqual({row["formation"] for row in rows}, set(FORMATIONS))
            for field in ("source_x_m", "source_y_m", "centre_x_m", "centre_y_m", "orientation_rad"):
                self.assertEqual(len({row[field] for row in rows}), 1)
        noise = {}
        for row in self.drones:
            key = (row["case_id"], row["drone_id"])
            noise.setdefault(key, []).append(row["noise_rad"])
        self.assertTrue(all(len(set(values)) == 1 for values in noise.values()))

    def test_centroids_scales_and_placement_bounds_match(self):
        config = self.config
        for seed in range(15):
            scenario = make_scenario(config, seed, 0)
            for n in config.drone_counts:
                for name in FORMATIONS:
                    positions = formation_positions(name, n, scenario.centre, scenario.orientation,
                                                    config.rms_radius, rng_for(seed, 0, 1000 + n))
                    assert_allclose(positions.mean(axis=0), scenario.centre, atol=1e-12)
                    rms = np.sqrt(np.mean(np.sum((positions - scenario.centre)**2, axis=1)))
                    self.assertAlmostEqual(rms, config.rms_radius)
                    self.assertTrue(np.all(positions >= 0))
                    self.assertTrue(np.all(positions <= [config.width, config.height]))

    def test_formation_api_cannot_access_hidden_source(self):
        self.assertNotIn("source", inspect.signature(formation_positions).parameters)
        scenario = make_scenario(self.config, 7, 0)
        changed_truth = replace(scenario, source=np.array([999., 999.]))
        def positions(s):
            return formation_positions("random", 5, s.centre, s.orientation, 100, rng_for(s.seed, s.trial, 1005))
        assert_array_equal(positions(scenario), positions(changed_truth))

    def test_station_estimator_excludes_unreachable_measurements(self):
        config = replace(self.config, station=(0., 0.), noise_std_deg=0.)
        scenario = replace(make_scenario(config, 7, 0), source=np.array([3., 4.]))
        positions = np.array([[0., 1.], [1., 0.], [100., 100.]])
        first = evaluate_trial(config, scenario, "test", positions, 2)
        noise = scenario.noise_rad.copy()
        noise[2] = 2.4
        second = evaluate_trial(config, replace(scenario, noise_rad=noise), "test", positions, 2)
        self.assertTrue(first.estimate.valid)
        self.assertEqual(first.record["used_measurements"], 2)
        self.assertEqual(first.record["unreachable_drones"], 1)
        self.assertFalse(first.drones[2]["used_by_estimator"])
        assert_array_equal(first.estimate.position, second.estimate.position)
        self.assertAlmostEqual(first.record["error_m"], 0)
        self.assertFalse(first.record["fully_connected"])
        self.assertEqual(first.record["quality"], 0)

    def test_zero_quality_for_disconnection_or_invalid_estimate(self):
        self.assertEqual(quality_score(0, False, True, 1000), 0)
        self.assertEqual(quality_score(None, True, False, 1000), 0)
        self.assertEqual(quality_score(None, False, False, 1000), 0)
        for row in self.records:
            if not row["fully_connected"] or not row["estimate_valid"]:
                self.assertEqual(row["quality"], 0)

    def test_quality_is_continuous_and_monotone_without_error_cutoff(self):
        values = [quality_score(error, True, True, 1000) for error in [0, 1, 10, 1000, 1e6]]
        self.assertEqual(values[0], 1)
        self.assertTrue(all(a > b > 0 for a, b in zip(values, values[1:])))
        self.assertLess(abs(quality_score(10, True, True, 1000) - quality_score(10.000001, True, True, 1000)), 1e-8)

    def test_quality_formula_uses_common_environment_diagonal(self):
        reference = np.hypot(self.config.width, self.config.height)
        for row in self.records:
            if row["fully_connected"] and row["estimate_valid"]:
                self.assertEqual(row["quality"], reference / (reference + row["error_m"]))

    def test_positions_and_noise_do_not_change_with_radio_range(self):
        cases = {}
        for row in self.drones:
            key = (row["scenario_id"], row["formation"], row["n_drones"], row["drone_id"])
            cases.setdefault(key, []).append((row["x_m"], row["y_m"], row["noise_rad"], row["bearing_rad"]))
        self.assertTrue(all(len(set(values)) == 1 for values in cases.values()))

    def test_summary_denominators_include_failed_trials(self):
        base = self.records[0].copy()
        rows = [dict(base, scenario_id="a", fully_connected=True, estimate_valid=True, error_m=3., quality=.8),
                dict(base, scenario_id="b", fully_connected=False, estimate_valid=True, error_m=0., quality=0.),
                dict(base, scenario_id="c", fully_connected=True, estimate_valid=False,
                     estimate_status="rank_deficient", error_m=None, quality=0.)]
        summary = summarize(rows, 80)
        self.assertEqual(summary["trials"], 3)
        self.assertEqual(summary["accuracy_trials"], 1)
        self.assertEqual(summary["mean_error_m"], 3)
        self.assertEqual(summary["rmse_m"], 3)
        self.assertAlmostEqual(summary["mean_quality"], .8 / 3)
        self.assertAlmostEqual(summary["connectivity_pct"], 200 / 3)
        self.assertAlmostEqual(summary["estimator_failure_pct"], 100 / 3)
        self.assertEqual(summary["connected_estimator_failure_pct"], 50)
        self.assertEqual(summary["excluded_disconnected_trials"], 1)
        self.assertEqual(summary["excluded_connected_invalid_trials"], 1)

    def test_no_successes_report_missing_accuracy_not_zero_error(self):
        row = dict(self.records[0], fully_connected=False, estimate_valid=False,
                   estimate_status="insufficient_measurements", error_m=None, quality=0)
        summary = summarize([row], 80)
        self.assertIsNone(summary["mean_error_m"])
        self.assertIsNone(summary["rmse_m"])
        self.assertEqual(summary["mean_quality"], 0)

    def test_bootstrap_and_paired_summaries_are_reproducible(self):
        assert_allclose(mean_ci([0, .3, 1], 80), mean_ci([0, .3, 1], 80))
        summaries = build_summaries(self.records, self.drones, 80)
        self.assertEqual(len(summaries["pairwise"]), 6)
        for row in summaries["per_drone"]:
            self.assertAlmostEqual(sum(row[f"{category}_pct"] for category in ("direct", "relay", "peer_only", "isolated")), 100)

    def test_identical_quality_does_not_produce_a_winner(self):
        rows = [dict(row, quality=.5) for row in self.records]
        summaries = build_summaries(rows, self.drones, 80)
        self.assertTrue(all(row["evidence"] == "inconclusive" for row in summaries["pairwise"]))

    def test_complete_output_contains_plots_logs_and_json(self):
        summaries = build_summaries(self.records, self.drones, 80)
        with tempfile.TemporaryDirectory() as directory:
            save_results(directory, self.config, self.records, self.drones, self.scenes, summaries)
            path = Path(directory)
            for name in ("scene.png", "metrics.png", "trials.csv", "drones.csv", "summary.json", "metadata.json",
                         "summary.md", "leaderboard.csv", "by_setting.csv", "paired_comparisons.csv", "per_drone_summary.csv"):
                self.assertGreater((path / name).stat().st_size, 20)
            metadata = json.loads((path / "metadata.json").read_text())
            self.assertEqual(metadata["reference_distance_m"], self.config.reference_distance)
            self.assertEqual(json.loads((path / "summary.json").read_text()), summaries)

    def test_invalid_configuration_is_rejected(self):
        for config in (replace(self.config, seeds=(7, 7)), replace(self.config, noise_std_deg=-1),
                       replace(self.config, rms_radius=1000), replace(self.config, communication_ranges=(0,)),
                       replace(self.config, station=(-1, 0))):
            with self.assertRaises(ValueError):
                config.validate()


class ControlledDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(benchmark="controlled", seeds=(7, 19), trials_per_seed=6,
                            drone_counts=(3, 5), communication_ranges=(10., 500.), bootstrap_samples=80)
        cls.records, cls.drones, cls.scenes = run_experiments(cls.config)

    def test_controlled_centre_pairing_and_equal_scales(self):
        self.assertEqual(len(self.records), 2 * 6 * 2 * 2 * 4)
        cases = {}
        for row in self.records:
            self.assertEqual((row["centre_x_m"], row["centre_y_m"]), self.config.deployment_centre)
            cases.setdefault(row["case_id"], []).append(row)
        for rows in cases.values():
            self.assertEqual({row["formation"] for row in rows}, set(FORMATIONS))
            for field in ("source_x_m", "source_y_m", "orientation_rad"):
                self.assertEqual(len({row[field] for row in rows}), 1)
        groups, noise = {}, {}
        for row in self.drones:
            groups.setdefault((row["case_id"], row["formation"]), []).append([row["x_m"], row["y_m"]])
            noise.setdefault((row["case_id"], row["drone_id"]), []).append(row["noise_rad"])
        self.assertTrue(all(len(set(values)) == 1 for values in noise.values()))
        for positions in groups.values():
            assert_allclose(np.mean(positions, axis=0), self.config.deployment_centre, atol=1e-12)
            rms = np.sqrt(np.mean(np.sum((np.array(positions) - self.config.deployment_centre)**2, axis=1)))
            self.assertAlmostEqual(rms, self.config.rms_radius)

    def test_disk_sampling_is_uniform_in_area_and_not_hull_conditioned(self):
        config = replace(self.config, source_radius=180.)
        offsets = np.array([make_scenario(config, 7, i).source - config.deployment_centre for i in range(1000)])
        radii = np.linalg.norm(offsets, axis=1)
        self.assertTrue(np.all(radii <= config.source_radius))
        self.assertAlmostEqual(float(np.mean((radii / config.source_radius)**2)), .5, delta=.04)
        # These sources lie beyond the radius of a polygon; they are still kept.
        self.assertTrue(np.any(radii > config.rms_radius))
        scenario = make_scenario(config, 7, 0)
        line = formation_positions("line", 3, scenario.centre, scenario.orientation, config.rms_radius, rng_for(7, 0, 1003))
        axis, offset = line[-1] - line[0], scenario.source - line[0]
        self.assertGreater(abs(axis[0] * offset[1] - axis[1] * offset[0]), 1e-6)

    def test_source_radius_does_not_change_placement_orientation_or_noise(self):
        small = make_scenario(replace(self.config, source_radius=25.), 7, 1)
        large = make_scenario(replace(self.config, source_radius=150.), 7, 1)
        assert_allclose(large.source - large.centre, 6 * (small.source - small.centre))
        self.assertEqual(small.orientation, large.orientation)
        assert_array_equal(small.noise_rad, large.noise_rad)
        for name in FORMATIONS:
            def positions(scene):
                return formation_positions(name, 5, scene.centre, scene.orientation, 100, rng_for(7, 1, 1005))
            assert_array_equal(positions(small), positions(large))

    def test_controlled_reproducibility_and_noise_pairing_with_randomized(self):
        records, drones, _ = run_experiments(self.config)
        self.assertEqual(records, self.records)
        self.assertEqual(drones, self.drones)
        controlled = make_scenario(self.config, 7, 0)
        randomized = make_scenario(replace(self.config, benchmark="randomized"), 7, 0)
        assert_array_equal(controlled.noise_rad, randomized.noise_rad)
        self.assertEqual(controlled.orientation, randomized.orientation)

    def test_randomized_v1_sampling_streams_are_unchanged(self):
        config = replace(self.config, benchmark="randomized")
        scene = make_scenario(config, 7, 2)
        assert_array_equal(scene.source, rng_for(7, 2, 0).uniform([0, 0], [config.width, config.height]))
        margin = config.placement_margin
        assert_array_equal(scene.centre, rng_for(7, 2, 1).uniform([margin, margin], [config.width-margin, config.height-margin]))
        self.assertEqual(scene.scenario_id, "s7-t0002")

    def test_core_accepts_positions_unrelated_to_benchmark_centre(self):
        config = replace(self.config, station=(300., 100.), noise_std_deg=0.)
        scene = replace(make_scenario(config, 7, 0), source=np.array([850., 800.]))
        positions = np.array([[100., 100.], [900., 100.], [100., 900.]])
        trial = evaluate_trial(config, scene, "arbitrary", positions, 2000.)
        self.assertTrue(trial.estimate.valid)
        self.assertTrue(trial.network.all_reachable)
        assert_allclose(trial.estimate.position, scene.source, atol=1e-10)
        disconnected = evaluate_trial(config, scene, "arbitrary", positions, 10.)
        self.assertFalse(disconnected.network.all_reachable)
        self.assertEqual(disconnected.record["quality"], 0)

    def test_outputs_identify_protocol_sampling_diagnostics_and_sensitivity(self):
        summaries = build_summaries(self.records, self.drones, 80, self.config.reference_distance)
        for row in summaries["quality_sensitivity"]:
            if row["reference_multiplier"] == 1:
                primary = next(r for r in summaries["overall"] if r["formation"] == row["formation"])
                self.assertEqual(row["mean_quality"], primary["mean_quality"])
                self.assertEqual(row["quality_ci95"], primary["quality_ci95"])
        with tempfile.TemporaryDirectory() as directory:
            save_results(directory, self.config, self.records, self.drones, self.scenes, summaries)
            path = Path(directory)
            metadata = json.loads((path / "metadata.json").read_text())
            self.assertEqual(metadata["benchmark_version"], "controlled-v1")
            self.assertEqual(metadata["source_sampling"]["radius_m"], self.config.source_radius)
            self.assertEqual(metadata["source_sampling"]["centre_m"], list(self.config.deployment_centre))
            for name in ("scene_disconnected.png", "quality_sensitivity.csv", "quality_sensitivity.png"):
                self.assertGreater((path / name).stat().st_size, 20)
            failure_case = metadata["disconnected_scene_case_id"]
            self.assertTrue(any(row["case_id"] == failure_case and not row["fully_connected"] for row in self.records))
            self.assertEqual(len([row for row in self.records if row["case_id"] == failure_case]), 4)

    def test_invalid_controlled_sampling_configuration_is_rejected(self):
        for config in (replace(self.config, source_radius=0), replace(self.config, source_radius=600),
                       replace(self.config, deployment_centre=(0., 0.)), replace(self.config, benchmark="unknown")):
            with self.assertRaises(ValueError):
                config.validate()


class SequentialTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(benchmark="controlled", seeds=(7, 19), trials_per_seed=2,
                             drone_counts=(3,), communication_ranges=(200., 350.), bootstrap_samples=50)
        self.scenario = replace(make_scenario(self.config, 7, 0), source=np.array([10., 10.]), noise_rad=np.zeros(3))
        self.positions = np.array([[1., 0.], [1., 1.], [0., 1.]])
        self.exact = replace(self.config, station=(0., 0.), noise_std_deg=0.)

    def test_initial_measurement_and_hold_still_history(self):
        episode, steps, drones, metrics = run_episode(self.exact, self.scenario, self.positions, 100, "hold_still", 4)
        self.assertEqual([row["time_s"] for row in steps], [0, 2, 4, 6, 8])
        self.assertEqual([row["cumulative_measurements"] for row in steps], [3, 6, 9, 12, 15])
        self.assertEqual(episode["generated_measurements"], 15)
        self.assertEqual(episode["duration_s"], 8)
        self.assertEqual(episode["swarm_travel_m"], 0)
        self.assertTrue(episode["terminal_connected"] and episode["continuously_connected"])
        self.assertTrue(all(row["measurement_delivered"] for row in drones))
        self.assertTrue(all(row["outage_duration_s"] == 0 for row in metrics))

    def test_reconnection_retains_old_history_without_backfilling_outage(self):
        actions = [np.tile([3., 0.], (3, 1)), np.tile([-3., 0.], (3, 1))]
        with patch("swarm_link.experiment.controller_displacements", side_effect=actions), \
             patch("swarm_link.experiment.estimate_position", wraps=estimate_position) as estimator:
            episode, steps, drones, metrics = run_episode(self.exact, self.scenario, self.positions, 1.5, "random_movement", 2)
        self.assertEqual([row["fully_connected"] for row in steps], [True, False, True])
        self.assertEqual([row["cumulative_measurements"] for row in steps], [3, 3, 6])
        self.assertTrue(steps[1]["estimate_valid"])
        self.assertEqual(episode["generated_measurements"], 9)
        self.assertTrue(episode["terminal_connected"])
        self.assertFalse(episode["continuously_connected"])
        self.assertAlmostEqual(episode["quality"], 1)
        self.assertEqual(episode["outage_duration_s"], 2)
        for row in metrics:
            self.assertEqual(row["disconnection_events"], 1)
            self.assertEqual(row["reconnection_events"], 1)
            self.assertEqual(row["outage_duration_s"], 2)
        self.assertTrue(all(not row["measurement_delivered"] for row in drones if row["step"] == 1))
        assert_array_equal(estimator.call_args.args[0], np.vstack((self.positions, self.positions)))

    def test_initially_disconnected_data_never_uploads_after_reconnection(self):
        initial = self.positions + [3, 0]
        with patch("swarm_link.experiment.controller_displacements", return_value=np.tile([-3., 0.], (3, 1))), \
             patch("swarm_link.experiment.estimate_position", wraps=estimate_position) as estimator:
            episode, steps, _, metrics = run_episode(self.exact, self.scenario, initial, 1.5, "random_movement", 1)
        self.assertEqual([row["cumulative_measurements"] for row in steps], [0, 3])
        self.assertEqual(steps[0]["estimate_status"], "insufficient_measurements")
        assert_array_equal(estimator.call_args.args[0], self.positions)
        self.assertTrue(episode["terminal_connected"])
        self.assertFalse(episode["continuously_connected"])
        self.assertTrue(all(row["disconnection_events"] == 0 and row["reconnection_events"] == 1 for row in metrics))

    def test_final_disconnection_zeros_score_but_keeps_valid_prior_estimate(self):
        with patch("swarm_link.experiment.controller_displacements", return_value=np.tile([3., 0.], (3, 1))):
            episode, steps, _, _ = run_episode(self.exact, self.scenario, self.positions, 1.5, "random_movement", 1)
        self.assertTrue(episode["estimate_valid"])
        self.assertFalse(episode["terminal_connected"])
        self.assertEqual(episode["quality"], 0)
        self.assertEqual(steps[-1]["cumulative_measurements"], 3)

    def test_persistent_disconnection_never_ends_episode_early(self):
        episode, steps, drones, metrics = run_episode(self.exact, self.scenario, self.positions + 100, 1, "hold_still", 5)
        self.assertEqual(len(steps), 6)
        self.assertEqual(len(drones), 18)
        self.assertEqual(episode["quality"], 0)
        self.assertEqual(episode["outage_duration_s"], 10)
        self.assertTrue(all(row["outage_duration_s"] == 10 for row in metrics))
        self.assertTrue(all(row["estimate_x_m"] is None for row in steps))

    def test_history_keeps_each_acquisition_position(self):
        action = np.tile([1., 0.], (3, 1))
        with patch("swarm_link.experiment.controller_displacements", return_value=action), \
             patch("swarm_link.experiment.estimate_position", wraps=estimate_position) as estimator:
            episode, _, _, _ = run_episode(self.exact, self.scenario, self.positions, 100, "random_movement", 2)
        expected = np.vstack((self.positions, self.positions + action, self.positions + 2*action))
        assert_array_equal(estimator.call_args.args[0], expected)
        self.assertLess(episode["error_m"], 1e-10)

    def test_motion_limit_wall_truncation_and_actual_travel(self):
        for seed in range(20):
            action = controller_displacements("random_movement", 7, rng_for(seed, 0, 99), 10)
            self.assertTrue(np.all(np.linalg.norm(action, axis=1) <= 10))
        assert_array_equal(move_positions([[999, 999]], [[6, 8]], [1000, 1000], 10), [[1000, 1000]])
        with self.assertRaises(ValueError):
            move_positions([[0, 0]], [[10, 10]], [1000, 1000], 10)
        with patch("swarm_link.experiment.controller_displacements", return_value=np.tile([6., 8.], (3, 1))):
            episode, _, drones, _ = run_episode(self.config, self.scenario, self.positions + 999, 2000, "random_movement", 1)
        self.assertTrue(all(row["boundary_limited"] for row in drones if row["step"] == 1))
        self.assertLess(episode["swarm_travel_m"], 30)

    def test_reachability_duration_and_event_conventions(self):
        metrics = reachability_metrics([True, False, False, True, False], 2)
        self.assertEqual(metrics["outage_duration_s"], 4)
        self.assertEqual(metrics["longest_outage_s"], 4)
        self.assertEqual(metrics["disconnection_events"], 2)
        self.assertEqual(metrics["reconnection_events"], 1)
        self.assertEqual(metrics["connected_step_fraction"], .4)
        self.assertEqual(reachability_metrics([True, False], 2)["outage_duration_s"], 0)
        self.assertEqual(reachability_metrics([False, False, True], 2)["outage_duration_s"], 4)

    def test_zero_horizon_and_other_swarm_sizes_are_supported(self):
        for n in (2, 7):
            scene = replace(self.scenario, noise_rad=np.zeros(n))
            positions = np.column_stack((np.arange(n)+1., np.arange(n) % 2))
            episode, steps, drones, _ = run_episode(self.exact, scene, positions, 100, "hold_still", 0)
            self.assertEqual(episode["generated_measurements"], n)
            self.assertEqual(episode["duration_s"], 0)
            self.assertEqual(len(steps), 1)
            self.assertEqual(len(drones), n)

    def test_matched_budgets_initial_conditions_noise_and_repeatability(self):
        # Separate purpose/time keys avoid collisions when configuring longer horizons.
        self.assertNotEqual(rng_for(7, 0, 5000, 1).random(), rng_for(7, 0, 3000, 2001).random())
        legacy = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7, 0, 3])))
        assert_array_equal(rng_for(7, 0, 3).normal(size=8), legacy.normal(size=8))
        output = run_sequential(self.config, movement_steps=3)
        self.assertEqual(output, run_sequential(self.config, movement_steps=3))
        episodes, _, drones, _ = output
        cases = {}
        for row in episodes:
            cases.setdefault(row["case_id"], []).append(row)
            self.assertEqual(row["generated_measurements"], 12)
        self.assertTrue(all(len(rows) == 2 for rows in cases.values()))
        paired = {}
        for row in drones:
            key = (row["case_id"], row["step"], row["drone_id"])
            paired.setdefault(key, []).append(row)
        for rows in paired.values():
            self.assertEqual(rows[0]["noise_rad"], rows[1]["noise_rad"])
            if rows[0]["step"] == 0:
                self.assertEqual((rows[0]["x_m"], rows[0]["y_m"]), (rows[1]["x_m"], rows[1]["y_m"]))

    def test_controller_actions_cannot_depend_on_hidden_source(self):
        changed = replace(self.scenario, source=np.array([800., 800.]))
        first = run_episode(self.config, self.scenario, self.positions, 1000, "random_movement", 5)[2]
        second = run_episode(self.config, changed, self.positions, 1000, "random_movement", 5)[2]
        self.assertEqual([(r["x_m"], r["y_m"]) for r in first], [(r["x_m"], r["y_m"]) for r in second])
        self.assertEqual(list(inspect.signature(controller_displacements).parameters), ["controller", "n_drones", "rng", "max_displacement"])

    def test_sequential_metric_denominators_use_terminal_not_continuous(self):
        base = run_episode(self.exact, self.scenario, self.positions, 100, "hold_still", 0)[0]
        rows = [dict(base, scenario_id="a", continuously_connected=False, quality=1., error_m=0.),
                dict(base, scenario_id="b", terminal_connected=False, continuously_connected=False, quality=0.),
                dict(base, scenario_id="c", estimate_valid=False, estimate_status="rank_deficient", quality=0., error_m=None)]
        summary = summarize_episodes(rows, 50)
        self.assertAlmostEqual(summary["mean_quality"], 1/3)
        self.assertAlmostEqual(summary["terminal_connectivity_pct"], 200/3)
        self.assertAlmostEqual(summary["continuous_connectivity_pct"], 100/3)
        self.assertEqual(summary["accuracy_episodes"], 1)

    def test_sequential_artifacts_and_static_directory_protection(self):
        settings = dict(movement_steps=2, step_seconds=2., max_speed=5., initial_formations=("polygon",))
        output = run_sequential(self.config, **settings)
        summary = build_sequential_summaries(output[0], 50)
        with tempfile.TemporaryDirectory() as directory:
            save_sequential_results(directory, self.config, settings, *output, summary)
            for name in ("metadata.json", "summary.json", "episodes.csv", "steps.csv", "drone_steps.csv",
                         "drone_episodes.csv", "trajectory.png", "diagnostics.png", "metrics.png", "summary.md"):
                self.assertGreater((Path(directory) / name).stat().st_size, 20)
            metadata = Path(directory) / "metadata.json"
            metadata.write_text('{"experiment":"static"}')
            with self.assertRaises(ValueError):
                save_sequential_results(directory, self.config, settings, *output, summary)
            self.assertEqual(metadata.read_text(), '{"experiment":"static"}')

    def test_invalid_movement_parameters_fail(self):
        for kwargs in (dict(movement_steps=-1), dict(step_seconds=0), dict(max_speed=-1)):
            with self.assertRaises(ValueError):
                run_episode(self.config, self.scenario, self.positions, 100, "hold_still", **kwargs)


if __name__ == "__main__":
    unittest.main()
