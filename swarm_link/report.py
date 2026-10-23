"""CSV/JSON evidence and static scientific figures for the baseline experiment."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .scenarios import FORMATIONS


COLORS = {"line": "#5766b4", "polygon": "#008776", "compact": "#cb7a18", "random": "#b74d75"}
NETWORK_COLORS = {"direct": "#008776", "relay": "#3467bd", "peer_only": "#d87913", "isolated": "#c54040"}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def plot_scenes(config, scenes, path, selection="Preselected first scene, largest N/range."):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    for ax, trial in zip(axes.flat, scenes):
        row = trial.record
        positions = trial.positions
        points = np.vstack((positions, config.station))
        for i, j in zip(*np.where(np.triu(trial.network.adjacency, 1))):
            ax.plot(points[[i, j], 0], points[[i, j], 1], color="#6e8591", alpha=.6, lw=1)
        for i, point in enumerate(positions):
            ax.scatter(*point, color=NETWORK_COLORS[trial.network.categories[i]], s=45, zorder=4)
            ax.annotate(str(i), point, xytext=(5, 5), textcoords="offset points", fontsize=8)
        usable = np.array([row["used_by_estimator"] for row in trial.drones])
        finite = np.isfinite(trial.bearings)
        for mask, color in [(usable & finite, "#6b6d75"), (~usable & finite, "#d18a8a")]:
            angles = trial.bearings[mask]
            direction = np.column_stack((np.cos(angles), np.sin(angles)))
            # Put arrowheads inside the map so the direction remains visible.
            lengths = np.full(len(angles), .7 * config.reference_distance)
            for axis, bound in enumerate((config.width, config.height)):
                component = direction[:, axis]
                distance = np.where(component >= 0, bound - positions[mask, axis], -positions[mask, axis])
                edge = np.full(len(angles), np.inf)
                np.divide(distance, component, out=edge, where=np.abs(component) > 1e-12)
                lengths = np.minimum(lengths, .94 * edge)
            ax.quiver(positions[mask, 0], positions[mask, 1], lengths * direction[:, 0], lengths * direction[:, 1],
                      angles="xy", scale_units="xy", scale=1, color=color, alpha=.45, width=.003)
        ax.scatter(*config.station, marker="s", color="#202b39", s=90, zorder=5)
        ax.scatter(row["source_x_m"], row["source_y_m"], marker="*", color="#202b39", s=170, zorder=6)
        ax.scatter(row["centre_x_m"], row["centre_y_m"], marker="+", color="#777777", s=65, zorder=5)
        if config.benchmark == "controlled":
            ax.add_patch(plt.Circle(config.deployment_centre, config.source_radius,
                                    fill=False, ls=":", color="#777777", alpha=.6))
        xlim, ylim = [0, config.width], [0, config.height]
        if trial.estimate.valid:
            estimate = trial.estimate.position
            ax.scatter(*estimate, marker="x", color="#b80052", s=90, lw=2, zorder=7)
            ax.plot([row["source_x_m"], estimate[0]], [row["source_y_m"], estimate[1]],
                    ls="--", color="#b80052", lw=1)
            # Keep an out-of-map estimate visible; do not clip it into plausible bounds.
            xlim = [min(0, estimate[0] - 20), max(config.width, estimate[0] + 20)]
            ylim = [min(0, estimate[1] - 20), max(config.height, estimate[1] + 20)]
        connected = "fully connected" if row["fully_connected"] else f'{row["unreachable_drones"]} unreachable'
        error_label = f'{row["error_m"]:.2f} m' if trial.estimate.valid else "unavailable"
        ax.set_title(f'{row["formation"]} | {connected}\n{row["estimate_status"].replace("_", " ")}\n'
                     f'error={error_label}; Q={row["quality"]:.3f}', fontsize=10)
        ax.set(xlabel="x [m]", ylabel="y [m]", xlim=xlim, ylim=ylim, aspect="equal")
        ax.grid(alpha=.18)
    handles = [Line2D([], [], marker="o", ls="", color=color, label=category.replace("_", " "))
               for category, color in NETWORK_COLORS.items()]
    handles.extend([Line2D([], [], marker=marker, color=color, ls="", label=label)
                    for marker, color, label in [("s", "#202b39", "station"), ("*", "#202b39", "true source"),
                                                  ("x", "#b80052", "valid estimate"), ("+", "#777777", "deployment centre")]])
    if config.benchmark == "controlled":
        handles.append(Line2D([], [], ls=":", color="#777777", label=f"source disk (r={config.source_radius:g} m)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)
    row = scenes[0].record
    fig.suptitle(f'{config.benchmark.title()} deployment | {row["scenario_id"]} | N={row["n_drones"]}, R={row["communication_range_m"]:g} m\n'
                 f'{selection}\nArrows: measured rays; pale red: excluded; dashed magenta: localization error.', fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics(config, records, summaries, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    overall = summaries["overall"]
    for i, row in enumerate(overall):
        low, high = row["quality_ci95"]
        axes[0, 0].barh(i, row["mean_quality"], color=COLORS[row["formation"]], height=.65)
        axes[0, 0].plot([low, high], [i, i], color="#202b39", lw=2)
    axes[0, 0].set(yticks=range(len(overall)), yticklabels=[row["formation"] for row in overall],
                   xlim=(0, 1), xlabel="Mean Q across every trial", title="Overall quality | scenario-bootstrap 95% CIs")
    axes[0, 0].invert_yaxis()
    for name in FORMATIONS:
        rows = [row for row in records if row["formation"] == name]
        values = np.sort([row["error_m"] for row in rows if row["fully_connected"] and row["estimate_valid"]])
        if len(values):
            axes[0, 1].plot(values, np.arange(1, len(values) + 1) / len(values), color=COLORS[name], label=name)
        for radius in config.communication_ranges:
            subset = [row for row in rows if row["communication_range_m"] == radius]
            axes[1, 0].scatter(radius, 100 * np.mean([row["fully_connected"] for row in subset]),
                               color=COLORS[name], s=32)
        rates = [100 * np.mean([row["fully_connected"] for row in rows if row["communication_range_m"] == radius])
                 for radius in config.communication_ranges]
        axes[1, 0].plot(config.communication_ranges, rates, color=COLORS[name], label=name)
        quality = [np.mean([row["quality"] for row in rows if row["n_drones"] == n]) for n in config.drone_counts]
        axes[1, 1].plot(config.drone_counts, quality, "o-", color=COLORS[name], label=name)
    axes[0, 1].set(xscale="symlog", xlabel="Localization error [m; linear near zero]", ylabel="Empirical CDF",
                   title="Error | valid, fully connected trials only")
    axes[1, 0].set(xlabel="Communication range [m]", ylabel="Fully connected trials [%]", ylim=(0, 100),
                   title="Strict connectivity | all drones must reach station")
    axes[1, 1].set(xlabel="Number of drones", ylabel="Mean Q across every trial", ylim=(0, 1),
                   xticks=config.drone_counts, title="Quality across swarm sizes | ranges equally weighted")
    for ax in axes.flat:
        ax.grid(alpha=.18, axis="x" if ax is axes[0, 0] else "both")
    axes[0, 1].legend(fontsize=9)
    fig.suptitle(f"{config.benchmark.title()} deployment | matched scenarios | {config.benchmark_version}", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_number(value, decimals=2):
    return "—" if value is None else f"{value:.{decimals}f}"


def summary_markdown(summaries, config):
    sampling = (f"Fixed centre {config.deployment_centre} m; source uniform in area within a {config.source_radius:g} m disk, independent of formation; no convex-hull conditioning."
                if config.benchmark == "controlled" else "Source uniform throughout the map; deployment centre independently uniform in the shared safe square.")
    lines = [f"# {config.benchmark.title()} deployment — {config.benchmark_version}", "", sampling, "",
             f"{len(config.seeds) * config.trials_per_seed:,} paired scenarios; "
             f"{len(config.drone_counts) * len(config.communication_ranges):,} settings per formation; "
             f"L = {config.reference_distance:.6f} m (environment diagonal).", "",
             "Q = L / (L + error) for valid, fully connected trials; Q = 0 otherwise. Every trial is included.", "",
             "| Formation | Trials | Mean Q [95% CI] | Connected % | Estimator failure % | Accuracy n | Mean error m | Median m | RMSE m |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries["overall"]:
        low, high = row["quality_ci95"]
        lines.append(f'| {row["formation"]} | {row["trials"]} | {row["mean_quality"]:.4f} [{low:.4f}, {high:.4f}] | '
                     f'{row["connectivity_pct"]:.2f} | {row["estimator_failure_pct"]:.2f} | {row["accuracy_trials"]} | '
                     f'{format_number(row["mean_error_m"])} | {format_number(row["median_error_m"])} | {format_number(row["rmse_m"])} |')
    lines.extend(["", "Errors use valid, fully connected trials only. Estimator failure % uses all trials, including those with insufficient station-reachable measurements.",
                  "See summary.json for connected-only estimator failure rates, excluded-case counts, failure reasons and per-seed variability.", "",
                  "## Per-drone connectivity", "", "Percentages below use drone observations, not whole trials. Per-drone IDs and settings are in per_drone_summary.csv; raw components and reachability are in drones.csv.", "",
                  "| Formation | Drone observations | Direct % | Relay % | Peer-only % | Isolated % | Mean unreachable drones/trial | Connected-only estimator failure % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summaries["overall"]:
        lines.append(f'| {row["formation"]} | {row["drone_observations"]} | {row["direct_drone_pct"]:.2f} | '
                     f'{row["relay_drone_pct"]:.2f} | {row["peer_only_drone_pct"]:.2f} | {row["isolated_drone_pct"]:.2f} | '
                     f'{row["mean_unreachable_drones"]:.3f} | {format_number(row["connected_estimator_failure_pct"], 3)} |')
    lines.extend(["",
                  "## Paired quality differences", "", "Exploratory 95% percentile intervals resample whole scenario IDs, keeping all N/range settings paired. No multiplicity correction or universal winner claim.", "",
                  "| A | B | Mean Q(A) − Q(B) | 95% CI | Evidence in this experiment |",
                  "|---|---|---:|---:|---|"])
    for row in summaries["pairwise"]:
        lines.append(f'| {row["formation_a"]} | {row["formation_b"]} | {row["mean_quality_difference"]:.4f} | '
                     f'[{row["ci95_low"]:.4f}, {row["ci95_high"]:.4f}] | {row["evidence"]} |')
    if "quality_sensitivity" in summaries:
        lines.extend(["", "## Reference-distance sensitivity", "",
                      "Diagnostic rescoring of the same trials at 0.01, 0.1, 1 and 10 times the map diagonal; primary Q remains at 1×. All failures still score zero. Orders below compare empirical means, not significance. See quality_sensitivity.csv for mean-Q intervals.", "",
                      "| L / diagonal | L [m] | Formation order by mean Q |", "|---:|---:|---|"])
        for multiplier in (0.01, 0.1, 1.0, 10.0):
            rows = [row for row in summaries["quality_sensitivity"] if row["reference_multiplier"] == multiplier]
            order = " > ".join(f'{row["formation"]} ({row["mean_quality"]:.4f})' for row in rows)
            lines.append(f'| {multiplier:g} | {rows[0]["reference_distance_m"]:.3f} | {order} |')
        lines.extend(["", "![Reference-distance sensitivity](quality_sensitivity.png)"])
    lines.extend(["", "![Matched scene](scene.png)", "", "![Metrics](metrics.png)", ""])
    return "\n".join(lines)


def plot_quality_sensitivity(config, rows, path):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for name in FORMATIONS:
        subset = [row for row in rows if row["formation"] == name]
        ax.plot([row["reference_distance_m"] for row in subset], [row["mean_quality"] for row in subset],
                "o-", color=COLORS[name], label=name)
    ax.axvline(config.reference_distance, color="#777777", ls="--", label="Primary L (map diagonal)")
    ax.set(xscale="log", xlabel="Reference distance L [m]", ylabel="Mean Q across every trial", ylim=(0, 1),
           title=f"{config.benchmark.title()} deployment | reference-distance sensitivity")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(output, config, records, drone_records, scenes, summaries):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_data = asdict(config)
    config_json = json.dumps(config_data, sort_keys=True, allow_nan=False)
    metadata = dict(config=config_data, config_sha256=hashlib.sha256(config_json.encode()).hexdigest(),
                    benchmark_version=config.benchmark_version, report_version="static-report-v2",
                    source_sampling=(dict(region="disk", centre_m=config.deployment_centre,
                                          radius_m=config.source_radius, distribution="uniform_area")
                                     if config.benchmark == "controlled" else
                                     dict(region="rectangle", lower_m=[0, 0], upper_m=[config.width, config.height], distribution="uniform_area")),
                    methods=list(FORMATIONS), method_information="Known deployment centre, N, RMS radius, source-independent orientation/placement RNG; no true transmitter position",
                    reference_distance_m=config.reference_distance, placement_margin_m=config.placement_margin,
                    versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
                    random_generator="PCG64 + SeedSequence([seed, trial, stream])",
                    uncertainty="95% percentile bootstrap of paired scenario IDs; all N/range settings stay in each block",
                    scene_case_id=scenes[0].record["case_id"],
                    disconnected_scene_case_id=scenes[4].record["case_id"] if len(scenes) > 4 else None,
                    disconnected_scene_selection="First encountered case with any formation disconnected; illustration only, no statistical exclusions")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    write_csv(output / "trials.csv", records)
    write_csv(output / "drones.csv", drone_records)
    write_csv(output / "leaderboard.csv", summaries["overall"])
    write_csv(output / "by_setting.csv", summaries["by_setting"])
    write_csv(output / "paired_comparisons.csv", summaries["pairwise"])
    write_csv(output / "per_drone_summary.csv", summaries["per_drone"])
    if "quality_sensitivity" in summaries:
        write_csv(output / "quality_sensitivity.csv", summaries["quality_sensitivity"])
        plot_quality_sensitivity(config, summaries["quality_sensitivity"], output / "quality_sensitivity.png")
    markdown = summary_markdown(summaries, config)
    if len(scenes) > 4:
        plot_scenes(config, scenes[4:], output / "scene_disconnected.png",
                    "First case with any formation disconnected; diagnostic selection only.")
        markdown += "\n![Disconnected-case illustration](scene_disconnected.png)\n"
    else:
        (output / "scene_disconnected.png").unlink(missing_ok=True)
        markdown += "\nNo disconnected case occurred in this run.\n"
    (output / "summary.md").write_text(markdown)
    plot_scenes(config, scenes[:4], output / "scene.png")
    plot_metrics(config, records, summaries, output / "metrics.png")
