"""Run with: uv run --locked python -m swarm_link."""

import argparse
from pathlib import Path

from .experiment import build_summaries, run_experiments, run_sequential, build_sequential_summaries
from .report import save_results, summary_markdown, save_sequential_results
from .scenarios import Config, FORMATIONS


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Reproduce matched DoA localization/connectivity experiments")
    parser.add_argument("--experiment", choices=("static", "sequential"), default="static")
    parser.add_argument("--benchmark", choices=("randomized", "controlled"), help="Initial sampling: static defaults to randomized; sequential to controlled")
    parser.add_argument("--output", type=Path, help="Output directory; sequential runs are kept separate from static results")
    parser.add_argument("--steps", type=int, help="Sequential movement steps (default: 60; measurements include t=0)")
    parser.add_argument("--step-seconds", type=float, help="Sequential seconds per movement (default: 2)")
    parser.add_argument("--max-speed", type=float, help="Sequential speed limit [m/s] (default: 5)")
    parser.add_argument("--initial-formations", choices=FORMATIONS, nargs="+", help="Sequential initial placements only (default: polygon)")
    parser.add_argument("--deployment-centre", nargs=2, type=float, metavar=("X", "Y"),
                        help="Controlled benchmark centre in metres (default: 500 350)")
    parser.add_argument("--source-radius", type=float, help="Controlled uniform-area source disk radius [m] (default: 75)")
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults.seeds)
    parser.add_argument("--trials-per-seed", type=int, help="Static default: 125; sequential: 25")
    parser.add_argument("--drone-counts", nargs="+", type=int, help="Static default: 3 5 8; sequential: 5")
    parser.add_argument("--ranges", nargs="+", type=float, help="Link radii [m]; static: 200 350 500; sequential: 200 350")
    parser.add_argument("--noise-std-deg", type=float, default=defaults.noise_std_deg)
    parser.add_argument("--rms-radius", type=float, default=defaults.rms_radius)
    parser.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    args = parser.parse_args()
    sequential = args.experiment == "sequential"
    benchmark = args.benchmark or ("controlled" if sequential else "randomized")
    if not sequential and any(value is not None for value in (args.steps, args.step_seconds, args.max_speed, args.initial_formations)):
        parser.error("Movement options apply only to --experiment sequential")
    if benchmark != "controlled" and (args.deployment_centre is not None or args.source_radius is not None):
        parser.error("--deployment-centre and --source-radius apply only to --benchmark controlled")
    config = Config(seeds=tuple(args.seeds), trials_per_seed=args.trials_per_seed if args.trials_per_seed is not None else (25 if sequential else defaults.trials_per_seed),
                    drone_counts=tuple(args.drone_counts or ((5,) if sequential else defaults.drone_counts)),
                    communication_ranges=tuple(args.ranges or ((200., 350.) if sequential else defaults.communication_ranges)),
                    noise_std_deg=args.noise_std_deg, rms_radius=args.rms_radius,
                    bootstrap_samples=args.bootstrap_samples, benchmark=benchmark,
                    deployment_centre=tuple(args.deployment_centre) if args.deployment_centre is not None else defaults.deployment_centre,
                    source_radius=args.source_radius if args.source_radius is not None else defaults.source_radius)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    if sequential:
        settings = dict(movement_steps=args.steps if args.steps is not None else 60,
                        step_seconds=args.step_seconds if args.step_seconds is not None else 2.,
                        max_speed=args.max_speed if args.max_speed is not None else 5.,
                        initial_formations=tuple(args.initial_formations or ("polygon",)))
        sizes = "-".join(str(n) for n in config.drone_counts)
        output = args.output if args.output is not None else Path("results/sequential") / benchmark / f'n{sizes}-t{settings["movement_steps"]}'
        try:
            episodes, steps, drones, drone_episodes = run_sequential(config, **settings)
            summaries = build_sequential_summaries(episodes, config.bootstrap_samples)
            save_sequential_results(output, config, settings, episodes, steps, drones, drone_episodes, summaries)
        except ValueError as error:
            parser.error(str(error))
        print(f"Sequential {benchmark}: {len(episodes)} episodes; {settings['movement_steps']} moves × {settings['step_seconds']:g} s")
        for row in summaries["overall"]:
            print(f'{row["controller"]}: Q={row["mean_quality"]:.4f}; terminal={row["terminal_connectivity_pct"]:.2f}%; continuous={row["continuous_connectivity_pct"]:.2f}%')
        print(f"All outputs: {output.resolve()}")
        return
    records, drones, scenes = run_experiments(config)
    summaries = build_summaries(records, drones, config.bootstrap_samples, config.reference_distance)
    output = args.output if args.output is not None else Path("results") / config.benchmark
    save_results(output, config, records, drones, scenes, summaries)
    print(summary_markdown(summaries, config).split("## Paired quality differences")[0])
    print(f"All outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
