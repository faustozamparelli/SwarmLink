"""Run with: python -m swarm_link --output results."""

import argparse
from pathlib import Path

from .experiment import build_summaries, run_experiments
from .report import save_results, summary_markdown
from .scenarios import Config


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Reproduce matched static DoA localization/connectivity baselines")
    parser.add_argument("--benchmark", choices=("randomized", "controlled"), default="randomized")
    parser.add_argument("--output", type=Path, help="Default: results/<benchmark>; existing files here are replaced")
    parser.add_argument("--deployment-centre", nargs=2, type=float, metavar=("X", "Y"),
                        help="Controlled benchmark centre in metres (default: 500 350)")
    parser.add_argument("--source-radius", type=float, help="Controlled uniform-area source disk radius [m] (default: 75)")
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults.seeds)
    parser.add_argument("--trials-per-seed", type=int, default=defaults.trials_per_seed)
    parser.add_argument("--drone-counts", nargs="+", type=int, default=defaults.drone_counts)
    parser.add_argument("--ranges", nargs="+", type=float, default=defaults.communication_ranges)
    parser.add_argument("--noise-std-deg", type=float, default=defaults.noise_std_deg)
    parser.add_argument("--rms-radius", type=float, default=defaults.rms_radius)
    parser.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    args = parser.parse_args()
    if args.benchmark != "controlled" and (args.deployment_centre is not None or args.source_radius is not None):
        parser.error("--deployment-centre and --source-radius apply only to --benchmark controlled")
    config = Config(seeds=tuple(args.seeds), trials_per_seed=args.trials_per_seed,
                    drone_counts=tuple(args.drone_counts), communication_ranges=tuple(args.ranges),
                    noise_std_deg=args.noise_std_deg, rms_radius=args.rms_radius,
                    bootstrap_samples=args.bootstrap_samples, benchmark=args.benchmark,
                    deployment_centre=tuple(args.deployment_centre) if args.deployment_centre is not None else defaults.deployment_centre,
                    source_radius=args.source_radius if args.source_radius is not None else defaults.source_radius)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    records, drones, scenes = run_experiments(config)
    summaries = build_summaries(records, drones, config.bootstrap_samples, config.reference_distance)
    output = args.output if args.output is not None else Path("results") / config.benchmark
    save_results(output, config, records, drones, scenes, summaries)
    print(summary_markdown(summaries, config).split("## Paired quality differences")[0])
    print(f"All outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
