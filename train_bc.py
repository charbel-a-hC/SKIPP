import argparse
import sys
import warnings

import yaml

import wandb
from bc.BCRunner import BCRunner
from configs.BCSweepConfig import BCSweepConfig

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def populate_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default="train_bc_sweep.config.yaml",
    )

    return parser


def main(args: argparse.ArgumentParser):
    with open(args.config, "r") as f:
        main_config = yaml.safe_load(f)

    sweep_config = main_config["sweep_params"]
    project, entity = main_config["project"], main_config["entity"]

    bc_config = BCSweepConfig.from_yaml(args.config)

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

    bc_runner = BCRunner(bc_config=bc_config, sweep_id=sweep_id)

    wandb.agent(
        sweep_id, project=project, entity=entity, function=bc_runner.train
    )


if __name__ == "__main__":
    parser = populate_parser(argparse.ArgumentParser(description="SKIPP"))
    args, _ = parser.parse_known_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(args)
