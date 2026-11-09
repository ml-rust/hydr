"""Freeze and compare actual subcatchment soil calls against the unchanged EPA driver."""

import argparse
import subprocess
from pathlib import Path

from subcatchment_soil_evidence import plan, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "plan", help="Freeze inputs and acceptance controls without native execution"
    )
    for name in ("source-root", "archive", "build-dir", "soil-calls", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument(
        "--mode", choices=("all", "smooth", "histories"), required=True
    )
    prepare.set_defaults(function=plan)
    execute = commands.add_parser(
        "run", help="Check the frozen plan and run repeated native protocols"
    )
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.set_defaults(function=run)
    args = parser.parse_args()
    try:
        args.function(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
