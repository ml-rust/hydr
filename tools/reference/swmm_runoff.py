"""CLI wiring for the pinned untouched EPA runoff reference."""

import argparse
import subprocess
from pathlib import Path

from swmm_runoff_build import build
from swmm_runoff_run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("build", build), ("run", run)):
        command = commands.add_parser(name)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--build-dir", type=Path, required=True)
        command.set_defaults(function=function)
        if name == "build":
            command.add_argument("--cc", default="gcc")
        else:
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
