#!/usr/bin/env python3
"""Run untouched Clawpack kernels on declared wet-bed or exactly dry dam cases."""

import argparse
import sys
from pathlib import Path

from clawpack_cases import CASES
from clawpack_execution import run


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyclaw-root", type=Path, required=True)
    parser.add_argument("--riemann-root", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--cells", type=int, choices=(100, 200, 400), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(arguments(), Path(__file__).resolve())
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        sys.exit(str(error))
