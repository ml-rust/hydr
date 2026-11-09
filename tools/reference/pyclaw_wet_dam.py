#!/usr/bin/env python3
"""Run an untouched PyClaw Python HLL solver on a declared wet-bed dam case."""

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import types
from pathlib import Path

import numpy as np

CASES = {"wet-quarter": 0.25, "wet-tenth": 0.1}
STEP = 0.01
END = 4.0
GRAVITY = 9.81


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyclaw-root", type=Path, required=True)
    parser.add_argument("--riemann-root", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--cells", type=int, choices=(100, 200, 400), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_solver(pyclaw_root, riemann_root):
    """Load original Python modules without compiled Riemann package imports."""
    for path in (pyclaw_root / "src/pyclaw", riemann_root / "riemann"):
        if not path.is_dir():
            raise ValueError(
                f"Source directory missing at {path}: supply extracted upstream sources"
            )
    namespace = types.ModuleType("clawpack")
    namespace.__path__ = [str(pyclaw_root / "src")]
    riemann = types.ModuleType("clawpack.riemann")
    riemann.__path__ = [str(riemann_root / "riemann")]
    namespace.riemann = riemann
    sys.modules["clawpack"] = namespace
    sys.modules["clawpack.riemann"] = riemann
    with contextlib.redirect_stdout(sys.stderr):
        importlib.import_module("clawpack.riemann.static")
        pyclaw = importlib.import_module("clawpack.pyclaw")
        shallow = importlib.import_module("clawpack.riemann.shallow_1D_py")
    return pyclaw, shallow.shallow_hll_1D


def source_hashes(roots):
    result = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        for label, root in roots.items():
            if path.is_relative_to(root) and path.is_file():
                result[f"{label}/{path.relative_to(root)}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    for label, root in roots.items():
        for name in ("LICENSE", "LICENSE.txt"):
            path = root / name
            if path.is_file():
                result[f"{label}/{name}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return dict(sorted(result.items()))


def check_state(state):
    if state.q.dtype != np.float64:
        raise ValueError(
            f"PyClaw state dtype={state.q.dtype}: require numpy.float64 reference fields"
        )
    if not np.isfinite(state.q).all() or not (state.q[0] > 0.0).all():
        raise ValueError(
            "PyClaw state contains nonfinite or nonpositive water: reject this wet-only reference run"
        )


def run(args):
    adapter_path = Path(__file__).resolve()
    pyclaw_root = args.pyclaw_root.resolve(strict=True)
    riemann_root = args.riemann_root.resolve(strict=True)
    output = args.output.resolve()
    if output.suffix != ".csv":
        raise ValueError(
            f"Output path={output}: supply a .csv file for its .json provenance sidecar"
        )
    sidecar = output.with_suffix(".json")
    for path in (output, sidecar):
        if os.path.lexists(path):
            raise FileExistsError(
                f"Output path={path} already exists: choose a fresh output path"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Upstream logging creates a local log file when its package imports.
    os.chdir(output.parent)
    pyclaw, riemann_solver = load_solver(pyclaw_root, riemann_root)
    np.seterr(divide="raise", invalid="raise", over="raise")
    solver = pyclaw.ClawSolver1D(riemann_solver)
    solver.kernel_language = "Python"
    solver.order = 1
    solver.num_eqn = 2
    solver.num_waves = 2
    solver.fwave = False
    solver.all_bcs = pyclaw.BC.wall
    solver.cfl_desired = 0.4
    solver.cfl_max = 0.45
    solver.dt_initial = STEP
    solver.dt = STEP
    solver.dt_max = STEP
    accepted_times = [0.0]

    def check_accepted_state(_, state):
        check_state(state)
        if state.t > accepted_times[-1]:
            accepted_times.append(float(state.t))

    solver.before_step = check_accepted_state
    dimension = pyclaw.Dimension(-100.0, 100.0, args.cells, name="x")
    domain = pyclaw.Domain(dimension)
    state = pyclaw.State(domain, 2)
    state.problem_data["grav"] = GRAVITY
    centres = state.grid.x.centers
    downstream = CASES[args.case]
    state.q[0] = np.where(centres < 0.0, 1.0, downstream)
    state.q[1] = 0.0
    check_state(state)
    dx = 200.0 / args.cells
    initial_storage = float(np.sum(state.q[0]) * dx)
    solution = pyclaw.Solution(state, domain)
    status = solver.evolve_to_time(solution, END)
    check_state(solution.state)
    if not math.isclose(solution.t, END, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"PyClaw endpoint={solution.t}: require completed horizon {END}"
        )
    if solution.t > accepted_times[-1]:
        accepted_times.append(float(solution.t))
    accepted_steps = np.diff(accepted_times)
    if len(accepted_steps) != status["numsteps"] or not (accepted_steps > 0.0).all():
        raise ValueError(
            f"PyClaw accepted time count={len(accepted_steps)}: require status numsteps={status['numsteps']}"
        )
    if float(np.max(accepted_steps)) > STEP + 1e-12:
        raise ValueError(
            f"PyClaw accepted timestep={np.max(accepted_steps)}: require max_step={STEP}"
        )
    storage = float(np.sum(solution.state.q[0]) * dx)
    residual = storage - initial_storage
    if abs(residual) > 1e-8 + 1e-12 * initial_storage:
        raise ValueError(
            f"PyClaw storage residual={residual}: require conserved unit-width water"
        )
    metadata = {
        "case": args.case,
        "cells": args.cells,
        "time": END,
        "gravity": GRAVITY,
        "left_depth": 1.0,
        "right_depth": downstream,
        "xmin": -100.0,
        "xmax": 100.0,
        "max_step": STEP,
        "method": "pyclaw-python-hll-order1",
    }
    with output.open("x", newline="") as stream:
        for key, value in metadata.items():
            stream.write(f"# {key}={value}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("x", "h", "qx"))
        writer.writerows(zip(centres, solution.state.q[0], solution.state.q[1]))
    status = {
        key: float(value) if math.isfinite(value) else None
        for key, value in status.items()
    }
    manifest = {
        **metadata,
        "width_m": 1.0,
        "initial_storage_per_width_m2": initial_storage,
        "storage_per_width_m2": storage,
        "water_residual_per_width_m2": residual,
        "min_depth_m": float(np.min(solution.state.q[0])),
        "max_depth_m": float(np.max(solution.state.q[0])),
        "status": status,
        "accepted_steps": len(accepted_steps),
        "accepted_dt_min_s": float(np.min(accepted_steps)),
        "accepted_dt_max_s": float(np.max(accepted_steps)),
        "positive_accepted_states_checked": len(accepted_times),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dtype": str(solution.state.q.dtype),
        "pyclaw_version": getattr(pyclaw, "__version__", "unconfirmed"),
        "bootstrap": "namespace packages load original Python HLL and static modules without compiled extensions",
        "source_roots": {"pyclaw": str(pyclaw_root), "riemann": str(riemann_root)},
        "source_sha256": source_hashes(
            {"pyclaw": pyclaw_root, "riemann": riemann_root}
        ),
        "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "command": sys.argv,
    }
    with sidecar.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        f"case={args.case} cells={args.cells} time={solution.t} water_residual={residual:.17e} output={output}"
    )


if __name__ == "__main__":
    try:
        run(arguments())
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        sys.exit(str(error))
