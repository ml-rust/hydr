"""Write exclusive Clawpack fields and execution provenance."""

import csv
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np
from clawpack_sources import adapter_hashes, source_hashes


def prepare_output(path):
    output = path.resolve()
    if output.suffix != ".csv":
        raise ValueError(
            f"Output path={output}: supply .csv with a .json provenance sidecar"
        )
    for candidate in (output, output.with_suffix(".json")):
        if os.path.lexists(candidate):
            raise FileExistsError(
                f"Output path={candidate} already exists: choose a fresh path"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_output(
    output,
    entry,
    case,
    cells,
    pyclaw,
    solution,
    monitor,
    status,
    initial_storage,
    roots,
    backend,
):
    profile = solution.state.q[:, :, 0] if case.allows_dry else solution.state.q
    storage = float(np.sum(profile[0]) * 200.0 / cells)
    residual = storage - initial_storage
    if abs(residual) > 1e-8 + 1e-12 * initial_storage:
        raise ValueError(
            f"PyClaw storage residual={residual}: require conserved unit-width water"
        )
    metadata = case.metadata(cells)
    with output.open("x", newline="") as stream:
        for key, value in metadata.items():
            stream.write(f"# {key}={value}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("x", "h", "qx"))
        writer.writerows(zip(solution.state.grid.x.centers, profile[0], profile[1]))
    intervals = monitor.intervals(status)
    manifest = {
        **metadata,
        "dimensions": 2 if case.allows_dry else 1,
        "width_m": 1.0,
        "ny": 1,
        "ymin": 0.0,
        "ymax": 1.0,
        "initial_storage_per_width_m2": initial_storage,
        "storage_per_width_m2": storage,
        "water_residual_per_width_m2": residual,
        "min_depth_m": float(np.min(profile[0])),
        "max_depth_m": float(np.max(profile[0])),
        "minimum_accepted_depth_m": monitor.min_depth,
        "max_abs_accepted_qy_m2_per_s": monitor.max_abs_qy,
        "negative_accepted_depths": 0,
        "zero_depth_discharge_cells": 0,
        "accepted_states_checked": len(monitor.times),
        "positive_accepted_states_checked": None
        if case.allows_dry
        else len(monitor.times),
        "status": {
            key: float(value) if math.isfinite(value) else None
            for key, value in status.items()
        },
        "accepted_steps": len(intervals),
        "accepted_dt_min_s": float(np.min(intervals)),
        "accepted_dt_max_s": float(np.max(intervals)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dtype": str(solution.state.q.dtype),
        "pyclaw_version": getattr(pyclaw, "__version__", "unconfirmed"),
        "bootstrap": "namespace packages load untouched upstream modules and requested original compiled extensions",
        "external_backend": backend,
        "source_roots": {name: str(root) for name, root in roots.items()},
        "source_sha256": source_hashes(roots, case.allows_dry),
        "adapter_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        "adapter_source_sha256": adapter_hashes(entry),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "command": sys.argv,
    }
    if case.allows_dry:
        manifest["external_numerical_policy"] = {
            "dry_tolerance_m": 1e-14,
            "both_dry_faces": "skip",
            "dry_velocity": "zero",
            "subthreshold_interface_state": "In non-both-dry interface problems, local h/hu/hv and velocities become zero when h<=1e-14 m",
            "negative_inputs": "upstream prints Negative input and zeroes negative depth and discharge",
            "negative_input_monitor": "accepted states checked; inspect captured Fortran stdout for internal Negative input messages",
            "dimensional_split": True,
            "fwave": True,
            "auxiliary_bed_m": 0.0,
        }
    with output.with_suffix(".json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        f"case={case.name} cells={cells} time={solution.t} water_residual={residual:.17e} output={output}"
    )
