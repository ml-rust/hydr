"""Frozen input identities and checked native execution evidence."""

import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from subcatchment_soil_compare import CONTROLS, FIELDS, compare
from subcatchment_soil_protocol import (
    check_initialization,
    check_restart,
    execute,
    full_case,
    isolated_case,
    restart_case,
    stream_hash,
)
from subcatchment_soil_records import OLD, bits, observer_files, read_sequences
from swmm_soil_sources import adapter_hashes, fresh, sha256, source_identity

DRIVER_SHA256 = "2a642cef22842de92a97756ed435803e95151e661c4bda142b6707609404434b"


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON={path}: duplicate key={key!r}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"JSON={path}: nonfinite token={value!r}")

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject)


def write_json(path, value):
    with path.open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def tooling_hashes():
    root = Path(__file__).parent
    return {
        path.name: sha256(path) for path in sorted(root.glob("subcatchment_soil*.py"))
    }


def frozen_inputs(paths, mode):
    paths = {key: str(Path(value).resolve(strict=True)) for key, value in paths.items()}
    source = source_identity(Path(paths["source_root"]), Path(paths["archive"]))
    build = Path(paths["build_dir"])
    manifest_path = build / "build-manifest.json"
    manifest = load_json(manifest_path)
    binary = build / "swmm-soil-reference"
    compiler = Path(manifest["compiler"])
    driver = Path(__file__).with_name("swmm_soil_driver.c")
    if (
        manifest["source"] != source
        or sha256(binary) != manifest["binary_sha256"]
        or sha256(driver) != DRIVER_SHA256
        or manifest["driver_sha256"] != DRIVER_SHA256
        or sha256(compiler) != manifest["compiler_sha256"]
        or manifest["native_factors"]
        != {"UCF": 1.0, "InfilFactor": 1.0, "recoveryFactor": 1.0}
    ):
        raise ValueError(
            "Cached native build identity differs: retain evidence and restore the canonical unchanged soil reference"
        )
    sidecar = Path(paths["soil_calls"])
    sequences = read_sequences(sidecar, mode)
    groups = []
    for sequence in sequences:
        groups.append(
            {
                "name": sequence.name,
                "case": sequence.rows[0].case,
                "dt_hex": sequence.dt.hex(),
                "steps": len(sequence.rows),
                "parameters_hex": list(bits(sequence.parameters)),
                "initial_state_hex": list(bits(sequence.rows[0].state(OLD))),
                "full_command_sha256": stream_hash(full_case(sequence)),
                "isolated_command_sha256": stream_hash(isolated_case(sequence)),
                "restart_midpoint": len(sequence.rows) // 2,
            }
        )
    frozen = {
        "scope": "Actual constructed soil calls only, independent of runoff drainage/reference accuracy",
        "paths": paths,
        "mode": mode,
        "controls": CONTROLS,
        "source": source,
        "build": manifest,
        "build_manifest_sha256": sha256(manifest_path),
        "native_binary_sha256": sha256(binary),
        "compiler_sha256": sha256(compiler),
        "driver_sha256": sha256(driver),
        "native_helper_sha256": adapter_hashes(),
        "adapter_sha256": tooling_hashes(),
        "observer_sha256": {str(path): sha256(path) for path in observer_files()},
        "soil_calls_sha256": sha256(sidecar),
        "python": sys.version,
        "platform": platform.platform(),
        "libc": list(platform.libc_ver()),
        "sequences": groups,
        "sequence_count": len(groups),
        "observed_step_count": sum(len(sequence.rows) for sequence in sequences),
    }
    return frozen, sequences


def plan(args):
    output = fresh(args.output)
    paths = {
        key: str(getattr(args, key))
        for key in ("source_root", "archive", "build_dir", "soil_calls")
    }
    frozen, _ = frozen_inputs(paths, args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"frozen": frozen, "invocation": sys.argv})
    print(
        f"Froze {frozen['sequence_count']} complete sequences and {frozen['observed_step_count']} actual calls in {output}"
    )


def compare_sequence(sequence, full, isolated):
    check_initialization(
        full[0], sequence.parameters, sequence.rows[0].state(OLD), sequence.name
    )
    rows = []
    for index, row in enumerate(sequence.rows):
        check_initialization(
            isolated[2 * index],
            sequence.parameters,
            row.state(OLD),
            f"{sequence.name} isolated step={row.step}",
        )
        rows.append(compare(row, full[index + 1], "sequence"))
        rows.append(compare(row, isolated[2 * index + 1], "isolated"))
    return rows


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, ".17g") if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def norms(rows):
    result = {}
    for metric in CONTROLS["absolute"]:
        differences = [row["absolute_difference_" + metric] for row in rows]
        finite = [value for value in differences if math.isfinite(value)]
        result[metric] = {
            "maximum_absolute_difference": max(finite, default=None),
            "nonfinite_difference_count": len(differences) - len(finite),
            "maximum_tolerance_ratio": max(
                (
                    row["absolute_difference_" + metric] / row["tolerance_" + metric]
                    for row in rows
                    if math.isfinite(row["absolute_difference_" + metric])
                ),
                default=None,
            ),
        }
    return result


def run(args):
    plan_path = args.plan.resolve(strict=True)
    plan_hash = sha256(plan_path)
    document = load_json(plan_path)
    if set(document) != {"frozen", "invocation"}:
        raise ValueError("Plan fields differ: create a fresh complete plan")
    frozen = document["frozen"]
    before, sequences = frozen_inputs(frozen["paths"], frozen["mode"])
    if before != frozen:
        raise ValueError(
            "Frozen plan inputs differ before execution: create a fresh plan after checking the change"
        )
    output = fresh(args.output)
    manifest_path = fresh(output.with_suffix(".json"))
    raw = fresh(output.with_suffix(".native"))
    if output.suffix != ".csv" or len({output, manifest_path, raw}) != 3:
        raise ValueError("Evidence output requires a fresh .csv path")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.mkdir()
    write_json(
        raw / "execution-controls.json",
        {"plan": str(plan_path), "plan_sha256": plan_hash, "frozen": frozen},
    )
    binary = Path(frozen["paths"]["build_dir"]) / "swmm-soil-reference"
    evidence, rows, restarts = [], [], []
    errors = []
    complete = False
    try:
        for sequence in sequences:
            full = execute(
                binary, raw, sequence.name + "-sequence", full_case(sequence), evidence
            )
            isolated = execute(
                binary,
                raw,
                sequence.name + "-isolated",
                isolated_case(sequence),
                evidence,
            )
            rows.extend(compare_sequence(sequence, full, isolated))
            midpoint, case = restart_case(sequence, full)
            restarted = execute(binary, raw, sequence.name + "-restart", case, evidence)
            check_restart(full, restarted, midpoint)
            restarts.append(
                {
                    "sequence": sequence.name,
                    "midpoint": midpoint,
                    "suffix_steps": len(restarted) - 1,
                    "exact_native_suffix": True,
                }
            )
        complete = True
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        errors.append(f"Native execution/protocol error: {error}")
    stable = False
    try:
        after, _ = frozen_inputs(frozen["paths"], frozen["mode"])
        stable = after == frozen and sha256(plan_path) == plan_hash
        if not stable:
            errors.append(
                "Execution inputs changed: retain raw evidence and create a fresh checked plan"
            )
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"Post-execution identity check error: {error}")
    write_csv(output, rows)
    failed = [row for row in rows if not row["passed"]]
    success = complete and stable and not errors and not failed
    report = {
        "scope": frozen["scope"],
        "passed": success,
        "complete": complete,
        "stable_inputs_before_after": stable,
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "invocation": sys.argv,
        "frozen": frozen,
        "native_streams": evidence,
        "native_midpoint_restarts": restarts,
        "comparison_row_count": len(rows),
        "failed_comparison_count": len(failed),
        "comparison_norms": norms(rows),
        "errors": errors,
        "csv": str(output),
        "csv_sha256": sha256(output),
        "raw_sha256": {
            path.name: sha256(path) for path in sorted(raw.iterdir()) if path.is_file()
        },
    }
    write_json(manifest_path, report)
    if not success:
        raise ValueError(
            f"Soil sequence acceptance rejected: complete={complete}, stable={stable}, differing_rows={len(failed)}, execution_errors={errors}. Read {output} and {manifest_path}"
        )
    print(
        f"Checked {len(sequences)} sequences, {len(rows)} full/isolated comparisons, and {len(restarts)} exact native restarts: {output}"
    )
