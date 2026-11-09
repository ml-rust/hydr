"""Run the frozen native runoff catalog with repetition and restricted-component restart."""

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

from swmm_runoff_cases import UNSUPPORTED, cases
from swmm_runoff_checks import check_case
from swmm_runoff_records import (
    CSV_FIELDS,
    NATIVE_FIELDS,
    command_stream,
    observations,
    read_native,
    state_values,
)
from swmm_runoff_sources import adapter_hashes
from swmm_soil_sources import clean_environment, fresh, sha256, source_identity


def execute(binary, name, calls, directory, suffix):
    commands = command_stream(calls)
    stdin_path = directory / f"{name}.{suffix}.stdin"
    stdout_path = directory / f"{name}.{suffix}.stdout.csv"
    stderr_path = directory / f"{name}.{suffix}.stderr"
    with stdin_path.open("x") as stream:
        stream.write(commands)
    with stdout_path.open("x") as out, stderr_path.open("x") as err:
        process = subprocess.run(
            [str(binary)],
            input=commands,
            text=True,
            stdout=out,
            stderr=err,
            env=clean_environment(),
        )
    if process.returncode != 0:
        raise ValueError(f"Case={name} exit={process.returncode}: read {stderr_path}")
    rows = read_native(name, calls, stdout_path.read_text())
    return rows, {
        "commands_hex": commands.splitlines(),
        "stdin_sha256": sha256(stdin_path),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }


def compare_tail(name, expected_rows, actual_rows, scope):
    if len(expected_rows) != len(actual_rows):
        raise ValueError(f"Case={name}: {scope} tail length differs")
    for expected, actual in zip(expected_rows, actual_rows):
        for key in NATIVE_FIELDS:
            if key == "index":
                continue
            same = (
                expected[key] == actual[key]
                if key == "op"
                else expected[key].hex() == actual[key].hex()
            )
            if not same:
                raise ValueError(
                    f"Case={name} field={key}: {scope} physical/diagnostic observation differs"
                )


def restart(binary, case, outputs, directory):
    calls = case.calls()
    positions = [i for i, (operation, _) in enumerate(calls) if operation == "STEP"]
    if len(positions) < 2:
        return {"scope": "No multi-step continuation in this case"}
    cut = positions[len(positions) // 2 - 1]
    remaining = calls[cut + 1 :]
    resumed_calls = [
        ("INIT", case.parameters),
        ("STATE", state_values(outputs[cut])),
    ] + remaining
    resumed, evidence = execute(binary, case.name, resumed_calls, directory, "restart")
    compare_tail(
        case.name, outputs[cut + 1 :], resumed[2:], "fresh restricted-component restart"
    )
    return {
        "scope": "Restricted-component physical observations only, not full SWMM driver rewind",
        "remaining_commands": len(remaining),
        "witness_hex": [value.hex() for value in state_values(outputs[cut])],
        **evidence,
    }


def discarded_trial(binary, case, outputs, directory):
    calls = case.calls()
    positions = [i for i, (operation, _) in enumerate(calls) if operation == "STEP"]
    if len(positions) < 2:
        return {"scope": "No multi-step continuation in this case"}
    cut = positions[len(positions) // 2 - 1]
    prefix, remaining = calls[: cut + 1], calls[cut + 1 :]
    witness = state_values(outputs[cut])
    trial = remaining[0]
    if trial[0] != "STEP":
        raise ValueError(
            f"Case={case.name}: discarded trial requires the next admitted STEP"
        )
    restored_calls = prefix + [trial, ("STATE", witness)] + remaining
    restored, evidence = execute(
        binary, case.name, restored_calls, directory, "discarded"
    )
    compare_tail(
        case.name,
        outputs[cut + 1 :],
        restored[len(prefix) + 2 :],
        "same-process discarded-trial restoration",
    )
    return {
        "scope": "Same-process restricted-component state restoration after actual native advancement",
        "trial_commands_hex": command_stream([trial]).splitlines(),
        "restore_witness_hex": [value.hex() for value in witness],
        "remaining_commands": len(remaining),
        **evidence,
    }


def run(args):
    identity = source_identity(args.source_root, args.archive)
    adapters = adapter_hashes()
    directory = args.build_dir.resolve(strict=True)
    manifest_path = directory / "build-manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    binary = directory / "swmm-runoff-reference"
    compiler = Path(manifest["compiler"])
    compiled = {
        name: digest
        for name, digest in adapters.items()
        if Path(name).suffix in (".c", ".h")
    }
    if (
        identity != manifest["source"]
        or sha256(binary) != manifest["binary_sha256"]
        or sha256(compiler) != manifest["compiler_sha256"]
        or compiled != manifest["compiled_adapter_sha256"]
    ):
        raise ValueError(
            "Native build inputs differ from the manifest: create a fresh canonical build"
        )
    output = fresh(args.output)
    sidecar = fresh(output.with_suffix(".json"))
    raw = fresh(output.with_suffix(".native"))
    if len({output, sidecar, raw}) != 3:
        raise ValueError("Evidence suffixes collide: supply a fresh .csv output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.mkdir()
    catalog = cases()
    if (
        len(catalog) != 29
        or sum(len(case.calls()) for case in catalog) != 214
        or len({case.name for case in catalog}) != len(catalog)
    ):
        raise ValueError(
            "Catalog differs from frozen 29 cases/214 rows: freeze revised inputs before native execution"
        )
    rows = []
    evidence = []
    for case in catalog:
        first, first_evidence = execute(binary, case.name, case.calls(), raw, "first")
        _, second_evidence = execute(binary, case.name, case.calls(), raw, "repeat")
        for key in ("stdout_sha256", "stderr_sha256"):
            if first_evidence[key] != second_evidence[key]:
                raise ValueError(f"Case={case.name}: repeated native {key} differs")
        measured = observations(case, first)
        gates = check_case(case, first, measured)
        continuation = restart(binary, case, first, raw)
        rollback = discarded_trial(binary, case, first, raw)
        rows.extend(measured)
        evidence.append(
            {
                "case": case.name,
                "first": first_evidence,
                "repeat": second_evidence,
                "restricted_restart": continuation,
                "discarded_trial_restore": rollback,
                "observations": gates,
            }
        )
    if (
        source_identity(args.source_root, args.archive) != identity
        or adapter_hashes() != adapters
        or sha256(binary) != manifest["binary_sha256"]
        or sha256(compiler) != manifest["compiler_sha256"]
        or sha256(manifest_path) != manifest_hash
    ):
        raise ValueError(
            "Execution inputs changed: preserve raw evidence and rerun with stable inputs"
        )
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, ".17g") if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    baseline = Path("resource/runoff-reference-acceptance-baseline.md")
    provenance = {
        "scope": "Untouched EPA runoff component evidence only; no hydr runoff API or conservative attachment claim",
        "source": identity,
        "build": manifest,
        "build_manifest_sha256": manifest_hash,
        "adapter_sha256": adapters,
        "python": sys.version,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "invocation": sys.argv,
        "case_count": len(catalog),
        "row_count": len(rows),
        "csv_sha256": sha256(output),
        "cases": evidence,
        "frozen_baseline_sha256": sha256(baseline) if baseline.is_file() else None,
        "unsupported": UNSUPPORTED,
        "accounting": "Endpoint discharge times dt, gross subarea transfer, external-only outlet transfer, routing lag, and competing losses remain separate native observations; no correction disguises material discrepancies",
        "checks": [
            "Pinned archive and every original file",
            "Stable compiler/source/C-header/binary envelope",
            "Hexadecimal inputs and seventeen-digit output",
            "Byte-identical repeated stdout and stderr",
            "Complete restricted-component witness restart",
            "Same-process discarded-trial restoration",
            "Frozen analytic and algebraic gates",
        ],
    }
    with sidecar.open("x") as stream:
        stream.write(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(
        f"Wrote {len(catalog)} native cases/{len(rows)} rows to {output}; native transfer/loss discrepancies remain explicit"
    )
