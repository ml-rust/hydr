"""Execute serialized native probes and preserve their complete evidence envelope."""

import csv
import io
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from swmm_soil_cases import ZERO, cases, check_case, check_pairs
from swmm_soil_sources import (
    adapter_hashes,
    clean_environment,
    fresh,
    sha256,
    source_identity,
)

NATIVE_FIELDS = [
    "op",
    "index",
    "rate",
    "F2",
    "S",
    "Ks",
    "IMDmax",
    "Lu",
    "IMD",
    "F",
    "Fu",
    "Sat",
    "T",
]
STATE_FIELDS = ["IMD", "F", "Fu", "Sat", "T"]
CSV_FIELDS = [
    "case",
    "index",
    "operation",
    "model",
    "initial_mode",
    "S_ft",
    "Ks_ft_per_s",
    "IMDmax",
    "initial_IMD",
    "initial_F_ft",
    "initial_Fu_ft",
    "initial_Sat",
    "initial_T_s",
    "before_IMD",
    "before_F_ft",
    "before_Fu_ft",
    "before_Sat",
    "before_T_s",
    "dt_s",
    "irate_ft_per_s",
    "ponded_depth_ft",
    "f1_ft",
    "c1_ft",
    "root_ks_ft_per_s",
    "root_ts_s",
    "rate_ft_per_s",
    "interval_infiltration_ft",
    "accumulated_interval_infiltration_ft",
    "F2_ft",
    "Lu_ft",
    "IMD",
    "F_ft",
    "Fu_ft",
    "Sat",
    "T_s",
]


def command_stream(case):
    return "".join(
        op + " " + " ".join(float(value).hex() for value in values) + "\n"
        for op, values in case.calls()
    )


def read_native(case, output):
    reader = csv.DictReader(io.StringIO(output))
    if reader.fieldnames != NATIVE_FIELDS:
        raise ValueError(
            f"Case={case.name}: native CSV header differs from the adapter contract"
        )
    rows = list(reader)
    calls = case.calls()
    if len(rows) != len(calls):
        raise ValueError(
            f"Case={case.name}: native output count differs from input commands"
        )
    parsed = []
    for index, (row, (operation, _)) in enumerate(zip(rows, calls), 1):
        if None in row or row["op"] != operation or row["index"] != str(index):
            raise ValueError(
                f"Case={case.name} index={index}: native command identity differs"
            )
        numeric = {}
        for key in NATIVE_FIELDS[2:]:
            value = row[key]
            if value is None or (value == "" and key not in ("rate", "F2")):
                raise ValueError(f"Case={case.name} field={key}: missing native value")
            numeric[key] = None if value == "" else float(value)
            if numeric[key] is not None and not math.isfinite(numeric[key]):
                raise ValueError(
                    f"Case={case.name} field={key}: nonfinite native value"
                )
        if numeric["Sat"] not in (0.0, 1.0):
            raise ValueError(f"Case={case.name}: native Sat differs from zero/one")
        if (numeric["rate"] is not None) != (operation == "STEP") or (
            numeric["F2"] is not None
        ) != (operation == "F2"):
            raise ValueError(
                f"Case={case.name}: native optional result fields differ from command"
            )
        parsed.append(numeric)
    return parsed


def evidence_rows(case, outputs):
    initial = outputs[0]
    calls = case.calls()
    amounts = []
    for index, ((operation, inputs), output) in enumerate(zip(calls, outputs)):
        row = dict.fromkeys(CSV_FIELDS, "")
        row.update(
            case=case.name,
            index=index + 1,
            operation=operation,
            model=case.model,
            initial_mode=calls[0][0],
            S_ft=case.parameters[0],
            Ks_ft_per_s=case.parameters[1],
            IMDmax=case.parameters[2],
        )
        for prefix, state in (
            ("initial_", initial),
            ("before_", outputs[index - 1] if index else None),
        ):
            if state is not None:
                for key, suffix in zip(
                    STATE_FIELDS, ("IMD", "F_ft", "Fu_ft", "Sat", "T_s")
                ):
                    row[prefix + suffix] = state[key]
        if operation == "STEP":
            _, dt, rain, depth = inputs
            amount = output["rate"] * dt
            amounts.append(amount)
            row.update(
                dt_s=dt,
                irate_ft_per_s=rain,
                ponded_depth_ft=depth,
                interval_infiltration_ft=amount,
            )
        if operation == "F2":
            row.update(zip(("f1_ft", "c1_ft", "root_ks_ft_per_s", "root_ts_s"), inputs))
        row.update(
            rate_ft_per_s=output["rate"],
            F2_ft=output["F2"],
            Lu_ft=output["Lu"],
            IMD=output["IMD"],
            F_ft=output["F"],
            Fu_ft=output["Fu"],
            Sat=output["Sat"],
            T_s=output["T"],
            accumulated_interval_infiltration_ft=math.fsum(amounts),
        )
        yield {
            key: format(value, ".17g") if isinstance(value, float) else value
            for key, value in row.items()
        }


def run(args):
    identity = source_identity(args.source_root, args.archive)
    adapters = adapter_hashes()
    build_path = args.build_dir.resolve(strict=True)
    manifest_path = build_path / "build-manifest.json"
    build_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    binary = build_path / "swmm-soil-reference"
    driver = Path(__file__).with_name("swmm_soil_driver.c")
    compiler = Path(manifest["compiler"])
    if (
        manifest["source"] != identity
        or sha256(binary) != manifest["binary_sha256"]
        or sha256(driver) != manifest["driver_sha256"]
        or sha256(compiler) != manifest["compiler_sha256"]
    ):
        raise ValueError(
            "Run inputs differ from the build manifest: create a fresh canonical build"
        )
    output = fresh(args.output)
    sidecar = fresh(output.with_suffix(".json"))
    raw_directory = fresh(output.with_suffix(".native"))
    if len({output, sidecar, raw_directory}) != 3:
        raise ValueError(
            "Output suffix collides with evidence files: supply a .csv output path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_directory.mkdir()
    results = {}
    evidence = []
    rows = []
    for case in cases():
        commands = command_stream(case)
        stdin_path = raw_directory / (case.name + ".stdin")
        with stdin_path.open("x") as stream:
            stream.write(commands)
        repeated = []
        for repeat in range(2):
            stdout = raw_directory / f"{case.name}.{repeat}.stdout.csv"
            stderr = raw_directory / f"{case.name}.{repeat}.stderr"
            with stdout.open("x") as out, stderr.open("x") as err:
                process = subprocess.run(
                    [str(binary)],
                    input=commands,
                    text=True,
                    stdout=out,
                    stderr=err,
                    env=clean_environment(),
                )
            if process.returncode != 0:
                raise ValueError(
                    f"Case={case.name} exit={process.returncode}: read {stderr}"
                )
            repeated.append(stdout)
        if repeated[0].read_bytes() != repeated[1].read_bytes():
            raise ValueError(
                f"Case={case.name}: repeated serialized native output differs"
            )
        parsed = read_native(case, repeated[0].read_text())
        check_case(case, parsed)
        results[case.name] = parsed
        rows.extend(evidence_rows(case, parsed))
        evidence.append(
            {
                "case": case.name,
                "commands_hex": commands.splitlines(),
                "stdin_sha256": sha256(stdin_path),
                "stdout_sha256": [sha256(path) for path in repeated],
            }
        )
    check_pairs(results)
    if (
        source_identity(args.source_root, args.archive) != identity
        or adapter_hashes() != adapters
        or sha256(binary) != manifest["binary_sha256"]
        or sha256(manifest_path) != build_hash
        or sha256(compiler) != manifest["compiler_sha256"]
    ):
        raise ValueError(
            "Run inputs changed during execution: retain raw evidence and rerun with stable inputs"
        )
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "scope": "Standalone EPA native soil reference only; no hydr conformance claim",
        "source": identity,
        "build": manifest,
        "build_manifest_sha256": build_hash,
        "adapter_sha256": adapters,
        "python": sys.version,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "invocation": sys.argv,
        "cases": evidence,
        "case_count": len(evidence),
        "row_count": len(rows),
        "csv_sha256": sha256(output),
        "native_factors": manifest["native_factors"],
        "ZERO_ft_per_s": ZERO,
        "native_root_rules": {
            "short_ts_strictly_below_s": 10.0,
            "front_strictly_above_c1_fraction": 0.01,
            "newton_increment_strictly_below_ft": 1e-5,
            "maximum_iterations": 20,
        },
        "checks": [
            "Pinned archive and all original files",
            "Stable build and execution inputs",
            "Byte-identical repeated serialized calls",
            "Finite full-state outputs",
            "Frozen branch consequences",
        ],
    }
    with sidecar.open("x") as stream:
        stream.write(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(evidence)} native cases and {len(rows)} rows to {output}")
