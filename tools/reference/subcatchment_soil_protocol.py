"""Unchanged EPA command protocol with repeated serialized sequence evidence."""

import hashlib
import subprocess
from dataclasses import dataclass

from subcatchment_soil_records import OLD, bits
from swmm_soil_cases import Case
from swmm_soil_run import STATE_FIELDS, command_stream, read_native
from swmm_soil_sources import clean_environment, sha256


@dataclass(frozen=True)
class Calls:
    name: str
    commands: tuple

    def calls(self):
        return self.commands


def full_case(sequence):
    return Case(
        sequence.name,
        parameters=sequence.parameters,
        state=sequence.rows[0].state(OLD),
        steps=tuple(row.native_input() for row in sequence.rows),
    )


def isolated_case(sequence):
    commands = []
    for row in sequence.rows:
        case = Case(
            sequence.name,
            parameters=sequence.parameters,
            state=row.state(OLD),
            steps=(row.native_input(),),
        )
        commands.extend(case.calls())
    return Calls(sequence.name + "-isolated", tuple(commands))


def restart_case(sequence, full_outputs):
    midpoint = len(sequence.rows) // 2
    state = tuple(full_outputs[midpoint][key] for key in STATE_FIELDS)
    return midpoint, Case(
        sequence.name + "-restart",
        parameters=sequence.parameters,
        state=state,
        steps=tuple(row.native_input() for row in sequence.rows[midpoint:]),
    )


def stream_hash(case):
    return hashlib.sha256(command_stream(case).encode("ascii")).hexdigest()


def execute(binary, raw, label, case, evidence):
    commands = command_stream(case)
    stdin_path = raw / f"{label}.stdin"
    with stdin_path.open("x", encoding="ascii", newline="") as stream:
        stream.write(commands)
    outputs = []
    errors = []
    invocation = [str(binary)]
    record = {
        "label": label,
        "invocation": invocation,
        "command_count": len(case.calls()),
        "stdin": str(stdin_path),
        "stdin_sha256": sha256(stdin_path),
        "repetitions": [],
    }
    evidence.append(record)
    for repeat in range(2):
        stdout = raw / f"{label}.{repeat}.stdout.csv"
        stderr = raw / f"{label}.{repeat}.stderr"
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.run(
                invocation,
                input=commands.encode("ascii"),
                stdout=out,
                stderr=err,
                env=clean_environment(),
                timeout=60,
                check=False,
            )
        record["repetitions"].append(
            {
                "stdout": str(stdout),
                "stdout_sha256": sha256(stdout),
                "stderr": str(stderr),
                "stderr_sha256": sha256(stderr),
                "exit_code": process.returncode,
            }
        )
        if process.returncode != 0:
            raise ValueError(
                f"Native stream={label} exit={process.returncode}: read {stderr}"
            )
        outputs.append(stdout)
        errors.append(stderr)
    first_output = outputs[0].read_bytes()
    if (
        first_output != outputs[1].read_bytes()
        or errors[0].read_bytes() != errors[1].read_bytes()
    ):
        raise ValueError(
            f"Native stream={label}: repeated stdout/stderr bytes differ, retain raw evidence"
        )
    return read_native(case, first_output.decode("ascii"))


def check_initialization(output, parameters, state, label):
    actual_parameters = tuple(output[key] for key in ("S", "Ks", "IMDmax"))
    actual_state = tuple(output[key] for key in STATE_FIELDS)
    if bits(actual_parameters) != bits(parameters) or bits(actual_state) != bits(state):
        raise ValueError(
            f"Native stream={label}: STATE changes supplied parameter/state bits"
        )


def check_restart(full, restarted, midpoint):
    state = tuple(full[midpoint][key] for key in STATE_FIELDS)
    parameters = tuple(full[midpoint][key] for key in ("S", "Ks", "IMDmax"))
    check_initialization(restarted[0], parameters, state, "midpoint restart")
    suffix = full[midpoint + 1 :]
    if len(suffix) != len(restarted) - 1:
        raise ValueError("Native midpoint restart suffix length differs")
    for index, (expected, actual) in enumerate(
        zip(suffix, restarted[1:]), midpoint + 1
    ):
        for key in expected:
            left, right = expected[key], actual[key]
            if (left is None) != (right is None) or (
                left is not None and left.hex() != right.hex()
            ):
                raise ValueError(
                    f"Native midpoint restart step={index} field={key}: numerical suffix bits differ"
                )
