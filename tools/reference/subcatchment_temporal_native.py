"""Bind complete processed native histories to their frozen original command streams."""

import csv
import io
import math
from itertools import islice

from subcatchment_temporal_cases import HISTORY_NAMES, HISTORY_TIMESTEPS, history_case
from swmm_runoff_cases import Case, antecedent
from swmm_runoff_checks import check_case
from swmm_runoff_records import (
    CSV_FIELDS,
    NATIVE_FIELDS,
    command_stream,
    observations,
    read_native,
)


def native_case(name, dt):
    spec = history_case(name)
    parameters = (
        1024.0,
        spec["impervious_fraction"],
        0.5,
        32.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 32.0,
        0.125,
        0.125,
        float(spec["route_code"]),
        spec["fraction"],
        0.125,
        1.0 / 1024.0,
        1.0,
        0.0,
    )
    return Case(
        f"{name}-dt{int(dt)}",
        parameters=parameters,
        state=antecedent(spec["initial_native"]),
        steps=((dt, spec["rain_native"], spec["evaporation_native"], 0.0),)
        * int(256.0 / dt),
    )


def check_processed(record, expected, name):
    for field in CSV_FIELDS:
        target = expected.get(field, "")
        actual = record[field]
        if isinstance(target, str):
            if actual != target:
                raise ValueError(
                    f"Native case={name} field={field}: command or optional support differs"
                )
        else:
            value = float(actual)
            if not math.isfinite(value) or value.hex() != float(target).hex():
                raise ValueError(
                    f"Native case={name} field={field}: processed observation differs from strict native records"
                )


def read_histories(path):
    cases = [
        native_case(name, dt) for name in HISTORY_NAMES for dt in HISTORY_TIMESTEPS
    ]
    expected_count = sum(len(case.calls()) for case in cases)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                "Native history CSV differs from the complete recorded schema"
            )
        records = list(islice(reader, expected_count + 1))
    if len(records) != expected_count:
        raise ValueError(
            f"Native histories require exactly {expected_count} complete command observations"
        )
    checked = {}
    offset = 0
    for case in cases:
        calls = case.calls()
        group = records[offset : offset + len(calls)]
        offset += len(calls)
        if any(
            None in record
            or any(value is None for value in record.values())
            or record["case"] != case.name
            for record in group
        ):
            raise ValueError(
                f"Native case={case.name}: malformed, missing, duplicate, or unordered observations"
            )
        raw = io.StringIO()
        writer = csv.DictWriter(raw, fieldnames=NATIVE_FIELDS)
        writer.writeheader()
        for record in group:
            writer.writerow({field: record[field] for field in NATIVE_FIELDS})
        parsed = read_native(case.name, calls, raw.getvalue())
        measured = observations(case, parsed)
        check_case(case, parsed, measured)
        for record, expected, call in zip(group, measured, calls):
            if record["command_hex"] != command_stream([call]).strip():
                raise ValueError(
                    f"Native case={case.name}: command receipt differs from frozen inputs"
                )
            check_processed(record, expected, case.name)
        checked[case.name] = [row for row in parsed if row["op"] == "STEP"]
    return checked
