"""Strict observed soil-call records and complete observer-catalog support."""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

FIELDS = (
    "case,dt_s,step,start_s,end_s,suction_ft,conductivity_ft_per_s,maximum_deficit,"
    "old_imd,old_f_ft,old_fu_ft,old_saturated,old_t_s,ponded_depth_m,"
    "constructed_rate_m_per_s,input_depth_m,available_depth_m,native_rate_ft_per_s,"
    "native_available_depth_ft,native_infiltration_depth_ft,infiltration_depth_m,"
    "adjustment_depth_m,imd,f_ft,fu_ft,saturated,t_s"
).split(",")
OLD = ("old_imd", "old_f_ft", "old_fu_ft", "old_saturated", "old_t_s")
ACCEPTED = ("imd", "f_ft", "fu_ft", "saturated", "t_s")
PARAMETERS = ("suction_ft", "conductivity_ft_per_s", "maximum_deficit")
CONVERSION = 0.3048
HISTORY_NAMES = (
    "pervious_sources",
    "impervious_to_pervious",
    "pervious_to_impervious",
)
SMOOTH_NAMES = ("smooth_impervious_to_pervious", "smooth_pervious_to_impervious")


@dataclass(frozen=True)
class Row:
    case: str
    step: int
    values: dict

    def __getitem__(self, key):
        return self.values[key]

    def state(self, fields):
        return tuple(self[field] for field in fields)

    def native_input(self):
        return (
            self["dt_s"],
            self["constructed_rate_m_per_s"] / CONVERSION,
            self["ponded_depth_m"] / CONVERSION,
        )


@dataclass(frozen=True)
class Sequence:
    name: str
    rows: tuple

    @property
    def parameters(self):
        return self.rows[0].state(PARAMETERS)

    @property
    def dt(self):
        return self.rows[0]["dt_s"]


def observer_files():
    root = Path(__file__).resolve().parents[2]
    return [root / "examples/subcatchment_reference.rs"] + [
        root / "examples/subcatchment_reference" / name
        for name in ("cases.rs", "records.rs", "observe.rs")
    ]


def catalog(mode):
    names = (SMOOTH_NAMES if mode in ("smooth", "all") else ()) + (
        HISTORY_NAMES if mode in ("histories", "all") else ()
    )
    if not names:
        raise ValueError(f"Mode={mode!r}: require all, smooth, or histories")
    result = {}
    for name in names:
        smooth = name in SMOOTH_NAMES
        horizon = 64.0 if smooth else 256.0
        timesteps = (
            (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
            if smooth
            else (
                64.0,
                32.0,
                16.0,
                8.0,
            )
        )
        parameters = (
            0.125,
            1.0 / (2048.0 if smooth else 1024.0),
            0.0 if smooth else 1.0,
        )
        for dt in timesteps:
            result[(name, dt.hex())] = (int(horizon / dt), horizon, parameters)
    return result


def check_schema():
    records = observer_files()[2].read_text()
    match = re.search(r'pub const SOIL_HEADER: &str = "([^"]+)";', records)
    if match is None or match[1].split(",") != FIELDS:
        raise ValueError(
            "Observer SOIL_HEADER differs: reconcile the adapter schema before planning"
        )


def bits(values):
    return tuple(float(value).hex() for value in values)


def parse_row(raw, line):
    if None in raw or any(value is None or value == "" for value in raw.values()):
        raise ValueError(
            f"Sidecar line={line}: require exactly the complete nonempty schema"
        )
    name = raw["case"]
    if not name or any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError(
            f"Sidecar line={line}: require a nonempty case without control characters"
        )
    try:
        step = int(raw["step"])
        values = {key: float(raw[key]) for key in FIELDS if key not in ("case", "step")}
    except ValueError as error:
        raise ValueError(
            f"Sidecar line={line}: invalid integer or F64 field"
        ) from error
    if (
        step <= 0
        or str(step) != raw["step"]
        or not all(map(math.isfinite, values.values()))
    ):
        raise ValueError(
            f"Sidecar line={line}: require canonical positive step and finite F64 fields"
        )
    row = Row(name, step, values)
    if row["dt_s"] <= 0.0 or row["end_s"] - row["start_s"] != row["dt_s"]:
        raise ValueError(
            f"Sidecar line={line}: require a positive exact represented interval"
        )
    signed = {"old_f_ft", "old_t_s", "f_ft", "t_s"}
    if any(value < 0.0 for key, value in values.items() if key not in signed):
        raise ValueError(
            f"Sidecar line={line}: negative value outside signed native F/T"
        )
    if row["old_saturated"] not in (0.0, 1.0) or row["saturated"] not in (0.0, 1.0):
        raise ValueError(f"Sidecar line={line}: saturation must be exactly zero or one")
    if row["conductivity_ft_per_s"] <= 0.0 or row["maximum_deficit"] > 1.0:
        raise ValueError(
            f"Sidecar line={line}: native soil parameters violate admitted support"
        )
    check_algebra(row, line)
    return row


def check_algebra(row, line):
    dt, rate, ponding = row.native_input()
    native_available = (rate + ponding / dt) * dt
    native_depth = row["native_rate_ft_per_s"] * dt
    si_depth = (row["native_rate_ft_per_s"] * CONVERSION) * dt
    incoming = row["constructed_rate_m_per_s"] * dt
    available = row["ponded_depth_m"] + incoming
    adjustment = max(si_depth - available, 0.0)
    expected = {
        "input_depth_m": incoming,
        "available_depth_m": available,
        "native_available_depth_ft": native_available,
        "native_infiltration_depth_ft": native_depth,
        "infiltration_depth_m": si_depth,
        "adjustment_depth_m": adjustment,
    }
    for key, value in expected.items():
        if not math.isfinite(value) or row[key] != value:
            raise ValueError(
                f"Sidecar line={line} field={key}: source algebra differs from observed inputs"
            )
    if adjustment > 0.0 and available + adjustment != si_depth:
        raise ValueError(
            f"Sidecar line={line}: adjustment does not exactly reconcile native infiltration"
        )


def read_sequences(path, mode):
    check_schema()
    expected = catalog(mode)
    grouped = {}
    closed = set()
    previous_key = None
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(
                "Soil sidecar header differs: supply the complete observer soil CSV"
            )
        for line, raw in enumerate(reader, 2):
            row = parse_row(raw, line)
            key = (row.case, row["dt_s"].hex())
            if key not in expected:
                raise ValueError(
                    f"Sidecar line={line} sequence={key}: require a frozen case/dt block"
                )
            if key != previous_key:
                if key in closed:
                    raise ValueError(
                        f"Sidecar line={line}: sequence reopens after another case/dt block"
                    )
                if previous_key is not None:
                    closed.add(previous_key)
                previous_key = key
            rows = grouped.setdefault(key, [])
            if len(rows) >= expected[key][0]:
                raise ValueError(
                    f"Sidecar line={line} sequence={key}: row count exceeds frozen support"
                )
            rows.append(row)
    if grouped.keys() != expected.keys():
        raise ValueError(
            f"Sequence support differs: missing={sorted(expected.keys()-grouped.keys())}, extra={sorted(grouped.keys()-expected.keys())}"
        )
    result = []
    for key, rows in grouped.items():
        length, horizon, parameters = expected[key]
        if len(rows) != length or rows[-1]["end_s"] != horizon:
            raise ValueError(
                f"Sequence={key}: require all {length} rows through time={horizon}"
            )
        previous_state = (parameters[2], 0.0, 0.0, 0.0, 0.0)
        for step, row in enumerate(rows, 1):
            if (
                row.step != step
                or row["start_s"] != (step - 1) * row["dt_s"]
                or row["end_s"] != step * row["dt_s"]
                or bits(row.state(PARAMETERS)) != bits(parameters)
                or bits(row.state(OLD)) != bits(previous_state)
            ):
                raise ValueError(
                    f"Sequence={key} step={step}: timing, parameters, or exact antecedent link differs"
                )
            previous_state = row.state(ACCEPTED)
        result.append(Sequence(f"sequence-{len(result):04d}", tuple(rows)))
    return tuple(result)
