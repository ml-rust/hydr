"""Check exact observer inputs and every smooth endpoint against the independent reference."""

import csv
import math

from subcatchment_temporal_cases import (
    HISTORY_NAMES,
    HISTORY_TIMESTEPS,
    HORIZON,
    INPUT_FIELDS,
    NAMES,
    OBSERVER_FIELDS,
    TIMESTEPS,
    VALUE_FIELDS,
    case,
    history_case,
    input_values,
    production_tolerance,
)
from subcatchment_temporal_native import read_histories


def read_observer(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != OBSERVER_FIELDS:
            raise ValueError(f"Observer={path}: complete CSV schema differs")
        rows = []
        for index, record in enumerate(reader, 2):
            if (
                index > 20000
                or None in record
                or any(value is None for value in record.values())
            ):
                raise ValueError(
                    f"Observer={path}:{index}: malformed or excessive records"
                )
            row = {"case": record["case"]}
            for field in OBSERVER_FIELDS[1:]:
                optional = field.startswith(("alpha", "rate")) or field in (
                    "soil_imd",
                    "soil_f_ft",
                    "soil_fu_ft",
                    "soil_saturated",
                    "soil_t_s",
                )
                if record[field] == "" and optional:
                    row[field] = None
                else:
                    value = float(record[field])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Observer={path}:{index} field={field}: nonfinite output"
                        )
                    row[field] = value
            if row["soil_saturated"] not in (None, 0.0, 1.0):
                raise ValueError(
                    f"Observer={path}:{index}: native saturation must be zero or one"
                )
            rows.append(row)
    return rows


def check_inputs(spec, row):
    differences = {}
    for field, expected in zip(INPUT_FIELDS, input_values(spec)):
        actual = row.get(field)
        if expected is None:
            if actual is not None:
                raise ValueError(
                    f"Case={spec['name']} field={field}: absent coefficient requires explicit missing support"
                )
            continue
        if actual is None or not math.isfinite(actual):
            raise ValueError(
                f"Case={spec['name']} field={field}: missing or nonfinite frozen input"
            )
        if field.startswith("alpha"):
            difference = abs(actual - expected)
            differences[field] = difference
            if difference > 8.0 * math.ulp(1.0) * abs(expected):
                raise ValueError(
                    f"Case={spec['name']} field={field}: coefficient differs from independently derived input"
                )
        elif actual.hex() != float(expected).hex():
            raise ValueError(
                f"Case={spec['name']} field={field}: observer input bits differ from the frozen recipe"
            )
    check_native_state(spec, row)
    return differences


def check_native_state(spec, row):
    fields = ("soil_imd", "soil_f_ft", "soil_fu_ft", "soil_saturated", "soil_t_s")
    values = [row.get(field) for field in fields]
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError(
            f"Case={spec['name']}: every endpoint requires complete finite native soil state"
        )
    imd, _, fu, saturation, _ = values
    maximum = row["soil_maximum_deficit"]
    conductivity = row["soil_conductivity_ft_per_s"]
    lu = 4.0 * math.sqrt(conductivity * 12.0 * 3600.0) / 12.0
    capacity = maximum * lu
    upper = max(maximum, capacity / lu)
    if (
        saturation not in (0.0, 1.0)
        or not 0.0 <= imd <= upper
        or not 0.0 <= fu <= capacity
    ):
        raise ValueError(
            f"Case={spec['name']}: native soil support exceeds its frozen parameters"
        )
    if row["step"] == 0.0:
        for field, actual, expected in zip(
            fields, values, (maximum, 0.0, 0.0, 0.0, 0.0)
        ):
            if actual.hex() != float(expected).hex():
                raise ValueError(
                    f"Case={spec['name']} field={field}: initial native state differs from the frozen recipe"
                )
        for index, expected in enumerate(spec["initial"]):
            if row[f"depth{index}_m"].hex() != expected.hex():
                raise ValueError(
                    f"Case={spec['name']}: initial surface state differs from the frozen recipe"
                )


def compare_smooth(rows, references):
    reports = []
    dynamic_fields = [
        field
        for field in VALUE_FIELDS
        if not field.startswith("rate")
        and field not in ("infiltration_m3", "evaporation_m3", "rain_m3")
    ]
    algebraic_fields = ("infiltration_m3", "evaporation_m3", "rain_m3")
    checked_fields = dynamic_fields + list(algebraic_fields)
    observed_groups = {
        (row["case"], row["dt_s"]) for row in rows if row["case"] in NAMES
    }
    required_groups = {(name, dt) for name in NAMES for dt in TIMESTEPS}
    if observed_groups != required_groups:
        raise ValueError(
            "Observer lacks the complete two-case, six-grid smooth refinement"
        )
    for name in NAMES:
        spec = case(name)
        reference = {row["time_s"]: row for row in references[name, "tighter"]}
        previous = math.inf
        for dt in TIMESTEPS:
            grid = [row for row in rows if row["case"] == name and row["dt_s"] == dt]
            required = int(HORIZON / dt) + 1
            if len(grid) != required:
                raise ValueError(
                    f"Case={name} dt={dt}: require all {required} endpoints"
                )
            maximum = 0.0
            component_maxima = {}
            coefficient_differences = {}
            for step, actual in enumerate(grid):
                time = step * dt
                if actual["step"] != step or actual["time_s"].hex() != time.hex():
                    raise ValueError(
                        f"Case={name} dt={dt}: endpoint identity or order differs"
                    )
                coefficient_differences = check_inputs(spec, actual)
                expected = reference[time]
                for field in checked_fields:
                    error = abs(actual[field] - expected[field])
                    tolerance = production_tolerance(spec, field, expected[field])
                    normalized = error / tolerance
                    component_maxima[field] = max(
                        component_maxima.get(field, 0.0), normalized
                    )
                    if field in dynamic_fields:
                        maximum = max(maximum, normalized)
                    if (
                        dt == TIMESTEPS[-1] or field in algebraic_fields
                    ) and normalized > 1.0:
                        raise ValueError(
                            f"Case={name} dt={dt} time={time} field={field}: error={error} exceeds frozen tolerance={tolerance}"
                        )
            if not maximum < previous:
                raise ValueError(
                    f"Case={name} dt={dt}: normalized dynamic error={maximum} does not decrease below {previous}"
                )
            previous = maximum
            reports.append(
                {
                    "case": name,
                    "dt_s": dt,
                    "endpoints": required,
                    "maximum_normalized_dynamic_error": maximum,
                    "component_maximum_normalized_errors": component_maxima,
                    "coefficient_differences": coefficient_differences,
                }
            )
    return reports


def native_departures(rows, native_path):
    histories = [row for row in rows if row["case"] not in NAMES]
    if not histories:
        return []
    if native_path is None:
        raise ValueError(
            "Historical observations require the unchanged native CSV for departure reporting"
        )
    native = read_histories(native_path)
    result = []
    for actual in histories:
        if actual["time_s"] != 256.0:
            continue
        native_case = f"{actual['case']}-dt{int(actual['dt_s'])}"
        matching = native[native_case]
        final = matching[-1]
        difference = {
            f"depth{k}_m": actual[f"depth{k}_m"] - float(final[f"a{k}_depth"]) * 0.3048
            for k in range(3)
        }
        for field, native_field in (
            ("infiltration_m3", "Vinfil"),
            ("evaporation_m3", "Vevap"),
            ("total_outlet_m3", "Voutflow"),
            ("total_routed_m3", "internal_routing_volume_ft3"),
        ):
            difference[field] = (
                actual[field]
                - math.fsum(float(row[native_field]) for row in matching) * 0.3048**3
            )
        for k in range(3):
            actual_rate = actual[f"rate{k}_m_per_s"]
            difference[f"rate{k}_m_per_s"] = (
                None
                if actual_rate is None
                else actual_rate - float(final[f"a{k}_runoff"]) * 0.3048
            )
        result.append(
            {
                "case": actual["case"],
                "dt_s": actual["dt_s"],
                "native_case": native_case,
                "differences": difference,
                "scope": "Method-departure observations, no coarse native identity gate",
            }
        )
    return result


def history_refinement(rows):
    names = HISTORY_NAMES
    steps = HISTORY_TIMESTEPS
    histories = [row for row in rows if row["case"] not in NAMES]
    if not histories:
        return []
    if {(row["case"], row["dt_s"]) for row in histories} != {
        (name, dt) for name in names for dt in steps
    }:
        raise ValueError("Historical observations require all twelve frozen grids")
    fields = [
        field
        for field in VALUE_FIELDS
        if field.startswith("depth")
        or field
        in (
            "total_storage_m3",
            "infiltration_m3",
            "evaporation_m3",
            "total_outlet_m3",
            "total_routed_m3",
        )
    ]
    result = []
    for name in names:
        previous = None
        for dt in steps:
            grid = [
                row for row in histories if row["case"] == name and row["dt_s"] == dt
            ]
            if len(grid) != int(256.0 / dt) + 1:
                raise ValueError(
                    f"Historical case={name} dt={dt}: missing endpoint observations"
                )
            for step, row in enumerate(grid):
                if row["step"] != step or row["time_s"] != step * dt:
                    raise ValueError(
                        f"Historical case={name} dt={dt}: endpoint order differs"
                    )
                check_inputs(history_case(name), row)
            final = grid[-1]
            physical = {field: final[field] for field in fields}
            difference = (
                None
                if previous is None
                else {field: physical[field] - previous[field] for field in fields}
            )
            result.append(
                {
                    "case": name,
                    "dt_s": dt,
                    "final": physical,
                    "adjacent_finer_minus_coarser": difference,
                    "scope": "Retained branch-crossing observations, no quantitative general-history accuracy gate",
                }
            )
            previous = physical
    return result
