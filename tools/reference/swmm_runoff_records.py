"""Strict native runoff schema and independent storage/transfer bookkeeping."""

import csv
import io
import math

NATIVE_FIELDS = (
    "op,index,time_s,dt_s,requested_rain_ft_s,actual_rain_ft_s,evap_ft_s,external_runon_cfs,"
    "area_ft2,frac_imperv,frac_zero_imperv,width_ft,slope,n_imperv,n_perv,"
    "dstore_imperv_ft,dstore_perv_ft,route_code,routed_fraction,evap_dry_only,"
    "S,Ks,IMDmax,Lu,IMD,F,Fu,Sat,T,oldRunoff,newRunoff,runon,rainfall,evapLoss,infilLoss"
).split(",")
for i in range(3):
    NATIVE_FIELDS.extend(
        f"a{i}_{field}"
        for field in (
            "fArea",
            "N",
            "dStore",
            "alpha",
            "routeTo",
            "fOutlet",
            "depth",
            "inflow",
            "runoff",
            "old_depth",
            "old_inflow",
            "old_runoff",
            "routed_inflow",
        )
    )
NATIVE_FIELDS.extend(
    (
        "Vinflow,Vevap,Vpevap,Vinfil,Voutflow,storage_before_ft3,storage_after_ft3,rain_volume_ft3,"
        "external_volume_ft3,internal_routing_volume_ft3,total_subarea_inflow_volume_ft3,native_runon_ft_s,endpoint_all_volume_ft3,endpoint_outlet_volume_ft3,"
        "storage_implied_outflow_ft3,outlet_storage_implied_outflow_ft3,storage_residual_all_ft3,storage_residual_outlet_ft3,internal_routing_lag_residual_ft3,runoff_return_ft_s,"
        "ode_calls0,ode_error0,ode_calls1,ode_error1,ode_calls2,ode_error2,native_error_code,"
        "probe_kind,probe_subarea,probe_time_s,probe_rate_ft_s,probe_depth_after_ft,probe_tolerance,probe_initial_step_s"
    ).split(",")
)
STATE_FIELDS = ["time_s"] + [
    f"a{i}_{field}" for i in range(3) for field in ("depth", "inflow", "runoff")
]
STATE_FIELDS.extend(
    (
        "oldRunoff",
        "newRunoff",
        "runon",
        "rainfall",
        "evapLoss",
        "infilLoss",
        "IMD",
        "F",
        "Fu",
        "Sat",
        "T",
    )
)
INIT_FIELDS = (
    "area_ft2",
    "frac_imperv",
    "frac_zero_imperv",
    "width_ft",
    "slope",
    "n_imperv",
    "n_perv",
    "dstore_imperv_ft",
    "dstore_perv_ft",
    "route_code",
    "routed_fraction",
    "S",
    "Ks",
    "IMDmax",
    "evap_dry_only",
)
INDEPENDENT_FIELDS = [
    "independent_storage_before_ft3",
    "independent_storage_after_ft3",
    "independent_rain_ft3",
    "independent_external_runon_ft3",
    "independent_internal_runon_ft3",
    "independent_endpoint_all_ft3",
    "independent_endpoint_outlet_ft3",
    "independent_storage_implied_all_ft3",
    "independent_storage_implied_outlet_ft3",
    "independent_residual_all_ft3",
    "independent_residual_outlet_ft3",
    "claimed_loss_excess_ft3",
]
CSV_FIELDS = (
    ["case", "command_hex"]
    + NATIVE_FIELDS
    + ["before_" + field for field in STATE_FIELDS]
    + INDEPENDENT_FIELDS
)
COMMAND_ARGUMENT_COUNTS = {"INIT": 15, "STATE": 21, "STEP": 4, "RATE": 2, "ODE": 4}


def command_stream(calls):
    return "".join(
        operation + " " + " ".join(float(value).hex() for value in values) + "\n"
        for operation, values in calls
    )


def read_native(name, calls, text):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != NATIVE_FIELDS:
        raise ValueError(
            f"Case={name}: native CSV header differs from the complete runoff schema"
        )
    records = list(reader)
    if len(records) != len(calls):
        raise ValueError(f"Case={name}: native output count differs from command count")
    result = []
    clock = None
    for index, (record, (operation, arguments)) in enumerate(zip(records, calls), 1):
        if (
            None in record
            or any(value is None for value in record.values())
            or record["op"] != operation
            or record["index"] != str(index)
        ):
            raise ValueError(
                f"Case={name} index={index}: native command identity differs"
            )
        row = {"op": operation, "index": index}
        for key in NATIVE_FIELDS[2:]:
            value = float(record[key])
            if not math.isfinite(value):
                raise ValueError(f"Case={name} field={key}: native value is nonfinite")
            row[key] = value
        if row["Sat"] not in (0.0, 1.0):
            raise ValueError(f"Case={name}: native Sat differs from zero/one")
        clock = check_command(name, row, operation, arguments, clock)
        result.append(row)
    return result


def same_input(name, row, field, expected, integer=False):
    actual, expected = row[field], float(expected)
    same = actual == expected if integer else actual.hex() == expected.hex()
    if not same:
        raise ValueError(
            f"Case={name} field={field}: emitted={actual.hex()} differs from command={expected.hex()}"
        )


def check_command(name, row, operation, arguments, clock):
    """Bind observations to submitted inputs, without replacing native numerical evolution."""
    if (
        operation not in COMMAND_ARGUMENT_COUNTS
        or len(arguments) != COMMAND_ARGUMENT_COUNTS[operation]
    ):
        raise ValueError(
            f"Case={name}: unsupported command or argument shape={operation}/{len(arguments)}"
        )
    if operation != "INIT" and clock is None:
        raise ValueError(f"Case={name}: command={operation} requires INIT")
    if operation == "INIT":
        for field, value in zip(INIT_FIELDS, arguments):
            same_input(
                name, row, field, value, field in ("route_code", "evap_dry_only")
            )
        clock = 0.0
        initial = (0.0,) * 16 + (float(arguments[13]), 0.0, 0.0, 0.0, 0.0)
        for field, value in zip(STATE_FIELDS, initial):
            same_input(name, row, field, value, field == "Sat")
    elif operation == "STATE":
        for field, value in zip(STATE_FIELDS, arguments):
            same_input(name, row, field, value, field == "Sat")
        clock = float(arguments[0])
    elif operation == "STEP":
        for field, value in zip(
            ("dt_s", "requested_rain_ft_s", "evap_ft_s", "external_runon_cfs"),
            arguments,
        ):
            same_input(name, row, field, value)
        clock += float(arguments[0])
        # US UCF(RAINFALL)=43200. IgnoreSnowmelt=TRUE removes snowFactor/temperature adjustments.
        # Original getNetPrecip adds its zero snowfall to the rain-gage division result.
        actual_rain = (float(arguments[1]) * 43200.0) / 43200.0 + 0.0
        same_input(name, row, "actual_rain_ft_s", actual_rain)
        same_input(name, row, "rainfall", actual_rain)
    if operation != "STEP":
        for field in (
            "dt_s",
            "requested_rain_ft_s",
            "actual_rain_ft_s",
            "evap_ft_s",
            "external_runon_cfs",
        ):
            same_input(name, row, field, 0.0)
    same_input(name, row, "time_s", clock)
    if operation in ("RATE", "ODE"):
        same_input(name, row, "probe_kind", 1.0 if operation == "RATE" else 2.0, True)
        same_input(name, row, "probe_subarea", arguments[0], True)
        same_input(name, row, "probe_time_s", arguments[1])
        same_input(
            name, row, "probe_tolerance", arguments[2] if operation == "ODE" else 0.0
        )
        same_input(
            name,
            row,
            "probe_initial_step_s",
            arguments[3] if operation == "ODE" else 0.0,
        )
    else:
        same_input(name, row, "probe_subarea", -1.0, True)
        for field in (
            "probe_kind",
            "probe_time_s",
            "probe_rate_ft_s",
            "probe_depth_after_ft",
            "probe_tolerance",
            "probe_initial_step_s",
        ):
            same_input(name, row, field, 0.0)
    return clock


def close(name, field, actual, expected, absolute=1e-10, relative=1e-12):
    tolerance = absolute + relative * max(abs(actual), abs(expected))
    if (
        not math.isfinite(actual)
        or not math.isfinite(expected)
        or abs(actual - expected) > tolerance
    ):
        raise ValueError(
            f"Case={name} field={field} actual={actual:.17g} expected={expected:.17g} tolerance={tolerance:.17g}"
        )


def observations(case, outputs):
    result = []
    calls = case.calls()
    for index, (call, row) in enumerate(zip(calls, outputs)):
        operation, _ = call
        combined = {
            "case": case.name,
            "command_hex": command_stream([call]).strip(),
            **row,
        }
        if index:
            combined.update(
                {"before_" + field: outputs[index - 1][field] for field in STATE_FIELDS}
            )
        if operation == "STEP":
            area, dt = row["area_ft2"], row["dt_s"]
            areas = [area * row[f"a{i}_fArea"] for i in range(3)]
            before = math.fsum(areas[i] * row[f"a{i}_old_depth"] for i in range(3))
            after = math.fsum(areas[i] * row[f"a{i}_depth"] for i in range(3))
            rain = row["actual_rain_ft_s"] * dt * area
            external = row["external_runon_cfs"] * dt
            delivered = math.fsum(
                row[f"a{i}_routed_inflow"] * areas[i] * dt for i in range(3)
            )
            internal = delivered - external
            endpoint_all = math.fsum(
                row[f"a{i}_runoff"] * areas[i] * dt for i in range(3)
            )
            endpoint_outlet = math.fsum(
                row[f"a{i}_fOutlet"] * row[f"a{i}_runoff"] * areas[i] * dt
                for i in range(3)
            )
            implied_outlet = (
                before + rain + external - row["Vevap"] - row["Vinfil"] - after
            )
            implied_all = implied_outlet + internal
            combined.update(
                zip(
                    INDEPENDENT_FIELDS,
                    (
                        before,
                        after,
                        rain,
                        external,
                        internal,
                        endpoint_all,
                        endpoint_outlet,
                        implied_all,
                        implied_outlet,
                        endpoint_all - implied_all,
                        endpoint_outlet - implied_outlet,
                        max(
                            0.0,
                            row["Vevap"] + row["Vinfil"] - (before + rain + delivered),
                        ),
                    ),
                )
            )
            for actual, expected in (
                ("storage_before_ft3", before),
                ("storage_after_ft3", after),
                ("rain_volume_ft3", rain),
                ("external_volume_ft3", external),
                ("internal_routing_volume_ft3", internal),
                ("endpoint_all_volume_ft3", endpoint_all),
                ("total_subarea_inflow_volume_ft3", delivered),
                ("native_runon_ft_s", row["external_runon_cfs"] / area),
                ("endpoint_outlet_volume_ft3", endpoint_outlet),
                ("storage_implied_outflow_ft3", implied_all),
                ("outlet_storage_implied_outflow_ft3", implied_outlet),
                ("storage_residual_all_ft3", endpoint_all - implied_all),
                ("storage_residual_outlet_ft3", endpoint_outlet - implied_outlet),
                (
                    "internal_routing_lag_residual_ft3",
                    internal - (endpoint_all - endpoint_outlet),
                ),
            ):
                close(case.name, actual, row[actual], expected)
        result.append(combined)
    return result


def state_values(row):
    return tuple(row[field] for field in STATE_FIELDS)
