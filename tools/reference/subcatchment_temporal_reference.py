"""Independent continuous routed equations and separately integrated physical transfers."""

import csv
import math

from subcatchment_temporal_cases import (
    CONTROLS,
    EXPONENT,
    HORIZON,
    NAMES,
    REFERENCE_FIELDS,
    TIMESTEPS,
    VALUE_FIELDS,
    VERSIONS,
    case,
    production_tolerance,
)


def dependencies():
    import numpy as np
    import scipy
    from scipy.integrate import solve_ivp

    actual = {"scipy": scipy.__version__, "numpy": np.__version__}
    if actual != VERSIONS:
        raise ValueError(f"Reference packages={actual}: require frozen {VERSIONS}")
    return np, solve_ivp, actual


def derivative(spec, np):
    area = np.asarray(spec["areas"], dtype=np.float64)
    depression = np.asarray(spec["depression"], dtype=np.float64)
    alpha = np.asarray(spec["alpha"], dtype=np.float64)

    def rhs(_time, state):
        mobile = state[:3] - depression
        if not np.all(np.isfinite(mobile)) or np.any(mobile <= 0.0):
            raise ValueError("Smooth reference leaves its positive mobile-depth regime")
        rate = alpha * mobile**EXPONENT
        gross = area * rate
        routed = np.zeros(3, dtype=np.float64)
        change = np.full(3, spec["rain"], dtype=np.float64) - rate
        change[2] -= spec["conductivity"]
        if spec["route_code"] == 2:
            routed[:2] = spec["fraction"] * gross[:2]
            change[2] += (routed[0] + routed[1]) / area[2]
        else:
            routed[2] = spec["fraction"] * gross[2]
            change[1] += routed[2] / area[1]
        outlet = gross - routed
        physical_sources = [
            area[2] * spec["conductivity"],
            0.0,
            spec["area"] * spec["rain"],
        ]
        return np.concatenate((change, gross, outlet, routed, physical_sources))

    return rhs


def row(spec, time, state):
    depth = list(state[:3])
    storage = [area * value for area, value in zip(spec["areas"], depth)]
    gross = list(state[3:6])
    outlet = list(state[6:9])
    routed = list(state[9:12])
    rate = [
        alpha * (value - depression) ** EXPONENT
        for alpha, value, depression in zip(spec["alpha"], depth, spec["depression"])
    ]
    values = [
        *depth,
        *storage,
        *gross,
        *outlet,
        *routed,
        *state[12:15],
        math.fsum(storage),
        math.fsum(gross),
        math.fsum(outlet),
        math.fsum(routed),
        *rate,
    ]
    return {
        "case": spec["name"],
        "time_s": float(time),
        **dict(zip(VALUE_FIELDS, map(float, values))),
    }


def check_reference_row(spec, record):
    if any(not math.isfinite(record[field]) for field in VALUE_FIELDS):
        raise ValueError(f"Case={spec['name']}: reference output is nonfinite")
    time = record["time_s"]
    if any(record[f"depth{k}_m"] <= spec["depression"][k] for k in range(3)):
        raise ValueError(
            f"Case={spec['name']} time={time}: reference leaves active storage"
        )
    expected = {
        "infiltration_m3": spec["areas"][2] * spec["conductivity"] * time,
        "evaporation_m3": 0.0,
        "rain_m3": spec["area"] * spec["rain"] * time,
    }
    for field, target in expected.items():
        if abs(record[field] - target) > 1e-9 + 1e-12 * abs(target):
            raise ValueError(
                f"Case={spec['name']} time={time}: independent reference {field} differs"
            )
    received = [0.0, 0.0, 0.0]
    if spec["route_code"] == 2:
        received[2] = record["routed0_m3"] + record["routed1_m3"]
    else:
        received[1] = record["routed2_m3"]
    for index in range(3):
        initial = spec["areas"][index] * spec["initial"][index]
        rain = spec["areas"][index] * spec["rain"] * time
        infiltration = record["infiltration_m3"] if index == 2 else 0.0
        residual = (record[f"storage{index}_m3"] - initial) - (
            rain - infiltration + received[index] - record[f"gross{index}_m3"]
        )
        scale = max(
            initial,
            record[f"storage{index}_m3"],
            rain,
            infiltration,
            received[index],
            record[f"gross{index}_m3"],
        )
        if abs(residual) > 1e-9 + 1e-12 * scale:
            raise ValueError(
                f"Case={spec['name']} time={time}: reference subarea{index} water residual={residual}"
            )
    initial = math.fsum(
        area * depth for area, depth in zip(spec["areas"], spec["initial"])
    )
    residual = (record["total_storage_m3"] - initial) - (
        record["rain_m3"] - record["infiltration_m3"] - record["total_outlet_m3"]
    )
    if abs(residual) > 1e-9 + 1e-12 * max(
        initial,
        record["total_storage_m3"],
        record["rain_m3"],
        record["total_outlet_m3"],
    ):
        raise ValueError(
            f"Case={spec['name']} time={time}: reference catchment residual={residual}"
        )


def integrate():
    np, solve_ivp, versions = dependencies()
    times = (
        np.arange(int(HORIZON / TIMESTEPS[-1]) + 1, dtype=np.float64) * TIMESTEPS[-1]
    )
    outputs = {}
    execution = []
    for name in NAMES:
        spec = case(name)
        initial = np.asarray(spec["initial"] + [0.0] * 12, dtype=np.float64)
        for label, control in CONTROLS.items():
            atol = np.asarray(
                [control["depth_atol"]] * 3 + [control["volume_atol"]] * 12
            )
            result = solve_ivp(
                derivative(spec, np),
                (0.0, HORIZON),
                initial,
                method="DOP853",
                t_eval=times,
                rtol=control["rtol"],
                atol=atol,
                max_step=1.0,
            )
            if (
                not result.success
                or result.y.shape != (15, len(times))
                or not np.all(np.isfinite(result.y))
            ):
                raise ValueError(
                    f"Case={name} reference={label}: DOP853 cannot supply the complete finite reference: {result.message}"
                )
            records = [
                row(spec, time, values) for time, values in zip(result.t, result.y.T)
            ]
            for record in records:
                check_reference_row(spec, record)
            outputs[name, label] = records
            execution.append(
                {
                    "case": name,
                    "reference": label,
                    "nfev": result.nfev,
                    "njev": result.njev,
                    "nlu": result.nlu,
                    "message": result.message,
                    "controls": control,
                }
            )
    return outputs, {"versions": versions, "execution": execution}


def check_reference_convergence(outputs):
    maximum = {}
    checked = [field for field in VALUE_FIELDS if not field.startswith("rate")]
    for name in NAMES:
        spec = case(name)
        for primary, tighter in zip(outputs[name, "primary"], outputs[name, "tighter"]):
            for field in checked:
                tolerance = production_tolerance(spec, field, tighter[field])
                ratio = abs(primary[field] - tighter[field]) / tolerance
                maximum[field] = max(maximum.get(field, 0.0), ratio)
                if ratio >= 0.01:
                    raise ValueError(
                        f"Case={name} time={primary['time_s']} field={field}: reference difference ratio={ratio} reaches one-hundredth of production tolerance"
                    )
    return maximum


def write_references(directory, outputs):
    for label in CONTROLS:
        with (directory / f"reference-{label}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REFERENCE_FIELDS)
            writer.writeheader()
            for name in NAMES:
                for record in outputs[name, label]:
                    writer.writerow(
                        {
                            key: value if key == "case" else format(value, ".17g")
                            for key, value in record.items()
                        }
                    )
