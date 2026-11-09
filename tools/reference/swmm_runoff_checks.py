"""Frozen analytic and algebraic gates; native transfer discrepancies remain observations."""

import math
from itertools import pairwise

from swmm_runoff_records import INIT_FIELDS, STATE_FIELDS, close

NONNEGATIVE_FIELDS = (
    "Vinflow",
    "Vevap",
    "Vpevap",
    "Vinfil",
    "Voutflow",
    "evapLoss",
    "infilLoss",
    "requested_rain_ft_s",
    "actual_rain_ft_s",
    "evap_ft_s",
    "external_runon_cfs",
    "rainfall",
    "oldRunoff",
    "newRunoff",
    "runon",
    "native_runon_ft_s",
    "runoff_return_ft_s",
    "storage_before_ft3",
    "storage_after_ft3",
    "rain_volume_ft3",
    "external_volume_ft3",
    "internal_routing_volume_ft3",
    "total_subarea_inflow_volume_ft3",
    "endpoint_all_volume_ft3",
    "endpoint_outlet_volume_ft3",
    "probe_rate_ft_s",
    "probe_depth_after_ft",
)


def recession(depth, storage, alpha, time):
    exponent = 1.6666667
    excess = (depth - storage) ** (1.0 - exponent)
    excess = (excess + (exponent - 1.0) * alpha * time) ** (1.0 / (1.0 - exponent))
    return storage + excess, alpha * excess**exponent


def check_case(case, outputs, measured):
    for row in outputs:
        for field, value in zip(INIT_FIELDS, case.parameters):
            if row[field].hex() != float(value).hex():
                raise ValueError(
                    f"Case={case.name} field={field}: stored original parameter differs from native input"
                )
        for field in NONNEGATIVE_FIELDS:
            if not math.isfinite(row[field]) or row[field] < 0.0:
                raise ValueError(
                    f"Case={case.name} field={field}: native loss/rate/volume must be finite and nonnegative"
                )
        lu, maximum = row["Lu"], row["IMDmax"]
        if (
            not math.isfinite(lu)
            or lu <= 0.0
            or not math.isfinite(maximum)
            or not 0.0 <= maximum <= 1.0
        ):
            raise ValueError(
                f"Case={case.name}: native Lu/IMDmax cannot define admitted soil bounds"
            )
        capacity = maximum * lu
        upper = max(maximum, capacity / lu)
        if (
            not math.isfinite(capacity)
            or not math.isfinite(upper)
            or not 0.0 <= row["IMD"] <= upper
            or not 0.0 <= row["Fu"] <= capacity
        ):
            raise ValueError(
                f"Case={case.name}: native soil IMD/Fu exceeds pinned recovery/capacity bounds"
            )
        if (
            not math.isfinite(row["F"])
            or not math.isfinite(row["T"])
            or row["Sat"] not in (0.0, 1.0)
        ):
            raise ValueError(
                f"Case={case.name}: require finite signed native F/T and independent zero/one Sat"
            )
        if row["native_error_code"] != 0.0 or any(
            row[f"ode_error{i}"] != 0.0 for i in range(3)
        ):
            raise ValueError(
                f"Case={case.name}: original native engine or ODE returns an error; retain raw evidence"
            )
        if row["a0_alpha"] != row["a1_alpha"]:
            raise ValueError(
                f"Case={case.name}: impervious subareas do not share combined-class alpha"
            )
        for i in range(3):
            if row[f"a{i}_depth"] < 0.0 or row[f"a{i}_runoff"] < 0.0:
                raise ValueError(
                    f"Case={case.name}: native depth or runoff is negative"
                )
    for previous, row in pairwise(outputs):
        if row["op"] in ("RATE", "ODE"):
            if any(row[field].hex() != previous[field].hex() for field in STATE_FIELDS):
                raise ValueError(
                    f"Case={case.name}: copied diagnostic probe changes accepted native context"
                )
    steps = [row for row in outputs if row["op"] == "STEP"]
    final = outputs[-1]
    if case.name.startswith("impervious_threshold") or case.name == "impervious_fill":
        expected = case.steps[0][0] * steps[0]["actual_rain_ft_s"]
        if final["a1_depth"] != expected or final["a1_runoff"] != 0.0:
            raise ValueError(
                f"Case={case.name}: exact storage-fill/endpoint threshold branch differs"
            )
    if case.name == "impervious_zero_slope":
        if final["a1_depth"] != 0.25 or final["a1_runoff"] != 0.0:
            raise ValueError(
                "impervious_zero_slope: original zero-alpha storage branch differs"
            )
    if case.name == "impervious_zero_roughness":
        if (
            final["a1_depth"] != 0.125
            or measured[-1]["independent_residual_outlet_ft3"] == 0.0
        ):
            raise ValueError(
                "impervious_zero_roughness: native storage reset/endpoint-volume discrepancy was not observed"
            )
    if case.name == "impervious_equilibrium":
        for row in steps:
            close(case.name, "equilibrium depth", row["a1_depth"], 0.25, 1e-10, 0.0)
            close(
                case.name,
                "equilibrium runoff",
                row["a1_runoff"],
                0.005820312096566695,
                1e-12,
                0.0,
            )
    if (
        case.name.startswith("impervious_recession_")
        or case.name == "impervious_variable_dt"
    ):
        for row in steps:
            depth, rate = recession(0.25, 0.125, row["a1_alpha"], row["time_s"])
            close(
                case.name,
                "analytic recession depth",
                row["a1_depth"],
                depth,
                1e-7,
                0.002,
            )
            close(
                case.name,
                "analytic recession runoff",
                row["a1_runoff"],
                rate,
                1e-10,
                0.005,
            )
    if case.name.startswith("ode_recession_"):
        depth, _ = recession(0.25, 0.125, final["a1_alpha"], final["probe_time_s"])
        close(
            case.name,
            "direct original ODE depth",
            final["probe_depth_after_ft"],
            depth,
            1e-7,
            0.002,
        )
        if final["probe_kind"] != 2.0 or final["ode_calls1"] != 1.0:
            raise ValueError(
                f"Case={case.name}: original ODE diagnostic was not executed"
            )
    if case.name.startswith("runoff_zero_threshold_"):
        if (final["probe_rate_ft_s"] > 0.0) != case.name.endswith("above") or final[
            "probe_kind"
        ] != 1.0:
            raise ValueError(
                f"Case={case.name}: strict original endpoint ZERO branch differs"
            )
    if case.name == "pervious_competing_losses":
        if final["a2_depth"] != 0.0 or measured[-1]["claimed_loss_excess_ft3"] <= 0.0:
            raise ValueError(
                "pervious_competing_losses: original competing evaporation/infiltration discrepancy was not observed"
            )
        if final["Vevap"] != 128.0 or final["Vinfil"] != 128.0:
            raise ValueError(
                "pervious_competing_losses: binary native loss claims differ from frozen inputs"
            )
    for row in steps:
        expected = [0.0, 0.0, 0.0]
        if row["route_code"] == 2.0:
            gross = (
                row["a0_old_runoff"] * row["a0_fArea"]
                + row["a1_old_runoff"] * row["a1_fArea"]
            )
            expected[2] = gross * (1.0 - row["a0_fOutlet"]) / row["a2_fArea"]
        elif row["route_code"] == 1.0:
            expected[1] = (
                row["a2_old_runoff"]
                * (1.0 - row["a2_fOutlet"])
                * row["a2_fArea"]
                / row["a1_fArea"]
            )
        for i in range(3):
            close(
                case.name,
                f"previous-endpoint routing inflow{i}",
                row[f"a{i}_routed_inflow"],
                expected[i],
                1e-12,
                1e-12,
            )
            if row[f"a{i}_fArea"] == 0.0 and row[f"a{i}_runoff"] != 0.0:
                raise ValueError(
                    f"Case={case.name}: zero-area subarea contributes runoff"
                )
    return {
        "step_count": len(steps),
        "elapsed_time_s": final["time_s"],
        "maximum_absolute_outlet_transfer_discrepancy_ft3": max(
            (abs(r.get("independent_residual_outlet_ft3", 0.0)) for r in measured),
            default=0.0,
        ),
        "maximum_claimed_loss_excess_ft3": max(
            (r.get("claimed_loss_excess_ft3", 0.0) for r in measured), default=0.0
        ),
        "total_native_evaporation_ft3": math.fsum(r["Vevap"] for r in steps),
        "total_native_infiltration_ft3": math.fsum(r["Vinfil"] for r in steps),
        "total_endpoint_outlet_volume_ft3": math.fsum(
            r["endpoint_outlet_volume_ft3"] for r in steps
        ),
    }
