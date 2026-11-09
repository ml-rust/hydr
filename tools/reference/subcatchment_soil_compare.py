"""Frozen standalone soil tolerances applied to actual constructed subcatchment inputs."""

import math

CONTROLS = {
    "relative": 1e-11,
    "absolute": {
        "rate": 1e-14,
        "interval_depth": 1e-12,
        "IMD": 1e-12,
        "F": 1e-12,
        "Fu": 1e-12,
        "T": 1e-9,
    },
    "rate_zero_nonzero_exact": True,
    "saturation_exact": True,
    "signed_F_T": "Retain original values and bits without nonnegative admission or clamping",
    "native_model": 3,
    "native_repetitions": 2,
    "native_timeout_s": 60,
    "native_restart": "Native emitted state after floor(sequence_steps/2), then identical suffix inputs",
    "input_bridge": "constructed_rate_m_per_s/0.3048 and original ponded_depth_m/0.3048 separately",
}
FIELDS = [
    "case",
    "dt_s",
    "step",
    "mode",
    "start_s",
    "end_s",
    "incoming_rate_ft_per_s",
    "ponded_depth_ft",
]
for metric in CONTROLS["absolute"]:
    FIELDS.extend(
        (
            "observed_" + metric,
            "native_" + metric,
            "absolute_difference_" + metric,
            "tolerance_" + metric,
        )
    )
FIELDS.extend(("observed_Sat", "native_Sat", "passed", "errors"))


def compare(row, native, mode):
    result = {
        "case": row.case,
        "dt_s": row["dt_s"],
        "step": row.step,
        "mode": mode,
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "incoming_rate_ft_per_s": row.native_input()[1],
        "ponded_depth_ft": row.native_input()[2],
        "observed_Sat": row["saturated"],
        "native_Sat": native["Sat"],
    }
    observed = {
        "rate": row["native_rate_ft_per_s"],
        "interval_depth": row["native_infiltration_depth_ft"],
        "IMD": row["imd"],
        "F": row["f_ft"],
        "Fu": row["fu_ft"],
        "T": row["t_s"],
    }
    values = {"rate": native["rate"], "interval_depth": native["rate"] * row["dt_s"]}
    values.update({key: native[key] for key in ("IMD", "F", "Fu", "T")})
    errors = []
    for metric, actual in observed.items():
        expected = values[metric]
        tolerance = CONTROLS["absolute"][metric] + CONTROLS["relative"] * max(
            abs(actual), abs(expected)
        )
        difference = abs(actual - expected)
        result.update(
            {
                "observed_" + metric: actual,
                "native_" + metric: expected,
                "absolute_difference_" + metric: difference,
                "tolerance_" + metric: tolerance,
            }
        )
        if not math.isfinite(expected) or difference > tolerance:
            errors.append(
                f"{metric}: observed={actual:.17e}, native={expected:.17e}, tolerance={tolerance:.17e}"
            )
    if (observed["rate"] == 0.0) != (values["rate"] == 0.0):
        errors.append("zero/nonzero infiltration-rate branch differs")
    if values["rate"] < 0.0:
        errors.append("native infiltration rate is negative")
    if row["saturated"] != native["Sat"]:
        errors.append("native saturation flag differs")
    result["passed"] = not errors
    result["errors"] = " | ".join(errors)
    return result
