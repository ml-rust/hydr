"""Frozen smooth routed cases, numerical controls, and physical comparison scales."""

import math

FOOT = 0.3048
EXPONENT = 1.6666667
TIMESTEPS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
HORIZON = 64.0
VERSIONS = {"scipy": "1.16.3", "numpy": "2.4.0"}
CONTROLS = {
    "primary": {"rtol": 1e-11, "depth_atol": 1e-13, "volume_atol": 1e-11},
    "tighter": {"rtol": 1e-13, "depth_atol": 1e-15, "volume_atol": 1e-13},
}
NAMES = ("smooth_impervious_to_pervious", "smooth_pervious_to_impervious")
HISTORY_NAMES = ("pervious_sources", "impervious_to_pervious", "pervious_to_impervious")
HISTORY_TIMESTEPS = (64.0, 32.0, 16.0, 8.0)
INPUT_FIELDS = (
    "area_m2,width_m,slope,impervious_fraction,zero_storage_fraction,impervious_manning,"
    "pervious_manning,impervious_depression_m,pervious_depression_m,soil_suction_ft,"
    "soil_conductivity_ft_per_s,soil_maximum_deficit,rain_m_per_s,evaporation_m_per_s,"
    "route_code,route_fraction,initial_depth0_m,initial_depth1_m,initial_depth2_m,"
    "area0_m2,area1_m2,area2_m2,alpha0_si,alpha1_si,alpha2_si"
).split(",")
VALUE_FIELDS = (
    "depth0_m,depth1_m,depth2_m,storage0_m3,storage1_m3,storage2_m3,"
    "gross0_m3,gross1_m3,gross2_m3,outlet0_m3,outlet1_m3,outlet2_m3,"
    "routed0_m3,routed1_m3,routed2_m3,infiltration_m3,evaporation_m3,rain_m3,"
    "total_storage_m3,total_gross_m3,total_outlet_m3,total_routed_m3,"
    "rate0_m_per_s,rate1_m_per_s,rate2_m_per_s"
).split(",")
REFERENCE_FIELDS = ["case", "time_s"] + VALUE_FIELDS
AUXILIARY_FIELDS = (
    "soil_imd,soil_f_ft,soil_fu_ft,soil_saturated,soil_t_s,soil_adjustment_m3,"
    "soil_adjustment_charge_m3,transfer_uncertainty_m3,physical_scale_lower_m3,water_residual_m3"
).split(",")
OBSERVER_FIELDS = (
    ["case", "dt_s", "step", "time_s"] + INPUT_FIELDS + VALUE_FIELDS + AUXILIARY_FIELDS
)


def case(name):
    if name not in NAMES:
        raise ValueError(f"Case={name}: require one of {NAMES}")
    area = 1024.0 * (FOOT * FOOT)
    areas = [area * 0.5 * 0.5, area * 0.5 * (1.0 - 0.5), area * (1.0 - 0.5)]
    alpha_native = 1.49 * 32.0 / 512.0 * math.sqrt(1.0 / 64.0) / (1.0 / 32.0)
    alpha_si = alpha_native * FOOT ** (1.0 - EXPONENT)
    depths = [FOOT / 32.0, 5.0 * FOOT / 32.0, 5.0 * FOOT / 32.0]
    depression = [0.0, FOOT / 8.0, FOOT / 8.0]
    return {
        "name": name,
        "route_code": 2 if name == NAMES[0] else 1,
        "fraction": 0.5,
        "area": area,
        "areas": areas,
        "alpha": [alpha_si] * 3,
        "depression": depression,
        "initial": depths,
        "mobile": [depth - store for depth, store in zip(depths, depression)],
        "rain": FOOT / 1024.0,
        "conductivity": FOOT / 2048.0,
    }


def input_values(spec):
    if spec["name"] in HISTORY_NAMES:
        return history_input_values(spec)
    return [
        spec["area"],
        32.0 * FOOT,
        1.0 / 64.0,
        0.5,
        0.5,
        1.0 / 32.0,
        1.0 / 32.0,
        FOOT / 8.0,
        FOOT / 8.0,
        0.125,
        1.0 / 2048.0,
        0.0,
        spec["rain"],
        0.0,
        float(spec["route_code"]),
        spec["fraction"],
        *spec["initial"],
        *spec["areas"],
        *spec["alpha"],
    ]


def history_case(name):
    if name not in HISTORY_NAMES:
        raise ValueError(f"Historical case={name}: require one of {HISTORY_NAMES}")
    area = 1024.0 * (FOOT * FOOT)
    pervious_only = name == HISTORY_NAMES[0]
    impervious_fraction = 0.0 if pervious_only else 0.5
    impervious = area * impervious_fraction
    areas = [
        impervious * 0.5,
        impervious * (1.0 - 0.5),
        area * (1.0 - impervious_fraction),
    ]
    class_areas_ft2 = [1024.0 * impervious_fraction] * 2 + [
        1024.0 * (1.0 - impervious_fraction)
    ]
    alpha = [
        None
        if class_area == 0.0
        else (1.49 * 32.0 / class_area * math.sqrt(1.0 / 64.0) / (1.0 / 32.0))
        * FOOT ** (1.0 - EXPONENT)
        for class_area in class_areas_ft2
    ]
    depths_native = (
        [0.0, 0.0, 0.125]
        if pervious_only
        else [0.125, 0.25, 0.0]
        if name == HISTORY_NAMES[1]
        else [0.0, 0.125, 0.25]
    )
    return {
        "name": name,
        "area": area,
        "impervious_fraction": impervious_fraction,
        "areas": areas,
        "alpha": alpha,
        "initial": [depth * FOOT for depth in depths_native],
        "initial_native": depths_native,
        "rain_native": 1.0 / 512.0 if pervious_only else 1.0 / 1024.0,
        "evaporation_native": 1.0 / 8192.0 if pervious_only else 0.0,
        "route_code": 0 if pervious_only else 2 if name == HISTORY_NAMES[1] else 1,
        "fraction": 0.0 if pervious_only else 0.5,
    }


def history_input_values(spec):
    return [
        spec["area"],
        32.0 * FOOT,
        1.0 / 64.0,
        spec["impervious_fraction"],
        0.5,
        1.0 / 32.0,
        1.0 / 32.0,
        FOOT / 8.0,
        FOOT / 8.0,
        0.125,
        1.0 / 1024.0,
        1.0,
        spec["rain_native"] * FOOT,
        spec["evaporation_native"] * FOOT,
        float(spec["route_code"]),
        spec["fraction"],
        *spec["initial"],
        *spec["areas"],
        *spec["alpha"],
    ]


def production_tolerance(spec, field, reference):
    if field.startswith("depth"):
        index = int(field[5])
        return 1e-9 + 0.005 * spec["mobile"][index]
    if field in ("infiltration_m3", "evaporation_m3", "rain_m3"):
        return 1e-9 + 1e-12 * abs(reference)
    if field.startswith(("storage", "gross", "outlet", "routed")):
        index = int(next(character for character in field if character.isdigit()))
        mobile = spec["areas"][index] * spec["mobile"][index]
        return 1e-9 + 0.005 * max(abs(reference), mobile)
    if field.startswith("total_"):
        mobile = math.fsum(
            area * depth for area, depth in zip(spec["areas"], spec["mobile"])
        )
        return 1e-9 + 0.005 * max(abs(reference), mobile)
    raise ValueError(f"Field={field}: no frozen physical tolerance")


def frozen_specification():
    return {
        "scope": "Independent smooth finite-Manning routed reference and pre-execution physical gates",
        "cases": {name: case(name) for name in NAMES},
        "timesteps_s": TIMESTEPS,
        "horizon_s": HORIZON,
        "exponent": EXPONENT,
        "versions": VERSIONS,
        "controls": CONTROLS,
        "max_reference_step_s": 1.0,
        "self_check_fraction": 0.01,
        "depth_tolerance": "1e-9 m + 0.005*initial_mobile_depth",
        "volume_tolerance": "1e-9 m3 + 0.005*max(abs(reference), relevant_initial_mobile_volume)",
        "algebraic_volume_tolerance": "1e-9 m3 + 1e-12*abs(reference)",
        "refinement_gate": "Strictly decreasing maximum normalized dynamic error over every endpoint and quantity",
        "finest_gate": "Every finest-grid physical error is within its frozen tolerance",
        "reference_states": "Three depths and independently integrated gross exports, donor outlets, donor routed volumes, infiltration, evaporation, and rainfall",
        "coefficient_input_tolerance_relative": 8.0 * math.ulp(1.0),
        "limits": "No quantitative arbitrary-soil-history or field accuracy claim",
    }
