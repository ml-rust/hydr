"""Frozen native runoff inputs. Checks expose original transfer discrepancies."""

import math
from dataclasses import dataclass

# INIT: A, fracImperv, fracZeroImperv, W, slope, nImp, nPerv, dStoreImp,
# dStorePerv, route, routedFraction, S, Ks, IMDmax, evapDryOnly.
DEFAULTS = (
    1024.0,
    1.0,
    0.0,
    32.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 32.0,
    0.125,
    0.125,
    0,
    0.0,
    0.125,
    1.0 / 1024.0,
    1.0,
    0,
)
RAIN = 1.0 / 1024.0
UNSUPPORTED = (
    "Snow, groundwater, LID, pollutants, seasonal patterns, and inter-subcatchment graphs",
    "Hydraulic feedback and automatic wet/dry runoff-driver timestep selection",
    "Full SWMM driver rollback or restart beyond the emitted restricted-component witness",
    "Nonpositive total area and unsupported parameter/state shapes",
    "Conservative attachment or arithmetic-adjustment treatment of endpoint/loss discrepancies",
)


@dataclass(frozen=True)
class Case:
    name: str
    parameters: tuple = DEFAULTS
    state: tuple | None = None
    steps: tuple = ()
    rate: tuple | None = None
    ode: tuple | None = None

    def calls(self):
        result = [("INIT", self.parameters)]
        if self.state is not None:
            result.append(("STATE", self.state))
        result.extend(("STEP", step) for step in self.steps)
        if self.rate is not None:
            result.append(("RATE", self.rate))
        if self.ode is not None:
            result.append(("ODE", self.ode))
        return result


def antecedent(depths, soil=(1.0, 0.0, 0.0, 0, 0.0)):
    # Complete STATE: time, three(depth,inflow,runoff), six catchment fields, soil.
    return (
        (0.0,)
        + tuple(value for depth in depths for value in (depth, 0.0, 0.0))
        + (0.0,) * 6
        + soil
    )


def forcing(dt, rain=0.0, evaporation=0.0):
    return (dt, rain, evaporation, 0.0)


def parameters(**changes):
    names = (
        "area",
        "impervious",
        "zero_impervious",
        "width",
        "slope",
        "n_impervious",
        "n_pervious",
        "storage_impervious",
        "storage_pervious",
        "route",
        "routed_fraction",
        "suction",
        "conductivity",
        "deficit",
        "evap_dry_only",
    )
    values = dict(zip(names, DEFAULTS))
    unknown = changes.keys() - values.keys()
    if unknown:
        raise ValueError(f"Unknown native runoff parameters: {sorted(unknown)}")
    values.update(changes)
    return tuple(values[name] for name in names)


def cases():
    result = [
        Case("impervious_fill", steps=(forcing(64.0, RAIN),)),
        Case("impervious_threshold", steps=(forcing(128.0, RAIN),)),
        Case(
            "impervious_threshold_below",
            steps=(forcing(math.nextafter(128.0, -math.inf), RAIN),),
        ),
        Case(
            "impervious_threshold_above",
            steps=(forcing(math.nextafter(128.0, math.inf), RAIN),),
        ),
    ]
    for dt in (64, 32, 16, 8):
        result.append(
            Case(
                f"impervious_overflow_dt{dt}",
                steps=(forcing(float(dt), RAIN),) * (256 // dt),
            )
        )
        result.append(
            Case(
                f"impervious_recession_dt{dt}",
                state=antecedent((0.0, 0.25, 0.0)),
                steps=(forcing(float(dt)),) * (256 // dt),
            )
        )
    result.extend(
        (
            Case(
                "impervious_equilibrium",
                state=antecedent((0.0, 0.25, 0.0)),
                steps=(forcing(64.0, 0.005820312096566695),) * 4,
            ),
            Case(
                "impervious_zero_roughness",
                parameters=parameters(n_impervious=0.0),
                steps=(forcing(256.0, RAIN),),
            ),
            Case(
                "impervious_zero_slope",
                parameters=parameters(slope=0.0),
                steps=(forcing(256.0, RAIN),),
            ),
            Case(
                "impervious_split_area",
                parameters=parameters(zero_impervious=0.25),
                state=antecedent((0.125, 0.25, 0.0)),
                steps=(forcing(64.0),),
            ),
            Case(
                "impervious_variable_dt",
                state=antecedent((0.0, 0.25, 0.0)),
                steps=tuple(forcing(dt) for dt in (16.0, 48.0, 32.0, 64.0, 96.0)),
            ),
            Case(
                "pervious_competing_losses",
                parameters=parameters(impervious=0.0),
                state=antecedent((0.0, 0.0, 0.125)),
                steps=(forcing(128.0, evaporation=RAIN),),
            ),
            Case(
                "pervious_storm_recovery",
                parameters=parameters(impervious=0.0),
                steps=(forcing(64.0, 1.0 / 512.0),) * 4
                + (forcing(512.0),) * 4
                + (forcing(64.0, 1.0 / 512.0),) * 4,
            ),
            Case(
                "route_impervious_to_pervious",
                parameters=parameters(
                    impervious=0.5, zero_impervious=0.5, route=2, routed_fraction=0.5
                ),
                state=antecedent((0.125, 0.25, 0.0)),
                steps=(forcing(64.0),) * 4,
            ),
            Case(
                "route_pervious_to_impervious",
                parameters=parameters(
                    impervious=0.5, zero_impervious=0.5, route=1, routed_fraction=0.5
                ),
                state=antecedent((0.0, 0.125, 0.25)),
                steps=(forcing(64.0),) * 4,
            ),
            Case(
                "impervious_zero_storage",
                parameters=parameters(storage_impervious=0.0),
                steps=(forcing(64.0, RAIN),),
            ),
            Case(
                "impervious_intermittent",
                steps=(
                    forcing(64.0, RAIN),
                    forcing(128.0),
                    forcing(32.0, RAIN),
                    forcing(64.0),
                    forcing(16.0, RAIN),
                ),
            ),
        )
    )
    for label, depth in (
        ("below", math.nextafter(1e-10, -math.inf)),
        ("equal", 1e-10),
        ("above", math.nextafter(1e-10, math.inf)),
    ):
        result.append(
            Case(
                f"runoff_zero_threshold_{label}",
                parameters=parameters(storage_impervious=0.0),
                state=antecedent((0.0, depth, 0.0)),
                rate=(1, 1.0),
            )
        )
    for label, dt, tolerance, initial_step in (
        ("full_initial_step", 64.0, 1e-4, 64.0),
        ("short_initial_step", 64.0, 1e-4, 1.0),
        ("tight_tolerance", 256.0, 1e-6, 256.0),
    ):
        result.append(
            Case(
                f"ode_recession_{label}",
                state=antecedent((0.0, 0.25, 0.0)),
                ode=(1, dt, tolerance, initial_step),
            )
        )
    return result
