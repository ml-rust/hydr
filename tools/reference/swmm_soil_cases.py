"""Frozen native-unit branch probes and stateful EPA infiltration sequences."""

import math
from dataclasses import dataclass

ORIGINAL = 2
MODIFIED = 3
ZERO = 1e-10
PARAMETERS = (0.4, 1e-5, 0.3)

# Input fixtures from the pinned EPA capacity-probe frozen plan, not expected outputs.
# Entries retain original F2 arguments and the native-evidence incoming-rate target.
CAPACITY_PROBES = (
    ("F2_ts_below", 0.02, 1.0, 1e-5, 9.999999999999998, "0x1.0b630a91537a1p-11"),
    ("F2_ts_equal", 0.02, 1.0, 1e-5, 10.0, "0x1.e16bbf5c509cdp-12"),
    ("F2_ts_above", 0.02, 1.0, 1e-5, 10.000000000000002, "0x1.e16bbf5c509cbp-12"),
    ("F2_front_below", 0.009999999999999998, 1.0, 1e-5, 1.0, "0x1.f979a30d0acc0p-11"),
    ("F2_front_equal", 0.01, 1.0, 1e-5, 1.0, "0x1.f979a30d0b270p-11"),
    ("F2_front_above", 0.010000000000000002, 1.0, 1e-5, 1.0, "0x1.08c3f3e0370d0p-10"),
    ("F2_zero_c1", 0.1, 0.0, 1e-5, 60.0, "0x1.4f8b588e36911p-17"),
    ("F2_newton", 0.1, 0.2, 1e-5, 60.0, "0x1.f458148155889p-16"),
    ("F2_singular_fallback", 0.0, 1.0, 1e-5, 10.0, "0x1.4f8b588e368f1p-17"),
)


@dataclass(frozen=True)
class Case:
    name: str
    model: int = MODIFIED
    parameters: tuple = PARAMETERS
    state: tuple | None = None
    steps: tuple = ()
    root: tuple | None = None

    def calls(self):
        initialization = (
            ("INIT", self.parameters)
            if self.state is None
            else ("STATE", self.parameters + self.state)
        )
        calls = [initialization]
        calls.extend(("STEP", (self.model,) + step) for step in self.steps)
        if self.root is not None:
            calls.append(("F2", self.root))
        return calls


def neighbors(value):
    return (
        ("below", math.nextafter(value, -math.inf)),
        ("equal", value),
        ("above", math.nextafter(value, math.inf)),
    )


def cases():
    result = []
    sequence = (
        (60.0, 0.0, 0.0),
        (60.0, 5e-6, 0.0),
        (300.0, 1e-3, 0.0),
        (600.0, 2e-4, 0.1),
        (3600.0, 0.0, 0.0),
        (60.0, 5e-6, 0.0),
    )
    for model, name in ((MODIFIED, "modified"), (ORIGINAL, "original")):
        result.append(Case(f"{name}_wetting_dry_renewal", model=model, steps=sequence))
        result.append(
            Case(
                f"{name}_expired_low_rain",
                model=model,
                state=(0.2, 0.01, 0.02, 0, 0.0),
                steps=((60.0, 5e-6, 0.0),),
            )
        )
    for saturation in (0, 1):
        for label, available in neighbors(ZERO):
            result.append(
                Case(
                    f"zero_{saturation}_{label}",
                    state=(0.2, 0.01, 0.02, saturation, 5.0),
                    steps=((60.0, available, 0.0),),
                )
            )
    for label, available in neighbors(PARAMETERS[1]):
        result.append(
            Case(
                f"ks_{label}",
                state=(0.2, 0.01, 0.02, 0, 0.0),
                steps=((60.0, available, 0.0),),
            )
        )

    binary = (0.75, 1.0 / 1024.0, 0.25)
    available = 2.0 * binary[1]
    fs = 3.0 / 16.0
    for label, f in neighbors(fs):
        result.append(
            Case(
                f"saturation_F_{label}",
                parameters=binary,
                state=(0.25, f, f, 0, 0.0),
                steps=((1.0, available, 0.0),),
            )
        )
    for label, f in neighbors(fs - available):
        result.append(
            Case(
                f"saturation_end_{label}",
                parameters=binary,
                state=(0.25, f, f, 0, 0.0),
                steps=((1.0, available, 0.0),),
            )
        )
    result.append(
        Case(
            "saturated_equal_supply",
            parameters=binary,
            state=(0.25, fs, fs, 1, 0.0),
            steps=((1.0, available, 0.0),),
        )
    )
    result.extend(
        (
            Case(
                "reachable_negative_F",
                steps=(
                    (60.0, 5e-6, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.001, 1e-3, 0.0),
                    (60.0, 0.0, 0.0),
                ),
            ),
            Case(
                "recovery_negative_F",
                state=(0.2, 1e-8, 0.02, 0, 1e6),
                steps=((60.0, 0.0, 0.0),),
            ),
            Case(
                "recovery_empty_upper_zone",
                state=(0.2, 0.01, 1e-8, 0, 1e6),
                steps=((3600.0, 0.0, 0.0),),
            ),
            Case(
                "recovery_expired_timer",
                state=(0.2, 0.01, 0.02, 0, 0.0),
                steps=((60.0, 0.0, 0.0),),
            ),
            Case(
                "saturated_dry_persistence",
                state=(0.2, 0.01, 0.02, 1, 5.0),
                steps=((60.0, 0.0, 0.0), (600.0, 0.0, 0.0)),
            ),
            Case(
                "zero_suction", parameters=(0.0, 1e-5, 0.3), steps=((60.0, 1e-3, 0.0),)
            ),
            Case(
                "zero_maximum_deficit",
                parameters=(0.4, 1e-5, 0.0),
                steps=((60.0, 1e-3, 0.0),),
            ),
        )
    )
    for label, ts in neighbors(10.0):
        result.append(Case(f"F2_ts_{label}", root=(0.02, 1.0, 1e-5, ts)))
    for label, f1 in neighbors(0.01):
        result.append(Case(f"F2_front_{label}", root=(f1, 1.0, 1e-5, 1.0)))
    result.extend(
        (
            Case("F2_zero_c1", root=(0.1, 0.0, 1e-5, 60.0)),
            Case("F2_newton", root=(0.1, 0.2, 1e-5, 60.0)),
            Case("F2_singular_fallback", root=(0.0, 1.0, 1e-5, 10.0)),
            Case("F2_zero_ks", root=(0.1, 0.2, 0.0, 60.0)),
        )
    )
    for label, timer in neighbors(60.0):
        result.append(
            Case(
                f"dry_timer_{label}",
                state=(0.2, 0.01, 0.02, 0, timer),
                steps=((60.0, 0.0, 0.0),),
            )
        )
    for label, rate in neighbors(available):
        if label != "equal":
            result.append(
                Case(
                    f"saturated_supply_{label}",
                    parameters=binary,
                    state=(0.25, fs, fs, 1, 0.0),
                    steps=((1.0, rate, 0.0),),
                )
            )
    for label, factor in (("below", 0.5), ("above", 2.0)):
        result.append(
            Case(
                f"ponding_zero_{label}",
                state=(0.2, 0.01, 0.02, 0, 1000.0),
                steps=((60.0, 0.0, factor * ZERO * 60.0),),
            )
        )
    result.append(
        Case(
            "saturated_signed_fallback",
            parameters=(0.005, 1e-5, 0.3),
            state=(0.2, -0.01, 0.02, 1, 5.0),
            steps=((60.0, 0.001, 0.0),),
        )
    )
    for name, f1, c1, ks, ts, target in CAPACITY_PROBES:
        for label, rate in neighbors(float.fromhex(target)):
            result.append(
                Case(
                    f"capacity_{name}_{label}",
                    parameters=(c1, ks, 1.0),
                    state=(1.0, f1, 0.0, 1, 0.0),
                    steps=((ts, rate, 0.0),),
                )
            )
    return result


def check_case(case, outputs):
    """Check source-declared branch consequences without another infiltration solver."""
    final = outputs[-1]
    if case.name.startswith("dry_timer_"):
        reset = not case.name.endswith("above")
        if (final["F"] == 0.0) != reset or (final["T"] <= 0.0) != reset:
            raise ValueError(f"{case.name}: strict recovery timer branch differs")
    if case.name.startswith("saturated_supply_"):
        if final["Sat"] != float(case.name.endswith("above")):
            raise ValueError(
                f"{case.name}: strict saturated availability comparison differs"
            )
    if case.name.startswith("ponding_zero_"):
        below = case.name.endswith("below")
        if (final["rate"] == 0.0) != below or (final["F"] < outputs[0]["F"]) != below:
            raise ValueError(
                f"{case.name}: positive ponding effective-ZERO branch differs"
            )
    if case.name == "saturated_signed_fallback":
        f1 = outputs[0]["F"]
        expected_f2 = f1 + case.parameters[1] * case.steps[0][0]
        expected_df = expected_f2 - f1
        if (
            final["F"] != f1 + expected_df
            or final["rate"] != expected_df / case.steps[0][0]
            or final["Sat"] != 1.0
        ):
            raise ValueError(
                "saturated_signed_fallback: native signed-event fallback differs"
            )
    if case.name in ("recovery_negative_F", "reachable_negative_F") and not (
        final["F"] < 0.0 < final["Fu"]
    ):
        raise ValueError(
            "recovery_negative_F: native negative event memory was not observed"
        )
    if case.name == "reachable_negative_F" and final["T"] <= 0.0:
        raise ValueError(
            "reachable_negative_F: renewed native recovery timer did not remain active"
        )
    if case.name == "recovery_empty_upper_zone":
        if (
            final["F"] != 0.0
            or final["Fu"] != 0.0
            or final["IMD"] != case.parameters[2]
        ):
            raise ValueError(
                "recovery_empty_upper_zone: native initialization recovery differs"
            )
    if case.name == "recovery_expired_timer" and final["F"] != 0.0:
        raise ValueError("recovery_expired_timer: native event memory did not reset")
    if case.name == "saturated_dry_persistence":
        for before, after, step in zip(outputs, outputs[1:], case.steps):
            if (
                any(before[key] != after[key] for key in ("IMD", "F", "Fu", "Sat"))
                or after["rate"] != 0.0
                or after["T"] != before["T"] - step[0]
            ):
                raise ValueError(
                    "saturated_dry_persistence: state changed outside the native timer decrement"
                )
    if case.name in ("saturation_end_equal", "saturated_equal_supply"):
        if final["Sat"] != 1.0 or final["rate"] != case.steps[0][1]:
            raise ValueError(
                f"{case.name}: exact saturation equality did not retain native Sat"
            )
    if case.name == "saturation_end_below" and final["Sat"] != 0.0:
        raise ValueError("saturation_end_below: strict unsaturated comparison differs")
    if case.name.startswith("zero_"):
        below = case.name.endswith("below")
        if below and final["rate"] != 0.0:
            raise ValueError(
                f"{case.name}: below-ZERO availability did not return zero"
            )
        if not below and final["rate"] <= 0.0:
            raise ValueError(
                f"{case.name}: exact/above-ZERO availability was suppressed"
            )
    if case.name.startswith("ks_"):
        if (final["T"] > 0.0) != case.name.endswith("above"):
            raise ValueError(f"{case.name}: strict conductivity timer branch differs")
    if case.root is not None:
        f1, c1, ks, ts = case.root
        minimum = f1 + ks * ts
        if final["F2"] < minimum:
            raise ValueError(f"{case.name}: F2 falls below native F2min")
        if (
            case.name in ("F2_zero_c1", "F2_singular_fallback", "F2_zero_ks")
            and final["F2"] != minimum
        ):
            raise ValueError(f"{case.name}: native minimum/fallback result differs")
        if case.name == "F2_newton" and final["F2"] <= minimum:
            raise ValueError(
                "F2_newton: ordinary Newton case returned only the minimum"
            )


def check_pairs(results):
    original = results["original_expired_low_rain"][-1]
    modified = results["modified_expired_low_rain"][-1]
    if (
        original["rate"] != modified["rate"]
        or original["F"] != 0.0
        or modified["F"] <= 0.0
        or original["IMD"] == modified["IMD"]
    ):
        raise ValueError(
            "Original/modified expired-event state divergence was not observed"
        )
