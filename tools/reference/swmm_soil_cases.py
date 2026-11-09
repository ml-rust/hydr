"""Frozen native-unit branch probes and stateful EPA infiltration sequences."""

import math
from dataclasses import dataclass

ORIGINAL = 2
MODIFIED = 3
ZERO = 1e-10
PARAMETERS = (0.4, 1e-5, 0.3)


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
    return result


def check_case(case, outputs):
    """Check source-declared branch consequences without another infiltration solver."""
    final = outputs[-1]
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
