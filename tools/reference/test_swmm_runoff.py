"""Self-contained runoff consumer regressions without a native compiler or evidence directory."""

import csv
import io
import math
import unittest

from swmm_runoff_cases import DEFAULTS, Case
from swmm_runoff_checks import NONNEGATIVE_FIELDS, check_case
from swmm_runoff_records import INIT_FIELDS, NATIVE_FIELDS, STATE_FIELDS, read_native


def initial():
    row = dict.fromkeys(NATIVE_FIELDS[2:], 0.0)
    row.update({key: float(value) for key, value in zip(INIT_FIELDS, DEFAULTS)})
    row.update(
        op="INIT",
        index=1,
        probe_subarea=-1.0,
        IMD=float(DEFAULTS[13]),
        Lu=4.0 * math.sqrt(DEFAULTS[12] * 12.0 * 3600.0) / 12.0,
    )
    for i in range(3):
        row[f"a{i}_fArea"] = 1.0 if i == 1 else 0.0
        row[f"a{i}_N"] = float(DEFAULTS[5 if i < 2 else 6])
        row[f"a{i}_dStore"] = 0.0 if i == 0 else 0.125
        row[f"a{i}_alpha"] = 0.18625 if i < 2 else 0.0
        row[f"a{i}_fOutlet"] = 1.0
    return row


def step(previous, arguments):
    row = previous.copy()
    dt, rain, evap, external = arguments
    actual_rain = (float(rain) * 43200.0) / 43200.0 + 0.0
    row.update(
        op="STEP",
        index=previous["index"] + 1,
        time_s=previous["time_s"] + dt,
        dt_s=dt,
        requested_rain_ft_s=rain,
        actual_rain_ft_s=actual_rain,
        rainfall=actual_rain,
        evap_ft_s=evap,
        external_runon_cfs=external,
    )
    return row


def restore(previous, witness):
    row = previous.copy()
    row.update(op="STATE", index=previous["index"] + 1)
    row.update(zip(STATE_FIELDS, witness))
    return row


def probe(previous, operation, arguments):
    row = previous.copy()
    row.update(
        op=operation,
        index=previous["index"] + 1,
        probe_kind=1.0 if operation == "RATE" else 2.0,
        probe_subarea=float(arguments[0]),
        probe_time_s=arguments[1],
        probe_tolerance=arguments[2] if operation == "ODE" else 0.0,
        probe_initial_step_s=arguments[3] if operation == "ODE" else 0.0,
    )
    return row


def parse(calls, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=NATIVE_FIELDS)
    writer.writeheader()
    writer.writerows(
        {
            key: format(value, ".17g") if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in rows
    )
    return read_native("consumer", calls, stream.getvalue())


class CommandBindingTests(unittest.TestCase):
    def test_valid_step_inputs_and_native_gauge_bridge_are_retained(self):
        args = (2.0, 0.0001, 2e-5, 3.0)
        rows = parse(
            [("INIT", DEFAULTS), ("STEP", args)], [initial(), step(initial(), args)]
        )
        self.assertEqual(
            rows[1]["actual_rain_ft_s"].hex(),
            ((args[1] * 43200.0) / 43200.0 + 0.0).hex(),
        )

    def test_every_step_input_clock_and_gauge_result_reject_corruption(self):
        args = (2.0, 0.0001, 2e-5, 3.0)
        for field in (
            "dt_s",
            "requested_rain_ft_s",
            "evap_ft_s",
            "external_runon_cfs",
            "time_s",
            "actual_rain_ft_s",
            "rainfall",
        ):
            with self.subTest(field=field):
                row = step(initial(), args)
                row[field] = math.nextafter(row[field], math.inf)
                with self.assertRaises(ValueError):
                    parse([("INIT", DEFAULTS), ("STEP", args)], [initial(), row])

    def test_clock_accumulates_from_submitted_intervals(self):
        args = (0.75, 0.0, 0.0, 0.0)
        first = step(initial(), args)
        second = step(first, args)
        calls = [("INIT", DEFAULTS), ("STEP", args), ("STEP", args)]
        self.assertEqual(parse(calls, [initial(), first, second])[-1]["time_s"], 1.5)
        second["time_s"] = math.nextafter(1.5, math.inf)
        with self.assertRaises(ValueError):
            parse(calls, [initial(), first, second])

    def test_state_checks_every_submitted_field_and_accepts_signed_memory(self):
        witness = (10.0,) + (0.0,) * 15 + (1.0, -0.01, 0.125, 1.0, -2.0)
        calls = [("INIT", DEFAULTS), ("STATE", witness)]
        parsed = parse(calls, [initial(), restore(initial(), witness)])
        check_case(Case("consumer"), parsed, [{} for _ in parsed])
        for field in STATE_FIELDS:
            with self.subTest(field=field):
                row = restore(initial(), witness)
                row[field] = (
                    0.0 if field == "Sat" else math.nextafter(row[field], math.inf)
                )
                with self.assertRaises(ValueError):
                    parse(calls, [initial(), row])

    def test_initialization_checks_clock_cold_state_and_config(self):
        parse([("INIT", DEFAULTS)], [initial()])
        for field in ("time_s", "IMD", "dt_s", "probe_kind", *INIT_FIELDS):
            with self.subTest(field=field):
                row = initial()
                row[field] = math.nextafter(row[field], math.inf)
                with self.assertRaises(ValueError):
                    parse([("INIT", DEFAULTS)], [row])

    def test_rate_and_ode_diagnostics_bind_all_arguments_without_advancing_clock(self):
        for operation, args in (("RATE", (1, 0.75)), ("ODE", (1, 64.0, 1e-4, 1.0))):
            calls = [("INIT", DEFAULTS), (operation, args)]
            parse(calls, [initial(), probe(initial(), operation, args)])
            fields = ["probe_kind", "probe_subarea", "probe_time_s", "time_s", "dt_s"]
            fields.extend(("probe_tolerance", "probe_initial_step_s"))
            for field in fields:
                with self.subTest(operation=operation, field=field):
                    row = probe(initial(), operation, args)
                    row[field] = math.nextafter(row[field], math.inf)
                    with self.assertRaises(ValueError):
                        parse(calls, [initial(), row])

    def test_exact_zero_assignment_and_integer_sat_semantics_are_distinct(self):
        args = (1.0, -0.0, 0.0, 0.0)
        parsed = parse(
            [("INIT", DEFAULTS), ("STEP", args)], [initial(), step(initial(), args)]
        )
        self.assertEqual(parsed[1]["requested_rain_ft_s"].hex(), (-0.0).hex())
        self.assertEqual(parsed[1]["actual_rain_ft_s"].hex(), (0.0).hex())
        witness = (0.0,) * 16 + (1.0, -0.01, 0.125, -0.0, -2.0)
        row = restore(initial(), witness)
        row["Sat"] = 0.0
        parse([("INIT", DEFAULTS), ("STATE", witness)], [initial(), row])


class NativeAdmissionTests(unittest.TestCase):
    def test_every_loss_rate_and_volume_requires_nonnegative_values(self):
        for field in NONNEGATIVE_FIELDS:
            with self.subTest(field=field):
                row = initial()
                row[field] = -math.ulp(0.0)
                with self.assertRaises(ValueError):
                    check_case(Case("consumer"), [row], [{}])

    def test_native_deficit_and_upper_zone_capacity_endpoints_are_checked(self):
        row = initial()
        capacity = row["IMDmax"] * row["Lu"]
        upper = max(row["IMDmax"], capacity / row["Lu"])
        row.update(IMD=upper, Fu=capacity, F=-1.0, T=-2.0, Sat=1.0)
        check_case(Case("consumer"), [row], [{}])
        for field, value in (
            ("IMD", math.nextafter(upper, math.inf)),
            ("IMD", -math.ulp(0.0)),
            ("Fu", math.nextafter(capacity, math.inf)),
            ("Fu", -math.ulp(0.0)),
            ("Lu", 0.0),
        ):
            with self.subTest(field=field, value=value):
                corrupted = {**row, field: value}
                with self.assertRaises(ValueError):
                    check_case(Case("consumer"), [corrupted], [{}])

    def test_signed_f_and_t_do_not_infer_saturation(self):
        for saturation in (0.0, 1.0):
            row = initial()
            row.update(F=-1.0, T=-2.0, Sat=saturation)
            check_case(Case("consumer"), [row], [{}])
        for field in ("F", "T"):
            with self.subTest(field=field):
                row = {**initial(), field: math.inf}
                with self.assertRaises(ValueError):
                    check_case(Case("consumer"), [row], [{}])


if __name__ == "__main__":
    unittest.main()
