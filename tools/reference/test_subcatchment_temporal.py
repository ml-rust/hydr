"""Archive-free input-binding regressions. Synthetic rows establish no numerical accuracy."""

import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path

from subcatchment_temporal_cases import (
    HISTORY_NAMES,
    HISTORY_TIMESTEPS,
    INPUT_FIELDS,
    NAMES,
    OBSERVER_FIELDS,
    case,
    history_case,
    input_values,
)
from subcatchment_temporal_compare import (
    check_inputs,
    history_refinement,
    native_departures,
)
from subcatchment_temporal_native import native_case, read_histories
from swmm_runoff_records import (
    CSV_FIELDS,
    INIT_FIELDS,
    NATIVE_FIELDS,
    STATE_FIELDS,
    observations,
)


def observer_row(spec, dt, step):
    row = dict.fromkeys(OBSERVER_FIELDS[1:], 0.0)
    row.update(zip(INPUT_FIELDS, input_values(spec)))
    row.update(case=spec["name"], dt_s=dt, step=float(step), time_s=step * dt)
    row.update({f"depth{i}_m": depth for i, depth in enumerate(spec["initial"])})
    row["soil_imd"] = row["soil_maximum_deficit"]
    return row


def observer_histories():
    return [
        observer_row(history_case(name), dt, step)
        for name in HISTORY_NAMES
        for dt in HISTORY_TIMESTEPS
        for step in range(int(256.0 / dt) + 1)
    ]


def native_records(case):
    initial = dict.fromkeys(NATIVE_FIELDS[2:], 0.0)
    initial.update(zip(INIT_FIELDS, case.parameters))
    initial.update(
        op="INIT",
        index=1,
        IMD=1.0,
        Lu=4.0 * math.sqrt((1.0 / 1024.0) * 12.0 * 3600.0) / 12.0,
        probe_subarea=-1.0,
    )
    fi, f0 = case.parameters[1:3]
    fractions = [fi * f0, fi * (1.0 - f0), 1.0 - fi]
    for index, fraction in enumerate(fractions):
        initial[f"a{index}_fArea"] = fraction
        initial[f"a{index}_N"] = 1.0 / 32.0
        initial[f"a{index}_dStore"] = 0.0 if index == 0 else 0.125
        initial[f"a{index}_alpha"] = 0.0
        donor = (
            case.parameters[9] == 2.0
            and index < 2
            or case.parameters[9] == 1.0
            and index == 2
        )
        initial[f"a{index}_fOutlet"] = 1.0 - case.parameters[10] if donor else 1.0
    outputs = [initial]
    for operation, arguments in case.calls()[1:]:
        previous = outputs[-1]
        row = previous.copy()
        row.update(op=operation, index=previous["index"] + 1)
        if operation == "STATE":
            row.update(zip(STATE_FIELDS, arguments))
        else:
            dt, rain, evaporation, external = arguments
            actual_rain = rain * 43200.0 / 43200.0 + 0.0
            row.update(
                dt_s=dt,
                time_s=previous["time_s"] + dt,
                requested_rain_ft_s=rain,
                actual_rain_ft_s=actual_rain,
                rainfall=actual_rain,
                evap_ft_s=evaporation,
                external_runon_cfs=external,
            )
            for index in range(3):
                row[f"a{index}_old_depth"] = previous[f"a{index}_depth"]
        storage = math.fsum(
            1024.0 * fractions[index] * row[f"a{index}_depth"] for index in range(3)
        )
        row.update(storage_before_ft3=storage, storage_after_ft3=storage)
        if operation == "STEP":
            rain = row["actual_rain_ft_s"] * row["dt_s"] * 1024.0
            implied = storage + rain - storage
            row.update(
                rain_volume_ft3=rain,
                storage_implied_outflow_ft3=implied,
                outlet_storage_implied_outflow_ft3=implied,
                storage_residual_all_ft3=-implied,
                storage_residual_outlet_ft3=-implied,
            )
        outputs.append(row)
    return observations(case, outputs)


def all_native_records():
    return [
        row
        for name in HISTORY_NAMES
        for dt in HISTORY_TIMESTEPS
        for row in native_records(native_case(name, dt))
    ]


def write_native(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, ".17g") if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


class ObserverBindingTests(unittest.TestCase):
    def test_complete_frozen_histories_are_retained(self):
        self.assertEqual(len(history_refinement(observer_histories())), 12)

    def test_every_history_input_binds_to_the_frozen_recipe(self):
        for name in HISTORY_NAMES:
            for field in INPUT_FIELDS:
                with self.subTest(case=name, field=field):
                    rows = observer_histories()
                    row = next(row for row in rows if row["case"] == name)
                    value = row[field]
                    row[field] = (
                        0.0
                        if value is None
                        else value + 1.0
                        if field.startswith("alpha")
                        else math.nextafter(value, math.inf)
                    )
                    with self.assertRaises(ValueError):
                        history_refinement(rows)

    def test_every_endpoint_requires_complete_finite_native_state(self):
        for field in (
            "soil_imd",
            "soil_f_ft",
            "soil_fu_ft",
            "soil_saturated",
            "soil_t_s",
        ):
            for value in (None, math.nan, math.inf):
                with self.subTest(field=field, value=value):
                    rows = observer_histories()
                    rows[1][field] = value
                    with self.assertRaisesRegex(ValueError, "native soil state"):
                        history_refinement(rows)
        for name in NAMES:
            for step in (0, 1):
                for field in (
                    "soil_imd",
                    "soil_f_ft",
                    "soil_fu_ft",
                    "soil_saturated",
                    "soil_t_s",
                ):
                    with self.subTest(case=name, step=step, field=field):
                        spec = case(name)
                        row = observer_row(spec, 1.0, step)
                        row[field] = None
                        with self.assertRaisesRegex(ValueError, "native soil state"):
                            check_inputs(spec, row)

    def test_smooth_and_history_initial_native_fields_are_exact(self):
        for spec in [
            *(case(name) for name in NAMES),
            *(history_case(name) for name in HISTORY_NAMES),
        ]:
            for field in (
                "soil_imd",
                "soil_f_ft",
                "soil_fu_ft",
                "soil_saturated",
                "soil_t_s",
            ):
                with self.subTest(case=spec["name"], field=field):
                    row = observer_row(spec, 1.0, 0)
                    check_inputs(spec, row)
                    row[field] = (
                        1.0
                        if field == "soil_saturated"
                        else math.nextafter(row[field], math.inf)
                    )
                    with self.assertRaises(ValueError):
                        check_inputs(spec, row)

    def test_initial_surface_state_and_absent_coefficients_remain_explicit(self):
        spec = history_case(HISTORY_NAMES[0])
        row = observer_row(spec, 64.0, 0)
        row["depth2_m"] = math.nextafter(row["depth2_m"], math.inf)
        with self.assertRaisesRegex(ValueError, "initial surface"):
            check_inputs(spec, row)
        row = observer_row(spec, 64.0, 0)
        row["alpha0_si"] = 0.0
        with self.assertRaisesRegex(ValueError, "absent coefficient"):
            check_inputs(spec, row)

    def test_signed_native_event_memory_remains_admitted_after_initialization(self):
        spec = history_case(HISTORY_NAMES[1])
        row = observer_row(spec, 64.0, 1)
        row.update(soil_f_ft=-1.0, soil_t_s=-2.0, soil_saturated=1.0)
        check_inputs(spec, row)


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "native.csv"
        self.rows = all_native_records()

    def test_complete_native_streams_supply_all_twelve_departure_observations(self):
        write_native(self.path, self.rows)
        self.assertEqual(len(read_histories(self.path)), 12)
        self.assertEqual(len(native_departures(observer_histories(), self.path)), 12)

    def test_missing_duplicate_and_reordered_steps_reject(self):
        for mutation in ("missing", "duplicate", "reordered", "replaced"):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self.rows)
                if mutation == "missing":
                    del rows[3]
                elif mutation == "duplicate":
                    rows.insert(3, copy.deepcopy(rows[3]))
                elif mutation == "reordered":
                    rows[3], rows[4] = rows[4], rows[3]
                else:
                    rows[3] = copy.deepcopy(rows[4])
                write_native(self.path, rows)
                with self.assertRaises(ValueError):
                    read_histories(self.path)

    def test_native_command_state_parameter_forcing_clock_and_volume_mutations_reject(
        self,
    ):
        mutations = [
            (1, "a2_depth"),
            (2, "area_ft2"),
            (2, "requested_rain_ft_s"),
            (2, "actual_rain_ft_s"),
            (2, "evap_ft_s"),
            (2, "time_s"),
            (2, "Vinfil"),
            (2, "independent_rain_ft3"),
        ]
        for index, field in mutations:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[index][field] += 1.0
                write_native(self.path, rows)
                with self.assertRaises(ValueError):
                    read_histories(self.path)

    def test_native_schema_and_command_receipts_are_required(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["command_hex"] += " 0x0.0p+0"
        write_native(self.path, rows)
        with self.assertRaisesRegex(ValueError, "command receipt"):
            read_histories(self.path)
        write_native(self.path, self.rows)
        text = self.path.read_text().replace(
            "case,command_hex", "case,other_command", 1
        )
        self.path.write_text(text)
        with self.assertRaisesRegex(ValueError, "schema"):
            read_histories(self.path)


if __name__ == "__main__":
    unittest.main()
