//! Individual frozen isolation-policy cases. Every recipe preserves its original numerical input bits.

use super::fixture::*;
use hydr::structures::*;

#[derive(Clone, Copy)]
pub(super) enum Recipe {
    ManualZero,
    ManualMinsub,
    ManualQuarter,
    PrescribedZero,
    PrescribedPlateau,
    PrescribedMinsub,
    ActualActive,
    ActualStalled,
    ActualPositive,
    ActualUnresolved,
    PumpStopped,
    PumpRunning,
    Padding,
}
#[derive(Clone, Copy)]
pub(super) enum Mutation {
    None,
    Nonfinite,
    Datum,
    Convention,
    Identity,
}

pub(super) fn device(id: &str, pump_device: bool) -> Result<StructureElement> {
    let mut element = if pump_device {
        let mut element = pump(id)?;
        element.law = StructureLaw::Pump(PumpParameters::new(
            PumpCurve::new(
                vec![
                    PumpCurveKnot {
                        head_m: 0.0,
                        discharge_m3_per_s: 1.0,
                    },
                    PumpCurveKnot {
                        head_m: 1.0,
                        discharge_m3_per_s: 0.0,
                    },
                ],
                "Explicit frozen isolation curve".into(),
            )?,
            -1.0,
            PumpSpeedPolicy::ReferenceOnly,
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent isolation endpoints".into(),
                connection_losses_and_velocity_head_account: "Curve includes connection losses"
                    .into(),
                stopped_isolation_assertion:
                    "The actual pump definition explicitly declares stopped isolation.".into(),
                operating_envelope: "Frozen isolation fixture".into(),
            },
        )?);
        element
    } else {
        gate(id)?
    };
    element.endpoint_a = "endpoint-a".into();
    element.endpoint_b = "endpoint-b".into();
    element.datum = "Analytical source isolation datum".into();
    Ok(element)
}

pub(super) fn prescribed(
    context: &StructureBatch,
    recipe: Recipe,
) -> Result<Option<GateTrajectoryBatch>> {
    let (knots, rate) = match recipe {
        Recipe::PrescribedZero | Recipe::Padding => (vec![(0.0, 0.0), (1.0, 1.0)], 1.0),
        Recipe::PrescribedPlateau => (vec![(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)], 1.0),
        Recipe::PrescribedMinsub => (
            vec![(0.0, f64::from_bits(1)), (1.0, f64::from_bits(1))],
            0.0,
        ),
        _ => return Ok(None),
    };
    let path = GateOpeningTrajectory::new(
        knots
            .into_iter()
            .map(|(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
            .collect(),
        GateMotionLimits {
            maximum_opening_speed_m_per_s: rate,
            maximum_closing_speed_m_per_s: rate,
        },
        GateTrajectoryAccuracy {
            maximum_absolute_opening_error_m: 0.0,
            maximum_relative_opening_error: 0.0,
        },
        "Frozen original isolation trajectory".into(),
    )?;
    Ok(Some(context.with_gate_trajectory_subset(vec![
        GateTrajectoryAssignment {
            structure_id: "device-0".into(),
            trajectory: path,
        },
    ])?))
}

pub(super) fn actual(
    context: &StructureBatch,
    recipe: Recipe,
) -> Result<Option<GateActuatorBatch>> {
    let (initial, target, opening, closing, time) = match recipe {
        Recipe::ActualActive => (0.0, 1.0, 0.3, 0.3, 0.0),
        Recipe::ActualStalled => (0.0, 1.0, 0.0, 1.0, 0.5),
        Recipe::ActualPositive => (1.0, 0.0, 1.0, 0.25, 0.0),
        Recipe::ActualUnresolved => (0.0, 1.0, 0.3, 0.3, f64::from_bits(2)),
        _ => return Ok(None),
    };
    let batch = context.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: "actuator-0".into(),
            structure_id: "device-0".into(),
            opening_rate_m_per_s: opening,
            closing_rate_m_per_s: closing,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: f64::from_bits(2),
                maximum_relative_opening_error: 0.0,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "Frozen original isolation rates".into(),
            ideal_response_applicability: "Ideal rate-limited response without inertia or backlash"
                .into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: "actuator-0".into(),
            opening_m: initial,
            target_opening_m: target,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: vec![],
            end_s: 1.0,
        },
    )?;
    let batch = if time == 0.0 {
        batch
    } else {
        advance(batch, time)?
    };
    if matches!(recipe, Recipe::ActualUnresolved) {
        let position = batch.outputs()?[0].position;
        assert_eq!(position.opening_m.to_bits(), 1);
        assert_eq!(position.lower_opening_m.to_bits(), 0);
        assert_eq!(position.upper_opening_m.to_bits(), 2);
        assert_eq!(position.absolute_error_m.to_bits(), 1);
    }
    Ok(Some(batch))
}
