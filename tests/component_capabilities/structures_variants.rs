//! Public structure-control variants mapped to declared capability ids.
//!
//! Each mapping is an exhaustive `match`, so a new public variant fails to compile until it
//! is classified.

use crate::gate_fixture::{self, Config};
use crate::pump_fixture;
use crate::structures_fixture::{
    Kind, Runtime, TestResult, Variant, actuator, actuators, built, orifice, weir,
};
use hydr::capability::UnsupportedReason;
use hydr::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlLatch, ControlStageSignal,
    ControlledGateBatch, ControlledPumpBatch, OperatingControlDefinition, PumpResponseModel,
    SampledControlTiming, StructureBatch, StructureInputErrorKind, StructureLaw,
};

/// Every public pump response model.
pub const PUMP_RESPONSES: [PumpResponseModel; 1] = [PumpResponseModel::IdealInstantaneous];

/// Value of the `pump-response-model` option that each model selects.
pub fn response_id(model: PumpResponseModel) -> &'static str {
    match model {
        PumpResponseModel::IdealInstantaneous => "ideal-instantaneous",
    }
}

/// Device family that a command or a structure law addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Device {
    Pump,
    Gate,
}

/// One sample of each public command.
const COMMANDS: [ControlCommand; 3] = [
    ControlCommand::PumpStoppedIsolated,
    ControlCommand::PumpRunning { speed_ratio: 1.0 },
    ControlCommand::GateTarget { opening_m: 0.5 },
];

fn command_device(command: ControlCommand) -> Device {
    match command {
        ControlCommand::PumpStoppedIsolated | ControlCommand::PumpRunning { .. } => Device::Pump,
        ControlCommand::GateTarget { .. } => Device::Gate,
    }
}

/// Device that a law admits commands and actuators for, or `None` for a passive law.
fn law_device(law: &StructureLaw) -> Option<Device> {
    match law {
        StructureLaw::HydrostaticRectangularGate(_) => Some(Device::Gate),
        StructureLaw::Pump(_) => Some(Device::Pump),
        StructureLaw::Culvert(_)
        | StructureLaw::SubmergedOrifice(_)
        | StructureLaw::SuppressedRectangularWeir(_) => None,
    }
}

const LATCHES: [ControlLatch; 2] = [ControlLatch::Low, ControlLatch::High];

/// Command target that `latch` selects from the switching fixture.
fn selected(latch: ControlLatch, config: &Config) -> f64 {
    match latch {
        ControlLatch::Low => config.low,
        ControlLatch::High => config.high,
    }
}

/// Asserts that `id` names a declared `RejectedByAdmission` entry of `kind`.
fn assert_refused(kind: Kind, id: &str) -> TestResult {
    let entry = kind
        .capabilities()
        .unsupported_by_id(id)
        .ok_or(format!("{kind:?}: `{id}` is undeclared"))?;
    assert_eq!(entry.reason, UnsupportedReason::RejectedByAdmission, "{id}");
    Ok(())
}

/// Asserts that `outcome` rejects with `kind` and a message containing `fragment`.
fn assert_rejects(
    outcome: crate::structures_fixture::Outcome,
    kind: StructureInputErrorKind,
    fragment: &str,
) -> TestResult {
    let (variant, reason) = outcome?.ok_or(format!("admitted, expected `{fragment}`"))?;
    assert_eq!(variant, Variant::Input(kind), "{reason}");
    assert!(reason.contains(fragment), "`{reason}` lacks `{fragment}`");
    Ok(())
}

/// One controller on `structure` that issues `command` from both latches.
fn controls(
    context: &StructureBatch<Runtime>,
    structure: &str,
    command: ControlCommand,
) -> Result<hydr::structures::OperatingControlBatch<Runtime>, hydr::structures::StructureError> {
    let config = Config::switching();
    context.with_operating_controls(
        vec![OperatingControlDefinition {
            controller_id: "relay".into(),
            structure_id: structure.into(),
            signal: ControlStageSignal {
                signal_id: "signal".into(),
                support_id: "support".into(),
                datum: gate_fixture::DATUM.into(),
            },
            lower_stage_m: 1.0,
            upper_stage_m: 2.0,
            high_qualification_delay_s: 0.0,
            low_qualification_delay_s: 0.0,
            low_command: command,
            high_command: command,
            provenance: "Explicit analytical classification relay".into(),
        }],
        vec![ControlInitialState {
            controller_id: "relay".into(),
            latch: ControlLatch::Low,
            pending: None,
            antecedent_sample: None,
            last_transition_at_s: None,
        }],
        ControlDecisionSchedule {
            initial_time_s: config.start,
            decision_times_s: config.times.clone(),
            end_s: config.end,
        },
        SampledControlTiming {
            maximum_decision_gap_s: config.maximum_gap,
            maximum_timer_lateness_s: config.lateness,
            sampled_applicability: "Explicit sampled-only analytical policy".into(),
        },
    )
}

#[test]
fn structure_public_variants_are_classified() -> TestResult {
    let option = Kind::ControlledPump
        .capabilities()
        .option("pump-response-model")
        .ok_or("controlled pump declares no response option")?;
    assert_eq!(option.values.len(), PUMP_RESPONSES.len(), "response values");
    for model in PUMP_RESPONSES {
        assert!(
            option.values.contains(&response_id(model)),
            "`{}` is undeclared",
            response_id(model)
        );
    }

    // Every command against every law: compatible pairs admit, others reject.
    let mut pump = pump_fixture::pump("pump")?;
    pump.datum = gate_fixture::DATUM.into();
    let elements = vec![
        gate_fixture::gate(0)?,
        weir("weir")?,
        orifice("orifice")?,
        pump,
    ];
    let context = StructureBatch::<Runtime>::new(crate::runtime::client(), elements.clone())?;
    let config = Config::switching();
    for element in &elements {
        let device = law_device(&element.law);
        for command in COMMANDS {
            let built_controls = controls(&context, &element.id, command);
            if device != Some(command_device(command)) {
                assert_rejects(
                    built(built_controls),
                    StructureInputErrorKind::Definition,
                    "supply a law-compatible command within actual device bounds",
                )?;
                assert_refused(Kind::OperatingControl, "law-incompatible-command")?;
                continue;
            }
            let built_controls = built_controls?;
            let response = pump_fixture::responses(&[element.id.as_str()]);
            let pumped = ControlledPumpBatch::<Runtime>::new(built_controls.clone(), response);
            match command_device(command) {
                Device::Pump => {
                    pumped?;
                }
                Device::Gate => {
                    assert_rejects(
                        built(pumped),
                        StructureInputErrorKind::Definition,
                        "requires an actual pump-only controlled subset",
                    )?;
                    assert_refused(Kind::ControlledPump, "non-pump-controller")?;
                    let mut matched = config.clone();
                    matched.target = 0.5;
                    ControlledGateBatch::<Runtime>::new(
                        built_controls,
                        matched.actuators(&context)?,
                    )?;
                }
            }
        }
    }

    // Only a gate law accepts actuator authority.
    for element in &elements {
        let outcome = built(actuators(&context, vec![actuator("a", &element.id)]));
        if law_device(&element.law) == Some(Device::Gate) {
            assert!(outcome?.is_none(), "{} actuator rejected", element.id);
        } else {
            assert_rejects(
                outcome,
                StructureInputErrorKind::Identity,
                "actuator requires an actual hydrostatic gate",
            )?;
            assert_refused(Kind::GateActuator, "non-gate-structure")?;
        }
    }

    // Each initial latch binds the target it selects and refuses the other.
    for latch in LATCHES {
        let mut changed = Config::switching();
        changed.initial_latch = latch;
        changed.target = selected(latch, &changed);
        changed.opening = changed.target;
        changed.build()?;
        let other = match latch {
            ControlLatch::Low => ControlLatch::High,
            ControlLatch::High => ControlLatch::Low,
        };
        let mut wrong = changed.clone();
        wrong.target = selected(other, &changed);
        let gates = changed.context()?;
        assert_rejects(
            built(ControlledGateBatch::<Runtime>::new(
                changed.controls(&gates)?,
                wrong.actuators(&gates)?,
            )),
            StructureInputErrorKind::Definition,
            "initial control command differs from actuator target",
        )?;
        assert_refused(Kind::ControlledGate, "initial-target-mismatch")?;
    }
    Ok(())
}
