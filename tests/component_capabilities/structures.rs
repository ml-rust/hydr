//! Structure control and response declarations checked against their admission code.

use crate::gate_fixture::{self, Config};
use crate::pump_fixture;
use crate::structures_fixture::{
    KINDS, Kind, Outcome, Runtime, TestResult, Variant, actuator, actuator_outcome, actuators,
    admitted, built, control_outcome, controlled_gate_outcome, controlled_pump_outcome, weir,
};
use crate::structures_variants::{PUMP_RESPONSES, response_id};
use hydr::capability::{ComponentId, UnsupportedReason};
use hydr::structures::{
    ControlCommand, ControlSample, ControlledGateBatch, ControlledPumpBatch, GateTargetUpdate,
    OperatingControlBatch, OperatingControlDefinition, StructureBatch, StructureError,
    StructureInputErrorKind as K,
};

/// Rebuilds `controls` on `context` after `change` edits its first definition.
fn redefined(
    context: &StructureBatch<Runtime>,
    controls: &OperatingControlBatch<Runtime>,
    change: fn(&mut OperatingControlDefinition),
) -> Result<OperatingControlBatch<Runtime>, StructureError> {
    let mut definitions = controls.definitions().to_vec();
    change(&mut definitions[0]);
    context.with_operating_controls(
        definitions,
        controls.initial_states().to_vec(),
        controls.schedule().clone(),
        controls.timing().clone(),
    )
}

/// One switching sample at the first decision after `change`.
fn sample(change: fn(&mut ControlSample)) -> Vec<ControlSample> {
    let mut samples = gate_fixture::samples(1, 1.0, 2.0);
    change(&mut samples[0]);
    samples
}

#[test]
fn structures_admit_declared_options() -> TestResult {
    for kind in KINDS {
        let capabilities = kind.capabilities();
        if kind != Kind::ControlledPump {
            assert!(capabilities.options.is_empty(), "{kind:?} options");
        }
        assert!(capabilities.regimes.is_empty(), "{kind:?} regimes");
        assert!(capabilities.sources.is_empty(), "{kind:?} sources");
        assert!(capabilities.endpoints.is_empty(), "{kind:?} endpoints");
    }
    let config = Config::switching();
    let context = config.context()?;
    let controls = config.controls(&context)?;
    admitted(
        control_outcome(&controls, 1.0, &gate_fixture::samples(1, 1.0, 2.0)),
        "gate-target control",
    )?;
    admitted(control_outcome(&controls, 0.5, &[]), "control hold")?;
    let pumps = StructureBatch::<Runtime>::new(
        crate::runtime::client(),
        vec![pump_fixture::pump("pump")?],
    )?;
    let pump_controls = pump_fixture::Config::switching().controls(&pumps, &["pump"])?;
    admitted(
        control_outcome(
            &pump_controls,
            1.0,
            &pump_fixture::samples(&["pump"], 1.0, 2.0),
        ),
        "pump control",
    )?;

    let gates = config.actuators(&context)?;
    admitted(
        actuator_outcome(&gates, 1.0, &gate_fixture::updates(1, 1.0, 1.0)),
        "actuator target",
    )?;
    admitted(
        controlled_gate_outcome(&config.build()?, 1.0, &gate_fixture::samples(1, 1.0, 2.0)),
        "controlled gate",
    )?;

    let option = Kind::ControlledPump
        .capabilities()
        .option("pump-response-model")
        .ok_or("controlled pump declares no response option")?;
    assert_eq!(option.default, None, "response model has no default");
    for value in option.values {
        let model = PUMP_RESPONSES
            .into_iter()
            .find(|m| response_id(*m) == *value)
            .ok_or(format!("`{value}` names no response model"))?;
        let mut responses = pump_fixture::responses(&["pump"]);
        responses[0].model = model;
        let batch = ControlledPumpBatch::<Runtime>::new(pump_controls.clone(), responses)?;
        admitted(
            controlled_pump_outcome(&batch, 1.0, &pump_fixture::samples(&["pump"], 1.0, 2.0)),
            "controlled pump",
        )?;
        assert_eq!(batch.outputs()?[0].model, model, "`{value}` output model");
    }
    Ok(())
}

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

fn input(kind: K) -> Variant {
    Variant::Input(kind)
}

/// Cases for each operating-control `RejectedByAdmission` entry.
fn control_cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let config = Config::switching();
    let context = config.context()?;
    let controls = config.controls(&context)?;
    let trial = |samples: Vec<ControlSample>| control_outcome(&controls, 1.0, &samples);
    Ok(Some(match id {
        "inverted-thresholds" => vec![(
            built(redefined(&context, &controls, |d| d.lower_stage_m = 2.0)),
            input(K::Definition),
            "control thresholds require finite increasing stages",
        )],
        "law-incompatible-command" => {
            let changes: [fn(&mut OperatingControlDefinition); 2] = [
                |d| d.low_command = ControlCommand::PumpStoppedIsolated,
                |d| d.high_command = ControlCommand::GateTarget { opening_m: 2.0 },
            ];
            changes
                .into_iter()
                .map(|change| {
                    (
                        built(redefined(&context, &controls, change)),
                        input(K::Definition),
                        "supply a law-compatible command within actual device bounds",
                    )
                })
                .collect()
        }
        "missing-sample" => vec![
            (
                trial(Vec::new()),
                input(K::OperatingPoint),
                "supply complete ordered decision inputs",
            ),
            (
                control_outcome(&controls, 0.5, &gate_fixture::samples(1, 0.5, 2.0)),
                input(K::OperatingPoint),
                "non-decision control interval requires an empty sample slice",
            ),
        ],
        "foreign-sample" => vec![
            (
                trial(sample(|s| s.signal_id = "other".into())),
                input(K::Identity),
                "control sample identity differs",
            ),
            (
                trial(sample(|s| s.datum = "other".into())),
                input(K::Datum),
                "control sample datum differs",
            ),
        ],
        "nonfinite-sample" => vec![(
            trial(sample(|s| s.stage_m = f64::NAN)),
            input(K::NonfiniteInput),
            "control sample requires finite stage",
        )],
        "stale-sample" => vec![(
            trial(gate_fixture::samples(1, 0.5, 2.0)),
            input(K::UnsupportedRegime),
            "supply original sample at this decision",
        )],
        "future-sample" => vec![
            (
                trial(gate_fixture::samples(1, 1.5, 2.0)),
                input(K::UnsupportedRegime),
                "supply original sample at this decision",
            ),
            (
                trial(sample(|s| s.available_at_s = 2.0)),
                input(K::UnsupportedRegime),
                "availability follows its original observation",
            ),
        ],
        _ => return Ok(None),
    }))
}

/// Cases for each gate-actuator `RejectedByAdmission` entry.
fn actuator_cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let mixed = StructureBatch::<Runtime>::new(
        crate::runtime::client(),
        vec![
            gate_fixture::gate(0)?,
            weir("weir")?,
            gate_fixture::gate(2)?,
        ],
    )?;
    let config = Config::switching();
    let batch = config.actuators(&config.context()?)?;
    let packet = |change: fn(&mut GateTargetUpdate)| {
        let mut updates = gate_fixture::updates(1, 1.0, 0.5);
        change(&mut updates[0]);
        actuator_outcome(&batch, 1.0, &updates)
    };
    Ok(Some(match id {
        "non-gate-structure" => vec![
            (
                built(actuators(&mixed, vec![actuator("a", "weir")])),
                input(K::Identity),
                "actuator requires an actual hydrostatic gate",
            ),
            (
                built(actuators(&mixed, vec![actuator("a", "missing")])),
                input(K::Identity),
                "actuator references an unknown actual structure",
            ),
        ],
        "duplicate-authority" => vec![
            (
                built(actuators(
                    &mixed,
                    vec![actuator("a", "gate-0"), actuator("b", "gate-0")],
                )),
                input(K::Identity),
                "actuator subset must follow unique original structure order",
            ),
            (
                built(actuators(
                    &mixed,
                    vec![actuator("a", "gate-0"), actuator("a", "gate-2")],
                )),
                input(K::Identity),
                "actuator IDs must be unique",
            ),
        ],
        "missing-target-packet" => vec![
            (
                actuator_outcome(&batch, 1.0, &[]),
                input(K::OperatingPoint),
                "target event requires",
            ),
            (
                actuator_outcome(&batch, 0.5, &gate_fixture::updates(1, 0.5, 0.5)),
                input(K::OperatingPoint),
                "non-target actuator interval requires",
            ),
        ],
        "stale-target" => vec![(
            packet(|u| u.issued_at_s = 0.5),
            input(K::UnsupportedRegime),
            "must be issued at the original event with causal availability",
        )],
        "future-target" => vec![(
            packet(|u| u.available_at_s = 2.0),
            input(K::UnsupportedRegime),
            "must be issued at the original event with causal availability",
        )],
        "target-outside-range" => vec![(
            packet(|u| u.target_opening_m = 2.0),
            input(K::OperatingPoint),
            "target opening lies outside actual gate range",
        )],
        _ => return Ok(None),
    }))
}

/// Cases for each controlled-gate `RejectedByAdmission` entry.
fn controlled_gate_cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let config = Config::switching();
    let context = config.context()?;
    let pair = |changed: &Config| -> Outcome {
        built(ControlledGateBatch::<Runtime>::new(
            config.controls(&context)?,
            changed.actuators(&context)?,
        ))
    };
    let mut changed = config.clone();
    Ok(Some(match id {
        "advanced-child" => vec![(
            built(ControlledGateBatch::<Runtime>::new(
                config.controls(&context)?.trial_to(0.5, &[])?.into_next(),
                config.actuators(&context)?,
            )),
            input(K::Definition),
            "requires fresh initial children",
        )],
        "foreign-structure-model" => {
            let mut elements = context.elements().to_vec();
            elements[0].endpoint_a.push_str(" different");
            let foreign = StructureBatch::<Runtime>::new(crate::runtime::client(), elements)?;
            vec![(
                built(ControlledGateBatch::<Runtime>::new(
                    config.controls(&context)?,
                    config.actuators(&foreign)?,
                )),
                input(K::Identity),
                "controlled gate models differ",
            )]
        }
        "mismatched-gate-subset" => {
            let mut wide = config.clone();
            wide.width = 2;
            let both = wide.context()?;
            let controls = wide.controls(&both)?;
            let second = both.with_operating_controls(
                controls.definitions()[1..].to_vec(),
                controls.initial_states()[1..].to_vec(),
                controls.schedule().clone(),
                controls.timing().clone(),
            )?;
            vec![
                (
                    built(ControlledGateBatch::<Runtime>::new(
                        controls,
                        config.actuators(&both)?,
                    )),
                    input(K::Identity),
                    "controlled gate authority is incomplete",
                ),
                (
                    built(ControlledGateBatch::<Runtime>::new(
                        second,
                        config.actuators(&both)?,
                    )),
                    input(K::Identity),
                    "controlled gate subsets differ",
                ),
            ]
        }
        "calendar-mismatch" => {
            changed.end = 5.0;
            vec![(pair(&changed), input(K::Definition), "calendars differ")]
        }
        "initial-target-mismatch" => {
            changed.target = 0.25;
            vec![(
                pair(&changed),
                input(K::Definition),
                "initial control command differs from actuator target",
            )]
        }
        _ => return Ok(None),
    }))
}

/// Cases for each controlled-pump `RejectedByAdmission` entry.
fn controlled_pump_cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let context = StructureBatch::<Runtime>::new(
        crate::runtime::client(),
        vec![pump_fixture::pump("a")?, pump_fixture::pump("b")?],
    )?;
    let controls = pump_fixture::Config::switching().controls(&context, &["a", "b"])?;
    let respond = |definitions| {
        built(ControlledPumpBatch::<Runtime>::new(
            controls.clone(),
            definitions,
        ))
    };
    Ok(Some(match id {
        "incomplete-response-definitions" => vec![
            (
                respond(pump_fixture::responses(&["a"])),
                input(K::Identity),
                "one complete ordered definition per original controller",
            ),
            (
                respond(pump_fixture::responses(&["b", "a"])),
                input(K::Identity),
                "must match every original controlled pump in order",
            ),
        ],
        "missing-response-evidence" => (0..2)
            .map(|field| {
                let mut definitions = pump_fixture::responses(&["a", "b"]);
                if field == 0 {
                    definitions[0].response_provenance.clear();
                } else {
                    definitions[0].ideal_response_applicability.clear();
                }
                (
                    respond(definitions),
                    input(K::Definition),
                    "supply nonempty explicit identity/evidence",
                )
            })
            .collect(),
        "non-pump-controller" => {
            let config = Config::switching();
            let gates = config.context()?;
            vec![(
                built(ControlledPumpBatch::<Runtime>::new(
                    config.controls(&gates)?,
                    pump_fixture::responses(&["gate-0"]),
                )),
                input(K::Definition),
                "requires an actual pump-only controlled subset",
            )]
        }
        _ => return Ok(None),
    }))
}

fn rejection_cases(
    kind: Kind,
    id: &str,
) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    match kind {
        Kind::OperatingControl => control_cases(id),
        Kind::GateActuator => actuator_cases(id),
        Kind::ControlledGate => controlled_gate_cases(id),
        Kind::ControlledPump => controlled_pump_cases(id),
    }
}

#[test]
fn structures_reject_declared_rejections() -> TestResult {
    for kind in KINDS {
        let mut count = 0;
        for entry in kind
            .capabilities()
            .unsupported
            .iter()
            .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
        {
            let cases = rejection_cases(kind, entry.id)?
                .ok_or(format!("{kind:?}: no case for `{}`", entry.id))?;
            assert!(!cases.is_empty(), "{kind:?} `{}` has no case", entry.id);
            for (outcome, variant, fragment) in cases {
                let (found, reason) =
                    outcome?.ok_or(format!("{kind:?}: `{}` admitted", entry.id))?;
                assert_eq!(found, variant, "{kind:?} `{}`: {reason}", entry.id);
                assert!(
                    reason.contains(fragment),
                    "{kind:?} `{}`: `{reason}` lacks `{fragment}`",
                    entry.id
                );
                count += 1;
            }
        }
        assert!(count > 0, "{kind:?} declares no admission rejection");
    }
    Ok(())
}

#[test]
fn structure_parents_run_nested_admission() -> TestResult {
    let control = Kind::OperatingControl.capabilities();
    for parent in [Kind::ControlledGate, Kind::ControlledPump] {
        let nested = parent.capabilities().nested;
        assert!(
            nested.contains(&ComponentId::OperatingControl),
            "{parent:?}"
        );
        assert!(
            control
                .attachable_to
                .contains(&parent.capabilities().component)
        );
    }
    assert!(
        Kind::ControlledGate
            .capabilities()
            .nested
            .contains(&ComponentId::GateActuator)
    );
    // Both owners reject a stale sample through nested control admission, changing nothing.
    let stale = "supply original sample at this decision";
    let gate = controlled_gate_outcome(
        &Config::switching().build()?,
        1.0,
        &gate_fixture::samples(1, 0.5, 2.0),
    )?
    .ok_or("controlled gate admitted a stale sample")?;
    assert_eq!(gate.0, input(K::UnsupportedRegime), "{}", gate.1);
    assert!(gate.1.contains(stale), "{}", gate.1);
    let pumps = StructureBatch::<Runtime>::new(
        crate::runtime::client(),
        vec![pump_fixture::pump("pump")?],
    )?;
    let pump = pump_fixture::Config::switching().build(&pumps, &["pump"])?;
    let pump = controlled_pump_outcome(&pump, 1.0, &pump_fixture::samples(&["pump"], 0.5, 2.0))?
        .ok_or("controlled pump admitted a stale sample")?;
    assert_eq!(pump.0, input(K::UnsupportedRegime), "{}", pump.1);
    assert!(pump.1.contains(stale), "{}", pump.1);
    // The controlled gate's nested actuator rejects a non-gate binding.
    let mixed = StructureBatch::<Runtime>::new(
        crate::runtime::client(),
        vec![gate_fixture::gate(0)?, weir("weir")?],
    )?;
    let (variant, reason) =
        built(actuators(&mixed, vec![actuator("a", "weir")]))?.ok_or("actuator admitted a weir")?;
    assert_eq!(variant, input(K::Identity), "{reason}");
    Ok(())
}
