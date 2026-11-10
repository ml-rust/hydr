use super::{
    fixture::*,
    isolation_matrix_fixture::{Mutation, Recipe, actual, device, prescribed},
};
use hydr::structures::*;

pub(super) fn run(recipe: Recipe, mutation: Mutation) -> Result {
    let pump_device = matches!(recipe, Recipe::PumpStopped | Recipe::PumpRunning);
    let mut elements = vec![device("device-0", pump_device)?];
    if matches!(recipe, Recipe::Padding) {
        elements.push(device("device-1", false)?);
    }
    let context = StructureBatch::new(elements)?;
    let prescribed = prescribed(&context, recipe)?;
    let actual = actual(&context, recipe)?;
    let mut manual = Vec::new();
    let bindings = if let Some(path) = &prescribed {
        let mut bindings = vec![OperatingSourceBinding::Prescribed {
            source_id: if matches!(recipe, Recipe::Padding) {
                "prescribed-source"
            } else {
                "source-0"
            }
            .into(),
            trajectory: path,
        }];
        if matches!(recipe, Recipe::Padding) {
            bindings.push(OperatingSourceBinding::Manual {
                source_id: "manual-source".into(),
                structure_ids: vec!["device-1".into()],
            });
            manual.push(opening("device-1", f64::from_bits(1)));
        }
        bindings
    } else if let Some(batch) = &actual {
        vec![OperatingSourceBinding::Actuated {
            source_id: "source-0".into(),
            actuator: batch,
        }]
    } else {
        manual.push(match recipe {
            Recipe::ManualZero => opening("device-0", 0.0),
            Recipe::ManualMinsub => opening("device-0", f64::from_bits(1)),
            Recipe::ManualQuarter => opening("device-0", 0.25),
            Recipe::PumpStopped => {
                operation("device-0", StructureOperatingPoint::PumpStoppedIsolated)
            }
            Recipe::PumpRunning => operation(
                "device-0",
                StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            ),
            _ => return Err("Frozen isolation recipe has no source".into()),
        });
        vec![OperatingSourceBinding::Manual {
            source_id: "source-0".into(),
            structure_ids: vec!["device-0".into()],
        }]
    };
    let registry = context.with_operating_sources(bindings)?;
    let mut packets = heads(&context);
    for packet in &mut packets {
        packet.endpoint_a.support = HeadSupport::Unknown;
        packet.endpoint_b.support = HeadSupport::Unknown;
    }
    let expected = match mutation {
        Mutation::None => match recipe {
            Recipe::ManualMinsub
            | Recipe::ManualQuarter
            | Recipe::PrescribedMinsub
            | Recipe::ActualPositive
            | Recipe::ActualUnresolved
            | Recipe::PumpRunning
            | Recipe::Padding => Some(StructureInputErrorKind::UnknownSupport),
            _ => None,
        },
        Mutation::Nonfinite => {
            packets[0].endpoint_a.support = HeadSupport::Wet { stage_m: f64::NAN };
            Some(StructureInputErrorKind::NonfiniteInput)
        }
        Mutation::Datum => {
            packets[0].endpoint_a.support = HeadSupport::Wet { stage_m: 1.0 };
            packets[0].datum = "Different datum".into();
            Some(StructureInputErrorKind::Datum)
        }
        Mutation::Convention => {
            packets[0].endpoint_a.support = HeadSupport::Wet { stage_m: 1.0 };
            packets[0].convention = HeadConvention::EnergyHead;
            Some(StructureInputErrorKind::HeadConvention)
        }
        Mutation::Identity => {
            packets[0].endpoint_a.support = HeadSupport::Wet { stage_m: 1.0 };
            packets[0].endpoint_a.id = "wrong-endpoint".into();
            Some(StructureInputErrorKind::Identity)
        }
    };
    let frames = actual
        .as_ref()
        .map(|batch| {
            vec![OperatingSourceFrame::ActuatedGate {
                source_id: "source-0",
                actuator: batch,
            }]
        })
        .unwrap_or_default();
    let time = actual.as_ref().map_or(
        if matches!(recipe, Recipe::PrescribedPlateau) {
            0.5
        } else {
            0.0
        },
        GateActuatorBatch::time,
    );
    let result = registry.propose_at(time, &packets, &manual, &frames);
    if let Some(expected) = expected {
        let error = rejected(result)?;
        assert_eq!(error.input_kind(), Some(expected), "{error}");
        assert_eq!(
            error.element(),
            Some(if matches!(recipe, Recipe::Padding) {
                1
            } else {
                0
            })
        );
    } else {
        let outputs = result?.outputs()?;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].hydraulic.input, packets[0]);
        assert_eq!(outputs[0].hydraulic.discharge_m3_per_s, 0.0);
        assert!(outputs[0].hydraulic.hydraulic_operands.is_none());
        assert_eq!(
            outputs[0].hydraulic.regime,
            if pump_device {
                StructureRegime::PumpIsolated
            } else {
                StructureRegime::GateClosed
            }
        );
    }
    Ok(())
}
