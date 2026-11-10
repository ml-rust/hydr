use super::{inputs as reference_inputs, reference_context};
use hydr::structures::*;

pub use reference_context::Result;

pub fn accuracy() -> GateActuatorAccuracy {
    GateActuatorAccuracy {
        maximum_absolute_opening_error_m: 1e-12,
        maximum_relative_opening_error: 1e-12,
        maximum_arrival_window_s: 1e-10,
    }
}

pub fn gate(id: &str, maximum: f64) -> Result<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: "reservoir".into(),
        endpoint_b: "river".into(),
        datum: "Analytical fixture datum".into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            1.0,
            0.0,
            maximum,
            1.0,
            0.5,
            GateApplicability {
                coefficient_provenance: "Analytical gate fixture".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints".into(),
                free_submerged_aperture_applicability: "Declared analytical aperture envelope"
                    .into(),
                unobstructed_aperture_assertion: "Both endpoints expose the aperture".into(),
                symmetric_reversal: true,
            },
        )?),
    })
}

pub fn definition(id: &str, structure: &str, opening: f64, closing: f64) -> GateActuatorDefinition {
    GateActuatorDefinition {
        actuator_id: id.into(),
        structure_id: structure.into(),
        opening_rate_m_per_s: opening,
        closing_rate_m_per_s: closing,
        accuracy: accuracy(),
        rate_provenance: "Explicit analytical directional rates".into(),
        ideal_response_applicability: "Ideal rate-limited response without inertia or backlash"
            .into(),
    }
}

pub fn initial(id: &str, opening: f64, target: f64) -> GateActuatorInitialState {
    GateActuatorInitialState {
        actuator_id: id.into(),
        opening_m: opening,
        target_opening_m: target,
    }
}

pub fn build(
    opening: f64,
    target: f64,
    rates: (f64, f64),
    times: &[f64],
    end: f64,
) -> Result<GateActuatorBatch> {
    Ok(
        StructureBatch::new(vec![gate("gate", 1.0)?])?.with_gate_actuators(
            vec![definition("actuator", "gate", rates.0, rates.1)],
            vec![initial("actuator", opening, target)],
            GateActuatorSchedule {
                initial_time_s: 0.0,
                target_times_s: times.to_vec(),
                end_s: end,
            },
        )?,
    )
}

pub fn update(time: f64, target: f64) -> GateTargetUpdate {
    GateTargetUpdate {
        actuator_id: "actuator".into(),
        structure_id: "gate".into(),
        target_opening_m: target,
        issued_at_s: time,
        available_at_s: time,
    }
}

pub fn walk(
    mut batch: GateActuatorBatch,
    end: f64,
    updates: &[GateTargetUpdate],
) -> Result<GateActuatorBatch> {
    while let Some(event) = batch.next_event()? {
        assert!(
            event.time_s > batch.time(),
            "Actuator fixture event does not advance: {event:?}"
        );
        if event.time_s >= end {
            break;
        }
        if event.target_index.is_some() {
            return Err(format!("Fixture walk crosses an unsupplied original target event: accepted={},requested={end},event={event:?}", batch.time()).into());
        }
        let start = batch.time();
        batch = batch
            .trial_to(event.time_s, &[])
            .map_err(|error| {
                format!(
                    "Actuator fixture intermediate trial start={start},event={event:?}: {error}"
                )
            })?
            .into_next();
    }
    let next = batch.next_event()?;
    if !updates.is_empty() {
        let target_index = batch
            .schedule()
            .target_times_s
            .iter()
            .position(|time| time.to_bits() == end.to_bits())
            .ok_or_else(|| {
                format!("Actuator fixture supplies updates outside original calendar at {end}")
            })?;
        let event = next.ok_or("Original target packet has no next event")?;
        assert_eq!(
            event.time_s.to_bits(),
            end.to_bits(),
            "Original target at {end} loses event priority after accepted time {}: {event:?}",
            batch.time()
        );
        assert_eq!(
            event.target_index,
            Some(target_index),
            "Original target at {end} lacks its exact calendar identity: {event:?}"
        );
    }
    let start = batch.time();
    Ok(batch.trial_to(end, updates)
        .map_err(|error| format!("Actuator fixture endpoint trial start={start},end={end},next={next:?},supplied_updates={}: {error}", updates.len()))?
        .into_next())
}

pub fn frozen(id: &str, policy: &str, width: usize) -> Result<GateActuatorBatch> {
    let case = reference_inputs::cases()
        .into_iter()
        .find(|case| case.id == id && case.policy == policy)
        .ok_or_else(|| format!("Missing actuator reference {id}/{policy}"))?;
    Ok(reference_context::build(&case, width)?)
}

pub fn position(batch: &GateActuatorBatch, exact: f64) -> Result<GateActuatorOutput> {
    let output = batch
        .outputs()?
        .into_iter()
        .next()
        .ok_or("Missing actuator output")?;
    assert!(output.position.lower_opening_m <= exact);
    assert!(exact <= output.position.upper_opening_m);
    assert!(output.position.lower_opening_m <= output.position.opening_m);
    assert!(output.position.opening_m <= output.position.upper_opening_m);
    assert!(output.position.absolute_error_m.is_finite());
    Ok(output)
}

pub fn rejected<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    result
        .err()
        .ok_or_else(|| "Expected actuator rejection".into())
}

pub fn kind<T>(
    result: std::result::Result<T, StructureError>,
    expected: StructureInputErrorKind,
) -> Result {
    let error = rejected(result)?;
    assert_eq!(error.input_kind(), Some(expected), "{error}");
    Ok(())
}

pub fn malformed_restore(
    destination: &mut GateActuatorBatch,
    data: &GateActuatorSnapshotData,
) -> Result {
    let before = destination.snapshot_data()?;
    kind(
        destination.restore_data(data),
        StructureInputErrorKind::Definition,
    )?;
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

pub fn mixed_context() -> Result<StructureBatch> {
    let passive = StructureElement {
        id: "weir".into(),
        endpoint_a: "reservoir".into(),
        endpoint_b: "river".into(),
        datum: "Analytical fixture datum".into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Analytical weir fixture".into(),
                free_nappe_and_approach_assertion: "Suppressed analytical crest".into(),
                submergence_correction_assertion: "Declared analytical correction".into(),
                symmetric_reversal: true,
            },
        )?),
    };
    Ok(StructureBatch::new(vec![
        gate("gate-0", 1.0)?,
        passive,
        gate("gate-2", 1.0)?,
    ])?)
}
