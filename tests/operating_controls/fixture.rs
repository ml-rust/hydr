use super::{
    reference_case,
    reference_inputs::{self, Case, Sample},
};
use hydr::structures::*;

pub use reference_case::Result;

pub fn case(id: &str) -> Result<&'static Case> {
    reference_inputs::cases()
        .find(|case| case.id == id)
        .ok_or_else(|| format!("Missing control fixture {id}").into())
}

pub fn batch(id: &str, width: usize) -> Result<OperatingControlBatch> {
    Ok(reference_case::build(case(id)?, width)?)
}

pub fn observe(time: f64, stage: f64, lane: usize) -> ControlSample {
    reference_case::sample(
        Sample {
            observed: time,
            available: time,
            stage,
        },
        lane,
    )
}

pub fn accept(
    batch: OperatingControlBatch,
    time: f64,
    stage: f64,
) -> Result<OperatingControlBatch> {
    Ok(batch
        .trial_to(time, &[observe(time, stage, 0)])?
        .into_next())
}

pub fn rejected<T>(result: std::result::Result<T, StructureError>) -> Result<StructureError> {
    result
        .err()
        .ok_or_else(|| "Expected control rejection".into())
}

pub fn malformed_restore(
    destination: &mut OperatingControlBatch,
    data: &OperatingControlSnapshotData,
) -> Result {
    let before = destination.snapshot_data()?;
    let error = rejected(destination.restore_data(data))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition),
        "{error}"
    );
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}
