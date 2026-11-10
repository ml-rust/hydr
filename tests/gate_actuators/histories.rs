use super::{fixture::Result, inputs, reference_context};
use hydr::structures::*;

fn query(
    case: &inputs::Case,
    time: f64,
) -> std::result::Result<GateActuatorBatch<crate::runtime::Runtime>, StructureError> {
    let mut batch = reference_context::build(case, 1)?;
    while batch.time() < time {
        let end = batch
            .next_event()?
            .map_or(time, |event| event.time_s.min(time));
        assert!(
            end > batch.time(),
            "Nonadvancing actuator event for {}",
            case.id
        );
        let packets = reference_context::updates(case, 1, end);
        batch = batch.trial_to(end, &packets)?.into_next();
    }
    Ok(batch)
}

#[test]
fn original_public_histories_keep_frozen_admission_categories_through_legal_event_walking() -> Result
{
    let cases = inputs::cases();
    assert_eq!(cases.len(), 20);
    for case in cases {
        assert!(!case.family.is_empty());
        match reference_context::build(&case, 1) {
            Ok(_) => assert_eq!(case.binding_expected, "Success", "{}", case.id),
            Err(error) => {
                assert_eq!(
                    reference_context::category(&error),
                    case.binding_expected,
                    "{}: {error}",
                    case.id
                );
                continue;
            }
        }
        for expected in case.queries {
            match query(&case, expected.time) {
                Ok(batch) => {
                    assert_eq!(
                        expected.expected, "Success",
                        "{}@{}",
                        case.id, expected.time
                    );
                    assert_eq!(batch.time().to_bits(), expected.time.to_bits());
                    let output = batch.outputs()?.remove(0);
                    let position = output.position;
                    assert!(position.opening_m.is_finite());
                    assert!(position.lower_opening_m >= 0.0);
                    assert!(position.lower_opening_m <= position.opening_m);
                    assert!(position.opening_m <= position.upper_opening_m);
                    assert!(position.upper_opening_m <= case.maximum);
                    assert!(position.absolute_error_m.is_finite());
                    if position.upper_opening_m > 0.0 {
                        assert!(position.opening_m > 0.0);
                    }
                    if output.support == GateActuatorSupport::ExactClosed {
                        assert_eq!(position.lower_opening_m, 0.0);
                        assert_eq!(position.upper_opening_m, 0.0);
                        assert_eq!(position.absolute_error_m.to_bits(), 0);
                    }
                    let mut restored = reference_context::build(&case, 1)?;
                    restored.restore_data(&batch.snapshot_data()?)?;
                    assert_eq!(restored.snapshot_data()?, batch.snapshot_data()?);
                }
                Err(error) => assert_eq!(
                    reference_context::category(&error),
                    expected.expected,
                    "{}@{}: {error}",
                    case.id,
                    expected.time
                ),
            }
        }
    }
    Ok(())
}
