use super::{fixture::*, references::cases};
use hydr::structures::*;

#[test]
fn immutable_proposals_preserve_original_operations_and_held_products() -> Result {
    for case in cases()
        .into_iter()
        .filter(|case| matches!(case.expected, "active" | "shutoff"))
    {
        for reverse in [false, true] {
            let batch = StructureBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                vec![definition(&case, case.id, reverse)?],
            )?;
            let packets = [case_heads(&case, case.id, reverse)];
            let operations = [running(case.id, case.speed)];
            let proposal = batch.propose_with_operations(&packets, &operations)?;
            let before = proposal.outputs()?;
            let held = proposal.held_volumes(HeldHeadInterval {
                start: 1.0,
                end: 4.0,
            })?;
            let held_before = held.outputs()?;
            for output in &held_before {
                assert_eq!(
                    output.volume_m3.to_bits(),
                    (output.proposal.discharge_m3_per_s * 3.0).to_bits(),
                    "{}",
                    case.id
                );
                assert_eq!(output.proposal.operation, operations[0].operating_point);
            }
            assert_eq!(
                batch
                    .propose_with_operations(&packets, &operations)?
                    .outputs()?,
                before
            );
            let isolated = batch.propose_with_operations(
                &[packet(case.id, HeadSupport::Unknown, HeadSupport::Unknown)],
                &[operation(
                    case.id,
                    StructureOperatingPoint::PumpStoppedIsolated,
                )],
            )?;
            assert_eq!(first(&isolated)?.discharge_m3_per_s, 0.0);
            assert!(
                batch
                    .propose_with_operations(&packets, &[running("foreign", case.speed)])
                    .is_err()
            );
            assert_eq!(proposal.outputs()?, before);
            assert_eq!(held.outputs()?, held_before);
        }
    }
    Ok(())
}

#[test]
fn held_positive_products_reject_overflow_and_underflow_without_changing_proposal() -> Result {
    for (id, duration) in [
        ("reference_zero_head", f64::MAX),
        ("tiny_speed_exact_zero_head", 0.0625),
    ] {
        let case = cases()
            .into_iter()
            .find(|case| case.id == id)
            .ok_or("Missing held product fixture")?;
        let batch = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            vec![definition(&case, id, false)?],
        )?;
        let proposal = batch
            .propose_with_operations(&[case_heads(&case, id, false)], &[running(id, case.speed)])?;
        let before = proposal.outputs()?;
        let error = rejection(proposal.held_volumes(HeldHeadInterval {
            start: 0.0,
            end: duration,
        }))?;
        assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
        assert_eq!(error.element(), Some(0));
        assert_eq!(proposal.outputs()?, before);
    }
    Ok(())
}

#[test]
fn curve_and_applicability_reject_undefined_physical_models() -> Result {
    for (heads, flows) in [
        (vec![0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![1.0, 0.0]),
        (vec![1.0, 0.0], vec![1.0, 0.0]),
        (vec![-f64::MAX, f64::MAX], vec![1.0, 0.0]),
        (vec![0.0, 1.0], vec![-1.0, 0.0]),
        (vec![0.0, 1.0], vec![1.0, 2.0]),
        (vec![0.0, 1.0], vec![0.0, 0.0]),
        (vec![0.0, f64::NAN], vec![1.0, 0.0]),
        (vec![0.0, 1.0], vec![1.0, f64::INFINITY]),
    ] {
        assert!(curve(&heads, &flows).is_err());
    }
    assert!(
        PumpCurve::new(
            vec![
                PumpCurveKnot {
                    head_m: 0.0,
                    discharge_m3_per_s: 1.0
                },
                PumpCurveKnot {
                    head_m: 1.0,
                    discharge_m3_per_s: 0.0
                }
            ],
            String::new()
        )
        .is_err()
    );
    for (minimum_ratio, maximum_ratio) in [
        (0.0, 1.0),
        (2.0, 3.0),
        (0.1, 0.9),
        (2.0, 1.0),
        (1.0, f64::INFINITY),
    ] {
        let policy = PumpSpeedPolicy::DeclaredAffinity {
            minimum_ratio,
            maximum_ratio,
            applicability: "Declared numerical range".into(),
        };
        assert!(
            PumpParameters::new(
                curve(&[0.0, 1.0], &[1.0, 0.0])?,
                -2.0,
                policy,
                applicability()
            )
            .is_err()
        );
    }
    for field in 0..4 {
        let mut evidence = applicability();
        match field {
            0 => evidence.endpoint_stage_head_assertion.clear(),
            1 => evidence.connection_losses_and_velocity_head_account.clear(),
            2 => evidence.stopped_isolation_assertion.clear(),
            _ => evidence.operating_envelope.clear(),
        }
        assert!(
            PumpParameters::new(curve(&[0.0, 1.0], &[1.0, 0.0])?, -2.0, affinity(), evidence)
                .is_err()
        );
    }
    assert!(
        PumpParameters::new(
            curve(&[0.0, 1.0], &[1.0, 0.0])?,
            f64::NAN,
            affinity(),
            applicability()
        )
        .is_err()
    );
    let mut wrong = standard("pump")?;
    wrong.direction = DirectionPolicy::Bidirectional;
    assert!(
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), vec![wrong])
            .is_err()
    );
    Ok(())
}
