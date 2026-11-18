//! Free-outlet lift `max(receiving stage, outlet) - donor stage`, dry delivery, and admission.
//!
//! Every row and tolerance below was frozen before the first run. The curve is
//! `Q_ref(h) = 4 - h / 2` over `[0, 8]` m, so exact binary rows expect exact bits.
use super::fixture::*;
use hydr::structures::*;

/// Relative flow tolerance of inexact rows, the same as the exact-rational corpus.
const TOLERANCE: f64 = 64.0 * f64::EPSILON;
/// Held-volume interval length, in seconds.
const HELD_S: f64 = 3.0;

enum Expected {
    /// Positive flow with exact bits.
    Exact(f64),
    /// Positive flow within `TOLERANCE` of the analytical value.
    Near(f64),
    /// Declared shutoff, zero flow.
    Shutoff,
    /// `UnsupportedRegime` whose message holds this text.
    Support(&'static str),
}

struct Row {
    id: &'static str,
    donor: HeadSupport,
    receiver: HeadSupport,
    outlet: f64,
    speed: f64,
    expected: Expected,
}

fn wet(stage_m: f64) -> HeadSupport {
    HeadSupport::Wet { stage_m }
}

fn dry(bed_elevation_m: f64) -> HeadSupport {
    HeadSupport::Dry { bed_elevation_m }
}

const DRY_DELIVERY: &str = "a known wet receiver or a declared free outlet";
const BURIED_OUTLET: &str = "dry receiver bed lies above its declared free outlet";
const SUCTION: &str = "strictly above its declared minimum suction stage";

fn rows() -> Vec<Row> {
    let row = |id, donor, receiver, outlet, speed, expected| Row {
        id,
        donor,
        receiver,
        outlet,
        speed,
        expected,
    };
    vec![
        row(
            "wet_below_outlet",
            wet(0.0),
            wet(1.0),
            4.0,
            1.0,
            Expected::Exact(2.0),
        ),
        row(
            "wet_above_outlet",
            wet(0.0),
            wet(4.0),
            1.0,
            1.0,
            Expected::Exact(2.0),
        ),
        row(
            "wet_at_outlet",
            wet(0.0),
            wet(2.0),
            2.0,
            1.0,
            Expected::Exact(3.0),
        ),
        row(
            "dry_below_outlet",
            wet(0.0),
            dry(1.0),
            4.0,
            1.0,
            Expected::Exact(2.0),
        ),
        row(
            "dry_at_outlet",
            wet(0.0),
            dry(4.0),
            4.0,
            1.0,
            Expected::Exact(2.0),
        ),
        row(
            "dry_half_speed",
            wet(0.0),
            dry(-1.0),
            1.0,
            0.5,
            Expected::Exact(1.0),
        ),
        row(
            "dry_double_speed",
            wet(0.0),
            dry(-1.0),
            16.0,
            2.0,
            Expected::Exact(4.0),
        ),
        row(
            "dry_shutoff",
            wet(0.0),
            dry(0.0),
            8.0,
            1.0,
            Expected::Shutoff,
        ),
        row(
            "wet_inexact",
            wet(0.1),
            wet(-1.0),
            3.3,
            1.0,
            Expected::Near(2.4),
        ),
        row(
            "dry_inexact",
            wet(0.1),
            dry(-1.0),
            3.3,
            1.0,
            Expected::Near(2.4),
        ),
        row(
            "dry_above_outlet",
            wet(0.0),
            dry(5.0),
            4.0,
            1.0,
            Expected::Support(BURIED_OUTLET),
        ),
        row(
            "dry_donor",
            dry(0.0),
            dry(0.0),
            4.0,
            1.0,
            Expected::Support(DRY_DELIVERY),
        ),
        row(
            "donor_at_suction",
            wet(-2.0),
            dry(0.0),
            4.0,
            1.0,
            Expected::Support(SUCTION),
        ),
    ]
}

fn outlet_parameters(elevation_m: f64) -> Result<PumpParameters> {
    Ok(PumpParameters::new(
        curve(&[0.0, 4.0, 8.0], &[4.0, 2.0, 0.0])?,
        -2.0,
        affinity(),
        applicability(),
    )?
    .with_free_outlet(PumpFreeOutlet {
        elevation_m,
        evidence: "Analytical delivery pipe discharging to atmosphere".into(),
    })?)
}

fn direction(reverse: bool) -> DirectionPolicy {
    if reverse {
        DirectionPolicy::BtoAOnly
    } else {
        DirectionPolicy::AtoBOnly
    }
}

fn batch(
    id: &str,
    law: PumpParameters,
    reverse: bool,
) -> Result<StructureBatch<crate::runtime::Runtime>> {
    Ok(StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![element(id, StructureLaw::Pump(law), direction(reverse))],
    )?)
}

fn oriented(row: &Row, reverse: bool) -> StructureHeads {
    if reverse {
        packet(row.id, row.receiver.clone(), row.donor.clone())
    } else {
        packet(row.id, row.donor.clone(), row.receiver.clone())
    }
}

/// Knuth TwoSum of `a + b`, independent of the runtime kernel.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// The stage the lift subtracts from: the receiver, its dry bed, or the outlet above either.
fn lifted(row: &Row) -> Result<f64> {
    let level = match row.receiver {
        HeadSupport::Wet { stage_m } => stage_m,
        HeadSupport::Dry { bed_elevation_m } => bed_elevation_m,
        HeadSupport::Unknown => return Err("frozen rows hold known support".into()),
    };
    Ok(level.max(row.outlet))
}

fn donor_stage(row: &Row) -> Result<f64> {
    match row.donor {
        HeadSupport::Wet { stage_m } => Ok(stage_m),
        _ => Err("admitted rows hold a wet donor".into()),
    }
}

fn check_admitted(
    row: &Row,
    proposal: &StructureProposal<crate::runtime::Runtime>,
    reverse: bool,
) -> Result {
    let output = first(proposal)?;
    let sign = if reverse { -1.0 } else { 1.0 };
    let q = sign * output.discharge_m3_per_s;
    assert_eq!(output.regime, StructureRegime::PumpCurve, "{}", row.id);
    let reference = match row.expected {
        Expected::Exact(reference) => {
            assert_eq!(q.to_bits(), reference.to_bits(), "{}", row.id);
            reference
        }
        Expected::Near(reference) => {
            let tolerance = TOLERANCE * reference.abs().max(q.abs());
            assert!(
                (q - reference).abs() <= tolerance,
                "{}: {q:e}, reference {reference:e}, tolerance {tolerance:e}",
                row.id
            );
            reference
        }
        Expected::Shutoff => {
            assert_eq!(q, 0.0, "{}", row.id);
            assert_eq!(output.status, StructureStatus::PumpShutoff, "{}", row.id);
            assert_eq!(output.proposed_direction, None, "{}", row.id);
            0.0
        }
        Expected::Support(_) => return Err(format!("{}: rejection row admitted", row.id).into()),
    };
    if reference != 0.0 {
        assert_eq!(output.status, StructureStatus::PumpRunning, "{}", row.id);
        let expected = if reverse {
            FlowDirection::BtoA
        } else {
            FlowDirection::AtoB
        };
        assert_eq!(output.proposed_direction, Some(expected), "{}", row.id);
    }
    let Some(StructureHydraulicOperands::Pump(operands)) = &output.hydraulic_operands else {
        return Err(format!("{}: missing pump operands", row.id).into());
    };
    let (high, low) = two_sum(lifted(row)?, -donor_stage(row)?);
    assert_eq!(
        operands.receiving_minus_donor_head_m().to_bits(),
        high.to_bits(),
        "{}",
        row.id
    );
    assert_eq!(
        operands.receiving_minus_donor_head_low_m().to_bits(),
        low.to_bits(),
        "{}",
        row.id
    );
    let held = proposal.held_volumes(HeldHeadInterval {
        start: 0.0,
        end: HELD_S,
    })?;
    let volume = held
        .outputs()?
        .first()
        .ok_or("missing held volume")?
        .volume_m3;
    let expected_volume = output.discharge_m3_per_s * HELD_S;
    assert_eq!(volume.to_bits(), expected_volume.to_bits(), "{}", row.id);
    if let Expected::Exact(reference) = row.expected {
        assert_eq!(
            volume.to_bits(),
            (sign * reference * HELD_S).to_bits(),
            "{}",
            row.id
        );
    }
    Ok(())
}

#[test]
fn free_outlet_rows_lift_to_the_higher_of_receiver_and_outlet_in_both_orientations() -> Result {
    let rows = rows();
    assert_eq!(rows.len(), 13);
    for row in &rows {
        for reverse in [false, true] {
            let batch = batch(row.id, outlet_parameters(row.outlet)?, reverse)?;
            let result = batch
                .propose_with_operations(&[oriented(row, reverse)], &[running(row.id, row.speed)]);
            match row.expected {
                Expected::Support(text) => {
                    let error = rejection(result)?;
                    assert_eq!(
                        error.input_kind(),
                        Some(StructureInputErrorKind::UnsupportedRegime),
                        "{}: {error}",
                        row.id
                    );
                    assert_eq!(error.element(), Some(0), "{}", row.id);
                    assert!(error.to_string().contains(text), "{}: {error}", row.id);
                }
                _ => check_admitted(row, &result?, reverse)?,
            }
        }
    }
    Ok(())
}

#[test]
fn an_outlet_below_a_wet_receiver_keeps_every_bit_of_the_plain_pump() -> Result {
    for (receiver, outlet) in [(4.0, 1.0), (2.0, 2.0), (3.0, -3.0)] {
        for speed in [1.0, 0.7, 2.0] {
            for reverse in [false, true] {
                let plain = batch(
                    "pump",
                    PumpParameters::new(
                        curve(&[0.0, 4.0, 8.0], &[4.0, 2.0, 0.0])?,
                        -2.0,
                        affinity(),
                        applicability(),
                    )?,
                    reverse,
                )?;
                let free = batch("pump", outlet_parameters(outlet)?, reverse)?;
                let stages = if reverse {
                    (receiver, 0.1)
                } else {
                    (0.1, receiver)
                };
                let heads = [heads("pump", stages.0, stages.1)];
                let operation = [running("pump", speed)];
                let a = first(&plain.propose_with_operations(&heads, &operation)?)?;
                let b = first(&free.propose_with_operations(&heads, &operation)?)?;
                assert_eq!(a, b, "receiver={receiver}, outlet={outlet}, speed={speed}");
                assert_eq!(
                    a.discharge_m3_per_s.to_bits(),
                    b.discharge_m3_per_s.to_bits()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn stopped_isolation_ignores_the_outlet_and_a_plain_pump_still_rejects_dry_delivery() -> Result {
    let free = batch("pump", outlet_parameters(4.0)?, false)?;
    let stopped = operation("pump", StructureOperatingPoint::PumpStoppedIsolated);
    let output = first(&free.propose_with_operations(
        &[packet("pump", HeadSupport::Unknown, dry(100.0))],
        &[stopped],
    )?)?;
    assert_eq!(output.status, StructureStatus::PumpStoppedIsolated);
    assert_eq!(output.discharge_m3_per_s, 0.0);
    let plain = batch("pump", standard_parameters()?, false)?;
    let error = rejection(plain.propose_with_operations(
        &[packet("pump", wet(0.0), dry(0.0))],
        &[running("pump", 1.0)],
    ))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::UnsupportedRegime)
    );
    assert!(error.to_string().contains(DRY_DELIVERY), "{error}");
    Ok(())
}

fn standard_parameters() -> Result<PumpParameters> {
    Ok(PumpParameters::new(
        curve(&[0.0, 4.0, 8.0], &[4.0, 2.0, 0.0])?,
        -2.0,
        affinity(),
        applicability(),
    )?)
}

#[test]
fn outlet_definition_needs_a_finite_elevation_and_evidence() -> Result {
    for (elevation_m, evidence) in [
        (f64::NAN, "Free outlet"),
        (f64::INFINITY, "Free outlet"),
        (-f64::INFINITY, "Free outlet"),
        (1.0, ""),
        (1.0, "   "),
        (1.0, "Free\noutlet"),
    ] {
        let error = rejection(standard_parameters()?.with_free_outlet(PumpFreeOutlet {
            elevation_m,
            evidence: evidence.into(),
        }))?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::Definition),
            "{error}"
        );
    }
    let first_outlet = PumpFreeOutlet {
        elevation_m: 1.0,
        evidence: "First outlet".into(),
    };
    let second_outlet = PumpFreeOutlet {
        elevation_m: 2.0,
        evidence: "Second outlet".into(),
    };
    let parameters = standard_parameters()?;
    assert_eq!(parameters.free_outlet(), None);
    let replaced = parameters
        .with_free_outlet(first_outlet)?
        .with_free_outlet(second_outlet.clone())?;
    assert_eq!(replaced.free_outlet(), Some(&second_outlet));
    Ok(())
}
