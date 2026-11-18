use super::boundary_common::{Side, TestResult, definition, initial, interval};
use super::fixtures::{FIXTURES, reservoir};
use super::snapshot_bits::assert_same;
use super::transport::{native, overwrite_aliases};
use hydr::channel::*;
use numr::dtype::DType;
use numr::ops::NextRepresentableOps;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, MidpointOps,
    ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
};

pub fn execute<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + NextRepresentableOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    for fixture in FIXTURES {
        for side in [Side::Left, Side::Right] {
            for view in 0..3 {
                let origin = ChannelReach::<R>::new(
                    client.clone(),
                    definition(fixture.shape, 3, 128.0, 4.0, 1.0),
                    initial(fixture.shape, &[128.0; 3], &[1.0; 3], &[0.0; 3]),
                )?;
                let before = origin.snapshot_data()?;
                let dt = 1.0 / 1024.0;
                let mut held = interval(side, fixture.boundary(), 0.0, dt);
                if fixture.loss == 0.0 && view == 1 {
                    let boundary = reservoir(fixture.elevation, -0.0);
                    if side.index() == 0 {
                        held.left = boundary;
                    } else {
                        held.right = boundary;
                    }
                }
                let tensor = native::<R>(&client, &held, view)?;
                let host = origin.step(dt, &held)?.next;
                let current = origin.step_tensor(dt, &tensor)?.next;
                let saved = current.snapshot_data()?;
                assert_same(&saved, &host.snapshot_data()?);
                let receipt = current
                    .last_step()?
                    .ok_or("Missing native reservoir receipt")?;
                let trace = receipt.boundary_traces[side.index()]
                    .as_ref()
                    .ok_or("Missing native reservoir trace")?;
                for (actual, reference) in [
                    (trace.depth_m, fixture.depth),
                    (trace.outward_velocity_m_per_s, fixture.velocity),
                    (trace.outward_discharge_m3_per_s, fixture.discharge),
                    (trace.momentum_flux_m4_per_s2, fixture.momentum),
                ] {
                    assert!(
                        (actual - reference).abs() <= 1e-12 + 1e-12 * reference.abs(),
                        "{}: actual={actual:e}, independent reference={reference:e}",
                        fixture.name
                    );
                }
                let velocity = trace.outward_velocity_m_per_s;
                let dissipation = trace.outward_discharge_m3_per_s
                    * (fixture.shape.gravity() * (trace.depth_m - fixture.elevation)
                        + velocity * velocity / 2.0);
                assert!(
                    (dissipation - fixture.dissipation).abs()
                        <= 1e-10 + 1e-12 * fixture.dissipation.abs(),
                    "{}: dissipation={dissipation:e}, independent reference={:e}",
                    fixture.name,
                    fixture.dissipation
                );
                let certificate = trace
                    .reservoir
                    .as_ref()
                    .ok_or("Missing native reservoir certificate")?;
                assert!(certificate.depth_lower_m <= f64::from_bits(fixture.root_floor_bits));
                assert!(certificate.depth_upper_m >= f64::from_bits(fixture.root_ceil_bits));
                let checkpoint = current.snapshot();
                overwrite_aliases::<R>(&client, &tensor)?;
                assert_same(&current.snapshot_data()?, &saved);
                assert_same(&checkpoint.data()?, &saved);
                let trusted = origin.restore(&checkpoint)?;
                let cold = origin.import_snapshot(&saved)?;
                assert_same(&trusted.snapshot_data()?, &saved);
                assert_same(&cold.snapshot_data()?, &saved);
                held.start_s = dt;
                held.end_s = 2.0 * dt;
                let fresh = native::<R>(&client, &held, (view + 1) % 3)?;
                let expected = current
                    .step_tensor(2.0 * dt, &fresh)?
                    .next
                    .snapshot_data()?;
                assert_same(
                    &host.step(2.0 * dt, &held)?.next.snapshot_data()?,
                    &expected,
                );
                for restarted in [trusted, cold] {
                    assert_same(
                        &restarted
                            .step_tensor(2.0 * dt, &fresh)?
                            .next
                            .snapshot_data()?,
                        &expected,
                    );
                }
                assert_same(&origin.snapshot_data()?, &before);
            }
        }
    }
    for fixture in FIXTURES.into_iter().take(2) {
        let origin = ChannelReach::<R>::new(
            client.clone(),
            definition(fixture.shape, 3, 128.0, 4.0, 1.0),
            initial(fixture.shape, &[128.0; 3], &[1.0; 3], &[0.0; 3]),
        )?;
        let policies = [
            ChannelBoundary::Stage { elevation_m: 0.81 },
            ChannelBoundary::Discharge {
                discharge_m3_per_s: -0.0,
            },
            fixture.boundary(),
        ];
        for (left_index, left) in policies.iter().enumerate() {
            for (right_index, right) in policies.iter().enumerate() {
                let dt = 1.0 / 1024.0;
                let held = ChannelForcingInterval {
                    start_s: 0.0,
                    end_s: dt,
                    left: left.clone(),
                    right: right.clone(),
                    lateral_inputs: Vec::new(),
                };
                let tensor = native::<R>(&client, &held, (left_index + right_index) % 3)?;
                let current = origin.step_tensor(dt, &tensor)?.next;
                let saved = current.snapshot_data()?;
                assert_same(&saved, &origin.step(dt, &held)?.next.snapshot_data()?);
                assert_same(
                    &origin.restore(&current.snapshot())?.snapshot_data()?,
                    &saved,
                );
                assert_same(&origin.import_snapshot(&saved)?.snapshot_data()?, &saved);
            }
        }
    }
    Ok(())
}

/// Forged snapshot fields must reject without changing the origin reach.
pub fn forged_snapshots_reject() -> TestResult {
    for fixture in [FIXTURES[3], FIXTURES[5]] {
        let origin = ChannelReach::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            definition(fixture.shape, 1, 128.0, 4.0, 1.0),
            initial(fixture.shape, &[128.0], &[1.0], &[0.0]),
        )?;
        let before = origin.snapshot_data()?;
        let dt = 1.0 / 1024.0;
        let current = origin
            .step(dt, &interval(Side::Right, fixture.boundary(), 0.0, dt))?
            .next;
        let saved = current.snapshot_data()?;
        for mutation in 0..10 {
            let mut altered = saved.clone();
            if mutation < 2 {
                let controls = &mut altered.definition.config.boundary;
                let value = if mutation == 0 {
                    &mut controls.reservoir_head_absolute_m
                } else {
                    &mut controls.reservoir_head_relative
                };
                *value = f64::from_bits(value.to_bits() + 1);
            } else {
                let last = altered
                    .last_interval
                    .as_mut()
                    .ok_or("Missing saved reservoir interval")?;
                if mutation < 4 {
                    last.forcing.right = reservoir(
                        fixture.elevation + if mutation == 2 { 0.01 } else { 0.0 },
                        fixture.loss + if mutation == 3 { 0.01 } else { 0.0 },
                    );
                } else {
                    let trace = last.diagnostics.boundary_traces[1]
                        .as_mut()
                        .ok_or("Missing saved reservoir trace")?;
                    if mutation == 9 {
                        trace.reservoir = None;
                    } else {
                        let certificate = trace
                            .reservoir
                            .as_mut()
                            .ok_or("Missing saved reservoir certificate")?;
                        let value = match mutation {
                            4 => &mut certificate.depth_lower_m,
                            5 => &mut certificate.depth_upper_m,
                            6 => &mut certificate.head_compatibility_abs_bound_m,
                            7 => &mut certificate.dissipation_head_lower_m,
                            _ => &mut certificate.dissipation_head_upper_m,
                        };
                        *value = f64::from_bits(value.to_bits() + 1);
                    }
                }
            }
            assert!(
                origin.import_snapshot(&altered).is_err(),
                "Accepted forged reservoir field {mutation}"
            );
            assert_same(&origin.snapshot_data()?, &before);
        }
    }
    Ok(())
}
