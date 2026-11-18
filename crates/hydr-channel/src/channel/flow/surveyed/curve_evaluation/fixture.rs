use super::super::PrismaticRiemannConfig;
use super::super::curves::{prepare_side, select_side};
use super::super::equation_ops::choose;
use super::super::error::{HydraulicResult, checked};
use super::super::integration::potential;
use super::super::potential_difference::direct_increment;
use super::super::potential_difference_test_data::PROFILES;
use super::super::potential_difference_test_support::{batch, values};
use super::super::types::{Oracle, SideState};
use super::super::work::{PrismaticRiemannWork, WorkLedger};
use super::{PairEvaluation, select_state};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(super) struct Fixture<R: Runtime<DType = DType>> {
    pub(super) geometry: HydraulicGeometry<R>,
    pub(super) bands: GatheredBands<R>,
    pub(super) left: SideState<R>,
    pub(super) right: SideState<R>,
    pub(super) state: Constitutive<R>,
    pub(super) config: PrismaticRiemannConfig,
    pub(super) work: WorkLedger,
}

pub(super) fn fixture<R>(
    client: &R::Client,
    left: &[f64],
    right: &[f64],
    depths: [&[f64]; 2],
    rows: &[i64],
) -> HydraulicResult<Fixture<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let config = PrismaticRiemannConfig::default();
    let work = WorkLedger::new(&config);
    let batch = batch::<R>(client, &PROFILES[4])?;
    let geometry = HydraulicGeometry::from_batch(&batch);
    let owners = values::<R, _>(client, &vec![0_i64; left.len()], false)?;
    let bands = checked(client, "curve scheduling original owners", |a| {
        geometry.gather(a, &owners)
    })?;
    let left_area = values::<R, _>(client, left, false)?;
    let right_area = values::<R, _>(client, right, false)?;
    let discharge = values::<R, _>(client, &vec![0.0; left.len()], false)?;
    let left = prepare_side(
        &geometry,
        &bands,
        &left_area,
        &discharge,
        &config,
        config.initial_simpson_panels,
        &work,
    )?;
    let right = prepare_side(
        &geometry,
        &bands,
        &right_area,
        &discharge,
        &config,
        config.initial_simpson_panels,
        &work,
    )?;
    let lower: Vec<_> = depths[0].iter().map(|depth| depth / 4.0).collect();
    let upper: Vec<_> = depths[1].iter().map(|depth| depth / 4.0).collect();
    let lower = values::<R, _>(client, &lower, false)?;
    let upper = values::<R, _>(client, &upper, false)?;
    let rows = values::<R, _>(client, rows, false)?;
    let (bands, state, left, right) =
        checked(client, "curve scheduling repeated query rows", |a| {
            let normalized = a.bounds(&lower, &upper)?;
            let state = geometry.at_normalized_depth(a, &bands, &normalized)?;
            Ok((
                bands.select(a, &rows)?,
                select_state(a, &state, &rows)?,
                select_side(a, &left, &rows)?,
                select_side(a, &right, &rows)?,
            ))
        })?;
    Ok(Fixture {
        geometry,
        bands,
        left,
        right,
        state,
        config,
        work,
    })
}

pub(super) fn oracle<R: Runtime<DType = DType>>(fixture: &Fixture<R>) -> Oracle<'_, R> {
    Oracle {
        geometry: &fixture.geometry,
        bands: &fixture.bands,
        left: &fixture.left,
        right: &fixture.right,
        config: &fixture.config,
        panels: fixture.config.initial_simpson_panels,
        work: &fixture.work,
    }
}

pub(super) fn callback_delta(
    before: PrismaticRiemannWork,
    after: PrismaticRiemannWork,
) -> [usize; 2] {
    [
        after.simpson_value_points - before.simpson_value_points,
        after.simpson_derivative_points - before.simpson_derivative_points,
    ]
}

pub(super) fn eager<R>(fixture: &Fixture<R>) -> HydraulicResult<(PairEvaluation<R>, [Vec<f64>; 2])>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let work = WorkLedger::new(&fixture.config);
    let panels = fixture.config.initial_simpson_panels;
    let phi = potential(
        &fixture.geometry,
        &fixture.bands,
        &fixture.state,
        &fixture.config,
        panels,
        &work,
    )?;
    let left = direct_increment(
        &fixture.geometry,
        &fixture.bands,
        &fixture.left.original_area,
        &fixture.state.area,
        &fixture.config,
        panels,
        &work,
    )?;
    let right = direct_increment(
        &fixture.geometry,
        &fixture.bands,
        &fixture.right.original_area,
        &fixture.state.area,
        &fixture.config,
        panels,
        &work,
    )?;
    let available = [
        left.available.try_to_vec::<f64>()?,
        right.available.try_to_vec::<f64>()?,
    ];
    let (left_increment, right_increment) =
        checked(fixture.geometry.client(), "eager curve composition", |a| {
            let left_full = a.sub(&phi, &fixture.left.potential)?;
            let right_full = a.sub(&phi, &fixture.right.potential)?;
            Ok((
                choose(a, &left.available, &left.bounds, &left_full)?,
                choose(a, &right.available, &right.bounds, &right_full)?,
            ))
        })?;
    Ok((
        PairEvaluation {
            dry_potential: phi,
            left_increment,
            right_increment,
        },
        available,
    ))
}

pub(super) fn endpoints<R: Runtime<DType = DType>>(
    interval: &Interval<R>,
) -> HydraulicResult<[Vec<f64>; 2]> {
    Ok([
        interval.lower().try_to_vec::<f64>()?,
        interval.upper().try_to_vec::<f64>()?,
    ])
}

pub(super) fn matches_eager<R>(
    fixture: &Fixture<R>,
    actual: &PairEvaluation<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let (expected, available) = eager(fixture)?;
    let lower = fixture.state.area.lower().try_to_vec::<f64>()?;
    let originals = [
        fixture.left.original_area.try_to_vec::<f64>()?,
        fixture.right.original_area.try_to_vec::<f64>()?,
    ];
    let actual_increments = [
        endpoints(&actual.left_increment)?,
        endpoints(&actual.right_increment)?,
    ];
    let expected_increments = [
        endpoints(&expected.left_increment)?,
        endpoints(&expected.right_increment)?,
    ];
    let actual_phi = endpoints(&actual.dry_potential)?;
    let expected_phi = endpoints(&expected.dry_potential)?;
    for (row, &area_lower) in lower.iter().enumerate() {
        let mut phi_needed = originals.iter().any(|side| side[row] == 0.0);
        for side in 0..2 {
            let needed = originals[side][row] > 0.0 && area_lower <= originals[side][row];
            phi_needed |= needed && available[side][row] == 0.0;
            for endpoint in 0..2 {
                let expected = if needed {
                    expected_increments[side][endpoint][row]
                } else {
                    0.0
                };
                assert_eq!(
                    actual_increments[side][endpoint][row], expected,
                    "row={row}, side={side}, endpoint={endpoint}"
                );
            }
        }
        for endpoint in 0..2 {
            let expected = if phi_needed {
                expected_phi[endpoint][row]
            } else {
                0.0
            };
            assert_eq!(
                actual_phi[endpoint][row], expected,
                "full potential row={row}, endpoint={endpoint}"
            );
        }
    }
    Ok(())
}
