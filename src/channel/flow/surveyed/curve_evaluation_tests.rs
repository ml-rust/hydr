use super::PrismaticRiemannConfig;
use super::curve_evaluation::{PairEvaluation, pair_evaluation, select_state};
use super::curves::{prepare_side, select_side};
use super::equation_ops::choose;
use super::error::{HydraulicResult, PrismaticRiemannError, checked, into_public};
use super::integration::potential;
use super::potential_difference::direct_increment;
use super::potential_difference_test_data::PROFILES;
use super::potential_difference_test_support::{batch, values};
use super::types::{Oracle, SideState};
use super::work::{PrismaticRiemannWork, WorkLedger};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;

struct Fixture<R: Runtime<DType = DType>> {
    geometry: HydraulicGeometry<R>,
    bands: GatheredBands<R>,
    left: SideState<R>,
    right: SideState<R>,
    state: Constitutive<R>,
    config: PrismaticRiemannConfig,
    work: WorkLedger,
}

fn fixture<R>(
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

fn oracle<R: Runtime<DType = DType>>(fixture: &Fixture<R>) -> Oracle<'_, R> {
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

fn callback_delta(before: PrismaticRiemannWork, after: PrismaticRiemannWork) -> [usize; 2] {
    [
        after.simpson_value_points - before.simpson_value_points,
        after.simpson_derivative_points - before.simpson_derivative_points,
    ]
}

fn eager<R>(fixture: &Fixture<R>) -> HydraulicResult<(PairEvaluation<R>, [Vec<f64>; 2])>
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

fn endpoints<R: Runtime<DType = DType>>(interval: &Interval<R>) -> HydraulicResult<[Vec<f64>; 2]> {
    Ok([
        interval.lower().try_to_vec::<f64>()?,
        interval.upper().try_to_vec::<f64>()?,
    ])
}

fn matches_eager<R>(fixture: &Fixture<R>, actual: &PairEvaluation<R>) -> HydraulicResult<()>
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

fn scheduling<R>(client: R::Client) -> HydraulicResult<()>
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
    let shock = fixture::<R>(&client, &[9.0], &[10.0], [&[2.2], &[2.2]], &[0])?;
    let before = shock.work.snapshot();
    let result = pair_evaluation(
        &oracle(&shock),
        &shock.bands,
        &shock.state,
        &shock.left,
        &shock.right,
    )?;
    assert_eq!(callback_delta(before, shock.work.snapshot()), [0, 0]);
    matches_eager(&shock, &result)?;

    let direct = fixture::<R>(&client, &[10.0], &[9.0], [&[1.99], &[1.99]], &[0])?;
    let before = direct.work.snapshot();
    let result = pair_evaluation(
        &oracle(&direct),
        &direct.bands,
        &direct.state,
        &direct.left,
        &direct.right,
    )?;
    let actual_work = callback_delta(before, direct.work.snapshot());
    let expected_work = WorkLedger::new(&direct.config);
    let expected = direct_increment(
        &direct.geometry,
        &direct.bands,
        &direct.left.original_area,
        &direct.state.area,
        &direct.config,
        direct.config.initial_simpson_panels,
        &expected_work,
    )?;
    assert_eq!(expected.available.try_to_vec::<f64>()?, [1.0]);
    let counts = callback_delta(Default::default(), expected_work.snapshot());
    assert!(counts[0] > 0 && counts[1] > 0);
    assert_eq!(
        actual_work, counts,
        "direct rarefaction plus shock runs direct callbacks only"
    );
    matches_eager(&direct, &result)?;

    let fallback = fixture::<R>(&client, &[10.0], &[12.0], [&[0.25], &[0.25]], &[0])?;
    let before = fallback.work.snapshot();
    let result = pair_evaluation(
        &oracle(&fallback),
        &fallback.bands,
        &fallback.state,
        &fallback.left,
        &fallback.right,
    )?;
    let actual_work = callback_delta(before, fallback.work.snapshot());
    let expected_work = WorkLedger::new(&fallback.config);
    let _phi = potential(
        &fallback.geometry,
        &fallback.bands,
        &fallback.state,
        &fallback.config,
        fallback.config.initial_simpson_panels,
        &expected_work,
    )?;
    let counts = callback_delta(Default::default(), expected_work.snapshot());
    assert!(counts[0] > 0 && counts[1] > 0);
    assert_eq!(
        actual_work, counts,
        "two fallback rarefactions share one target potential"
    );
    let (_, available) = eager(&fallback)?;
    assert_eq!(available, [vec![0.0], vec![0.0]]);
    matches_eager(&fallback, &result)?;

    let mixed = fixture::<R>(
        &client,
        &[10.0, 9.0, 10.0, 0.0, 10.0],
        &[9.0, 10.0, 12.0, 10.0, 10.0],
        [
            &[1.99, 2.2, 0.25, 0.25, 1.99],
            &[1.99, 2.2, 0.25, 0.25, 2.01],
        ],
        &[4, 0, 3, 2, 0, 1, 4, 2],
    )?;
    let result = pair_evaluation(
        &oracle(&mixed),
        &mixed.bands,
        &mixed.state,
        &mixed.left,
        &mixed.right,
    )?;
    matches_eager(&mixed, &result)?;
    let left = endpoints(&result.left_increment)?;
    let right = endpoints(&result.right_increment)?;
    for interval in [left, right] {
        assert!(
            interval[0][0] < 0.0 && interval[1][0] > 0.0,
            "ambiguous area interval retains both increment signs"
        );
        assert_eq!(interval[0][0], interval[0][6]);
        assert_eq!(interval[1][0], interval[1][6]);
        assert_eq!(interval[0][3], interval[0][7]);
        assert_eq!(interval[1][3], interval[1][7]);
    }
    Ok(())
}

#[test]
fn curve_demand_schedules_only_required_potentials_cpu() -> Result<(), PrismaticRiemannError> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    scheduling::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
}

#[cfg(feature = "cuda")]
#[test]
fn curve_demand_schedules_only_required_potentials_cuda() -> Result<(), Box<dyn std::error::Error>>
{
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    scheduling::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        .map_err(|error| into_public(error, Default::default()))?;
    Ok(())
}
