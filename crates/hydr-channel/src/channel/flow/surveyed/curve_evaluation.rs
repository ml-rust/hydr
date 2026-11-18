#[cfg(test)]
mod fixture;

use super::equation_ops::{scalar, select_interval};
use super::error::{Arithmetic, HydraulicResult, checked};
use super::integration::potential;
use super::potential_difference::{DirectIncrement, direct_increment};
use super::types::{Oracle, SideState};
use crate::channel::section::{Constitutive, GatheredBands};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(crate) struct PairEvaluation<R: Runtime<DType = DType>> {
    pub dry_potential: Interval<R>,
    pub left_increment: Interval<R>,
    pub right_increment: Interval<R>,
}

pub(crate) fn pair_evaluation<R>(
    oracle: &Oracle<'_, R>,
    bands: &GatheredBands<R>,
    state: &Constitutive<R>,
    left: &SideState<R>,
    right: &SideState<R>,
) -> HydraulicResult<PairEvaluation<R>>
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
    let c = oracle.geometry.client();
    let (left_needed, right_needed) = checked(c, "surveyed curve demand", |a| {
        let zero = scalar(a, 0.0)?;
        let left_shock = c.gt(state.area.lower(), &left.original_area)?;
        let right_shock = c.gt(state.area.lower(), &right.original_area)?;
        Ok((
            c.mul(&left.wet, &c.eq(&left_shock, zero.lower())?)?,
            c.mul(&right.wet, &c.eq(&right_shock, zero.lower())?)?,
        ))
    })?;
    let left_direct = selected_direct(oracle, bands, &state.area, left, &left_needed)?;
    let right_direct = selected_direct(oracle, bands, &state.area, right, &right_needed)?;
    let (left_fallback, right_fallback, full_needed) =
        checked(c, "surveyed curve fallback demand", |a| {
            let zero = scalar(a, 0.0)?;
            let left_fallback =
                c.mul(&left_needed, &c.eq(&left_direct.available, zero.lower())?)?;
            let right_fallback =
                c.mul(&right_needed, &c.eq(&right_direct.available, zero.lower())?)?;
            let dry = c.maximum(&left.dry, &right.dry)?;
            let full_needed = c.maximum(&dry, &c.maximum(&left_fallback, &right_fallback)?)?;
            Ok((left_fallback, right_fallback, full_needed))
        })?;
    let phi = selected_potential(oracle, bands, state, &full_needed)?;
    let left_increment =
        complete_increment(oracle, left_direct.bounds, &left_fallback, &phi, left)?;
    let right_increment =
        complete_increment(oracle, right_direct.bounds, &right_fallback, &phi, right)?;
    Ok(PairEvaluation {
        dry_potential: phi,
        left_increment,
        right_increment,
    })
}

pub(crate) fn fan_increment<R>(
    oracle: &Oracle<'_, R>,
    bands: &GatheredBands<R>,
    state: &Constitutive<R>,
    side: &SideState<R>,
) -> HydraulicResult<Interval<R>>
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
    let c = oracle.geometry.client();
    let direct = direct_increment(
        oracle.geometry,
        bands,
        &side.original_area,
        &state.area,
        oracle.config,
        oracle.panels,
        oracle.work,
    )?;
    let fallback = checked(c, "surveyed fan fallback demand", |a| {
        let zero = scalar(a, 0.0)?;
        Ok(c.eq(&direct.available, zero.lower())?)
    })?;
    let phi = selected_potential(oracle, bands, state, &fallback)?;
    complete_increment(oracle, direct.bounds, &fallback, &phi, side)
}

fn selected_direct<R>(
    oracle: &Oracle<'_, R>,
    bands: &GatheredBands<R>,
    area: &Interval<R>,
    side: &SideState<R>,
    needed: &Tensor<R>,
) -> HydraulicResult<DirectIncrement<R>>
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
    let c = oracle.geometry.client();
    let rows = c.nonzero(needed)?.flatten()?;
    let empty = Tensor::zeros(&[side.original_area.numel()], DType::F64, c.device())?;
    if rows.numel() == 0 {
        return checked(c, "surveyed absent curve increment", |a| {
            Ok(DirectIncrement {
                bounds: a.point(&empty)?,
                available: empty,
            })
        });
    }
    let (selected_bands, selected_area, original) =
        checked(c, "surveyed selected curve increment", |a| {
            Ok((
                bands.select(a, &rows)?,
                select_interval(a, area, &rows)?,
                c.index_select(&side.original_area, 0, &rows)?,
            ))
        })?;
    let direct = direct_increment(
        oracle.geometry,
        &selected_bands,
        &original,
        &selected_area,
        oracle.config,
        oracle.panels,
        oracle.work,
    )?;
    checked(c, "surveyed curve increment scatter", |a| {
        let dense = a.point(&empty)?;
        Ok(DirectIncrement {
            bounds: scatter(a, &dense, &rows, &direct.bounds)?,
            available: c.index_put(&empty, 0, &rows, &direct.available)?,
        })
    })
}

fn selected_potential<R>(
    oracle: &Oracle<'_, R>,
    bands: &GatheredBands<R>,
    state: &Constitutive<R>,
    needed: &Tensor<R>,
) -> HydraulicResult<Interval<R>>
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
    let c = oracle.geometry.client();
    let rows = c.nonzero(needed)?.flatten()?;
    let empty = Tensor::zeros(&[state.area.lower().numel()], DType::F64, c.device())?;
    if rows.numel() == 0 {
        return checked(
            c,
            "surveyed absent full potential",
            |a| Ok(a.point(&empty)?),
        );
    }
    let (selected_bands, selected_state) = checked(c, "surveyed selected full potential", |a| {
        Ok((bands.select(a, &rows)?, select_state(a, state, &rows)?))
    })?;
    let phi = potential(
        oracle.geometry,
        &selected_bands,
        &selected_state,
        oracle.config,
        oracle.panels,
        oracle.work,
    )?;
    checked(c, "surveyed full potential scatter", |a| {
        let dense = a.point(&empty)?;
        scatter(a, &dense, &rows, &phi)
    })
}

fn complete_increment<R>(
    oracle: &Oracle<'_, R>,
    direct: Interval<R>,
    fallback: &Tensor<R>,
    phi: &Interval<R>,
    side: &SideState<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = oracle.geometry.client();
    let rows = c.nonzero(fallback)?.flatten()?;
    if rows.numel() == 0 {
        return Ok(direct);
    }
    checked(c, "surveyed fallback potential difference", |a| {
        let target = select_interval(a, phi, &rows)?;
        let original = select_interval(a, &side.potential, &rows)?;
        let difference = a.sub(&target, &original)?;
        scatter(a, &direct, &rows, &difference)
    })
}

fn scatter<R>(
    a: &mut Arithmetic<'_, R>,
    dense: &Interval<R>,
    rows: &Tensor<R>,
    values: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(a.bounds(
        &c.index_put(dense.lower(), 0, rows, values.lower())?,
        &c.index_put(dense.upper(), 0, rows, values.upper())?,
    )?)
}

pub(crate) fn select_state<R>(
    a: &mut Arithmetic<'_, R>,
    state: &Constitutive<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<Constitutive<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(Constitutive {
        depth: select_interval(a, &state.depth, rows)?,
        forward_area: select_interval(a, &state.forward_area, rows)?,
        area: select_interval(a, &state.area, rows)?,
        top_width: select_interval(a, &state.top_width, rows)?,
        integral: select_interval(a, &state.integral, rows)?,
        hydraulic_depth: select_interval(a, &state.hydraulic_depth, rows)?,
        wet_possible: c.index_select(&state.wet_possible, 0, rows)?,
        dry_possible: c.index_select(&state.dry_possible, 0, rows)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::error::{PrismaticRiemannError, into_public};
    use super::super::work::WorkLedger;
    use super::fixture::{callback_delta, eager, endpoints, fixture, matches_eager, oracle};
    use super::*;
    use numr::ops::{HypotOps, MidpointOps};

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
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        scheduling::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn curve_demand_schedules_only_required_potentials_cuda()
    -> Result<(), Box<dyn std::error::Error>> {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        scheduling::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
            .map_err(|error| into_public(error, Default::default()))?;
        Ok(())
    }
}
