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
