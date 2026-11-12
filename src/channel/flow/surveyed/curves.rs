use super::curve_evaluation::{fan_increment, pair_evaluation, select_state};
use super::equation_ops::{choose, hull, reject_mask, scalar, scale, select_interval};
use super::error::{Arithmetic, HydraulicResult, checked};
use super::integration::potential;
use super::pressure::{pressure, pressure_secant};
use super::shock;
use super::types::{Oracle, SideState, WaveSide};
use super::work::WorkLedger;
use super::{PrismaticRiemannConfig, PrismaticRiemannRejection};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(crate) fn sound<R>(
    a: &mut Arithmetic<'_, R>,
    state: &Constitutive<R>,
    gravity: f64,
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
    let squared = scale(a, &state.hydraulic_depth, gravity)?;
    Ok(a.sqrt_nonnegative(&squared)?)
}

pub(crate) fn prepare_side<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    area: &Tensor<R>,
    discharge: &Tensor<R>,
    config: &PrismaticRiemannConfig,
    panels: usize,
    work: &WorkLedger,
) -> HydraulicResult<SideState<R>>
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
    let c = geometry.client();
    let state = checked(c, "surveyed side recovery", |a| {
        geometry.recover(a, bands, area, &config.recovery)
    })?;
    let phi = potential(geometry, bands, &state, config, panels, work)?;
    checked(c, "surveyed side state", |a| {
        let zero = scalar(a, 0.0)?;
        let one = scalar(a, 1.0)?;
        let wet = c.gt(area, zero.upper())?;
        let dry = c.eq(area, zero.lower())?;
        let exact_area = a.point(area)?;
        let denominator = choose(a, &wet, &exact_area, &one)?;
        let q = a.point(discharge)?;
        let velocity = a.div_positive_interval(&q, &denominator)?;
        let velocity = choose(a, &wet, &velocity, &zero)?;
        let sound = sound(a, &state, config.gravity_m_per_s2)?;
        let pressure = pressure(a, &state, config.gravity_m_per_s2)?;
        Ok(SideState {
            original_area: area.clone(),
            original_discharge: discharge.clone(),
            geometry: state,
            velocity,
            sound,
            pressure,
            potential: phi,
            wet,
            dry,
        })
    })
}

pub(crate) fn star_residual<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
    t: &Tensor<R>,
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
    let (bands, state, left, right) = checked(c, "surveyed star query geometry", |a| {
        let bands = oracle.bands.select(a, face_owner)?;
        let t = a.point(t)?;
        let state = oracle.geometry.at_normalized_depth(a, &bands, &t)?;
        Ok((
            bands,
            state,
            select_side(a, oracle.left, face_owner)?,
            select_side(a, oracle.right, face_owner)?,
        ))
    })?;
    let evaluated = pair_evaluation(oracle, &bands, &state, &left, &right)?;
    checked(c, "surveyed star equation", |a| {
        let left_curve = side_curve(
            a,
            oracle,
            &bands,
            &state,
            &evaluated.dry_potential,
            &evaluated.left_increment,
            &left,
        )?;
        let right_curve = side_curve(
            a,
            oracle,
            &bands,
            &state,
            &evaluated.dry_potential,
            &evaluated.right_increment,
            &right,
        )?;
        let waves = a.add(&left_curve, &right_curve)?;
        let jump = a.sub(&right.velocity, &left.velocity)?;
        Ok(a.add(&waves, &jump)?)
    })
}

pub(crate) fn star_velocity<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
    t: &Interval<R>,
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
    let (bands, state, left, right) = checked(c, "surveyed star velocity geometry", |a| {
        let bands = oracle.bands.select(a, face_owner)?;
        let state = oracle.geometry.at_normalized_depth(a, &bands, t)?;
        Ok((
            bands,
            state,
            select_side(a, oracle.left, face_owner)?,
            select_side(a, oracle.right, face_owner)?,
        ))
    })?;
    let evaluated = pair_evaluation(oracle, &bands, &state, &left, &right)?;
    checked(c, "surveyed star velocity intersection", |a| {
        let fl = side_curve(
            a,
            oracle,
            &bands,
            &state,
            &evaluated.dry_potential,
            &evaluated.left_increment,
            &left,
        )?;
        let fr = side_curve(
            a,
            oracle,
            &bands,
            &state,
            &evaluated.dry_potential,
            &evaluated.right_increment,
            &right,
        )?;
        let ul = a.sub(&left.velocity, &fl)?;
        let ur = a.add(&right.velocity, &fr)?;
        let lower = c.maximum(ul.lower(), ur.lower())?;
        let upper = c.minimum(ul.upper(), ur.upper())?;
        reject_mask(
            a,
            &c.gt(&lower, &upper)?,
            PrismaticRiemannRejection::SamplingUnresolved,
            "Surveyed star velocities have an empty intersection. Refine the certified star enclosure",
        )?;
        Ok(a.bounds(&lower, &upper)?)
    })
}

pub(crate) fn fan_velocity<R>(
    oracle: &Oracle<'_, R>,
    side: WaveSide,
    face_owner: &Tensor<R>,
    state: &Constitutive<R>,
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
    let (bands, selected) = checked(c, "surveyed fan owners", |a| {
        let selected = match side {
            WaveSide::Left => oracle.left,
            WaveSide::Right => oracle.right,
        };
        Ok((
            oracle.bands.select(a, face_owner)?,
            select_side(a, selected, face_owner)?,
        ))
    })?;
    let difference = fan_increment(oracle, &bands, state, &selected)?;
    checked(c, "surveyed fan velocity", |a| {
        Ok(match side {
            WaveSide::Left => a.sub(&selected.velocity, &difference)?,
            WaveSide::Right => a.add(&selected.velocity, &difference)?,
        })
    })
}

pub(crate) fn sonic_residual<R>(
    oracle: &Oracle<'_, R>,
    side: WaveSide,
    face_owner: &Tensor<R>,
    t: &Tensor<R>,
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
    let state = checked(c, "surveyed sonic geometry", |a| {
        let bands = oracle.bands.select(a, face_owner)?;
        let t = a.point(t)?;
        oracle.geometry.at_normalized_depth(a, &bands, &t)
    })?;
    let velocity = fan_velocity(oracle, side, face_owner, &state)?;
    checked(c, "surveyed increasing sonic equation", |a| {
        let sound = sound(a, &state, oracle.config.gravity_m_per_s2)?;
        Ok(match side {
            WaveSide::Left => a.sub(&sound, &velocity)?,
            WaveSide::Right => a.add(&velocity, &sound)?,
        })
    })
}

fn side_curve<R>(
    a: &mut Arithmetic<'_, R>,
    oracle: &Oracle<'_, R>,
    bands: &GatheredBands<R>,
    state: &Constitutive<R>,
    phi: &Interval<R>,
    increment: &Interval<R>,
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
    let c = a.client();
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let rare = a.bounds(
        &c.minimum(increment.lower(), zero.lower())?,
        &c.minimum(increment.upper(), zero.upper())?,
    )?;
    let original = a.point(&side.original_area)?;
    let shock_area = a.bounds(
        &c.maximum(state.area.lower(), original.lower())?,
        &c.maximum(state.area.upper(), original.upper())?,
    )?;
    let shock_area = choose(a, &side.wet, &shock_area, &one)?;
    let side_area = choose(a, &side.wet, &original, &one)?;
    let secant = pressure_secant(
        a,
        oracle.geometry,
        bands,
        state,
        &side.geometry,
        oracle.config.gravity_m_per_s2,
    )?;
    let shock = shock::curve(a, &shock_area, side_area.lower(), &secant)?;
    let both = hull(a, &rare, &shock)?;
    let definitely_rare = c.lt(state.area.upper(), original.lower())?;
    let definitely_shock = c.gt(state.area.lower(), original.upper())?;
    let wet_curve = choose(a, &definitely_rare, &rare, &both)?;
    let wet_curve = choose(a, &definitely_shock, &shock, &wet_curve)?;
    choose(a, &side.wet, &wet_curve, phi)
}

pub(crate) fn select_side<R>(
    a: &mut Arithmetic<'_, R>,
    side: &SideState<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<SideState<R>>
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
    Ok(SideState {
        original_area: c.index_select(&side.original_area, 0, rows)?,
        original_discharge: c.index_select(&side.original_discharge, 0, rows)?,
        geometry: select_state(a, &side.geometry, rows)?,
        velocity: select_interval(a, &side.velocity, rows)?,
        sound: select_interval(a, &side.sound, rows)?,
        pressure: select_interval(a, &side.pressure, rows)?,
        potential: select_interval(a, &side.potential, rows)?,
        wet: c.index_select(&side.wet, 0, rows)?,
        dry: c.index_select(&side.dry, 0, rows)?,
    })
}
