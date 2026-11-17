use super::{
    point_flux_types::{PointFluxKind, SelectedPointFlux},
    types::CapturedPointInput,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Face state of one selection, in the face order of its candidates.
pub(super) struct ReflectedFaces<'f, R: Runtime<DType = DType>> {
    /// Invocation owner of each face.
    pub(super) owners: &'f Tensor<R>,
    /// Face position of each invocation owner.
    pub(super) association: &'f Tensor<R>,
    /// One where the face holds a complete selected trace.
    pub(super) ready: &'f Tensor<R>,
    pub(super) physical_water: &'f mut Interval<R>,
    pub(super) joint_water: &'f mut Interval<R>,
    pub(super) joint_momentum: &'f Interval<R>,
}

/// Applies the exact reflection identity to the ready mirror faces of one selection.
///
/// A mirror face has both sides wet, equal captured areas in every component, and captured
/// discharges that are exact negatives in every component, with equal radii.
/// The reflected problem is the same problem, so the interface discharge is exactly 0.
/// A certified mirror face takes +0.0 water bounds, a +0.0 selected discharge and velocity,
/// and the pressure-only momentum of the same trace. The momentum bounds stay unchanged.
///
/// Returns the zero-one mask of mirror faces the identity cannot certify: water bounds that
/// exclude 0, a trace that is neither star nor dry, or a pressure enclosure outside the
/// momentum bound. Those faces and every other face keep their values bit for bit.
pub(super) fn apply<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CapturedPointInput<R>,
    invocation_rows: &Tensor<R>,
    faces: ReflectedFaces<'_, R>,
    selected: &mut [SelectedPointFlux<R>],
) -> HydraulicResult<Tensor<R>>
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
    let zero = Tensor::zeros(faces.owners.shape(), a.dtype(), c.device())?;
    if faces.owners.numel() == 0 {
        return Ok(zero);
    }
    let mirror = c.mul(
        &mirror_mask(a, input, invocation_rows, faces.owners)?,
        faces.ready,
    )?;
    // Joint and physical water must already enclose the exact value 0.
    let mut excluded = zero.clone();
    for bound in [&*faces.physical_water, &*faces.joint_water] {
        let contains = c.mul(&c.le(bound.lower(), &zero)?, &c.ge(bound.upper(), &zero)?)?;
        excluded = c.maximum(&excluded, &c.eq(&contains, &zero)?)?;
    }
    let mut rejected = c.mul(&mirror, &excluded)?;
    // The selection loop already checked every selected owner's position.
    let mut positions = Vec::with_capacity(selected.len());
    for point in selected.iter() {
        let at = c.index_select(faces.association, 0, &point.invocation_owner)?;
        let rows = c.index_select(&mirror, 0, &at)?;
        let row_zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
        let enclosure = match (point.kind, &point.pressure) {
            (PointFluxKind::Star, Some(pressure)) => {
                let chosen = a.expansion_interval(&pressure.chosen)?;
                Some(a.bounds(
                    &c.minimum(chosen.lower(), pressure.evaluation.lower())?,
                    &c.maximum(chosen.upper(), pressure.evaluation.upper())?,
                )?)
            }
            (PointFluxKind::Dry, None) => Some(a.point(&row_zero)?),
            // A mirror face has velocity 0 at the interface, so a fan or exterior trace
            // contradicts the identity.
            _ => None,
        };
        let failed = match enclosure {
            Some(enclosure) => {
                let lower = c.index_select(faces.joint_momentum.lower(), 0, &at)?;
                let upper = c.index_select(faces.joint_momentum.upper(), 0, &at)?;
                let inside = c.mul(
                    &c.le(&lower, enclosure.lower())?,
                    &c.le(enclosure.upper(), &upper)?,
                )?;
                c.mul(&rows, &c.eq(&inside, &row_zero)?)?
            }
            None => rows,
        };
        let previous = c.index_select(&rejected, 0, &at)?;
        rejected = c.index_put(&rejected, 0, &at, &c.maximum(&previous, &failed)?)?;
        positions.push(at);
    }
    let certified = c.gt(&c.mul(&mirror, &c.eq(&rejected, &zero)?)?, &zero)?;
    for bound in [&mut *faces.physical_water, &mut *faces.joint_water] {
        *bound = a.bounds(
            &c.where_cond(&certified, &zero, bound.lower())?,
            &c.where_cond(&certified, &zero, bound.upper())?,
        )?;
    }
    for (point, at) in selected.iter_mut().zip(&positions) {
        let take = c.index_select(&certified, 0, at)?;
        let row_zero = Tensor::zeros(take.shape(), a.dtype(), c.device())?;
        let replace = |a: &mut Arithmetic<'_, R>,
                       value: &CertifiedExpansion<R>,
                       by: [&Tensor<R>; 3]|
         -> HydraulicResult<CertifiedExpansion<R>> {
            Ok(a.expansion_bounds(
                &c.where_cond(&take, by[0], value.high())?,
                &c.where_cond(&take, by[1], value.low())?,
                &c.where_cond(&take, by[2], value.abs_error())?,
            )?)
        };
        let zeros = [&row_zero, &row_zero, &row_zero];
        point.discharge = replace(a, &point.discharge, zeros)?;
        point.velocity = replace(a, &point.velocity, zeros)?;
        point.reflected = c.maximum(&point.reflected, &take)?;
        // A dry trace already holds zero momentum.
        if let (PointFluxKind::Star, Some(pressure)) = (point.kind, &point.pressure) {
            let chosen = &pressure.chosen;
            point.momentum = replace(
                a,
                &point.momentum,
                [chosen.high(), chosen.low(), chosen.abs_error()],
            )?;
        }
    }
    Ok(rejected)
}

/// Zero-one mask, in face order, of the faces whose captured states are exact mirrors.
/// Components compare by value, so +0.0 and -0.0 low parts count as equal.
fn mirror_mask<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CapturedPointInput<R>,
    invocation_rows: &Tensor<R>,
    owners: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
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
    let source = c.index_select(invocation_rows, 0, owners)?;
    let take = |value: &Tensor<R>| c.index_select(value, 0, &source);
    let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
    let (left_area, right_area) = (&input.left_area, &input.right_area);
    let (left_q, right_q) = (&input.left_discharge, &input.right_discharge);
    let mut mask = c.gt(&take(left_area.high())?, &zero)?;
    for (left, right) in [
        (left_area.high(), right_area.high()),
        (left_area.low(), right_area.low()),
        (left_area.abs_error(), right_area.abs_error()),
        // A negated expansion keeps its radius.
        (left_q.abs_error(), right_q.abs_error()),
    ] {
        mask = c.mul(&mask, &c.eq(&take(left)?, &take(right)?)?)?;
    }
    for (left, right) in [
        (left_q.high(), right_q.high()),
        (left_q.low(), right_q.low()),
    ] {
        mask = c.mul(&mask, &c.eq(&take(left)?, &c.neg(&take(right)?)?)?)?;
    }
    Ok(mask)
}
