use super::{
    paired_sources::SelectedRows,
    wave_curve_types::CurveValue,
    wet_star_endpoint::exact_zero,
    wet_star_types::{WetStarDomain, WetStarStatus},
    wet_wave_types::WetWaveSide,
    wet_waves_endpoint::{offsets, select},
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar},
        error::{Arithmetic, HydraulicResult},
    },
    section::source_difference,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    stage: &SelectedRows<R>,
    star: &WetStarDomain<R>,
    sound: &[CurveValue<R>; 2],
    gravity: f64,
    side: usize,
) -> HydraulicResult<WetWaveSide<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let shape = star.parent_rows.shape();
    let zero = Tensor::zeros(shape, a.dtype(), c.device())?;
    let one = Tensor::ones(shape, a.dtype(), c.device())?;
    let active = c.where_cond(
        &c.eq(
            &star.status,
            &Tensor::full_scalar(
                shape,
                DType::I64,
                WetStarStatus::Certified as i64 as f64,
                c.device(),
            )?,
        )?,
        &one,
        &zero,
    )?;
    let difference = source_difference(
        a,
        &stage.bands,
        &star.query,
        &stage.sources[side],
        &stage.boundaries,
    )?;
    let lower_difference = a.expansion_interval(&difference.lower)?;
    let upper_difference = a.expansion_interval(&difference.upper)?;
    let structural_equal = c.mul(
        &exact_zero(a, &difference.lower)?,
        &exact_zero(a, &difference.upper)?,
    )?;
    // Captured uniform identity and strict wet-root uniqueness prove A*=As throughout a broader bracket.
    let exact_equal = c.maximum(&structural_equal, &stage.velocity.uniform)?;
    let nonidentity = c.eq(&exact_equal, &zero)?;
    let shock_possible = c.mul(
        &active,
        &c.mul(&nonidentity, &c.gt(upper_difference.upper(), &zero)?)?,
    )?;
    let rarefaction_possible = c.mul(
        &active,
        &c.mul(&nonidentity, &c.lt(lower_difference.lower(), &zero)?)?,
    )?;
    let equality_possible = c.mul(
        &active,
        &c.maximum(
            &exact_equal,
            &c.mul(
                &c.le(lower_difference.lower(), &zero)?,
                &c.ge(upper_difference.upper(), &zero)?,
            )?,
        )?,
    )?;
    let shock_actual = c.mul(&shock_possible, &c.gt(lower_difference.lower(), &zero)?)?;
    let shock_conditional = c.mul(&shock_possible, &c.eq(&shock_actual, &zero)?)?;
    let gravity = scalar(a, gravity)?;
    let physical = gravity_depth_root(a, &gravity, stage.sources[side].hydraulic_depth_m(), None)?;
    let source_sound = CurveValue {
        retained: a.expansion_from_interval(&physical)?,
        physical,
        available: active.clone(),
    };
    let velocity = if side == 0 {
        &stage.velocity.left
    } else {
        &stage.velocity.right
    };
    let head_retained = characteristic(a, velocity, &source_sound.retained, side)?;
    let head = CurveValue {
        physical: a.expansion_interval(&head_retained)?,
        retained: head_retained,
        available: c.maximum(&rarefaction_possible, &equality_possible)?,
    };
    let first_tail = characteristic(
        a,
        &star.endpoints[0].sides[side].velocity,
        &sound[0].retained,
        side,
    )?;
    let last_tail = characteristic(
        a,
        &star.endpoints[1].sides[side].velocity,
        &sound[1].retained,
        side,
    )?;
    let first_tail = a.expansion_interval(&first_tail)?;
    let last_tail = a.expansion_interval(&last_tail)?;
    let (tail_lower, tail_upper) = if side == 0 {
        (last_tail.lower(), first_tail.upper())
    } else {
        (first_tail.lower(), last_tail.upper())
    };
    let tail_ordered = c.le(tail_lower, tail_upper)?;
    let tail_available = c.mul(&rarefaction_possible, &tail_ordered)?;
    let physical = a.bounds(
        &c.where_cond(&tail_available, tail_lower, &zero)?,
        &c.where_cond(&tail_available, tail_upper, &zero)?,
    )?;
    let tail = CurveValue {
        retained: a.expansion_from_interval(&physical)?,
        physical,
        available: tail_available,
    };
    let lower = offsets(a, star, 0, side)?;
    let upper = offsets(a, star, 1, side)?;
    // Convex pressure holds globally: certified banks reject benches, and pressure derivatives join continuously.
    // K(A,As), A/As, and B increase on A>=As. B(As) equals source sound.
    let lower_offset = select(
        a,
        &lower.actual.available,
        &lower.actual.retained,
        &source_sound.retained,
    )?;
    let upper_offset = select(
        a,
        &upper.actual.available,
        &upper.actual.retained,
        &upper.conditional.retained,
    )?;
    let upper_available = c.maximum(&upper.actual.available, &upper.conditional.available)?;
    let lower_bounds = a.expansion_interval(&lower_offset)?;
    let upper_bounds = a.expansion_interval(&upper_offset)?;
    let offset_ordered = c.le(lower_bounds.lower(), upper_bounds.upper())?;
    let shock_available = c.mul(&shock_possible, &c.mul(&upper_available, &offset_ordered)?)?;
    let physical = a.bounds(
        &c.where_cond(&shock_available, lower_bounds.lower(), &zero)?,
        &c.where_cond(&shock_available, upper_bounds.upper(), &zero)?,
    )?;
    let shock_offset = CurveValue {
        retained: a.expansion_from_interval(&physical)?,
        physical,
        available: shock_available.clone(),
    };
    // Compose each characteristic before projection, preserving cancellation against source velocity.
    let first_speed = characteristic(a, velocity, &lower_offset, side)?;
    let last_speed = characteristic(a, velocity, &upper_offset, side)?;
    let first_bounds = a.expansion_interval(&first_speed)?;
    let last_bounds = a.expansion_interval(&last_speed)?;
    let (speed_lower, speed_upper) = if side == 0 {
        (last_bounds.lower(), first_bounds.upper())
    } else {
        (first_bounds.lower(), last_bounds.upper())
    };
    let speed_ordered = c.le(speed_lower, speed_upper)?;
    let speed_available = c.mul(&shock_available, &speed_ordered)?;
    let physical = a.bounds(
        &c.where_cond(&speed_available, speed_lower, &zero)?,
        &c.where_cond(&speed_available, speed_upper, &zero)?,
    )?;
    let shock_stationary = c.mul(
        &speed_available,
        &c.mul(&exact_zero(a, &first_speed)?, &exact_zero(a, &last_speed)?)?,
    )?;
    let shock_speed = CurveValue {
        retained: a.expansion_from_interval(&physical)?,
        physical,
        available: speed_available,
    };
    let equality_stationary = c.mul(&equality_possible, &exact_zero(a, &head.retained)?)?;
    let mut evidence_available = one;
    for endpoint in &star.endpoints {
        let evidence = &endpoint.waves;
        let mut available = evidence.far_rarefaction[side].clone();
        if let Some(local) = &evidence.local[side] {
            available = c.index_put(&available, 0, &local.rows, &local.endpoint.curve.available)?;
        }
        evidence_available = c.mul(&evidence_available, &available)?;
    }
    let shock_valid = c.maximum(&c.eq(&shock_possible, &zero)?, &shock_speed.available)?;
    let rare_valid = c.maximum(&c.eq(&rarefaction_possible, &zero)?, &tail.available)?;
    let covered = c.maximum(
        &shock_possible,
        &c.maximum(&rarefaction_possible, &equality_possible)?,
    )?;
    let valid = c.mul(
        &active,
        &c.mul(
            &evidence_available,
            &c.mul(&covered, &c.mul(&shock_valid, &rare_valid)?)?,
        )?,
    )?;
    Ok(WetWaveSide {
        shock_possible,
        rarefaction_possible,
        equality_possible,
        shock_actual,
        shock_conditional,
        source_sound,
        shock_offset,
        shock_speed,
        head,
        tail,
        shock_stationary,
        equality_stationary,
        valid,
    })
}

fn characteristic<R>(
    a: &mut Arithmetic<'_, R>,
    velocity: &CertifiedExpansion<R>,
    offset: &CertifiedExpansion<R>,
    side: usize,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if side == 0 {
        Ok(a.expansion_sub(velocity, offset)?)
    } else {
        Ok(a.expansion_add(velocity, offset)?)
    }
}
