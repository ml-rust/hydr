//! Original-difference motion certificates and conservative directional-speed admission.

use super::motion_bounds::{down, finite, up};
use crate::cpu::structures::CpuStructureContext;
use crate::structures::{GateTrajectoryAssignment, StructureError, StructureInputErrorKind};
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, TwoSumOps, UnaryOps};

pub(crate) fn check_motion(
    context: &CpuStructureContext,
    assignments: &[GateTrajectoryAssignment],
    slots: &[usize],
) -> Result<(), StructureError> {
    let (mut t0, mut t1, mut a0, mut a1, mut limit, mut witness, mut power, mut origins) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for (assignment, &slot) in assignments.iter().zip(slots) {
        let limits = assignment.trajectory.motion_limits();
        for (segment, pair) in assignment.trajectory.knots().windows(2).enumerate() {
            let increasing = pair[1].opening_m > pair[0].opening_m;
            let speed_limit = if increasing {
                limits.maximum_opening_speed_m_per_s
            } else {
                limits.maximum_closing_speed_m_per_s
            };
            t0.push(pair[0].time_s);
            t1.push(pair[1].time_s);
            a0.push(pair[0].opening_m.min(pair[1].opening_m));
            a1.push(pair[0].opening_m.max(pair[1].opening_m));
            limit.push(speed_limit);
            let span = pair[1].time_s - pair[0].time_s;
            witness.push(span);
            power.push(
                if span.is_finite()
                    && span > 0.0
                    && crate::cpu::structures::power_metadata::power_of_two(span)
                {
                    1.0
                } else {
                    0.0
                },
            );
            origins.push((slot, segment));
        }
    }
    let c = &context.client;
    let zero = context.tensor(&vec![0.0; origins.len()])?;
    let one = c.add_scalar(&zero, 1.0)?;
    let (h, hl) = c.two_sum(&context.tensor(&t1)?, &c.neg(&context.tensor(&t0)?)?)?;
    let (d, dl) = c.two_sum(&context.tensor(&a1)?, &c.neg(&context.tensor(&a0)?)?)?;
    let moving = c.gt(&d, &zero)?;
    let limit = context.tensor(&limit)?;
    let matching = c.eq(&h, &context.tensor(&witness)?)?;
    let duration_valid = c.mul(&finite(context, &h)?, &c.gt(&h, &zero)?)?;
    let usable = c.mul(&matching, &duration_valid)?;
    let active = c.mul(&moving, &usable)?;
    let q = c.div(
        &c.where_cond(&active, &d, &zero)?,
        &c.where_cond(&active, &h, &one)?,
    )?;
    let zero_lows = c.mul(&c.eq(&hl, &zero)?, &c.eq(&dl, &zero)?)?;
    let positive_q = c.mul(&finite(context, &q)?, &c.gt(&q, &zero)?)?;
    let base = c.mul(
        &active,
        &c.mul(&context.tensor(&power)?, &c.mul(&zero_lows, &positive_q)?)?,
    )?;
    let normal = c.ge(&q, &c.mul_scalar(&one, f64::MIN_POSITIVE)?)?;
    let direct = c.mul(&base, &c.maximum(&c.le(&h, &one)?, &normal)?)?;
    let back_candidate = c.mul(&base, &c.mul(&c.gt(&h, &one)?, &c.sub(&one, &normal)?)?)?;
    let back = c.mul(
        &c.where_cond(&back_candidate, &q, &zero)?,
        &c.where_cond(&back_candidate, &h, &one)?,
    )?;
    let exact = c.maximum(
        &direct,
        &c.mul(
            &back_candidate,
            &c.mul(&finite(context, &back)?, &c.eq(&back, &d)?)?,
        )?,
    )?;
    let exact_accept = c.mul(&exact, &c.le(&q, &limit)?)?;
    let exact_excess = c.mul(&exact, &c.gt(&q, &limit)?)?;
    let prohibited = c.mul(&moving, &c.eq(&limit, &zero)?)?;
    let unresolved = c.mul(&active, &c.sub(&one, &c.maximum(&exact, &prohibited)?)?)?;

    let abs_dl = c.abs(&dl)?;
    let abs_hl = c.abs(&hl)?;
    let has_dl = c.mul(&unresolved, &c.ne(&dl, &zero)?)?;
    let has_hl = c.mul(&unresolved, &c.ne(&hl, &zero)?)?;
    let du_sum = c.add(
        &c.where_cond(&has_dl, &d, &zero)?,
        &c.where_cond(&has_dl, &abs_dl, &zero)?,
    )?;
    let du = c.where_cond(&has_dl, &up(context, &du_sum)?, &d)?;
    let hl_difference = c.sub(
        &c.where_cond(&has_hl, &h, &zero)?,
        &c.where_cond(&has_hl, &abs_hl, &zero)?,
    )?;
    let lower_h = c.where_cond(&has_hl, &down(context, &hl_difference)?, &h)?;
    let upper_valid = c.mul(
        &unresolved,
        &c.mul(
            &finite(context, &du)?,
            &c.mul(&finite(context, &lower_h)?, &c.gt(&lower_h, &zero)?)?,
        )?,
    )?;
    let upper_q = c.div(
        &c.where_cond(&upper_valid, &du, &zero)?,
        &c.where_cond(&upper_valid, &lower_h, &one)?,
    )?;
    let upper_q = up(context, &upper_q)?;
    let upper_accept = c.mul(
        &upper_valid,
        &c.mul(&finite(context, &upper_q)?, &c.le(&upper_q, &limit)?)?,
    )?;

    let remaining = c.mul(&unresolved, &c.sub(&one, &upper_accept)?)?;
    let lower_dl = c.mul(&remaining, &c.ne(&dl, &zero)?)?;
    let upper_hl = c.mul(&remaining, &c.ne(&hl, &zero)?)?;
    let lower_difference = c.sub(
        &c.where_cond(&lower_dl, &d, &zero)?,
        &c.where_cond(&lower_dl, &abs_dl, &zero)?,
    )?;
    let lower_d = c.where_cond(
        &lower_dl,
        &c.maximum(&zero, &down(context, &lower_difference)?)?,
        &d,
    )?;
    let upper_sum = c.add(
        &c.where_cond(&upper_hl, &h, &zero)?,
        &c.where_cond(&upper_hl, &abs_hl, &zero)?,
    )?;
    let upper_h = c.where_cond(&upper_hl, &up(context, &upper_sum)?, &h)?;
    let lower_valid = c.mul(
        &remaining,
        &c.mul(
            &finite(context, &lower_d)?,
            &c.mul(&finite(context, &upper_h)?, &c.gt(&upper_h, &zero)?)?,
        )?,
    )?;
    let lower_active = c.mul(&lower_valid, &c.gt(&lower_d, &zero)?)?;
    let lower_q = c.div(
        &c.where_cond(&lower_active, &lower_d, &zero)?,
        &c.where_cond(&lower_active, &upper_h, &one)?,
    )?;
    let lower_q = c.where_cond(
        &lower_active,
        &c.maximum(&zero, &down(context, &lower_q)?)?,
        &zero,
    )?;
    let lower_excess = c.mul(
        &lower_valid,
        &c.mul(&finite(context, &lower_q)?, &c.gt(&lower_q, &limit)?)?,
    )?;
    let definition = c.maximum(&prohibited, &c.maximum(&exact_excess, &lower_excess)?)?;
    let accepted = c.maximum(
        &c.sub(&one, &moving)?,
        &c.maximum(&exact_accept, &upper_accept)?,
    )?;
    let numerical = c.maximum(
        &c.sub(&one, &usable)?,
        &c.sub(&one, &c.maximum(&accepted, &definition)?)?,
    )?;
    let rejected = c.maximum(&definition, &numerical)?;
    if context.any(&rejected)? {
        let flags = rejected.try_to_vec::<f64>()?;
        let kinds = definition.try_to_vec::<f64>()?;
        let lane = flags.iter().position(|v| *v != 0.0).ok_or_else(|| {
            StructureError::numerical("motion rejection mask contradicts reduction")
        })?;
        let (slot, segment) = origins[lane];
        let message = format!(
            "gate trajectory segment={segment}: supply motion within directional limits and representable arithmetic bounds"
        );
        return Err(if kinds[lane] != 0.0 {
            StructureError::input(StructureInputErrorKind::Definition, message).at(slot)
        } else {
            StructureError::numerical(message).at(slot)
        });
    }
    Ok(())
}
