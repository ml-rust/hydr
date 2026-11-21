//! Jump classification at the outlet, and the S1 node bracket of the momentum zero.

use super::super::March;
use super::super::depth::DepthEnclosure;
use super::super::integrand::{area, coordinate, depth, floor, shape, slopes};
use super::super::jet;
use super::super::partition::LANES;
use super::super::trig::fill;
use super::steep::{Steep, column, momentum, owners, pair};
use crate::structures::StructureError;
use crate::structures::numerics::culvert::constants::CONVEYANCE_PEAK;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};

/// Bracket of the momentum zero between two S1 nodes, with the solver slope floor.
pub(super) struct Bracket<R: Runtime<DType = DType>> {
    /// Rows whose momentum gap at the outlet is certified positive.
    pub(super) inside: Tensor<R>,
    /// Rows whose supercritical flow certainly leaves the outlet.
    pub(super) outside: Tensor<R>,
    pub(super) bisect: Tensor<R>,
    pub(super) direct: Tensor<R>,
    /// Lower S1 coordinate of the bracket.
    pub(super) low: Tensor<R>,
    /// Upper S1 coordinate of the bracket.
    pub(super) high: Tensor<R>,
    /// S1 distance from the outlet at `high`.
    pub(super) reach: Bounds<R>,
    pub(super) slope: Tensor<R>,
    /// Supercritical depth at the outlet.
    pub(super) outlet: Bounds<R>,
}

pub(super) struct ScanInputs<'s, R: Runtime<DType = DType>> {
    pub(super) barrel: &'s TensorBarrelContext<R>,
    pub(super) outlet: &'s TensorOutletContext<R>,
    pub(super) march: &'s March<R>,
    pub(super) steep: &'s Steep<R>,
    pub(super) tailwater: &'s Tensor<R>,
    /// S2 coordinates at the outlet and at the distance of every S1 node, per row.
    pub(super) found: &'s DepthEnclosure<R>,
}

/// Classifies each row at the outlet and brackets the momentum zero between S1 nodes.
pub(super) fn scan<R>(
    c: &R::Client,
    inputs: &ScanInputs<'_, R>,
) -> Result<Bracket<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let ScanInputs {
        barrel,
        outlet,
        march,
        steep,
        tailwater,
        found,
    } = *inputs;
    let m = barrel.width();
    let count = LANES + 1;
    let per_row = count + 1;
    let device = &barrel.device;
    let profile = &march.profile;
    let lane_rows = owners::<R>(device, m, count)?;
    let lanes = barrel.select(&lane_rows)?;
    let table =
        |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> { value.reshape(&[m, per_row]) };
    let v1_low = table(&found.lower)?;
    let v1_high = table(&found.upper)?;
    let solved = table(&found.solved)?;
    let tail = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value
            .narrow(1, 1, count)?
            .contiguous()?
            .reshape(&[m * count])
    };
    let head = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value.narrow(1, 0, 1)?.contiguous()?.reshape(&[m])
    };
    // A coordinate of 1/2 lies inside both the circle and the box domain.
    let harmless = |value: &Tensor<R>, ok: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        c.where_cond(ok, value, &fill(ok, 0.5)?)
    };
    let s1_params = &profile.spec.params;

    let values = with_interval_arithmetic(c, DType::F64, "culvert outlet jump scan", |a| {
        let scale = a.bounds(&steep.momentum_scale[0], &steep.momentum_scale[1])?;
        // The supercritical depth at the outlet and the momentum gap there.
        let ok0 = head(&solved)?;
        let v_out = a.bounds(
            &harmless(&head(&v1_low)?, &ok0)?,
            &harmless(&head(&v1_high)?, &ok0)?,
        )?;
        let y_out = depth(a, &barrel.circular, &barrel.rise, &v_out)?;
        // Rows without a subcritical outlet take a harmless wet depth.
        let wet = c.where_cond(&steep.candidates, tailwater, &barrel.half_rise)?;
        let tw = a.point(&c.minimum(&wet, &barrel.rise)?)?;
        let m_tail = momentum(a, barrel, &scale, &tw)?;
        let m_out = momentum(a, barrel, &scale, &y_out)?;
        let gap = a.sub(&m_tail, &m_out)?;
        let inside = c.mul(
            &c.mul(&steep.candidates, &ok0)?,
            &c.gt(gap.lower(), &barrel.zero)?,
        )?;
        let outside = c.mul(
            &c.mul(&steep.candidates, &ok0)?,
            &c.lt(gap.upper(), &barrel.zero)?,
        )?;
        // A tailwater at or below critical depth leaves no subcritical profile at the outlet.
        let no_profile = c.mul(&steep.supercritical, &march.classes.critical_now)?;
        let outside = c.maximum(&outside, &c.mul(&no_profile, &ok0)?)?;

        // The gap at every S1 node.
        let lane_scale = a.bounds(
            &c.index_select(&steep.momentum_scale[0], 0, &lane_rows)?,
            &c.index_select(&steep.momentum_scale[1], 0, &lane_rows)?,
        )?;
        let ok = tail(&solved)?;
        let v_node = profile.nodes.reshape(&[m * count])?;
        let node = a.point(&harmless(&v_node, &ok)?)?;
        let y2 = depth(a, &lanes.circular, &lanes.rise, &node)?;
        let v1 = a.bounds(
            &harmless(&tail(&v1_low)?, &ok)?,
            &harmless(&tail(&v1_high)?, &ok)?,
        )?;
        let y1 = depth(a, &lanes.circular, &lanes.rise, &v1)?;
        let m2 = momentum(a, &lanes, &lane_scale, &y2)?;
        let m1 = momentum(a, &lanes, &lane_scale, &y1)?;
        let rest = a.sub(&m2, &m1)?;
        Ok((inside, outside, pair(&rest), ok, pair(&y_out)))
    })?;
    let (inside, outside, [rest_low, rest_high], ok, outlet_depth) = values;
    let table =
        |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> { value.reshape(&[m, count]) };
    let zero = &barrel.zero;
    let ok = table(&ok)?;
    let positive = c.mul(&c.gt(&table(&rest_low)?, &fill(&ok, 0.0)?)?, &ok)?;
    let negative = c.mul(&c.lt(&table(&rest_high)?, &fill(&ok, 0.0)?)?, &ok)?;
    let positives = c.sum(&positive, &[1], false)?;
    let negatives = c.sum(&negative, &[1], false)?;
    let one = &barrel.one;
    let total = fill(zero, count as f64)?;
    let last = fill(zero, LANES as f64)?;
    let low_index = c.maximum(&c.sub(&positives, one)?, zero)?;
    let high_index = c.minimum(&c.sub(&total, &negatives)?, &last)?;
    let low_ok = column(c, &positive, &low_index)?;
    let high_ok = column(c, &negative, &high_index)?;
    let ordered = c.gt(&high_index, &low_index)?;
    let bracketed = c.mul(
        &c.mul(&inside, &c.gt(&positives, zero)?)?,
        &c.mul(&ordered, &c.mul(&low_ok, &high_ok)?)?,
    )?;
    // S1 runs from the outlet toward yc, so v falls with the node index.
    let v_high = column(c, &profile.nodes, &low_index)?;
    let v_low = column(c, &profile.nodes, &high_index)?;
    let reach = [
        column(c, &profile.reach[0], &low_index)?,
        column(c, &profile.reach[1], &low_index)?,
    ];

    let slope = with_interval_arithmetic(c, DType::F64, "culvert outlet jump floor", |a| {
        let g1 = floor(a, s1_params, &v_low, &v_high)?;
        let rise = &barrel.rise;
        let phi = |a: &mut Arithmetic<'_, R>, v: &Interval<R>| -> ArithmeticResult<Interval<R>> {
            let shaped = shape(a, &barrel.circular, &outlet.beta, v, 1)?;
            let friction = slopes(a, s1_params, &shaped)?;
            let sf = jet::coefficient(a, &friction.friction, 0, rise)?;
            let s = a.point(&outlet.slope)?;
            let excess = a.sub(&s, &sf)?;
            let section = area(a, &barrel.circular, &outlet.beta, rise, v)?;
            a.mul(&section, &excess)
        };
        let low = a.point(&v_low)?;
        let at_low = phi(a, &low)?;
        let critical = a.bounds(&march.classes.critical[0], &march.classes.critical[1])?;
        let v_c = coordinate(a, &barrel.circular, rise, &critical)?;
        let at_critical = phi(a, &v_c)?;
        let gain = c.sub(at_low.lower(), at_critical.upper())?;
        // Friction falls with depth below the conveyance peak, and on every box depth.
        let high = a.point(&v_high)?;
        let top = depth(a, &barrel.circular, rise, &high)?;
        let peak = c.mul_scalar(rise, f64::from_bits(CONVEYANCE_PEAK[0]))?;
        let monotone = c.maximum(&barrel.boxed, &c.le(top.upper(), &peak)?)?;
        let usable = c.mul(&c.mul(&c.gt(&gain, zero)?, &c.gt(&g1, zero)?)?, &monotone)?;
        let gain = a.point(&c.where_cond(&usable, &gain, one)?)?;
        let g1 = a.point(&c.where_cond(&usable, &g1, one)?)?;
        let bound = a.mul(&gain, &g1)?;
        Ok(c.where_cond(&usable, bound.lower(), zero)?)
    })?;
    let usable = c.gt(&slope, zero)?;
    let bisect = c.mul(&bracketed, &usable)?;
    let direct = c.sub(&bracketed, &bisect)?;
    Ok(Bracket {
        inside,
        outside,
        bisect,
        direct,
        low: v_low,
        high: v_high,
        reach,
        slope: c.where_cond(&usable, &slope, one)?,
        outlet: outlet_depth,
    })
}
