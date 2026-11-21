//! Discharge, regime, status and sensitivity of every culvert lane, written into the result.
//!
//! The per-barrel enclosure is `[min(Q_i lower, Q_o lower), min(Q_i upper, Q_o upper)]`. Its
//! midpoint, kept inside the enclosure, times the barrel count and the direction is the
//! signed discharge. The inlet governs when `Q_i` lies certainly below `Q_o`, the outlet when
//! `Q_o` lies certainly below `Q_i`, and the lane is a tie when the enclosures overlap.

use super::Selected;
use super::inlet::InletDischarge;
use super::lanes::Lanes;
use super::outlet::OutletDischarge;
use super::sensitivity::{self, Governs};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{TensorCulvertLaw, TensorCulvertResult};
use crate::structures::tensor::{TensorStructureContext, TensorStructureResult};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Writes the culvert lanes of `result`.
pub(crate) fn assemble<R>(
    law: &TensorCulvertLaw<R>,
    context: &TensorStructureContext<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    inlet: &InletDischarge<R>,
    outlet: &OutletDischarge<R>,
    mut result: TensorStructureResult<R>,
) -> Result<TensorStructureResult<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let one = &barrel.one;
    let zero = &barrel.zero;
    let not = |mask: &Tensor<R>| c.sub(one, mask);
    let flowing = c.mul(
        &c.mul(&lanes.active, &not(&inlet.zero)?)?,
        &c.mul(&not(&outlet.level)?, &not(&outlet.dead)?)?,
    )?;
    let [inlet_low, inlet_high] = &inlet.bounds;
    let [outlet_low, outlet_high] = &outlet.bounds;
    let lower = c.where_cond(&flowing, &c.minimum(inlet_low, outlet_low)?, zero)?;
    let upper = c.where_cond(&flowing, &c.minimum(inlet_high, outlet_high)?, zero)?;
    let below = c.lt(inlet_high, outlet_low)?;
    let inlet_governs = c.mul(&flowing, &c.maximum(&not(&outlet.solved)?, &below)?)?;
    let outlet_governs = c.mul(
        &c.mul(&flowing, &outlet.solved)?,
        &c.lt(outlet_high, inlet_low)?,
    )?;
    let tie = c.sub(&c.sub(&flowing, &inlet_governs)?, &outlet_governs)?;
    let middle = c.add(&c.mul_scalar(&lower, 0.5)?, &c.mul_scalar(&upper, 0.5)?)?;
    let point = c.minimum(&c.maximum(&middle, &lower)?, &upper)?;

    let code = |value: f64| barrel.constant(value);
    let mut regime = lanes.regime.clone();
    regime = c.where_cond(&outlet.level, &code(codes::EQUAL_HEADS)?, &regime)?;
    regime = c.where_cond(&outlet.dead, &code(codes::CULVERT_OUTLET_CONTROL)?, &regime)?;
    regime = c.where_cond(
        &c.mul(&lanes.active, &inlet.zero)?,
        &code(codes::CULVERT_INLET_CONTROL)?,
        &regime,
    )?;
    regime = c.where_cond(
        &inlet_governs,
        &code(codes::CULVERT_INLET_CONTROL)?,
        &regime,
    )?;
    regime = c.where_cond(
        &outlet_governs,
        &code(codes::CULVERT_OUTLET_CONTROL)?,
        &regime,
    )?;
    regime = c.where_cond(&tie, &code(codes::CULVERT_CONTROL_TIE)?, &regime)?;
    let status = c.where_cond(
        &flowing,
        &code(codes::ACTIVE)?,
        &c.where_cond(&lanes.active, &code(codes::PHYSICAL_ZERO)?, &lanes.status)?,
    )?;
    let direction = c.where_cond(&outlet.level, zero, &lanes.direction)?;
    let magnitude = c.mul(&point, &law.count)?;
    let discharge = c.where_cond(&flowing, &c.mul(&direction, &magnitude)?, zero)?;
    let governs = Governs {
        inlet: inlet_governs,
        outlet: outlet_governs,
        tie,
    };
    let bound = sensitivity::bound(law, rows, lanes, inlet, outlet, &governs, &point)?;

    let put = |wide: &Tensor<R>, lane: &Tensor<R>| c.index_put(wide, 0, &law.lanes, lane);
    let empty = &context.zero;
    let headwater = c.where_cond(&lanes.active, &lanes.headwater, zero)?;
    result.discharge = put(&result.discharge, &discharge)?;
    context.finite(&result.discharge, "final signed culvert proposal")?;
    result.head_sensitivity = put(&result.head_sensitivity, &bound)?;
    context.finite(&result.head_sensitivity, "culvert head sensitivity")?;
    result.regime = put(&result.regime, &regime)?;
    result.status = put(&result.status, &status)?;
    result.direction = put(&result.direction, &direction)?;
    result.head_gap = put(&result.head_gap, &lanes.gap)?;
    result.upstream_head = put(&result.upstream_head, &headwater)?;
    result.downstream_head = put(&result.downstream_head, &lanes.tailwater)?;
    result.ratio = put(&result.ratio, zero)?;
    result.correction_k = put(&result.correction_k, zero)?;
    result.log_magnitude = put(&result.log_magnitude, zero)?;
    result.culvert = Some(TensorCulvertResult {
        headwater: put(empty, &headwater)?,
        tailwater: put(empty, &lanes.tailwater)?,
        lower: put(empty, &lower)?,
        upper: put(empty, &upper)?,
    });
    Ok(result)
}
