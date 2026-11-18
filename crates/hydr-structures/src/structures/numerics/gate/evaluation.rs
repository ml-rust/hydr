//! Positive pressure-strip logarithms combined through a correctly rounded two-input logaddexp.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::CorrectlyRoundedOps;
use numr::ops::ScalarOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult, gate::TensorGateResult,
};

pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
    mut result: TensorStructureResult<R>,
) -> Result<TensorStructureResult<R>, StructureError>
where
    R::Client: numr::ops::CumulativeOps<R>
        + numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let Some(gate) = &context.gate else {
        return Ok(result);
    };
    let operation = input.gate.as_ref().ok_or_else(|| {
        StructureError::numerical("checked gate proposal lacks its explicit opening tensors")
    })?;
    let class = super::classification::classify(context, gate, input, operation)?;
    let geometry = super::geometry::evaluate(context, gate, operation, &class)?;
    let c = &context.client;
    let submerged_height =
        c.where_cond(&class.submerged, &geometry.submerged_height, &context.one)?;
    let gap = c.where_cond(&class.submerged, &geometry.gap, &context.one)?;
    let submerged_log = c.add(
        &c.cr_log(&submerged_height)?,
        &c.mul_scalar(&c.cr_log(&gap)?, 0.5)?,
    )?;
    let free_height = c.where_cond(&class.free, &geometry.free_height, &context.one)?;
    let lower = c.where_cond(&class.free, &geometry.lower, &context.one)?;
    let upper = c.where_cond(&class.free, &geometry.upper, &context.zero)?;
    let ratio = c.div(&upper, &lower)?;
    context.finite(&ratio, "active gate free pressure-head ratio is nonfinite")?;
    context.reject(
        &c.maximum(&c.lt(&ratio, &context.zero)?, &c.gt(&ratio, &context.one)?)?,
        "gate free pressure-head ratio lies outside zero to one",
    )?;
    // No outward bound covers this square root, so no runtime square-root contract is checked.
    // Its error reaches only `k`, which feeds the correctly rounded log/exp path to the discharge.
    // Downstream finite and positive checks reject any nonfinite or vanished result.
    let k = c.add_scalar(&c.div(&ratio, &c.add_scalar(&c.sqrt(&ratio)?, 1.0)?)?, 1.0)?;
    let free_log = c.add(
        &c.add(
            &c.cr_log(&c.add_scalar(&context.zero, 2.0 / 3.0)?)?,
            &c.cr_log(&free_height)?,
        )?,
        &c.add(&c.mul_scalar(&c.cr_log(&lower)?, 0.5)?, &c.cr_log(&k)?)?,
    )?;
    let absent = c.add_scalar(&context.zero, f64::NEG_INFINITY)?;
    let integral_log = super::super::log_add_exp::log_add_exp(
        c,
        &c.where_cond(&class.submerged, &submerged_log, &absent)?,
        &c.where_cond(&class.free, &free_log, &absent)?,
        &absent,
    )?;
    let raw = c.add(
        &c.add(&gate.log_scale, &gate.log_root_gravity)?,
        &integral_log,
    )?;
    let (magnitude, logarithm) = super::super::orifice::exp_active(context, &class.active, &raw)?;
    let discharge = c.mul(&class.direction, &magnitude)?;
    context.finite(&discharge, "final signed gate proposal is nonfinite")?;
    let sensitivity = head_sensitivity(context, gate, &class, &geometry.gap)?;
    result.head_sensitivity = c.where_cond(&gate.mask, &sensitivity, &result.head_sensitivity)?;
    result.discharge = c.where_cond(&gate.mask, &discharge, &result.discharge)?;
    result.status = c.where_cond(&gate.mask, &class.status, &result.status)?;
    result.regime = c.where_cond(&gate.mask, &class.regime, &result.regime)?;
    result.direction = c.where_cond(&gate.mask, &class.direction, &result.direction)?;
    result.gate = Some(TensorGateResult {
        geometry,
        ratio: c.where_cond(&class.free, &ratio, &context.zero)?,
        k: c.where_cond(&class.free, &k, &context.zero)?,
        logarithm,
    });
    Ok(result)
}

/// Gate sensitivity `1.5 / h_u + 0.5 / gap`.
///
/// `h_u` is the donor head above the sill. `gap` is the donor-receiver head difference where
/// the receiver stands above the sill, else zero.
///
/// - Free aperture: `d ln Q / dH = 1.5 / (a + sqrt(ab) + b) <= 1.5 / a`, with `a`, `b` the donor
///   heads above sill and crown.
/// - Submerged strip: `Q ∝ h_s sqrt(gap)` with `h_s` its height, so `0.5 / gap` per head.
/// - Mixed aperture: the free strip over `[R, min(D, crown)]` moves `ln Q` by at most
///   `1.5 / h_u + 0.5 / gap`, since `h_u = h_s + gap`.
fn head_sensitivity<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    gate: &crate::structures::tensor::gate::TensorGateContext<R>,
    class: &super::classification::Classification<R>,
    gap: &numr::tensor::Tensor<R>,
) -> Result<numr::tensor::Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R> + numr::ops::ConditionalOps<R>,
{
    use super::super::sensitivity::{GAP_EXPONENT, HEAD_EXPONENT, per_metre};
    let c = &context.client;
    let above_sill = c.where_cond(
        &class.active,
        &c.sub(&class.donor, &gate.sill)?,
        &context.zero,
    )?;
    let sensitivity = c.add(
        &per_metre(context, &above_sill, HEAD_EXPONENT)?,
        &per_metre(context, gap, GAP_EXPONENT)?,
    )?;
    Ok(c.where_cond(&class.active, &sensitivity, &context.zero)?)
}

#[cfg(test)]
mod tests {
    use super::super::super::sensitivity::fixture::{
        TestResult, check_bound, grid, lanes, structures,
    };
    use crate::structures::{
        DirectionPolicy, GateApplicability, GateParameters, StructureLaw, StructureOperatingPoint,
    };

    #[test]
    fn gate_sensitivity_bounds_free_mixed_and_submerged_apertures() -> TestResult {
        let gate = GateParameters::new(
            2.0,
            0.0,
            2.0,
            0.6,
            9.81,
            GateApplicability {
                coefficient_provenance: "Analytical coefficient".into(),
                hydrostatic_quiescent_pool_assertion: "Level analytical pools".into(),
                free_submerged_aperture_applicability: "Declared aperture envelope".into(),
                unobstructed_aperture_assertion: "Complete unobstructed aperture".into(),
                symmetric_reversal: true,
            },
        )?;
        // Sill at 0 m, crown at 1 m: the grid covers each side of both planes.
        let values = grid(-0.6, 3.0, 48);
        let mut cases = lanes(&values, &[0.0, 1.0], [true, true]);
        cases.extend(lanes(&values, &[0.0, 1.0], [true, false]));
        let structures = structures(
            |_| {
                (
                    StructureLaw::HydrostaticRectangularGate(gate.clone()),
                    DirectionPolicy::Bidirectional,
                    StructureOperatingPoint::GateOpening { opening_m: 1.0 },
                )
            },
            cases.len(),
        )?;
        let flowing = check_bound(&structures, &cases)?;
        assert!(flowing > cases.len() / 2, "{flowing} of {}", cases.len());
        Ok(())
    }
}
