use std::cell::{Cell, RefCell};

use super::paired_sources::PairedSources;
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar},
        error::{HydraulicError, HydraulicResult, checked},
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::optimize::{
    CertifiedRootDomainQuery, CertifiedRootMappedValues, CertifiedRootTolerance,
};

/// A selected star parameter bracket stops at this fraction of its face's jump scale.
pub(super) const JUMP_FRACTION: CertifiedRootTolerance = CertifiedRootTolerance {
    atol: 1. / (1u64 << 44) as f64,
    rtol: 0.,
};

/// The jump scale floor as a fraction of the larger source area.
/// With `JUMP_FRACTION`, a bracket at the floor is near the two-part rounding of that area,
/// so a narrower bracket moves the selected area by less than its own rounding.
const AREA_RESOLUTION: f64 = 1. / (1u64 << 62) as f64;

/// Per-row jump scale J = max(|A_R - A_L| + A_max |u_R - u_L| / c, A_max AREA_RESOLUTION),
/// in square metres.
///
/// The floor bounds the sharpening rounds when the jump lies far below the area resolution.
/// It only sets where bisection stops, so it is an F64 value, never a certificate.
/// Each term takes its outward magnitude, and c takes the lower bound of the smaller
/// source celerity. A row without a positive celerity bound holds +inf, so it never sharpens.
/// The second tensor is one on rows whose two sources are bit-identical.
pub(super) fn scale<R>(
    sources: &PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let stage = sources.select_rows(parent_rows, query)?;
    let gravity = sources.evaluation.config.gravity_m_per_s2;
    checked::<R, _>(sources.geometry.client(), "wet star jump scale", |a| {
        let c = a.client();
        let [left, right] = [0, 1].map(|side| stage.sources[side].requested_areas_m2());
        let area = a.expansion_sub(right, left)?;
        let area = a.expansion_interval(&area)?;
        let area = c.maximum(area.upper(), &c.mul_scalar(area.lower(), -1.)?)?;
        let velocity = a.expansion_interval(&stage.velocity.difference)?;
        let velocity = c.maximum(velocity.upper(), &c.mul_scalar(velocity.lower(), -1.)?)?;
        let left = a.expansion_interval(left)?;
        let right = a.expansion_interval(right)?;
        let largest = c.maximum(left.upper(), right.upper())?;
        let gravity = scalar(a, gravity)?;
        let [left, right] = [0, 1].map(|side| stage.sources[side].hydraulic_depth_m());
        let left = gravity_depth_root(a, &gravity, left, None)?;
        let right = gravity_depth_root(a, &gravity, right, None)?;
        let celerity = c.minimum(left.lower(), right.lower())?;
        let zero = Tensor::zeros(velocity.shape(), a.dtype(), c.device())?;
        let infinite = Tensor::full_scalar(velocity.shape(), a.dtype(), f64::INFINITY, c.device())?;
        let positive = c.gt(&celerity, &zero)?;
        let quotient = c.div(
            &c.mul(&largest, &velocity)?,
            &c.where_cond(&positive, &celerity, &infinite)?,
        )?;
        let term = c.where_cond(
            &c.eq(&velocity, &zero)?,
            &zero,
            &c.where_cond(&positive, &quotient, &infinite)?,
        )?;
        let jump = c.maximum(
            &c.add(&area, &term)?,
            &c.mul_scalar(&largest, AREA_RESOLUTION)?,
        )?;
        let uniform = Tensor::empty(stage.velocity.uniform.shape(), a.dtype(), c.device())?;
        c.copy_into(&uniform, &stage.velocity.uniform)?;
        Ok((jump, uniform))
    })
}

/// Maps each parameter domain to its outward enclosure of p / J.
///
/// A zero scale maps to [0, 1], which never meets the tolerance, so the bracket stops only at
/// its arithmetic floor. An infinite scale maps to [0, 0], which meets it at once.
pub(super) fn evaluate<R>(
    sources: &RefCell<&mut PairedSources<'_, R>>,
    scale: &Tensor<R>,
    query: &CertifiedRootDomainQuery<R>,
    cause: &Cell<Option<HydraulicError>>,
) -> numr::error::Result<CertifiedRootMappedValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let calculation = (|| -> HydraulicResult<_> {
        let pair = sources
            .try_borrow()
            .map_err(|_| invalid("mapping overlaps another source evaluation"))?;
        checked::<R, _>(pair.geometry.client(), "wet star jump mapping", |a| {
            let c = a.client();
            let scale = c.index_select(scale, 0, &query.owner)?;
            let zero = Tensor::zeros(scale.shape(), a.dtype(), c.device())?;
            let one = Tensor::ones(scale.shape(), a.dtype(), c.device())?;
            let empty = c.eq(&scale, &zero)?;
            let finite = c.lt(
                &scale,
                &Tensor::full_scalar(scale.shape(), a.dtype(), f64::INFINITY, c.device())?,
            )?;
            let divisor = c.where_cond(&c.mul(&finite, &c.eq(&empty, &zero)?)?, &scale, &one)?;
            let parameter = a.bounds(&query.lower, &query.upper)?;
            let divisor = a.point(&divisor)?;
            let mapped = a.div_positive_interval(&parameter, &divisor)?;
            let pick = |value: &Tensor<R>, at_zero: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                Ok(c.where_cond(&empty, at_zero, &c.where_cond(&finite, value, &zero)?)?)
            };
            Ok(CertifiedRootMappedValues {
                lower: pick(mapped.lower(), &zero)?,
                upper: pick(mapped.upper(), &one)?,
            })
        })
    })();
    calculation.map_err(|error| {
        let first = cause.take().unwrap_or(error);
        cause.set(Some(first));
        numr::error::Error::InvalidArgument {
            arg: "wet star jump mapping",
            reason: "Hydraulic mapping rejected. Read the retained typed cause".into(),
        }
    })
}

fn invalid(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Wet star jump scale {message}. Discard the sharpened bracket"),
    }
}
