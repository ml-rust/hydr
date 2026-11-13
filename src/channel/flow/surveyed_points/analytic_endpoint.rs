use super::analytic_bottom::AnalyticSource;
use super::analytic_bottom_arithmetic::{AnalyticOperands, evaluate as evaluate_bottom};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::ClippedBandEndpoints,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

#[derive(Clone, Copy)]
pub(super) enum EndpointSide {
    Source,
    Target,
}

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    endpoints: &ClippedBandEndpoints<R>,
    side: EndpointSide,
    gravity: f64,
) -> HydraulicResult<AnalyticSource<R>>
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
    let endpoint = match side {
        EndpointSide::Source => endpoints.source(),
        EndpointSide::Target => endpoints.target(),
    };
    let raw = endpoint.retained_area();
    let available = endpoint.raw_available_mask();
    let zero = Tensor::zeros(raw.high().shape(), DType::F64, c.device())?;
    let one = Tensor::ones(raw.high().shape(), DType::F64, c.device())?;
    let safe_area = a.expansion_bounds(
        &c.where_cond(available, raw.high(), &zero)?,
        &c.where_cond(available, raw.low(), &zero)?,
        &c.where_cond(available, raw.abs_error(), &zero)?,
    )?;
    let raw_bounds = a.expansion_interval(&safe_area)?;
    let rectangular = c.mul(endpoints.rectangular_band0_mask(), available)?;
    let rectangular = c.mul(&rectangular, &c.ge(raw_bounds.lower(), &zero)?)?;
    let pointed = c.mul(endpoints.pointed_band0_mask(), available)?;
    let eligible = c.maximum(&rectangular, &pointed)?;
    let local_depth = endpoint.coordinates().depth_above_start_m();
    let local_depth = a.bounds(
        &local_depth.lower().reshape(raw.high().shape())?,
        &local_depth.upper().reshape(raw.high().shape())?,
    )?;
    evaluate_bottom(
        a,
        AnalyticOperands {
            eligible,
            rectangular,
            pointed,
            exact_area: &safe_area,
            complete_depth: &local_depth,
            width: endpoints.compiled_bottom_width(),
            depth_identity: None,
            zero: &zero,
            one: &one,
            gravity,
        },
    )
}
