use crate::channel::{
    flow::surveyed::{
        analytic_bottom::kinds,
        band_lanes::IntegrationLanes,
        equation_ops::{choose, scalar},
        error::{Arithmetic, HydraulicResult},
    },
    section::PreparedAreaPrefix,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct PointPotentialLanes<R: Runtime<DType = DType>> {
    pub(super) lanes: IntegrationLanes<R>,
    pub(super) pointed: Tensor<R>,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) original_rows: Tensor<R>,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) section_owner: Tensor<R>,
}

pub(super) fn build<R>(
    a: &mut Arithmetic<'_, R>,
    prefix: &PreparedAreaPrefix<R>,
    gravity: f64,
) -> HydraulicResult<PointPotentialLanes<R>>
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
    let physical = prefix.integration_geometry(a)?;
    let m = physical.extent.lower().numel();
    let shape = [physical.face_count, physical.band_count];
    let face = c
        .arange(0.0, physical.face_count as f64, 1.0, DType::I64)?
        .reshape(&[physical.face_count, 1])?;
    let face_owner = face.broadcast_to(&shape)?.contiguous()?.reshape(&[m])?;
    let bottom = c.cast(
        &c.eq(
            &physical.band_index,
            &Tensor::zeros(&[], DType::I64, c.device())?,
        )?,
        DType::F64,
    )?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let (rectangular, pointed) = kinds(
        a,
        &bottom,
        &physical.pointed,
        &physical.start_width,
        &physical.slope,
    )?;
    let zero_extent = c.eq(physical.extent.upper(), zero.upper())?;
    let analytic = c.maximum(&zero_extent, &c.maximum(&pointed, &rectangular)?)?;
    let active = c.mul(&physical.valid, &c.eq(&analytic, zero.lower())?)?;
    let safe_width = choose(a, &active, &physical.start_width, &one)?;
    let safe_area = choose(a, &active, &physical.start_area, &one)?;
    let safe_slope = choose(a, &active, &physical.slope, &zero)?;
    Ok(PointPotentialLanes {
        lanes: IntegrationLanes {
            face_owner,
            band_index: physical.band_index,
            bottom,
            active,
            analytic,
            extent: physical.extent,
            width: safe_width,
            slope: safe_slope,
            area: safe_area,
            gravity: scalar(a, gravity)?,
            face_count: physical.face_count,
            band_count: physical.band_count,
        },
        pointed: physical.pointed,
        original_rows: physical.original_rows,
        section_owner: physical.section_owner,
    })
}

#[cfg(test)]
#[path = "potential_lanes_tests.rs"]
mod tests;
