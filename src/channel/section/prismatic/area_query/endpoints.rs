use super::super::hydraulic_geometry::GatheredBands;
use super::{AreaDirection, AreaQuery, SupportedAreaQuery, admission, range, selection};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) enum PhysicalEndpoint {
    Lower,
    Upper,
}

impl<R> SupportedAreaQuery<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Preserve the continuous area expression at one physical endpoint.
    /// Its transformed area retains every arithmetic radius.
    pub(crate) fn endpoint(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        endpoint: PhysicalEndpoint,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        let parameter = match (self.query.direction, endpoint) {
            (AreaDirection::Add, PhysicalEndpoint::Lower)
            | (AreaDirection::Subtract, PhysicalEndpoint::Upper) => self.query.parameter.lower(),
            (AreaDirection::Add, PhysicalEndpoint::Upper)
            | (AreaDirection::Subtract, PhysicalEndpoint::Lower) => self.query.parameter.upper(),
        };
        let parameter = a.point(parameter)?;
        self.reconstruct(a, bands, parameter)
    }

    /// Rebind the retained affine map and prove support for its complete new parameter image.
    pub(crate) fn with_parameter_bounds(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        admission::chart(a, bands, &self.query.anchor, &self.query.scale, parameter)?;
        admission::metadata(
            a,
            &self.query.original_rows,
            bands.section_owner.shape(),
            DType::I64,
        )?;
        let parameter = a.bounds(parameter.lower(), parameter.upper())?;
        self.reconstruct(a, bands, parameter)
    }

    fn reconstruct(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        parameter: Interval<R>,
    ) -> HydraulicResult<Self> {
        let offsets = range::offsets(a, &self.query.scale, &parameter, self.query.direction)?;
        let anchor = selection::capture_anchor(a, &self.query.anchor)?;
        let scale = a.expansion_bounds(
            self.query.scale.high(),
            self.query.scale.low(),
            self.query.scale.abs_error(),
        )?;
        let c = a.client();
        let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
            c.copy_into(&owned, value)?;
            Ok(owned)
        };
        AreaQuery {
            model: Arc::clone(&self.query.model),
            anchor,
            offsets,
            direction: self.query.direction,
            scale,
            parameter,
            owners: copy(&self.query.owners)?,
            original_rows: copy(&self.query.original_rows)?,
            band_count: self.query.band_count,
        }
        .prove_support(a, bands)
    }
}
