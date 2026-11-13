use super::super::hydraulic_geometry::GatheredBands;
use super::{
    admission, boundaries,
    chart::{AreaAnchor, AreaDirection, AreaQuery, EndpointDifferences, SupportedAreaQuery},
    range,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        ShapeOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

impl<R> AreaQuery<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn prove_support(
        self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
    ) -> HydraulicResult<SupportedAreaQuery<R>> {
        admission::matching_model(Arc::ptr_eq(&self.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.owners)?;
        let c = a.client();
        let n = self.owners.numel();
        let zero_tensor = Tensor::zeros(&[n], DType::F64, c.device())?;
        let zero = a.point(&zero_tensor)?;
        let relative = boundaries::relative(a, bands, &self)?;
        let mut image = Vec::with_capacity(relative.len());
        let mut lower_image = Vec::with_capacity(relative.len());
        let mut upper_image = Vec::with_capacity(relative.len());
        for (index, base) in relative.into_iter().enumerate() {
            let base = if index == 0 {
                if let AreaAnchor::Source { area, .. } = &self.anchor {
                    let exact = a.expansion_interval(area)?;
                    a.bounds(
                        &c.maximum(base.lower(), exact.lower())?,
                        &c.minimum(base.upper(), exact.upper())?,
                    )?
                } else {
                    base
                }
            } else {
                base
            };
            let lower = range::add_endpoint(a, &base, &self.offsets.lower)?;
            let upper = range::add_endpoint(a, &base, &self.offsets.upper)?;
            image.push(a.bounds(lower.lower(), upper.upper())?);
            lower_image.push(lower);
            upper_image.push(upper);
        }
        let low: Vec<_> = image.iter().map(Interval::lower).collect();
        let high: Vec<_> = image.iter().map(Interval::upper).collect();
        let low = c.stack(&low, 1)?;
        let high = c.stack(&high, 1)?;
        let count = boundaries::counts(a, bands)?.reshape(&[n, 1])?;
        let cap_low = c.gather(&low, 1, &count)?.reshape(&[n])?;
        let cap_high = c.gather(&high, 1, &count)?.reshape(&[n])?;
        let cap_relative = a.bounds(&cap_low, &cap_high)?;
        let deficit = a.sub(&zero, &cap_relative)?;
        let area = if let AreaAnchor::Source { area, .. } = &self.anchor {
            let low = a.expansion_add(area, &self.offsets.lower)?;
            let high = a.expansion_add(area, &self.offsets.upper)?;
            let low = a.expansion_interval(&low)?;
            let high = a.expansion_interval(&high)?;
            a.bounds(low.lower(), high.upper())?
        } else {
            a.bounds(image[0].lower(), image[0].upper())?
        };
        let area = if matches!(&self.anchor, AreaAnchor::Zero) {
            // Admission proves the exact scale nonnegative. Parameter sign fixes the physical offset sign despite a symmetric roundoff radius.
            let nonnegative = match self.direction {
                AreaDirection::Add => c.ge(self.parameter.lower(), &zero_tensor)?,
                AreaDirection::Subtract => c.le(self.parameter.upper(), &zero_tensor)?,
            };
            a.bounds(
                &c.where_cond(
                    &nonnegative,
                    &c.maximum(area.lower(), &zero_tensor)?,
                    area.lower(),
                )?,
                area.upper(),
            )?
        } else {
            area
        };
        admission::reject(
            a,
            &c.lt(area.lower(), &zero_tensor)?,
            "area query image is not proved nonnegative. Change the chart or tighten its certificates",
        )?;
        admission::reject(
            a,
            &c.lt(deficit.lower(), &zero_tensor)?,
            "area query image is not proved below its physical cap. Change the chart or tighten its certificates",
        )?;
        let above_start = a.bounds(
            &low.narrow(1, 0, bands.band_count)?,
            &high.narrow(1, 0, bands.band_count)?,
        )?;
        let end_relative = a.bounds(
            &low.narrow(1, 1, bands.band_count)?,
            &high.narrow(1, 1, bands.band_count)?,
        )?;
        let zero_matrix = a.point(&Tensor::zeros(
            &[n, bands.band_count],
            DType::F64,
            c.device(),
        )?)?;
        let below_end = a.sub(&zero_matrix, &end_relative)?;
        let lower_endpoint = endpoint_differences(a, &lower_image, bands.band_count, &zero_matrix)?;
        let upper_endpoint = endpoint_differences(a, &upper_image, bands.band_count, &zero_matrix)?;
        let candidate = c.mul(
            &bands.valid,
            &c.mul(
                &c.mul(
                    &c.ge(above_start.upper(), zero_matrix.lower())?,
                    &c.ge(below_end.upper(), zero_matrix.lower())?,
                )?,
                &c.mul(
                    &c.le(above_start.lower(), bands.area_increment.upper())?,
                    &c.le(below_end.lower(), bands.area_increment.upper())?,
                )?,
            )?,
        )?;
        // Whole-image support and the complete closed partition establish candidate coverage.
        // These four necessary conditions cannot exclude any physical state in an intersected band.
        Ok(SupportedAreaQuery {
            query: self,
            area,
            above_start,
            below_end,
            candidate,
            lower_endpoint,
            upper_endpoint,
        })
    }
}

fn endpoint_differences<R>(
    a: &mut Arithmetic<'_, R>,
    image: &[Interval<R>],
    bands: usize,
    zero: &Interval<R>,
) -> HydraulicResult<EndpointDifferences<R>>
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
    let lower: Vec<_> = image.iter().map(Interval::lower).collect();
    let upper: Vec<_> = image.iter().map(Interval::upper).collect();
    let lower = c.stack(&lower, 1)?;
    let upper = c.stack(&upper, 1)?;
    let above_start = a.bounds(&lower.narrow(1, 0, bands)?, &upper.narrow(1, 0, bands)?)?;
    let end_relative = a.bounds(&lower.narrow(1, 1, bands)?, &upper.narrow(1, 1, bands)?)?;
    let below_end = a.sub(zero, &end_relative)?;
    Ok(EndpointDifferences {
        above_start,
        below_end,
    })
}
