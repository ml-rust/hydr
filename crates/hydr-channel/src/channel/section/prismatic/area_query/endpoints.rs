use super::super::hydraulic_geometry::GatheredBands;
use super::{AreaDirection, AreaQuery, SupportedAreaQuery, admission, range, selection};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, TypeConversionOps,
    },
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
        if self.at_point && self.query.band_count == bands.band_count {
            // Both endpoints are the singleton parameter, and its reconstruction repeats these
            // certificates from the same normalized scale, anchor, and owner bands.
            return self.retained(a);
        }
        let parameter = match (self.query.direction, endpoint) {
            (AreaDirection::Add, PhysicalEndpoint::Lower)
            | (AreaDirection::Subtract, PhysicalEndpoint::Upper) => self.query.parameter.lower(),
            (AreaDirection::Add, PhysicalEndpoint::Upper)
            | (AreaDirection::Subtract, PhysicalEndpoint::Lower) => self.query.parameter.upper(),
        };
        let parameter = a.point(parameter)?;
        let mut query = self.reconstruct(a, bands, parameter)?;
        query.at_point = true;
        Ok(query)
    }

    /// Owned copies of every certificate. Copies keep each bit and the singleton marker.
    fn retained(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Self> {
        let interval =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| a.bounds(value.lower(), value.upper());
        let expansion =
            |a: &mut Arithmetic<'_, R>,
             value: &solvr::common::arithmetic::CertifiedExpansion<R>| {
                a.expansion_bounds(value.high(), value.low(), value.abs_error())
            };
        let c = a.client();
        let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
            c.copy_into(&owned, value)?;
            Ok(owned)
        };
        Ok(Self {
            query: AreaQuery {
                model: Arc::clone(&self.query.model),
                anchor: selection::capture_anchor(a, &self.query.anchor)?,
                offsets: super::chart::ExpandedBounds {
                    lower: expansion(a, &self.query.offsets.lower)?,
                    upper: expansion(a, &self.query.offsets.upper)?,
                },
                direction: self.query.direction,
                scale: expansion(a, &self.query.scale)?,
                parameter: interval(a, &self.query.parameter)?,
                owners: copy(&self.query.owners)?,
                original_rows: copy(&self.query.original_rows)?,
                band_count: self.query.band_count,
            },
            area: interval(a, &self.area)?,
            above_start: interval(a, &self.above_start)?,
            below_end: interval(a, &self.below_end)?,
            candidate: copy(&self.candidate)?,
            lower_endpoint: super::chart::EndpointDifferences {
                above_start: interval(a, &self.lower_endpoint.above_start)?,
                below_end: interval(a, &self.lower_endpoint.below_end)?,
            },
            upper_endpoint: super::chart::EndpointDifferences {
                above_start: interval(a, &self.upper_endpoint.above_start)?,
                below_end: interval(a, &self.upper_endpoint.below_end)?,
            },
            at_point: true,
        })
    }

    /// Reprove only changed rows. Preserve every untouched certificate and the complete original affine expression.
    pub(crate) fn with_selected_parameter_bounds(
        mut self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        rows: &Tensor<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        admission::metadata(a, rows, &[rows.numel()], DType::I64)?;
        admission::metadata(a, parameter.lower(), &[rows.numel()], a.dtype())?;
        admission::metadata(a, parameter.upper(), &[rows.numel()], a.dtype())?;
        let c = a.client();
        let count = i64::try_from(self.query.owners.numel()).map_err(|_| {
            crate::channel::section::CrossSectionError::input(
                "selected parameter row count exceeds I64 indexing. Split the query",
            )
        })?;
        let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
        let upper = Tensor::from_slice(&[count], &[], c.device())?;
        let invalid = c.maximum(&c.lt(rows, &zero)?, &c.ge(rows, &upper)?)?;
        admission::reject(
            a,
            &c.cast(&invalid, a.dtype())?,
            "selected parameter row lies outside the supported query. Supply admitted rows",
        )?;
        if rows.numel() > 1 {
            let sorted = c.sort(rows, 0, false)?;
            let n = rows.numel() - 1;
            let duplicates = c.eq(&sorted.narrow(0, 0, n)?, &sorted.narrow(0, 1, n)?)?;
            admission::reject(
                a,
                &c.cast(&duplicates, a.dtype())?,
                "selected parameter rows repeat a destination. Supply unique rows",
            )?;
        }
        if rows.numel() == 0 {
            return Ok(self);
        }
        self.at_point = false;
        let selected_bands = bands.select(a, rows)?;
        let selected =
            self.select(a, rows)?
                .with_parameter_bounds(a, &selected_bands, parameter)?;
        let interval = |a: &mut Arithmetic<'_, R>,
                        old: &Interval<R>,
                        new: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.index_put(old.lower(), 0, rows, new.lower())?,
                &c.index_put(old.upper(), 0, rows, new.upper())?,
            )?)
        };
        let expansion =
            |a: &mut Arithmetic<'_, R>,
             old: &solvr::common::arithmetic::CertifiedExpansion<R>,
             new: &solvr::common::arithmetic::CertifiedExpansion<R>|
             -> HydraulicResult<solvr::common::arithmetic::CertifiedExpansion<R>> {
                Ok(a.expansion_bounds(
                    &c.index_put(old.high(), 0, rows, new.high())?,
                    &c.index_put(old.low(), 0, rows, new.low())?,
                    &c.index_put(old.abs_error(), 0, rows, new.abs_error())?,
                )?)
            };
        self.query.parameter = interval(a, &self.query.parameter, &selected.query.parameter)?;
        self.query.offsets.lower =
            expansion(a, &self.query.offsets.lower, &selected.query.offsets.lower)?;
        self.query.offsets.upper =
            expansion(a, &self.query.offsets.upper, &selected.query.offsets.upper)?;
        self.area = interval(a, &self.area, &selected.area)?;
        self.above_start = interval(a, &self.above_start, &selected.above_start)?;
        self.below_end = interval(a, &self.below_end, &selected.below_end)?;
        self.candidate = c.index_put(&self.candidate, 0, rows, &selected.candidate)?;
        self.lower_endpoint.above_start = interval(
            a,
            &self.lower_endpoint.above_start,
            &selected.lower_endpoint.above_start,
        )?;
        self.lower_endpoint.below_end = interval(
            a,
            &self.lower_endpoint.below_end,
            &selected.lower_endpoint.below_end,
        )?;
        self.upper_endpoint.above_start = interval(
            a,
            &self.upper_endpoint.above_start,
            &selected.upper_endpoint.above_start,
        )?;
        self.upper_endpoint.below_end = interval(
            a,
            &self.upper_endpoint.below_end,
            &selected.upper_endpoint.below_end,
        )?;
        Ok(self)
    }

    pub(super) fn reconstruct(
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

#[cfg(test)]
mod tests {
    use super::super::{fixtures, test_support as support};
    use super::*;
    use crate::channel::flow::surveyed::error;
    use crate::channel::section::HydraulicGeometry;
    use std::collections::BTreeMap;
    use support::{Client, Result, hydraulic};

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(&client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let mut groups = BTreeMap::new();
        for case in fixtures::CASES.iter().rev() {
            groups
                .entry((case.chart.kind, case.chart.add))
                .or_insert_with(Vec::new)
                .push(case);
        }
        let mut count = 0;
        for cases in groups.values() {
            let charts: Vec<_> = cases.iter().map(|case| case.chart).collect();
            let input = support::input::<R>(&client, &section, &charts, true, false)?;
            let last = i64::try_from(cases.len() - 1)?;
            let rows = Tensor::from_slice(&[last, 0, last], &[3], client.device())?;
            for (physical, upper) in [
                (PhysicalEndpoint::Lower, false),
                (PhysicalEndpoint::Upper, true),
            ] {
                let (endpoint, selected, value, parent_rows, selected_rows) = hydraulic(
                    error::checked::<R, _>(&client, "physical area endpoints", |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let parent = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                        let endpoint = parent.endpoint(a, &bands, physical)?;
                        let value = geometry.recover_query(a, &bands, &endpoint)?;
                        let selected_bands = bands.select(a, &rows)?;
                        let selected_parent = parent.select(a, &rows)?;
                        let selected = selected_parent.endpoint(a, &selected_bands, physical)?;
                        let parent_rows =
                            selected_parent.query.original_rows.try_to_vec::<i64>()?;
                        let selected_rows = selected.query.original_rows.try_to_vec::<i64>()?;
                        Ok((endpoint, selected, value, parent_rows, selected_rows))
                    }),
                )?;
                assert_eq!(parent_rows, [last, 0, last]);
                assert_eq!(selected_rows, parent_rows);
                let fields = [
                    &value.state.area,
                    &value.state.forward_area,
                    &value.state.depth,
                    &value.state.top_width,
                    &value.state.integral,
                    &value.state.hydraulic_depth,
                ];
                for (field, reference_index) in fields.into_iter().zip([0, 0, 1, 2, 3, 4]) {
                    let lower = field.lower().try_to_vec::<f64>()?;
                    let higher = field.upper().try_to_vec::<f64>()?;
                    for (row, case) in cases.iter().enumerate() {
                        let reference = case.fields[reference_index];
                        let expected = if upper {
                            reference.maximum
                        } else {
                            reference.minimum
                        }
                        .map(f64::from_bits);
                        let actual = [lower[row], higher[row]];
                        assert!(
                            actual[0] <= expected[0] && expected[1] <= actual[1],
                            "{}, upper={upper}, field={reference_index}: {actual:?} excludes {expected:?}",
                            case.id
                        );
                        let error = (actual[0] - expected[1])
                            .abs()
                            .max((actual[1] - expected[0]).abs());
                        assert!(
                            error <= f64::from_bits(reference.allowances[usize::from(upper)]),
                            "{}, upper={upper}, field={reference_index}: error={error}",
                            case.id
                        );
                    }
                }
                let selected_owners = selected.query.owners.try_to_vec::<i64>()?;
                client.copy_into(
                    &endpoint.query.owners,
                    &Tensor::zeros(endpoint.query.owners.shape(), DType::I64, client.device())?,
                )?;
                assert_eq!(selected.query.owners.try_to_vec::<i64>()?, selected_owners);
                let selected_parameters = selected.query.parameter.lower().try_to_vec::<f64>()?;
                client.copy_into(
                    endpoint.query.parameter.lower(),
                    &Tensor::zeros(
                        endpoint.query.parameter.lower().shape(),
                        DType::F64,
                        client.device(),
                    )?,
                )?;
                assert_eq!(
                    selected.query.parameter.lower().try_to_vec::<f64>()?,
                    selected_parameters
                );
            }
            count += cases.len();
        }
        assert_eq!(count, 17);
        Ok(())
    }

    #[test]
    fn cpu_physical_endpoints_preserve_map_geometry_and_original_rows() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let previous = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        matrix::<CpuRuntime>(client)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), previous);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_physical_endpoints_preserve_map_geometry_and_original_rows() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
