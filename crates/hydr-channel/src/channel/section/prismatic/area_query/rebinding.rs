use super::super::hydraulic_geometry::GatheredBands;
use super::{SupportedAreaQuery, admission};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

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
    /// Rebind the retained affine map and prove support for its complete new parameter image.
    pub(crate) fn with_parameter_bounds(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        admission::chart(
            a,
            bands,
            #[cfg(test)]
            &self.query.anchor,
            &self.query.scale,
            parameter,
        )?;
        admission::metadata(
            a,
            &self.query.original_rows,
            bands.section_owner.shape(),
            DType::I64,
        )?;
        let parameter = a.bounds(parameter.lower(), parameter.upper())?;
        self.reconstruct(a, bands, parameter)
    }

    /// Rebind to the singleton parameter `point`. Its endpoints then reuse these certificates.
    pub(crate) fn with_parameter_point(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        point: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        let parameter = a.point(point)?;
        let mut query = self.with_parameter_bounds(a, bands, &parameter)?;
        query.at_point = true;
        Ok(query)
    }
}

#[cfg(test)]
mod tests {
    use super::super::endpoints::PhysicalEndpoint;
    use super::super::query_fixture::{float_fields, integer_fields, snapshot};
    use super::super::{fixtures, test_support as support};
    use super::*;
    use crate::channel::{
        flow::surveyed::error::{self, HydraulicError},
        section::{CrossSectionError, HydraulicGeometry},
    };
    use numr::{ops::BinaryOps, tensor::Tensor};
    use solvr::common::arithmetic::with_interval_arithmetic;
    use std::collections::{BTreeMap, BTreeSet};
    use support::{Client, Result, hydraulic};

    fn parameter<R>(
        client: &R::Client,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<(Interval<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let storage = lower
            .iter()
            .zip(upper)
            .flat_map(|(&lo, &hi)| [f64::NAN, lo, f64::NAN, hi])
            .collect::<Vec<_>>();
        let storage = Tensor::from_slice(&storage, &[lower.len(), 4], client.device())?;
        let lo = storage.narrow(1, 1, 1)?.squeeze(Some(1));
        let hi = storage.narrow(1, 3, 1)?.squeeze(Some(1));
        assert!(lo.offset() > 0 && hi.offset() > 0);
        if lower.len() > 1 {
            assert!(!lo.is_contiguous() && !hi.is_contiguous());
        }
        let interval = hydraulic(error::checked::<R, _>(
            client,
            "strided chart parameter capture",
            |a| Ok(a.bounds(&lo, &hi)?),
        ))?;
        Ok((interval, storage))
    }

    fn rejects<T>(result: HydraulicResult<T>) -> Result {
        let error = result
            .err()
            .ok_or("Parameter rebinding accepts an invalid request")?;
        assert!(
            matches!(
                error,
                HydraulicError::Geometry(CrossSectionError::InvalidInput { .. })
            ),
            "{error:?}"
        );
        Ok(())
    }

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
        let mut opaque = false;
        for cases in groups.values() {
            let charts = cases.iter().map(|case| case.chart).collect::<Vec<_>>();
            let input = support::input::<R>(&client, &section, &charts, true, false)?;
            let (bands, parent) = hydraulic(error::checked::<R, _>(
                &client,
                "chart rebinding parent",
                |a| {
                    let bands = geometry.gather(a, &input.owners)?;
                    let parent = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                    Ok((bands, parent))
                },
            ))?;
            let before = snapshot(&parent)?;
            let lo = parent.query.parameter.lower().try_to_vec::<f64>()?;
            let hi = parent.query.parameter.upper().try_to_vec::<f64>()?;
            let (parameters, storage) = parameter::<R>(&client, &lo, &hi)?;
            let rebound = hydraulic(error::checked::<R, _>(
                &client,
                "same parameter chart rebinding",
                |a| parent.with_parameter_bounds(a, &bands, &parameters),
            ))?;
            assert!(Arc::ptr_eq(&parent.query.model, &rebound.query.model));
            assert_eq!(before, snapshot(&rebound)?);
            let value = hydraulic(error::checked::<R, _>(
                &client,
                "rebound chart frozen geometry",
                |a| geometry.recover_query(a, &bands, &rebound),
            ))?;
            support::check(&support::snapshot(&value)?, cases);
            for physical in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
                let (old, new, widened) = hydraulic(error::checked::<R, _>(
                    &client,
                    "rebound singleton chart",
                    |a| {
                        let old = parent.endpoint(a, &bands, physical)?;
                        let parameter =
                            a.bounds(old.query.parameter.lower(), old.query.parameter.upper())?;
                        let new = parent.with_parameter_bounds(a, &bands, &parameter)?;
                        let widened =
                            new.with_parameter_bounds(a, &bands, &parent.query.parameter)?;
                        Ok((old, new, widened))
                    },
                ))?;
                assert_eq!(snapshot(&old)?, snapshot(&new)?);
                assert_eq!(before, snapshot(&widened)?);
                let (old_value, new_value, raw) = hydraulic(error::checked::<R, _>(
                    &client,
                    "singleton chart geometry identity",
                    |a| {
                        let witnesses = geometry.boundary_areas(a)?.gather(a, &input.owners)?;
                        Ok((
                            geometry.recover_query(a, &bands, &old)?,
                            geometry.recover_query(a, &bands, &new)?,
                            new.retained_endpoint_area(a, &bands, &witnesses)?,
                        ))
                    },
                ))?;
                assert_eq!(
                    support::snapshot(&old_value)?,
                    support::snapshot(&new_value)?
                );
                opaque |= raw
                    .abs_error()
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .any(|value| value > 0.0);
            }
            let last = i64::try_from(cases.len() - 1)?;
            let rows = Tensor::from_slice(&[last, 0, last], &[3], client.device())?;
            let selected = hydraulic(error::checked::<R, _>(
                &client,
                "selected chart parameter rebinding",
                |a| {
                    let bands = bands.select(a, &rows)?;
                    let selected = parent.select(a, &rows)?;
                    let rebound =
                        selected.with_parameter_bounds(a, &bands, &selected.query.parameter)?;
                    Ok((selected, rebound))
                },
            ))?;
            assert_eq!(snapshot(&selected.0)?, snapshot(&selected.1)?);
            assert_eq!(
                selected.1.query.original_rows.try_to_vec::<i64>()?,
                [last, 0, last]
            );
            let rebound_before = snapshot(&rebound)?;
            for value in [parameters.lower(), parameters.upper(), &storage] {
                client.copy_into(
                    value,
                    &Tensor::zeros(value.shape(), value.dtype(), client.device())?,
                )?;
            }
            assert_eq!(rebound_before, snapshot(&rebound)?);
            for value in float_fields(&rebound)
                .into_iter()
                .chain(integer_fields(&rebound))
            {
                client.copy_into(
                    value,
                    &Tensor::zeros(value.shape(), value.dtype(), client.device())?,
                )?;
            }
            assert_eq!(before, snapshot(&parent)?);
            count += cases.len();
        }
        assert_eq!(count, 17);
        assert_eq!(
            fixtures::CASES
                .iter()
                .map(|case| case.group)
                .collect::<BTreeSet<_>>()
                .len(),
            15
        );
        opaque |= uncertain_endpoints::<R>(&client, &section)?;
        assert!(
            opaque,
            "Retained non-point endpoint uncertainty must survive rebinding"
        );
        admission::<R>(&client, &section)?;
        Ok(())
    }

    /// Frozen fixture endpoints resolve exactly in two components.
    /// This chart anchors at the decimal knot and offsets far below its low component.
    /// Each singleton endpoint must keep the same nonzero compression radius after rebinding.
    fn uncertain_endpoints<R>(
        client: &R::Client,
        section: &crate::channel::section::PrismaticSectionBatch<R>,
    ) -> Result<bool>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let case = fixtures::CASES
            .iter()
            .find(|case| case.id == "decimal_knot_signed_offsets")
            .ok_or("Missing decimal knot chart")?;
        let mut chart = case.chart;
        // 2^-130 lies more than one double below the anchor's low component.
        chart.scale[0] = 0x37d0_0000_0000_0000;
        let input = support::input::<R>(client, section, &[chart], true, false)?;
        let geometry = HydraulicGeometry::from_batch(section);
        let (bands, parent) = hydraulic(error::checked::<R, _>(
            client,
            "uncertain endpoint parent",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let parent = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                Ok((bands, parent))
            },
        ))?;
        let before = snapshot(&parent)?;
        for physical in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
            let (old, new, widened, old_raw, new_raw) = hydraulic(error::checked::<R, _>(
                client,
                "uncertain endpoint rebinding",
                |a| {
                    let old = parent.endpoint(a, &bands, physical)?;
                    let parameter =
                        a.bounds(old.query.parameter.lower(), old.query.parameter.upper())?;
                    let new = parent.with_parameter_bounds(a, &bands, &parameter)?;
                    let widened = new.with_parameter_bounds(a, &bands, &parent.query.parameter)?;
                    let witnesses = geometry.boundary_areas(a)?.gather(a, &input.owners)?;
                    let old_raw = old.retained_endpoint_area(a, &bands, &witnesses)?;
                    let new_raw = new.retained_endpoint_area(a, &bands, &witnesses)?;
                    Ok((old, new, widened, old_raw, new_raw))
                },
            ))?;
            assert_eq!(snapshot(&old)?, snapshot(&new)?);
            assert_eq!(before, snapshot(&widened)?);
            let bits = |value: &Tensor<R>| -> Result<Vec<u64>> {
                Ok(value
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect())
            };
            for (old, new) in [
                (old_raw.high(), new_raw.high()),
                (old_raw.low(), new_raw.low()),
                (old_raw.abs_error(), new_raw.abs_error()),
            ] {
                assert_eq!(bits(old)?, bits(new)?);
            }
            if new_raw
                .abs_error()
                .try_to_vec::<f64>()?
                .into_iter()
                .any(|value| value <= 0.0)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn admission<R>(
        client: &R::Client,
        section: &crate::channel::section::PrismaticSectionBatch<R>,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let case = fixtures::CASES
            .iter()
            .find(|case| case.chart.kind == fixtures::AnchorKind::Zero && case.chart.scale[0] != 0)
            .ok_or("Missing positive zero-anchor chart")?;
        let input = support::input::<R>(client, section, &[case.chart], false, false)?;
        let geometry = HydraulicGeometry::from_batch(section);
        let foreign_section = support::batch::<R>(client)?;
        let foreign_geometry = HydraulicGeometry::from_batch(&foreign_section);
        let (bands, parent, foreign, wrong_owners) = hydraulic(error::checked::<R, _>(
            client,
            "rebind admission graph",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let parent = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                let foreign = foreign_geometry.gather(a, &input.owners)?;
                let wrong = i64::try_from((case.chart.owner + 1) % fixtures::PROFILES.len())
                    .map_err(|e| HydraulicError::Numerical {
                        context: e.to_string(),
                    })?;
                let wrong_owners =
                    geometry.gather(a, &Tensor::from_slice(&[wrong], &[1], client.device())?)?;
                Ok((bands, parent, foreign, wrong_owners))
            },
        ))?;
        let before = snapshot(&parent)?;
        rejects(error::checked::<R, _>(client, "foreign rebind", |a| {
            parent.with_parameter_bounds(a, &foreign, &parent.query.parameter)
        }))?;
        rejects(error::checked::<R, _>(client, "owner rebind", |a| {
            parent.with_parameter_bounds(a, &wrong_owners, &parent.query.parameter)
        }))?;
        for shape in [&[][..], &[1, 1][..], &[2][..]] {
            let bad = hydraulic(error::checked::<R, _>(
                client,
                "malformed parameter shape",
                |a| Ok(a.point(&Tensor::zeros(shape, a.dtype(), client.device())?)?),
            ))?;
            rejects(error::checked::<R, _>(
                client,
                "rebind parameter shape",
                |a| parent.with_parameter_bounds(a, &bands, &bad),
            ))?;
        }
        let bad_dtype = with_interval_arithmetic::<R, _, _, _>(
            client,
            DType::F32,
            "foreign parameter dtype",
            |a| a.point(&Tensor::from_slice(&[0.0_f32], &[1], client.device())?),
        )?;
        rejects(error::checked::<R, _>(
            client,
            "rebind parameter dtype",
            |a| parent.with_parameter_bounds(a, &bands, &bad_dtype),
        ))?;
        for (lo, hi) in [
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (2.0, 1.0),
            (-1.0, -1.0),
            (f64::MAX, f64::MAX),
        ] {
            let (parameter, _) = parameter::<R>(client, &[0.0], &[1.0])?;
            client.copy_into(
                parameter.lower(),
                &Tensor::from_slice(&[lo], &[1], client.device())?,
            )?;
            client.copy_into(
                parameter.upper(),
                &Tensor::from_slice(&[hi], &[1], client.device())?,
            )?;
            rejects(error::checked::<R, _>(
                client,
                "rebind parameter admission",
                |a| parent.with_parameter_bounds(a, &bands, &parameter),
            ))?;
        }
        assert_eq!(before, snapshot(&parent)?);
        Ok(())
    }

    #[test]
    fn configured_cpu_parameter_rebinding_preserves_supported_charts_and_owned_witnesses() -> Result
    {
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
    fn cuda_parameter_rebinding_preserves_supported_charts_and_owned_witnesses() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
