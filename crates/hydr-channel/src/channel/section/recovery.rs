use super::{
    CrossSectionBatch, CrossSectionError, CrossSectionQuery, CrossSectionRecovery,
    CrossSectionRecoveryConfig, evaluation,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::OptimizeError;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};
use std::cell::RefCell;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> CrossSectionBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Recover static stages within surveyed support and independently admit forward areas.
    pub fn recover_stages(
        &self,
        areas_m2: &[f64],
        config: &CrossSectionRecoveryConfig,
    ) -> Result<CrossSectionRecovery<R>, CrossSectionError> {
        if areas_m2.len() != self.context.definitions.len() {
            return Err(CrossSectionError::input(format!(
                "area count={}, section count={}: supply one area per ordered section",
                areas_m2.len(),
                self.context.definitions.len()
            )));
        }
        self.recover_stages_tensor(&self.context.tensor(areas_m2)?, config)
    }

    pub(crate) fn recover_stages_tensor(
        &self,
        areas: &Tensor<R>,
        config: &CrossSectionRecoveryConfig,
    ) -> Result<CrossSectionRecovery<R>, CrossSectionError> {
        config.check()?;
        let context = &self.context;
        let c = &context.client;
        if areas.shape() != [context.definitions.len()]
            || areas.dtype() != DType::F64
            || !areas.device().is_same(c.device())
        {
            return Err(CrossSectionError::input(format!(
                "area shape={:?}, dtype={:?}: supply one F64 area per ordered section on the configured device",
                areas.shape(),
                areas.dtype()
            )));
        }
        for (mask, requirement) in [
            (context.invalid(areas)?, "finite"),
            (c.lt(areas, &context.zero)?, "nonnegative"),
        ] {
            if let Some(index) = context.rejected_lane(&mask)? {
                let value = areas.narrow(0, index, 1)?.item::<f64>()?;
                return Err(CrossSectionError::input(format!(
                    "requested area={value}: supply a {requirement} area"
                ))
                .at(index));
            }
        }
        let capacity = evaluation::evaluate(context, &context.cap)?.area;
        if let Some(index) = context.rejected_lane(&c.gt(areas, &capacity)?)? {
            let value = areas.narrow(0, index, 1)?.item::<f64>()?;
            let maximum = capacity.narrow(0, index, 1)?.item::<f64>()?;
            return Err(CrossSectionError::input(format!(
                "requested area={value}, supported area={maximum}: supply an area within the surveyed support"
            ))
            .at(index));
        }
        let positive = c.gt(areas, &context.zero)?;
        let height_cap = c.sub(&context.cap, &context.bed)?;
        let area_tolerance =
            c.add_scalar(&c.mul_scalar(areas, config.area_rtol)?, config.area_atol_m2)?;
        context.section_finite(&area_tolerance, "recovery area tolerance")?;
        context.reject_section(
            &c.mul(&positive, &c.le(&area_tolerance, &context.zero)?)?,
            &area_tolerance,
            "positive recovery area tolerance",
        )?;
        let stage = |height: &Tensor<R>| -> Result<Tensor<R>, CrossSectionError> {
            let at_bed = c.eq(height, &context.zero)?;
            let at_cap = c.eq(height, &height_cap)?;
            let interior = c.add(&context.bed, height)?;
            Ok(c.where_cond(
                &at_bed,
                &context.bed,
                &c.where_cond(&at_cap, &context.cap, &interior)?,
            )?)
        };
        let diagnostic = RefCell::new(None);
        let root = c.bisect_batched(
            |height| {
                let result = (|| {
                    let fields = evaluation::evaluate(context, &stage(height)?)?;
                    Ok::<_, CrossSectionError>(c.sub(&fields.area, areas)?)
                })();
                match result {
                    Ok(residual) => Ok(residual),
                    Err(error) => {
                        let mut slot = diagnostic.try_borrow_mut().map_err(|_| {
                            numr::error::Error::InvalidArgument {
                                arg: "cross-section recovery diagnostic",
                                reason: "callback diagnostic channel is already borrowed".into(),
                            }
                        })?;
                        if slot.is_none() {
                            *slot = Some(error);
                        }
                        Err(numr::error::Error::InvalidArgument {
                            arg: "cross-section recovery callback",
                            reason: "full section evaluation rejected the candidate".into(),
                        })
                    }
                }
            },
            &context.zero_sections,
            &height_cap,
            &BatchedBisectionOptions {
                atol: config.height_atol_m,
                rtol: config.height_rtol,
                max_iter: config.max_iter,
                max_expansions: 0,
                expansion_factor: 2.0,
            },
        );
        if let Some(error) = diagnostic.into_inner() {
            return Err(error);
        }
        let root = root.map_err(|error| {
            let message = format!(
                "solvr bisect_batched: {error}. Increase max_iter, select representable tolerances, or rescale geometry"
            );
            match error {
                OptimizeError::NumrError(_) => CrossSectionError::Runtime {
                    section: None,
                    context: message,
                },
                _ => CrossSectionError::numerical(message),
            }
        })?;
        let lower_stage = stage(&root.lower)?;
        let upper_stage = stage(&root.upper)?;
        let lower = evaluation::evaluate(context, &lower_stage)?;
        let upper = evaluation::evaluate(context, &upper_stage)?;
        let zero_area = c.eq(areas, &context.zero)?;
        let lower_eligible = c.maximum(&zero_area, &lower.wet)?;
        let upper_eligible = c.maximum(&zero_area, &upper.wet)?;
        context.reject_section(
            &c.eq(&c.maximum(&lower_eligible, &upper_eligible)?, &context.zero)?,
            areas,
            "positive requested area has no represented wet endpoint",
        )?;
        let use_lower = c.mul(
            &lower_eligible,
            &c.maximum(
                &c.eq(&upper_eligible, &context.zero)?,
                &c.le(&c.abs(&root.f_lower)?, &c.abs(&root.f_upper)?)?,
            )?,
        )?;
        let selected_height = c.where_cond(&use_lower, &root.lower, &root.upper)?;
        let height_tolerance = c.add_scalar(
            &c.mul_scalar(&selected_height, config.height_rtol)?,
            config.height_atol_m,
        )?;
        context.section_finite(&height_tolerance, "recovery height tolerance")?;
        context.reject_section(
            &c.mul(&positive, &c.le(&height_tolerance, &context.zero)?)?,
            &height_tolerance,
            "positive recovery height tolerance",
        )?;
        let selected = c.where_cond(&use_lower, &lower_stage, &upper_stage)?;
        let fields = evaluation::evaluate(context, &selected)?;
        context.reject_section(
            &c.mul(&positive, &c.le(&fields.wet, &context.zero)?)?,
            areas,
            "positive requested area requires represented wet geometry",
        )?;
        let residual = c.abs(&c.sub(&fields.area, areas)?)?;
        context.section_finite(&residual, "recovered forward-area residual")?;
        context.reject_section(
            &c.gt(&residual, &area_tolerance)?,
            &residual,
            "recovered forward-area residual exceeds the configured area tolerance",
        )?;
        Ok(CrossSectionRecovery {
            requested_area: areas.clone(),
            geometry: CrossSectionQuery {
                context: Arc::clone(context),
                fields,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CrossSectionDefinition, CrossSectionPoint};
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn native_recovery<R: Runtime<DType = DType>>(client: R::Client) -> TestResult
    where
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + HypotOps<R>
            + IndexingOps<R>
            + MidpointOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>
            + BatchedBisectionAlgorithms<R>,
    {
        let definitions = ["first", "second"]
            .into_iter()
            .map(|id| CrossSectionDefinition {
                id: id.into(),
                vertical_datum: "Analytical section datum".into(),
                points: [(0.0, 4.0), (0.0, 0.0), (2.0, 0.0), (2.0, 4.0)]
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 4.0,
                profile_provenance: "Explicit two-metre rectangular geometry".into(),
            })
            .collect();
        let batch = CrossSectionBatch::<R>::new(client, definitions)?;
        let context = Arc::clone(&batch.context);
        let config = CrossSectionRecoveryConfig::default();
        let strided = |values: [f64; 2]| -> Result<Tensor<R>, numr::error::Error> {
            Ok(Tensor::from_slice(
                &[f64::NAN, values[0], f64::INFINITY, values[1]],
                &[2, 2],
                context.client.device(),
            )?
            .narrow(1, 1, 1)?
            .squeeze(Some(1)))
        };
        let original = strided([-0.0, 2.0])?;
        assert!(!original.is_contiguous());
        let recovery = batch.recover_stages_tensor(&original, &config)?;
        assert!(Arc::ptr_eq(&context, &recovery.geometry.context));
        assert!(!recovery.requested_area.is_contiguous());
        let retained = recovery.requested_area.contiguous()?.try_to_vec::<f64>()?;
        assert_eq!(retained[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(retained[1].to_bits(), 2.0_f64.to_bits());
        let properties = recovery.properties()?;
        assert_eq!(
            properties[0].requested_area_m2.to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            properties[0].geometry.query_stage_m.to_bits(),
            0.0_f64.to_bits()
        );
        assert!(properties[0].geometry.dry);
        assert_eq!(properties[1].geometry.query_stage_m, 1.0);
        assert_eq!(properties[1].geometry.area_m2, 2.0);
        let wrong_rank =
            Tensor::<R>::from_slice(&[0.0_f64, 2.0], &[1, 2], context.client.device())?;
        let wrong_dtype = Tensor::<R>::from_slice(&[0.0_f32, 2.0], &[2], context.client.device())?;
        for areas in [&wrong_rank, &wrong_dtype] {
            let error = batch
                .recover_stages_tensor(areas, &config)
                .err()
                .ok_or("Invalid native area layout was accepted")?;
            assert!(matches!(
                error,
                CrossSectionError::InvalidInput { section: None, .. }
            ));
        }
        for (values, expected) in [
            ([0.0, f64::NAN], 1),
            ([f64::NAN, f64::INFINITY], 0),
            ([9.0, f64::NAN], 1),
            ([-1.0, f64::NAN], 1),
        ] {
            let error = batch
                .recover_stages_tensor(&strided(values)?, &config)
                .err()
                .ok_or("Nonfinite native area was accepted")?;
            assert!(
                matches!(error, CrossSectionError::InvalidInput { section: Some(index), .. } if index == expected)
            );
        }
        assert_eq!(recovery.properties()?, properties);
        assert_eq!(
            batch.recover_stages(&[0.0, 2.0], &config)?.properties()?[1]
                .geometry
                .area_m2,
            2.0
        );
        Ok(())
    }

    #[test]
    fn cpu_strided_recovery_retains_operands_context_and_admission() -> TestResult {
        native_recovery::<CpuRuntime>(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_strided_recovery_retains_operands_context_and_admission() -> TestResult {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        native_recovery::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
