use super::{
    CrossSectionDefinition, CrossSectionError, CrossSectionQuery, context::Context, evaluation,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Immutable surveyed profiles evaluated through the caller's configured Numr client.
#[derive(Clone, Debug)]
pub struct CrossSectionBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
}

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
        + UnaryOps<R>,
{
    /// Construct declared profiles without inferred bank extensions or a default runtime.
    pub fn new(
        client: R::Client,
        definitions: Vec<CrossSectionDefinition>,
    ) -> Result<Self, CrossSectionError> {
        let context = Arc::new(Context::<R>::new(client, definitions)?);
        // The declared upper support must itself resolve finite positive section properties.
        evaluation::evaluate(&context, &context.cap)?;
        Ok(Self { context })
    }
    /// Original immutable definitions in query order.
    pub fn definitions(&self) -> &[CrossSectionDefinition] {
        &self.context.definitions
    }
    /// Evaluate one finite stage per section in that section's declared vertical datum.
    pub fn evaluate(&self, stages_m: &[f64]) -> Result<CrossSectionQuery<R>, CrossSectionError> {
        if stages_m.len() != self.definitions().len() {
            return Err(CrossSectionError::input(format!(
                "stage count={}, section count={}: supply one stage per ordered section",
                stages_m.len(),
                self.definitions().len()
            )));
        }
        for (index, value) in stages_m.iter().enumerate() {
            if !value.is_finite() {
                return Err(CrossSectionError::input(format!(
                    "stage={value}: supply a finite query stage"
                ))
                .at(index));
            }
        }
        self.evaluate_tensor(&self.context.tensor(stages_m)?)
    }
    pub(crate) fn evaluate_tensor(
        &self,
        stages: &Tensor<R>,
    ) -> Result<CrossSectionQuery<R>, CrossSectionError> {
        Ok(CrossSectionQuery {
            context: Arc::clone(&self.context),
            fields: evaluation::evaluate(&self.context, stages)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::CrossSectionPoint;
    use super::*;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn sections(client: CpuClient) -> Result<CrossSectionBatch<CpuRuntime>, CrossSectionError> {
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
        CrossSectionBatch::new(client, definitions)
    }

    fn strided(
        client: &CpuClient,
        values: [f64; 2],
    ) -> Result<Tensor<CpuRuntime>, numr::error::Error> {
        Ok(Tensor::from_slice(
            &[f64::NAN, values[0], f64::INFINITY, values[1]],
            &[2, 2],
            client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)))
    }

    #[test]
    fn native_strided_stages_preserve_original_bits_and_ignore_padding() -> TestResult {
        let client = client();
        let batch = sections(client.clone())?;
        let stages = strided(&client, [-0.0, 1.0])?;
        assert!(!stages.is_contiguous());
        let properties = batch.evaluate_tensor(&stages)?.properties()?;
        assert_eq!(properties[0].query_stage_m.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(properties[1].query_stage_m.to_bits(), 1.0_f64.to_bits());
        assert!(properties[0].dry);
        assert_eq!(properties[0].area_m2, 0.0);
        assert_eq!(properties[0].wetted_perimeter_m, 0.0);
        assert_eq!(properties[0].hydrostatic_integral_m3, 0.0);
        assert_eq!(properties[0].top_width_m, None);
        assert_eq!(properties[0].hydraulic_radius_m, None);
        assert_eq!(properties[0].hydraulic_depth_m, None);
        assert!(!properties[1].dry);
        assert_eq!(properties[1].area_m2, 2.0);
        assert_eq!(properties[1].top_width_m, Some(2.0));
        assert_eq!(properties[1].wetted_perimeter_m, 4.0);
        assert_eq!(properties[1].hydraulic_radius_m, Some(0.5));
        assert_eq!(properties[1].hydraulic_depth_m, Some(1.0));
        assert_eq!(properties[1].hydrostatic_integral_m3, 1.0);
        Ok(())
    }

    #[test]
    fn native_stage_rank_and_dtype_require_the_declared_layout() -> TestResult {
        let client = client();
        let batch = sections(client.clone())?;
        let wrong_rank =
            Tensor::<CpuRuntime>::from_slice(&[0.0_f64, 1.0], &[1, 2], client.device())?;
        let wrong_dtype = Tensor::<CpuRuntime>::from_slice(&[0.0_f32, 1.0], &[2], client.device())?;
        for stages in [&wrong_rank, &wrong_dtype] {
            let error = batch
                .evaluate_tensor(stages)
                .err()
                .ok_or("Invalid native stage layout was accepted")?;
            assert!(matches!(
                error,
                CrossSectionError::InvalidInput { section: None, .. }
            ));
        }
        assert_eq!(batch.evaluate(&[1.0, 1.0])?.properties()?[0].area_m2, 2.0);
        Ok(())
    }

    #[test]
    fn native_nonfinite_strided_stages_report_the_first_section() -> TestResult {
        let client = client();
        let batch = sections(client.clone())?;
        for (values, expected) in [([0.0, f64::NAN], 1), ([f64::NAN, f64::INFINITY], 0)] {
            let stages = strided(&client, values)?;
            let error = batch
                .evaluate_tensor(&stages)
                .err()
                .ok_or("Nonfinite native stage was accepted")?;
            assert!(matches!(
                error,
                CrossSectionError::InvalidInput { section: Some(index), .. } if index == expected
            ));
        }
        let properties = batch.evaluate(&[1.0, 1.0])?.properties()?;
        assert!(properties.iter().all(|property| property.area_m2 == 2.0));
        Ok(())
    }

    #[test]
    fn native_nonfinite_admission_precedes_an_earlier_unsupported_stage() -> TestResult {
        let client = client();
        let batch = sections(client.clone())?;
        let stages = strided(&client, [5.0, f64::NAN])?;
        let error = batch
            .evaluate_tensor(&stages)
            .err()
            .ok_or("Nonfinite native stage was accepted")?;
        assert!(matches!(
            error,
            CrossSectionError::InvalidInput {
                section: Some(1),
                ..
            }
        ));
        let properties = batch.evaluate(&[1.0, 1.0])?.properties()?;
        assert!(properties.iter().all(|property| property.area_m2 == 2.0));
        Ok(())
    }
}
