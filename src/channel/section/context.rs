use super::{CrossSectionDefinition, CrossSectionError, definition};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) struct Context<R: Runtime<DType = DType>> {
    pub definitions: Vec<CrossSectionDefinition>,
    pub client: R::Client,
    pub bed: Tensor<R>,
    pub cap: Tensor<R>,
    pub owners: Tensor<R>,
    pub segment_sections: Vec<usize>,
    pub left_elevation: Tensor<R>,
    pub right_elevation: Tensor<R>,
    pub low: Tensor<R>,
    pub high: Tensor<R>,
    pub span: Tensor<R>,
    pub width: Tensor<R>,
    pub length: Tensor<R>,
    pub zero: Tensor<R>,
    pub one: Tensor<R>,
    pub zero_sections: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub fn new(
        client: R::Client,
        definitions: Vec<CrossSectionDefinition>,
    ) -> Result<Self, CrossSectionError> {
        let inputs = definition::check(&definitions)?;
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(CrossSectionError::input(error.message("cross-section")));
        }
        let device = client.device();
        let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], device);
        let left_station = tensor(&inputs.left_station)?;
        let right_station = tensor(&inputs.right_station)?;
        let left_elevation = tensor(&inputs.left_elevation)?;
        let right_elevation = tensor(&inputs.right_elevation)?;
        let width = client.sub(&right_station, &left_station)?;
        let span = client.abs(&client.sub(&right_elevation, &left_elevation)?)?;
        let length = client.hypot(&width, &span)?;
        let result = Self {
            bed: tensor(&inputs.bed)?,
            cap: tensor(
                &definitions
                    .iter()
                    .map(|d| d.maximum_stage_m)
                    .collect::<Vec<_>>(),
            )?,
            owners: Tensor::<R>::from_slice(&inputs.owners, &[inputs.owners.len()], device)?,
            segment_sections: inputs.segment_sections,
            low: client.minimum(&left_elevation, &right_elevation)?,
            high: client.maximum(&left_elevation, &right_elevation)?,
            zero: Tensor::zeros(&[], DType::F64, device)?,
            one: Tensor::full_scalar(&[], DType::F64, 1.0, device)?,
            zero_sections: Tensor::zeros(&[definitions.len()], DType::F64, device)?,
            definitions,
            left_elevation,
            right_elevation,
            span,
            width,
            length,
            client,
        };
        result.segment_finite(&result.width, "horizontal segment extent")?;
        result.segment_finite(&result.span, "vertical segment extent")?;
        result.segment_finite(&result.length, "segment contact length")?;
        result.reject_segment(
            &result.client.le(&result.length, &result.zero)?,
            &result.length,
            "positive segment contact length",
        )?;
        Ok(result)
    }
    pub fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, CrossSectionError> {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            self.client.device(),
        )?)
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for Context<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossSectionContext")
            .field("definitions", &self.definitions)
            .finish_non_exhaustive()
    }
}
