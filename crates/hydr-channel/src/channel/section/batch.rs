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
