use super::{ChannelError, ChannelInitialState, ChannelReachDefinition};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

pub(super) struct Context<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub definition: Arc<ChannelReachDefinition>,
    pub original_initial: ChannelInitialState,
    pub length: Tensor<R>,
    pub width: Tensor<R>,
    pub bed: Tensor<R>,
    pub cap: Tensor<R>,
    pub manning: Tensor<R>,
    pub face_width: Tensor<R>,
    pub charge_reduction: crate::soil::admission::AdjustmentReduction,
    pub lateral: super::sources::LateralContext<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn new(
        client: R::Client,
        definition: ChannelReachDefinition,
        original_initial: ChannelInitialState,
    ) -> Result<Self, ChannelError> {
        definition.check()?;
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(ChannelError::input(error.message("channel reach")));
        }
        let device = client.device();
        let make = |values: Vec<f64>| Tensor::<R>::from_slice(&values, &[values.len()], device);
        let length = make(definition.cells.iter().map(|c| c.length_m).collect())?;
        let width = make(definition.cells.iter().map(|c| c.width_m).collect())?;
        let bed = make(definition.cells.iter().map(|c| c.bed_elevation_m).collect())?;
        let cap = make(definition.cells.iter().map(|c| c.maximum_depth_m).collect())?;
        let manning = make(definition.cells.iter().map(|c| c.manning_n).collect())?;
        let count = definition.cells.len();
        let charge_reduction = crate::soil::admission::AdjustmentReduction::new(count)
            .map_err(|error| ChannelError::numerical(error.to_string()))?;
        let first = width.narrow(0, 0, 1)?;
        let last = width.narrow(0, count - 1, 1)?;
        let face_width = if count == 1 {
            client.cat(&[&first, &last], 0)?
        } else {
            let middle = client.midpoint(
                &width.narrow(0, 0, count - 1)?,
                &width.narrow(0, 1, count - 1)?,
            )?;
            client.cat(&[&first, &middle, &last], 0)?
        };
        let definition = Arc::new(definition);
        let lateral = super::sources::LateralContext::new(&client, Arc::clone(&definition))?;
        Ok(Self {
            client,
            definition,
            original_initial,
            length,
            width,
            bed,
            cap,
            manning,
            face_width,
            charge_reduction,
            lateral,
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R> {
    pub fn count(&self) -> usize {
        self.definition.cells.len()
    }

    pub fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, ChannelError> {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            self.client.device(),
        )?)
    }

    pub fn constant(&self, like: &Tensor<R>, value: f64) -> Result<Tensor<R>, ChannelError> {
        Ok(Tensor::full_scalar(
            like.shape(),
            DType::F64,
            value,
            self.client.device(),
        )?)
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for Context<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelContext")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}
