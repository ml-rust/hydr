use super::{ChannelError, ChannelInitialState, ChannelReachDefinition};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

pub(super) struct Context<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub definition: Arc<ChannelReachDefinition>,
    pub original_initial: ChannelInitialState,
    pub length: Tensor<R>,
    pub geometry: super::geometry::TensorGeometry<R>,
    pub bed: Tensor<R>,
    pub cap: Tensor<R>,
    pub manning: Tensor<R>,
    pub charge_reduction: crate::numerics::outward::AdjustmentReduction,
    pub lateral: super::sources::LateralContext<R>,
    /// Recent wave-bound evaluations, shared by every state of this reach.
    pub speed_memo: super::speed_memo::SpeedMemo<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + TensorOps<R>
        + NextRepresentableOps<R>,
{
    pub fn new(
        client: R::Client,
        definition: ChannelReachDefinition,
        original_initial: ChannelInitialState,
    ) -> Result<Self, ChannelError> {
        Self::with_binding_owner(client, definition, original_initial, false)
    }

    /// `coordinator_bindings` admits lateral identities with the reserved surface-exchange prefix.
    pub fn with_binding_owner(
        client: R::Client,
        definition: ChannelReachDefinition,
        original_initial: ChannelInitialState,
        coordinator_bindings: bool,
    ) -> Result<Self, ChannelError> {
        if matches!(
            &definition.geometry,
            super::ChannelGeometry::PrismaticTriangular { .. }
        ) && R::square_root_contract(DType::F64)
            != numr::runtime::SquareRootContract::Ieee754RoundToNearest
        {
            return Err(ChannelError::input(
                "triangular F64 square roots lack round-to-nearest guarantees: select an admitted runtime",
            ));
        }
        definition.check(coordinator_bindings)?;
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(ChannelError::input(error.message("channel reach")));
        }
        let device = client.device();
        let make = |values: Vec<f64>| Tensor::<R>::from_slice(&values, &[values.len()], device);
        let length = make(definition.geometry.cells().map(|c| c.length_m).collect())?;
        let bed = make(
            definition
                .geometry
                .cells()
                .map(|c| c.bed_elevation_m)
                .collect(),
        )?;
        let cap = make(
            definition
                .geometry
                .cells()
                .map(|c| c.maximum_depth_m)
                .collect(),
        )?;
        let manning = make(definition.geometry.cells().map(|c| c.manning_n).collect())?;
        let count = definition.geometry.len();
        let charge_reduction = crate::numerics::outward::AdjustmentReduction::new(count)
            .map_err(|error| ChannelError::numerical(error.to_string()))?;
        let geometry = super::geometry::TensorGeometry::new(&client, &definition, &cap, &length)?;
        let definition = Arc::new(definition);
        let lateral = super::sources::LateralContext::new(&client, Arc::clone(&definition))?;
        Ok(Self {
            client,
            definition,
            original_initial,
            length,
            geometry,
            bed,
            cap,
            manning,
            charge_reduction,
            lateral,
            speed_memo: super::speed_memo::SpeedMemo::new(),
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R> {
    pub fn count(&self) -> usize {
        self.definition.geometry.len()
    }

    pub fn surveyed(&self) -> bool {
        matches!(
            self.definition.geometry,
            super::ChannelGeometry::PrismaticSurveyed { .. }
        )
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
