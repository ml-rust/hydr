use super::CoupledLevelPoolError;
use crate::numerics::water_transfers::{self, AggregationKind, TransferTree as SharedTransferTree};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Debug)]
pub(super) struct TransferTree<R: Runtime<DType = DType>> {
    shared: SharedTransferTree<R>,
    pub degree_margin: Tensor<R>,
}

pub(super) struct Aggregate<R: Runtime<DType = DType>> {
    pub volume: Tensor<R>,
    pub bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TransferTree<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn new(
        client: &R::Client,
        pool_count: usize,
        endpoints: &[(usize, usize)],
        ids: &[String],
    ) -> Result<Self, CoupledLevelPoolError> {
        if ids.len() != endpoints.len() {
            return Err(CoupledLevelPoolError::input(
                "structure identities and endpoints differ in count",
            ));
        }
        let input_count = endpoints
            .len()
            .checked_mul(2)
            .ok_or_else(|| CoupledLevelPoolError::input("endpoint count overflows"))?;
        let mut targets = Vec::with_capacity(input_count);
        for (edge, (&(a, b), id)) in endpoints.iter().zip(ids).enumerate() {
            targets.push((a, edge, id.as_str()));
            targets.push((b, endpoints.len() + edge, id.as_str()));
        }
        Self::new_grouped(client, pool_count, input_count, &targets)
    }

    pub fn new_grouped(
        client: &R::Client,
        pool_count: usize,
        input_count: usize,
        targets: &[(usize, usize, &str)],
    ) -> Result<Self, CoupledLevelPoolError> {
        let shared = SharedTransferTree::new_grouped(client, pool_count, input_count, targets)
            .map_err(CoupledLevelPoolError::from_transfer)?;
        Ok(Self {
            degree_margin: shared.degree_margin.clone(),
            shared,
        })
    }

    pub fn aggregate(
        &self,
        client: &R::Client,
        endpoint_values: &Tensor<R>,
    ) -> Result<Aggregate<R>, CoupledLevelPoolError> {
        let aggregate = self
            .shared
            .aggregate(client, endpoint_values, AggregationKind::Water)
            .map_err(CoupledLevelPoolError::from_transfer)?;
        Ok(Aggregate {
            volume: aggregate.value,
            bound: aggregate.bound,
        })
    }
}

pub(super) fn finite<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
    field: &str,
) -> Result<(), CoupledLevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    water_transfers::finite(client, value, field).map_err(CoupledLevelPoolError::from_transfer)
}

pub(super) fn reject<R: Runtime<DType = DType>>(
    client: &R::Client,
    mask: &Tensor<R>,
    message: &str,
) -> Result<(), CoupledLevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    water_transfers::reject(client, mask, message).map_err(CoupledLevelPoolError::from_transfer)
}

pub(super) fn upward_add<R: Runtime<DType = DType>>(
    client: &R::Client,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, CoupledLevelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    water_transfers::upward_add(client, a, b).map_err(CoupledLevelPoolError::from_transfer)
}
