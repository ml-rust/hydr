use super::{
    point_flux_association,
    point_flux_faces::{Continuation, Faces},
    point_flux_masks::{equals, not},
    point_flux_types::PointFluxStatus,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Face owners of one round: every unsettled face, and the owners of each continuation.
pub(super) struct Demand<R: Runtime<DType = DType>> {
    pub(super) all: Tensor<R>,
    /// Indexed by `Continuation::index`.
    pub(super) owners: [Tensor<R>; 3],
}

impl<R> Faces<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Over-budget faces demand every possible continuation that is not terminal.
    /// A face with no such continuation is settled and demands nothing.
    pub(super) fn demand(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Demand<R>> {
        let c = a.client();
        let budget = equals(a, &self.status, PointFluxStatus::FluxBudgetExceeded as i64)?;
        let empty = Tensor::empty(&[0], DType::I64, c.device())?;
        let mut owners = [empty.clone(), empty.clone(), empty];
        let mut any = Tensor::zeros(budget.shape(), a.dtype(), c.device())?;
        for continuation in Continuation::ALL {
            let index = continuation.index();
            let open = not(a, &self.terminal[index])?;
            let mask = c.mul(&c.mul(&budget, &self.possible[index])?, &open)?;
            any = c.maximum(&any, &mask)?;
            owners[index] = c.index_select(&self.owner, 0, &c.nonzero(&mask)?.flatten()?)?;
        }
        let all = c.index_select(&self.owner, 0, &c.nonzero(&any)?.flatten()?)?;
        Ok(Demand { all, owners })
    }

    /// The requested face owners whose continuation depth was never offered the tightest tolerance.
    pub(super) fn untightened(
        &self,
        a: &mut Arithmetic<'_, R>,
        continuation: Continuation,
        owners: &Tensor<R>,
    ) -> HydraulicResult<Tensor<R>> {
        let c = a.client();
        let positions = point_flux_association::positions(a, &self.association, owners)?;
        let tightened = &self.tightened[continuation.index()];
        let fresh = not(a, &c.index_select(tightened, 0, &positions)?)?;
        Ok(c.index_select(owners, 0, &c.nonzero(&fresh)?.flatten()?)?)
    }
}
