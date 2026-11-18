use super::{CoupledLevelPoolError, context::Context, topology::finite};
use crate::coupling::links::LinkError;
use crate::coupling::links::donors::{self, DonorTexts, Donors};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

#[derive(Clone, Debug)]
pub(super) struct Exchange<R: Runtime<DType = DType>> {
    pub proposed: Tensor<R>,
    pub signed: Tensor<R>,
    pub factor: Tensor<R>,
    pub incoming: Tensor<R>,
    pub outgoing: Tensor<R>,
    pub aggregation_bound: f64,
}

/// Coupled level pool donor-limit rejection texts.
const TEXTS: DonorTexts = DonorTexts {
    unrepresentable: "positive donor storage cannot represent a conservative shared allocation. Rescale storage",
    factor_underflow: "positive donor allocation factor underflows. Rescale the competing demands",
    accepted: "accepted connection volume",
    accepted_underflow: "positive accepted connection volume underflows. Rescale the exchange",
    totals: "represented donor totals exceed original accepted storage",
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    pub fn exchange(
        &self,
        proposed: Tensor<R>,
        available: &Tensor<R>,
    ) -> Result<Exchange<R>, CoupledLevelPoolError> {
        let c = &self.client;
        finite(c, &proposed, "held connection volume")?;
        let donors = Donors {
            tree: self.tree.shared(),
            a: &self.a,
            b: &self.b,
            groups: self.pool_count(),
        };
        let limit = donors::limit(
            c, &donors, &proposed, available, &self.zero, &self.one, &TEXTS,
        )
        .map_err(LinkError::level_pool)?;
        Ok(Exchange {
            proposed,
            signed: limit.signed,
            factor: limit.factor,
            incoming: limit.incoming,
            outgoing: limit.outgoing,
            aggregation_bound: limit.aggregation_bound,
        })
    }
}
