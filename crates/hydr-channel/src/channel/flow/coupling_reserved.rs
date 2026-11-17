//! Coordinator forcing bound to a reservation, so trials reuse its accepted-start operands.

use super::{
    ChannelError, ChannelReach, ChannelTensorLateralInput, coupling::ChannelCouplingForcing,
    coupling_reservation::ChannelCouplingReservation,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> ChannelReach<R>
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
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Bind lateral operands to the endpoints and accepted-start operands of `reservation`.
    ///
    /// Every trial with the returned forcing reuses the reservation's accepted-start fluxes
    /// instead of calculating them again. The bits match an unreserved trial.
    ///
    /// # Errors
    /// Returns [`ChannelError::Input`] when `reservation` belongs to another accepted state,
    /// plus every `bind_coupling_tensor` rejection.
    #[doc(hidden)]
    pub fn bind_coupling_reserved(
        &self,
        reservation: &ChannelCouplingReservation<R>,
        end_s: f64,
        lateral: &[ChannelTensorLateralInput<R>],
    ) -> Result<ChannelCouplingForcing<R>, ChannelError> {
        let reserved = &reservation.reserved;
        if !reserved.belongs_to(&self.accepted) {
            return Err(ChannelError::input(
                "coupling reservation belongs to another accepted state: reserve from this reach",
            ));
        }
        let c = &self.context.client;
        let mut forcing = self.bind_coupling_owned(
            self.time(),
            end_s,
            Some(reserved.left.clone()),
            Some(reserved.right.clone()),
            lateral
                .iter()
                .map(|input| input.clone().into_owned(c))
                .collect::<Result<_, _>>()?,
        )?;
        forcing.reserved = Some(Arc::clone(reserved));
        Ok(forcing)
    }
}
