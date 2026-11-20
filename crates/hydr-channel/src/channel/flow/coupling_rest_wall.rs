//! Private coordinator read: the wall flux of a surveyed reach end cell at rest.

use super::{ChannelEnd, ChannelError, ChannelReach, geometry::TensorGeometry};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{
            BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms,
            CertifiedUpperBracketAlgorithms,
        },
    },
};

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: TensorOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Two-part momentum flux of a `Wall` face at `end`, when the accepted end cell rests.
    ///
    /// Returns `[high, low]` for a surveyed reach whose end cell holds exactly zero discharge
    /// in both parts. Returns `None` for any other reach or end state.
    ///
    /// A coordinator hands such an end a `Wall` for an interval in which its port moves no
    /// water. The pair is the momentum rate the reach then applies at that face.
    ///
    /// # Errors
    /// Returns field recovery and point Riemann errors.
    #[doc(hidden)]
    pub fn coupling_rest_wall(&self, end: ChannelEnd) -> Result<Option<[f64; 2]>, ChannelError> {
        let TensorGeometry::Surveyed(geometry) = &self.context.geometry else {
            return Ok(None);
        };
        let fields = self
            .context
            .fields(&self.accepted.volume, &self.accepted.momentum)?;
        let end = match end {
            ChannelEnd::Left => 0,
            ChannelEnd::Right => 1,
        };
        self.context
            .surveyed_wall_rest_point(geometry, &fields, end)
    }
}
