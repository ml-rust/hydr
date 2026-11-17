//! Triangular dry-front states for the channel CFL rule.
//!
//! A triangle has `A ∝ h²` and `c ∝ A^(1/4)`, so `φ(A) = 4c`. The side bounds enclose `u ± 4c`
//! with directed rounding.

use super::super::{
    ChannelError,
    boundary::ScalarBoundaryFrame,
    cfl_fronts::{DryFronts, SideFronts, side_fronts},
    context::Context,
    geometry::TriangleGeometry,
};
use super::faces::{Face, TriangularFaces};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    /// Fronts from the face-side states, reconstructed when `LimitedLinear` is active.
    pub(super) fn triangular_fronts(
        &self,
        geometry: &TriangleGeometry<R>,
        faces: &TriangularFaces<R>,
        frame: Option<&ScalarBoundaryFrame<R>>,
    ) -> Result<Option<DryFronts<R>>, ChannelError> {
        let c = &self.client;
        let bounds = |face: &Face<R>| -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
            let side = self.triangular_side_bounds(geometry, face)?;
            Ok((side.lower_signal, side.upper_signal))
        };
        let side = |face: &Face<R>| -> Result<SideFronts<R>, ChannelError> {
            let (lower, upper) = bounds(face)?;
            let wet = c.gt(&face.area, &self.constant(&face.area, 0.0)?)?;
            side_fronts(c, &lower, &upper, &wet)
        };
        let trace = |depth: &Tensor<R>,
                     area: &Tensor<R>,
                     velocity: &Tensor<R>|
         -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
            bounds(&Face {
                area: area.clone(),
                discharge: c.mul(area, velocity)?,
                depth: depth.clone(),
                velocity: velocity.clone(),
            })
        };
        self.dry_fronts(side(&faces.left)?, side(&faces.right)?, frame, &trace)
    }
}
