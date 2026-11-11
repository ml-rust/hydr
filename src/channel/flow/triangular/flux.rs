use super::super::{
    ChannelBoundary, ChannelError, admission::Fields, context::Context, flux::Flux,
    geometry::TriangleGeometry,
};
use super::godunov::RiemannInput;
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
    },
    runtime::Runtime,
};
use solvr::optimize::roots::BatchedBisectionAlgorithms;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(in crate::channel::flow) fn triangular_flux(
        &self,
        geometry: &TriangleGeometry<R>,
        fields: &Fields<R>,
        left_boundary: &ChannelBoundary,
        right_boundary: &ChannelBoundary,
    ) -> Result<Flux<R>, ChannelError> {
        let c = &self.client;
        let faces = self.triangular_faces(geometry, fields, left_boundary, right_boundary)?;
        let left = faces.left;
        let right = faces.right;
        let (sl, sr) = self.triangular_bounds(geometry, &left, &right)?;
        let ls = self.triangular_side_bounds(geometry, &left)?;
        let rs = self.triangular_side_bounds(geometry, &right)?;
        let sample = self.triangular_godunov(
            geometry,
            RiemannInput {
                left: &left,
                right: &right,
                left_bounds: &ls,
                right_bounds: &rs,
                lower_signal: &sl,
                upper_signal: &sr,
            },
        )?;
        let water = sample.water;
        let momentum = sample.momentum;
        let zero = self.constant(&water, 0.0)?;
        let (left_pressure, left_pressure_bound) = self.triangular_pressure_value(&left, &zero)?;
        let (right_pressure, right_pressure_bound) =
            self.triangular_pressure_value(&right, &zero)?;
        for (boundary, index) in [(left_boundary, 0), (right_boundary, self.count())] {
            if matches!(boundary, ChannelBoundary::Wall)
                && water.narrow(0, index, 1)?.item::<f64>()? != 0.0
            {
                return Err(ChannelError::numerical(format!(
                    "triangular wall face={index} has nonzero water flux: inspect reflected Godunov arithmetic"
                )));
            }
        }
        let speed = c.maximum(&c.abs(&sl)?, &c.abs(&sr)?)?;
        Ok(Flux {
            water,
            momentum,
            speed,
            internal_speed: faces.internal_speed,
            momentum_bound: sample.momentum_bound,
            water_bound: sample.water_bound,
            riemann: Some(sample.diagnostics),
            left_pressure: zero.clone(),
            right_pressure: zero.clone(),
            left_pressure_bound: zero.clone(),
            right_pressure_bound: zero,
            left_reconstructed_pressure: left_pressure,
            right_reconstructed_pressure: right_pressure,
            left_reconstructed_pressure_bound: left_pressure_bound,
            right_reconstructed_pressure_bound: right_pressure_bound,
        })
    }
}
