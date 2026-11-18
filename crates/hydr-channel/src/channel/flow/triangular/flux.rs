use super::super::{
    ChannelError, ChannelTensorBoundary, admission::Fields, context::Context, flux::Flux,
    geometry::TriangleGeometry,
};
use super::{faces::narrow_face, godunov::RiemannInput};
use crate::channel::flow::{
    boundary::assembly::numerical_range, riemann_diagnostics::RiemannDiagnosticState,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps, ShapeOps,
        TwoSumOps, UnaryOps,
    },
    runtime::Runtime,
};
use solvr::optimize::roots::BatchedBisectionAlgorithms;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
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
    pub(in crate::channel::flow) fn triangular_flux(
        &self,
        geometry: &TriangleGeometry<R>,
        fields: &Fields<R>,
        left_boundary: &ChannelTensorBoundary<R>,
        right_boundary: &ChannelTensorBoundary<R>,
    ) -> Result<Flux<R>, ChannelError> {
        let faces = self.triangular_faces(geometry, fields, left_boundary, right_boundary)?;
        let interior = self.triangular_endpoint_interior(&faces)?;
        self.check_flux_ports(fields, &interior, left_boundary, right_boundary)?;
        let frame = self.direct_boundary_frame(&interior, left_boundary, right_boundary)?;
        let (start, len) = numerical_range(self.count(), left_boundary, right_boundary);
        let numerical = if len > 0 {
            let left = narrow_face(&faces.left, start, len)?;
            let right = narrow_face(&faces.right, start, len)?;
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
            Some((sample, self.directional_speeds(&sl, &sr)?))
        } else {
            None
        };
        let water = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical.as_ref().map(|(sample, _)| &sample.water),
            frame.as_ref().map(|f| &f.longitudinal_water),
        )?;
        let momentum = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical.as_ref().map(|(sample, _)| &sample.momentum),
            frame.as_ref().map(|f| &f.momentum_flux_m4_per_s2),
        )?;
        let speed =
            self.assemble_face_speeds(frame.as_ref(), numerical.as_ref().map(|(_, speed)| speed))?;
        let dry_fronts = self.triangular_fronts(geometry, &faces, frame.as_ref())?;
        let water_bound = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical.as_ref().map(|(sample, _)| &sample.water_bound),
            frame.as_ref().map(|f| &f.water_underflow_bound),
        )?;
        let momentum_bound = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical.as_ref().map(|(sample, _)| &sample.momentum_bound),
            frame.as_ref().map(|f| &f.momentum_underflow_bound),
        )?;
        let water_flux_width = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical
                .as_ref()
                .map(|(sample, _)| &sample.diagnostics.water_flux_width),
            frame.as_ref().map(|f| &f.water_flux_width_m3_per_s),
        )?;
        let momentum_flux_width = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical
                .as_ref()
                .map(|(sample, _)| &sample.diagnostics.momentum_flux_width),
            frame.as_ref().map(|f| &f.momentum_flux_width_m4_per_s2),
        )?;
        let root_iterations = numerical
            .as_ref()
            .map_or(0, |(sample, _)| sample.diagnostics.root_iterations)
            .max(frame.as_ref().map_or(0, |f| f.root_iterations));
        let zero = self.constant(&water, 0.0)?;
        let (left_pressure, left_pressure_bound) =
            self.triangular_pressure_value(&faces.left, &zero)?;
        let (right_pressure, right_pressure_bound) =
            self.triangular_pressure_value(&faces.right, &zero)?;
        let prescribed_water = self.prescribed_water_mask(
            frame.as_ref(),
            numerical.as_ref().map(|(sample, _)| &sample.water),
            left_boundary,
            right_boundary,
        )?;
        for (boundary, index) in [(left_boundary, 0), (right_boundary, self.count())] {
            if matches!(boundary, ChannelTensorBoundary::Wall)
                && water.narrow(0, index, 1)?.item::<f64>()? != 0.0
            {
                return Err(ChannelError::numerical(format!(
                    "triangular wall face={index} has nonzero water flux: inspect reflected Godunov arithmetic"
                )));
            }
        }
        Ok(Flux {
            boundary_traces: frame.and_then(|frame| frame.traces),
            surveyed: None,
            prescribed_water,
            water,
            momentum,
            speed,
            internal_speed: faces.internal_speed,
            dry_fronts,
            momentum_bound,
            water_bound,
            riemann: Some(RiemannDiagnosticState {
                water_flux_width,
                momentum_flux_width,
                root_iterations,
                surveyed_work: None,
            }),
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
