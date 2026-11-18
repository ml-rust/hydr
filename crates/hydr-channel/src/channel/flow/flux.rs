use super::{
    ChannelError, ChannelTensorBoundary, admission::Fields, conserved_update::Parts,
    context::Context, geometry::TensorGeometry,
};
use super::{
    boundary::{NativeBoundaryTraces, assembly::numerical_range},
    cfl_fronts::DryFronts,
    cfl_speeds::FaceSpeeds,
    reconstruction::narrow_face,
};
use crate::numerics::homogeneous_hll_tensor;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};
use std::sync::Arc;

/// Two-part surveyed face rates, each from one selected point trace.
#[derive(Clone, Debug)]
pub(super) struct SurveyedRates<R: Runtime<DType = DType>> {
    /// Water flux point and its radius, in m³/s.
    pub water: Parts<R>,
    /// Momentum flux point and its radius, in m⁴/s².
    pub momentum: Parts<R>,
}

/// Face rates for one interval.
///
/// `water` and `momentum` hold the F64 rates. On surveyed geometry they are the point high parts.
pub(super) struct Flux<R: Runtime<DType = DType>> {
    pub boundary_traces: Option<Arc<NativeBoundaryTraces<R>>>,
    /// Two-part rates on surveyed geometry. Other geometry holds `None`.
    pub surveyed: Option<SurveyedRates<R>>,
    pub prescribed_water: Option<Tensor<R>>,
    pub water: Tensor<R>,
    pub water_bound: Tensor<R>,
    pub riemann: Option<super::riemann_diagnostics::RiemannDiagnosticState<R>>,
    pub momentum: Tensor<R>,
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub left_reconstructed_pressure: Tensor<R>,
    pub right_reconstructed_pressure: Tensor<R>,
    pub momentum_bound: Tensor<R>,
    pub left_pressure_bound: Tensor<R>,
    pub right_pressure_bound: Tensor<R>,
    pub left_reconstructed_pressure_bound: Tensor<R>,
    pub right_reconstructed_pressure_bound: Tensor<R>,
    /// Directional signal speeds at the cell faces.
    pub speed: FaceSpeeds<R>,
    /// Directional signal speeds at each half-cell split face, when one exists.
    pub internal_speed: Option<FaceSpeeds<R>>,
    /// Dry-front states at the cell faces. `None` when no face is wet on exactly one side.
    pub dry_fronts: Option<DryFronts<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
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
    pub fn flux(
        &self,
        fields: &Fields<R>,
        left_boundary: &ChannelTensorBoundary<R>,
        right_boundary: &ChannelTensorBoundary<R>,
    ) -> Result<Flux<R>, ChannelError> {
        let face_width = match &self.geometry {
            TensorGeometry::Rectangular(section) => &section.face_width,
            TensorGeometry::Triangular(geometry) => {
                return self.triangular_flux(geometry, fields, left_boundary, right_boundary);
            }
            TensorGeometry::Surveyed(geometry) => {
                return self.surveyed_flux(geometry, fields, left_boundary, right_boundary);
            }
        };
        let c = &self.client;
        let count = self.count();
        let reconstructed = self.reconstruct(fields, left_boundary, right_boundary)?;
        self.check_flux_ports(
            fields,
            &reconstructed.endpoints,
            left_boundary,
            right_boundary,
        )?;
        let frame =
            self.direct_boundary_frame(&reconstructed.endpoints, left_boundary, right_boundary)?;
        let (start, len) = numerical_range(count, left_boundary, right_boundary);
        let left = reconstructed.left;
        let right = reconstructed.right;
        let left_h = &left.h;
        let right_h = &right.h;
        let homogeneous = if len > 0 {
            Some(homogeneous_hll_tensor(
                c,
                &narrow_face(&left, start, len)?,
                &narrow_face(&right, start, len)?,
                self.definition.config.gravity,
                &self.definition.config.wave_config(),
            )?)
        } else {
            None
        };
        let numerical_water = homogeneous
            .as_ref()
            .map(|flux| {
                let water = c.mul(&flux.shared[0], &face_width.narrow(0, start, len)?)?;
                let required = c.gt(&c.abs(&flux.shared[0])?, &self.constant(&water, 0.0)?)?;
                self.represented(&required, &water, "full-width shared water rate")?;
                Ok::<_, ChannelError>(water)
            })
            .transpose()?;
        let numerical_momentum = homogeneous
            .as_ref()
            .map(|flux| {
                Ok::<_, ChannelError>(c.mul(&flux.shared[1], &face_width.narrow(0, start, len)?)?)
            })
            .transpose()?;
        let water = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical_water.as_ref(),
            frame.as_ref().map(|f| &f.longitudinal_water),
        )?;
        let momentum = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical_momentum.as_ref(),
            frame.as_ref().map(|f| &f.momentum_flux_m4_per_s2),
        )?;
        let numerical_speed = homogeneous
            .as_ref()
            .map(|flux| self.directional_speeds(&flux.lower, &flux.upper))
            .transpose()?;
        let speed = self.assemble_face_speeds(frame.as_ref(), numerical_speed.as_ref())?;
        let dry_fronts = self.rectangular_fronts(&left, &right, frame.as_ref())?;
        let left_reconstructed = self.pressure(left_h, face_width)?;
        let right_reconstructed = self.pressure(right_h, face_width)?;
        let pressure_bound = c.maximum(
            &left_reconstructed.unit_bound,
            &right_reconstructed.unit_bound,
        )?;
        let numerical_momentum_bound =
            if let (Some(homogeneous), Some(momentum)) = (&homogeneous, &numerical_momentum) {
                Some(self.product_bound(
                    &homogeneous.shared[1],
                    &pressure_bound.narrow(0, start, len)?,
                    &face_width.narrow(0, start, len)?,
                    momentum,
                    "full-width shared momentum rate",
                )?)
            } else {
                None
            };
        let momentum_bound = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical_momentum_bound.as_ref(),
            frame.as_ref().map(|f| &f.momentum_underflow_bound),
        )?;
        let numerical_water_bound = numerical_water
            .as_ref()
            .map(|water| self.constant(water, 0.0))
            .transpose()?;
        let water_bound = self.assemble_boundary_faces(
            frame.as_ref(),
            numerical_water_bound.as_ref(),
            frame.as_ref().map(|f| &f.water_underflow_bound),
        )?;
        let prescribed_water = self.prescribed_water_mask(
            frame.as_ref(),
            numerical_water.as_ref(),
            left_boundary,
            right_boundary,
        )?;
        let left_original = self.pressure(&reconstructed.left_depth, &reconstructed.left_width)?;
        let right_original =
            self.pressure(&reconstructed.right_depth, &reconstructed.right_width)?;
        let left_pressure_bound =
            self.bound_add(&left_original.bound, &left_reconstructed.bound)?;
        let right_pressure_bound =
            self.bound_add(&right_original.bound, &right_reconstructed.bound)?;
        let left_reconstructed_pressure = left_reconstructed.value;
        let right_reconstructed_pressure = right_reconstructed.value;
        let left_pressure = c.sub(&left_original.value, &left_reconstructed_pressure)?;
        let right_pressure = c.sub(&right_original.value, &right_reconstructed_pressure)?;
        for (value, name) in [
            (&water, "shared water flux"),
            (&momentum, "shared momentum flux"),
            (&left_pressure, "left pressure correction"),
            (&right_pressure, "right pressure correction"),
        ] {
            self.finite(value, name)?;
        }
        for (boundary, index) in [(left_boundary, 0), (right_boundary, count)] {
            if matches!(boundary, ChannelTensorBoundary::Wall)
                && water.narrow(0, index, 1)?.item::<f64>()? != 0.0
            {
                return Err(ChannelError::numerical(format!(
                    "wall face={index} has nonzero water flux: inspect reflected HLL arithmetic"
                )));
            }
        }
        Ok(Flux {
            boundary_traces: frame.and_then(|frame| frame.traces),
            surveyed: None,
            prescribed_water,
            water_bound,
            riemann: None,
            water,
            momentum,
            left_pressure,
            right_pressure,
            left_reconstructed_pressure,
            right_reconstructed_pressure,
            momentum_bound,
            left_pressure_bound,
            right_pressure_bound,
            left_reconstructed_pressure_bound: left_reconstructed.bound,
            right_reconstructed_pressure_bound: right_reconstructed.bound,
            speed,
            internal_speed: None,
            dry_fronts,
        })
    }
}
