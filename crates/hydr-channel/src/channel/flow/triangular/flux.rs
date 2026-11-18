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

#[cfg(test)]
mod tests {
    use super::super::fixture::{TestResult, context};
    use super::*;
    use crate::channel::flow::{
        ChannelBoundary, ChannelInitialState,
        geometry::{ChannelGeometry, TensorGeometry},
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn reflected_wall_water_is_exact_zero_and_rest_pressure_uses_triangular_section() -> TestResult
    {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        for velocity in [-1.0, 0.0, 1.0] {
            let fields = ctx.fields_high(&ctx.tensor(&[1.0])?, &ctx.tensor(&[velocity])?)?;
            let flux = native_flux(
                &ctx,
                geometry,
                &fields,
                &ChannelBoundary::Wall,
                &ChannelBoundary::Wall,
            )?;
            assert_eq!(flux.water.try_to_vec::<f64>()?, [0.0, 0.0]);
            assert_eq!(flux.left_pressure.try_to_vec::<f64>()?, [0.0, 0.0]);
            if velocity == 0.0 {
                for value in flux.momentum.try_to_vec::<f64>()? {
                    assert!((value - 8.0 / 3.0).abs() < 1e-14);
                }
            }
        }
        let dry = ctx.fields_high(&ctx.tensor(&[0.0])?, &ctx.tensor(&[0.0])?)?;
        let flux = native_flux(
            &ctx,
            geometry,
            &dry,
            &ChannelBoundary::Wall,
            &ChannelBoundary::Wall,
        )?;
        assert_eq!(flux.speed.rightward.try_to_vec::<f64>()?, [0.0, 0.0]);
        assert_eq!(flux.speed.leftward.try_to_vec::<f64>()?, [0.0, 0.0]);
        assert_eq!(flux.momentum.try_to_vec::<f64>()?, [0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn one_sided_motion_retains_finite_flux_for_extreme_represented_velocity() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let fields = ctx.fields_high(&ctx.tensor(&[1e-300])?, &ctx.tensor(&[1e-100])?)?;
        let boundary = ChannelBoundary::Exterior {
            depth_m: 1e-150,
            velocity_m_per_s: 1e200,
        };
        let flux = native_flux(&ctx, geometry, &fields, &boundary, &boundary)?;
        for value in flux.momentum.try_to_vec::<f64>()? {
            assert!(value.is_finite() && value > 0.0);
        }
        for value in flux.momentum_bound.try_to_vec::<f64>()? {
            assert!(value.is_finite() && value >= 0.0);
        }
        Ok(())
    }

    #[test]
    fn erased_pressure_retains_a_positive_momentum_charge() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let fields = ctx.fields_high(&ctx.tensor(&[1e-300])?, &ctx.tensor(&[0.0])?)?;
        let boundary = ChannelBoundary::Exterior {
            depth_m: 1e-150,
            velocity_m_per_s: 0.0,
        };
        let flux = native_flux(&ctx, geometry, &fields, &boundary, &boundary)?;
        assert_eq!(flux.momentum.try_to_vec::<f64>()?, [0.0, 0.0]);
        for value in flux.momentum_bound.try_to_vec::<f64>()? {
            assert!(value.is_finite() && value > 0.0);
        }
        Ok(())
    }

    #[test]
    fn certified_vacuum_interface_selects_both_zero_fluxes_and_keeps_original_wave_bounds()
    -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let velocity = 10.0;
        let fields = ctx.fields_high(&ctx.tensor(&[1.0])?, &ctx.tensor(&[velocity])?)?;
        let left = ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: -velocity,
        };
        let right = ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: velocity,
        };
        let flux = native_flux(&ctx, geometry, &fields, &left, &right)?;
        assert_eq!(flux.water.narrow(0, 0, 1)?.item::<f64>()?, 0.0);
        assert_eq!(flux.momentum.narrow(0, 0, 1)?.item::<f64>()?, 0.0);
        assert_eq!(flux.momentum_bound.narrow(0, 0, 1)?.item::<f64>()?, 0.0);
        // The left exterior moves away leftward and the cell moves away rightward.
        for speed in [&flux.speed.leftward, &flux.speed.rightward] {
            let speed = speed.narrow(0, 0, 1)?.item::<f64>()?;
            assert!(speed.is_finite() && speed >= velocity + 2.0);
        }
        let pressure = flux
            .left_reconstructed_pressure
            .narrow(0, 0, 1)?
            .item::<f64>()?;
        assert!((pressure - 8.0 / 3.0).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn exact_vacuum_discards_only_shared_flux_underflow_charges() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let fields = ctx.fields_high(&ctx.tensor(&[1e-300])?, &ctx.tensor(&[1e-300])?)?;
        let left = ChannelBoundary::Exterior {
            depth_m: 1e-150,
            velocity_m_per_s: -1.0,
        };
        let right = ChannelBoundary::Exterior {
            depth_m: 1e-150,
            velocity_m_per_s: 1.0,
        };
        let flux = native_flux(&ctx, geometry, &fields, &left, &right)?;
        assert_eq!(flux.water.narrow(0, 0, 1)?.item::<f64>()?, 0.0);
        assert_eq!(flux.momentum.narrow(0, 0, 1)?.item::<f64>()?, 0.0);
        assert_eq!(flux.momentum_bound.narrow(0, 0, 1)?.item::<f64>()?, 0.0);
        assert!(
            flux.left_reconstructed_pressure_bound
                .narrow(0, 0, 1)?
                .item::<f64>()?
                > 0.0
        );
        assert!(
            flux.right_reconstructed_pressure_bound
                .narrow(0, 0, 1)?
                .item::<f64>()?
                > 0.0
        );
        assert!(flux.momentum_bound.narrow(0, 1, 1)?.item::<f64>()? > 0.0);
        Ok(())
    }

    fn interface_flux(
        ctx: &Context<CpuRuntime>,
        left_depth: f64,
        left_velocity: f64,
        right_depth: f64,
        right_velocity: f64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let area = right_depth * right_depth;
        let fields = ctx.fields_high(
            &ctx.tensor(&[area])?,
            &ctx.tensor(&[area * right_velocity])?,
        )?;
        let left = ChannelBoundary::Exterior {
            depth_m: left_depth,
            velocity_m_per_s: left_velocity,
        };
        let right = ChannelBoundary::Exterior {
            depth_m: right_depth,
            velocity_m_per_s: right_velocity,
        };
        let flux = native_flux(ctx, geometry, &fields, &left, &right)?;
        Ok((
            flux.water.narrow(0, 0, 1)?.item::<f64>()?,
            flux.momentum.narrow(0, 0, 1)?.item::<f64>()?,
        ))
    }

    #[test]
    fn godunov_vacuum_onset_has_continuous_zero_momentum_flux() -> TestResult {
        let ctx = context()?;
        for speed in [8.0 - 1e-8, 8.0, 8.0 + 1e-8] {
            let (water, momentum) = interface_flux(&ctx, 1.0, -speed, 1.0, speed)?;
            assert!(water.abs() <= 1e-12);
            assert!(momentum.abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn vanishing_right_states_share_the_unique_left_fan_flux() -> TestResult {
        let ctx = context()?;
        for epsilon in [0.1, 0.01, 0.0001, 1e-8] {
            for fraction in [0.0, 0.5, 1.0] {
                let depth = epsilon * epsilon / 4.0;
                let velocity = 8.0 + (4.0 - 8.0 * fraction) * epsilon;
                let (water, momentum) = interface_flux(&ctx, 1.0, 0.0, depth, velocity)?;
                assert!((water - 0.65536).abs() <= 1e-12 + 1e-12 * 0.65536);
                assert!(
                    (momentum - 1.7476266666666667).abs() <= 1e-12 + 1e-12 * 1.7476266666666667
                );
            }
        }
        Ok(())
    }

    #[test]
    fn stationary_entropy_shock_flux_is_continuous_under_galilean_boosts() -> TestResult {
        let ctx = context()?;
        let discharge = (2.0 / 3.0) * (7.0_f64 * 8.0).sqrt();
        let reference_momentum = discharge * discharge + 8.0 / 3.0;
        for boost in [-1e-8, 0.0, 1e-8] {
            let (water, momentum) =
                interface_flux(&ctx, 1.0, discharge + boost, 2.0, discharge / 4.0 + boost)?;
            let area = if boost < 0.0 { 4.0 } else { 1.0 };
            let expected_water = discharge + area * boost;
            let expected_momentum =
                reference_momentum + 2.0 * discharge * boost + area * boost * boost;
            assert!((water - expected_water).abs() <= 1e-12 + 1e-12 * expected_water.abs());
            assert!(
                (momentum - expected_momentum).abs() <= 1e-12 + 1e-12 * expected_momentum.abs()
            );
        }
        Ok(())
    }

    #[test]
    fn sonic_fan_boundary_fluxes_agree_from_both_sides() -> TestResult {
        let ctx = context()?;
        // Left rarefaction head reaches xi=0 at u=c=2.
        let expected_water = 2.0;
        let expected_momentum = 4.0 + 8.0 / 3.0;
        for shift in [-1e-8, 0.0, 1e-8] {
            let (water, momentum) = interface_flux(&ctx, 1.0, 2.0 + shift, 0.0, 0.0)?;
            let limit_water = expected_water + shift;
            let limit_momentum = expected_momentum + 4.0 * shift + shift * shift;
            assert!((water - limit_water).abs() <= 1e-12 + 1e-12 * limit_water.abs());
            assert!((momentum - limit_momentum).abs() <= 1e-12 + 1e-12 * limit_momentum.abs());
        }
        Ok(())
    }

    #[test]
    fn exhausted_godunov_root_returns_an_actionable_typed_rejection() -> TestResult {
        let original = context()?;
        let mut definition = original.definition.as_ref().clone();
        definition.config.root_iterations = 1;
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let ctx = Context::new(client, definition, original.original_initial.clone())?;
        let result = interface_flux(&ctx, 1.0, 2.0, 1.0, -2.0);
        let error = result.err().ok_or("root exhaustion requires rejection")?;
        let channel = error
            .downcast_ref::<ChannelError>()
            .ok_or("typed channel rejection required")?;
        assert!(matches!(channel, ChannelError::Numerical { .. }));
        assert!(channel.to_string().contains("root"));
        Ok(())
    }

    #[test]
    fn finite_extreme_upwind_flux_skips_unselected_sonic_residuals() -> TestResult {
        let original = context()?;
        let mut definition = original.definition.as_ref().clone();
        let ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v,
            right_bank_slope_h_per_v,
            cells,
            ..
        } = &mut definition.geometry
        else {
            return Err("triangular geometry required".into());
        };
        *left_bank_slope_h_per_v = 1e308;
        *right_bank_slope_h_per_v = 1e308;
        cells[0].maximum_depth_m = 1e-150;
        let tiny = f64::from_bits(1);
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let ctx = Context::<CpuRuntime>::new(
            client,
            definition,
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![tiny],
                longitudinal_momentum_m4_per_s: vec![tiny],
            },
        )?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let fields = ctx.fields_high(&ctx.tensor(&[tiny])?, &ctx.tensor(&[tiny])?)?;
        let depth = fields.depth.item::<f64>()?;
        let exterior = ChannelBoundary::Exterior {
            depth_m: depth,
            velocity_m_per_s: 1.0,
        };
        let flux = native_flux(&ctx, geometry, &fields, &exterior, &exterior)?;
        for value in flux.water.try_to_vec::<f64>()? {
            assert_eq!(value, tiny);
        }
        for value in flux.momentum.try_to_vec::<f64>()? {
            assert_eq!(value, tiny);
        }
        for value in flux.speed.rightward.try_to_vec::<f64>()? {
            assert!(value.is_finite() && value >= 1.0);
        }
        for value in flux.speed.leftward.try_to_vec::<f64>()? {
            assert!(value.is_finite() && value >= 0.0);
        }
        Ok(())
    }

    fn native_flux(
        context: &Context<CpuRuntime>,
        geometry: &TriangleGeometry<CpuRuntime>,
        fields: &Fields<CpuRuntime>,
        left: &ChannelBoundary,
        right: &ChannelBoundary,
    ) -> Result<Flux<CpuRuntime>, ChannelError> {
        context.triangular_flux(
            geometry,
            fields,
            &ChannelTensorBoundary::from_host(&context.client, left)?,
            &ChannelTensorBoundary::from_host(&context.client, right)?,
        )
    }
}
