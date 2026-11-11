use super::{
    super::{
        ChannelBoundary, ChannelConfig, ChannelInitialState, ChannelReachDefinition,
        context::Context,
        geometry::{ChannelGeometry, TensorGeometry, TriangularChannelCell},
    },
    faces::Face,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn context() -> Result<Context<CpuRuntime>, Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    Ok(Context::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "test datum".into(),
            provenance: "triangular invariant reference fixtures".into(),
            geometry: ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v: 1.0,
                right_bank_slope_h_per_v: 1.0,
                bed_elevation_m: 0.0,
                cells: vec![TriangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    maximum_depth_m: 2.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig {
                gravity: 8.0,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![1.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )?)
}

fn face(
    ctx: &Context<CpuRuntime>,
    depth: f64,
    velocity: f64,
) -> Result<Face<CpuRuntime>, Box<dyn std::error::Error>> {
    Ok(Face {
        area: ctx.tensor(&[depth * depth])?,
        discharge: ctx.tensor(&[depth * depth * velocity])?,
        depth: ctx.tensor(&[depth])?,
        velocity: ctx.tensor(&[velocity])?,
    })
}

#[test]
fn film_signals_remain_at_the_wet_vacuum_front_scale() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    let wet = face(&ctx, 1.0, 0.0)?;
    for exponent in 1..=150 {
        let film_depth = 10_f64.powi(-exponent);
        let film = face(&ctx, film_depth, 0.0)?;
        let (sl, sr) = ctx.triangular_bounds(geometry, &wet, &film)?;
        // For g=8, the resting sound floor is exactly 2*sqrt(h_right).
        let reference_speed = 8.0 - 6.0 * film_depth.sqrt();
        let sl = sl.item::<f64>()?;
        let sr = sr.item::<f64>()?;
        assert!(sl.is_finite() && sr.is_finite());
        assert!(sl <= -reference_speed && (sl + reference_speed).abs() < 1e-12);
        assert!(sr >= reference_speed && (sr - reference_speed).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn invariant_signals_enclose_compression_vacuum_and_characteristic_cancellation() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    for (ul, ur, reference_lower, reference_upper) in [
        (3.0, -3.0, -5.0, 5.0),
        (-10.0, 10.0, -12.0, 12.0),
        (8.0, 8.0, 6.0, 10.0),
        (2.0, 2.0, 0.0, 4.0),
        (0.0, 0.0, -2.0, 2.0),
        (-3.0, 3.0, -5.0, 5.0),
    ] {
        let (sl, sr) =
            ctx.triangular_bounds(geometry, &face(&ctx, 1.0, ul)?, &face(&ctx, 1.0, ur)?)?;
        assert!(sl.item::<f64>()? <= reference_lower);
        assert!(sr.item::<f64>()? >= reference_upper);
        assert!((sl.item::<f64>()? - reference_lower).abs() < 1e-12);
        assert!((sr.item::<f64>()? - reference_upper).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn reflected_wall_water_is_exact_zero_and_rest_pressure_uses_triangular_section() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    for velocity in [-1.0, 0.0, 1.0] {
        let fields = ctx.fields(&ctx.tensor(&[1.0])?, &ctx.tensor(&[velocity])?)?;
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
    let dry = ctx.fields(&ctx.tensor(&[0.0])?, &ctx.tensor(&[0.0])?)?;
    let flux = native_flux(
        &ctx,
        geometry,
        &dry,
        &ChannelBoundary::Wall,
        &ChannelBoundary::Wall,
    )?;
    assert_eq!(flux.speed.try_to_vec::<f64>()?, [0.0, 0.0]);
    assert_eq!(flux.momentum.try_to_vec::<f64>()?, [0.0, 0.0]);
    Ok(())
}

#[test]
fn one_sided_motion_retains_finite_flux_for_extreme_represented_velocity() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    let fields = ctx.fields(&ctx.tensor(&[1e-300])?, &ctx.tensor(&[1e-100])?)?;
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
    let fields = ctx.fields(&ctx.tensor(&[1e-300])?, &ctx.tensor(&[0.0])?)?;
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
fn uncertain_rarefaction_boundary_retains_the_enclosure_until_certified() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    let left = face(&ctx, 1.0, 0.0)?;
    // The exact transition has c_left=2, c_right=1, and velocity jump 4.
    for right_velocity in [f64::from_bits(4.0_f64.to_bits() - 1), 4.0] {
        let (sl, sr) =
            ctx.triangular_bounds(geometry, &left, &face(&ctx, 0.25, right_velocity)?)?;
        assert!(sl.item::<f64>()? < -4.9);
        assert!(sl.item::<f64>()? <= -2.0);
        assert!(sr.item::<f64>()? >= right_velocity + 1.0);
    }
    let right_velocity = 4.0 + 1e-10;
    let (sl, sr) = ctx.triangular_bounds(geometry, &left, &face(&ctx, 0.25, right_velocity)?)?;
    assert!(sl.item::<f64>()? <= -2.0);
    assert!((sl.item::<f64>()? + 2.0).abs() < 1e-12);
    assert!(sr.item::<f64>()? >= right_velocity + 1.0);
    assert!((sr.item::<f64>()? - (right_velocity + 1.0)).abs() < 1e-12);
    Ok(())
}

#[test]
fn certified_vacuum_interface_selects_both_zero_fluxes_and_keeps_original_wave_bounds() -> TestResult
{
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    let velocity = 10.0;
    let fields = ctx.fields(&ctx.tensor(&[1.0])?, &ctx.tensor(&[velocity])?)?;
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
    let speed = flux.speed.narrow(0, 0, 1)?.item::<f64>()?;
    assert!(speed.is_finite() && speed >= velocity + 2.0);
    let pressure = flux
        .left_reconstructed_pressure
        .narrow(0, 0, 1)?
        .item::<f64>()?;
    assert!((pressure - 8.0 / 3.0).abs() < 1e-14);
    Ok(())
}

#[test]
fn vacuum_front_certificate_remains_false_when_zero_is_uncertain() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    // At g=8 and h=1 the two vacuum fronts cross zero at velocities -8 and +8.
    for (velocity, expected) in [(8.0 - 1e-10, 0.0), (8.0, 0.0), (8.0 + 1e-10, 1.0)] {
        let bounds = ctx.triangular_wave_bounds(
            geometry,
            &face(&ctx, 1.0, -velocity)?,
            &face(&ctx, 1.0, velocity)?,
        )?;
        assert_eq!(bounds.vacuum.item::<f64>()?, expected);
    }
    let one_dry =
        ctx.triangular_wave_bounds(geometry, &face(&ctx, 1.0, -10.0)?, &face(&ctx, 0.0, 0.0)?)?;
    assert_eq!(one_dry.vacuum.item::<f64>()?, 0.0);
    Ok(())
}

#[test]
fn unselected_vacuum_advection_uses_benign_operands_before_product_checks() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    let left = face(&ctx, 1.0, -1e200)?;
    let right = face(&ctx, 1.0, 1e200)?;
    let bounds = ctx.triangular_wave_bounds(geometry, &left, &right)?;
    assert!(bounds.lower.item::<f64>()?.is_finite() && bounds.upper.item::<f64>()?.is_finite());
    assert_eq!(bounds.vacuum.item::<f64>()?, 1.0);
    let physical = ctx.triangular_physical(&left, &ctx.tensor(&[0.0])?)?;
    assert_eq!(physical.momentum.item::<f64>()?, 0.0);
    assert_eq!(physical.momentum_bound.item::<f64>()?, 0.0);
    assert!((physical.pressure.item::<f64>()? - 8.0 / 3.0).abs() < 1e-14);
    Ok(())
}

#[test]
fn exact_vacuum_discards_only_shared_flux_underflow_charges() -> TestResult {
    let ctx = context()?;
    let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
        return Err("triangular geometry required".into());
    };
    let fields = ctx.fields(&ctx.tensor(&[1e-300])?, &ctx.tensor(&[1e-300])?)?;
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
    let fields = ctx.fields(
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
            assert!((momentum - 1.7476266666666667).abs() <= 1e-12 + 1e-12 * 1.7476266666666667);
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
        let expected_momentum = reference_momentum + 2.0 * discharge * boost + area * boost * boost;
        assert!((water - expected_water).abs() <= 1e-12 + 1e-12 * expected_water.abs());
        assert!((momentum - expected_momentum).abs() <= 1e-12 + 1e-12 * expected_momentum.abs());
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
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let ctx = Context::new(client, definition, original.original_initial.clone())?;
    let result = interface_flux(&ctx, 1.0, 2.0, 1.0, -2.0);
    let error = result.err().ok_or("root exhaustion requires rejection")?;
    let channel = error
        .downcast_ref::<super::super::ChannelError>()
        .ok_or("typed channel rejection required")?;
    assert!(matches!(
        channel,
        super::super::ChannelError::Numerical { .. }
    ));
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
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
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
    let fields = ctx.fields(&ctx.tensor(&[tiny])?, &ctx.tensor(&[tiny])?)?;
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
    for value in flux.speed.try_to_vec::<f64>()? {
        assert!(value.is_finite() && value >= 1.0);
    }
    Ok(())
}

#[test]
fn pressure_charge_retains_joint_erased_area_and_coefficient_uncertainty() -> TestResult {
    let original = context()?;
    let mut definition = original.definition.as_ref().clone();
    definition.config.gravity = f64::from_bits(1);
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let ctx = Context::<CpuRuntime>::new(client, definition, original.original_initial.clone())?;
    // For k=1 and h=1e-200, exact positive area rounds to zero within one subnormal.
    let zero = ctx.tensor(&[0.0])?;
    let sample = Face {
        area: zero.clone(),
        discharge: zero.clone(),
        depth: ctx.tensor(&[1e-200])?,
        velocity: zero,
    };
    let area_bound = ctx.tensor(&[f64::from_bits(1)])?;
    let (pressure, bound) = ctx.triangular_pressure_value(&sample, &area_bound)?;
    assert_eq!(pressure.item::<f64>()?, 0.0);
    assert!(bound.item::<f64>()? > 0.0);
    Ok(())
}

fn native_flux(
    context: &Context<CpuRuntime>,
    geometry: &super::super::geometry::TriangleGeometry<CpuRuntime>,
    fields: &super::super::admission::Fields<CpuRuntime>,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> Result<super::super::flux::Flux<CpuRuntime>, super::super::ChannelError> {
    use crate::channel::ChannelTensorBoundary;
    context.triangular_flux(
        geometry,
        fields,
        &ChannelTensorBoundary::from_host(&context.client, left)?,
        &ChannelTensorBoundary::from_host(&context.client, right)?,
    )
}
