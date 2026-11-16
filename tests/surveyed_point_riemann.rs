//! Public point-flux Riemann checks on exact rectangle states.

use hydr::channel::{
    ChannelError, CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint,
    PrismaticPointRiemannResult, PrismaticPointTraceKind, PrismaticRiemannConfig,
    PrismaticRiemannError, PrismaticSectionBatch,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval, with_interval_arithmetic};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::{
    CertifiedBisectionAlgorithms, CertifiedMappedBisectionAlgorithms,
    CertifiedUpperBracketAlgorithms,
};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + HypotOps<R>
    + MidpointOps<R>
    + RuntimeClient<R>
    + CertifiedSimpsonAlgorithms<R>
    + CertifiedBisectionAlgorithms<R>
    + CertifiedMappedBisectionAlgorithms<R>
    + CertifiedUpperBracketAlgorithms<R>
{
}

impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
}

/// One rectangle of the given width and cap depth.
fn rectangle<R>(client: &R::Client, width: f64, depth: f64) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definition = CrossSectionDefinition {
        id: "exact_rectangle".into(),
        vertical_datum: "Exact rectangle datum".into(),
        points: [(0.0, depth), (0.0, 0.0), (width, 0.0), (width, depth)]
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: depth,
        profile_provenance: "Exact dyadic rectangle".into(),
    };
    Ok(CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?)
}

fn config(gravity: f64) -> PrismaticRiemannConfig {
    PrismaticRiemannConfig {
        gravity_m_per_s2: gravity,
        ..PrismaticRiemannConfig::default()
    }
}

/// Exact points with the given radius.
fn points<R>(client: &R::Client, values: &[f64], radius: f64) -> Result<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let n = values.len();
    let high = Tensor::from_slice(values, &[n], client.device())?;
    let low = Tensor::zeros(&[n], DType::F64, client.device())?;
    let radius = Tensor::from_slice(&vec![radius; n], &[n], client.device())?;
    Ok(with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::F64,
        "test conserved points",
        |a| a.expansion_bounds(&high, &low, &radius),
    )?)
}

/// Faces as `[left area, left discharge, right area, right discharge]`, all on section 0.
fn solve<R>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    faces: &[[f64; 4]],
    config: &PrismaticRiemannConfig,
) -> Result<std::result::Result<PrismaticPointRiemannResult<R>, PrismaticRiemannError>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let owners = Tensor::from_slice(&vec![0_i64; faces.len()], &[faces.len()], client.device())?;
    let inputs = (0..4)
        .map(|field| {
            points::<R>(
                client,
                &faces.iter().map(|f| f[field]).collect::<Vec<_>>(),
                0.,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(section.riemann_flux_points_tensor(
        &owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3], config,
    ))
}

fn components<R: Runtime<DType = DType>>(value: &CertifiedExpansion<R>) -> Result<Vec<[f64; 3]>> {
    let high = value.high().try_to_vec::<f64>()?;
    let low = value.low().try_to_vec::<f64>()?;
    let radius = value.abs_error().try_to_vec::<f64>()?;
    Ok((0..high.len())
        .map(|i| [high[i], low[i], radius[i]])
        .collect())
}

fn endpoints<R: Runtime<DType = DType>>(value: &Interval<R>) -> Result<Vec<[f64; 2]>> {
    let lower = value.lower().try_to_vec::<f64>()?;
    let upper = value.upper().try_to_vec::<f64>()?;
    Ok(lower.into_iter().zip(upper).map(|(l, u)| [l, u]).collect())
}

/// Checks that `bounds` hold `value` and lie within `allowance` of it.
fn near(bounds: [f64; 2], value: f64, allowance: f64, label: &str) {
    assert!(
        bounds[0] - allowance <= value && value <= bounds[1] + allowance,
        "{label}: {bounds:?} misses {value}"
    );
    assert!(
        (bounds[0] - value).abs() <= allowance && (bounds[1] - value).abs() <= allowance,
        "{label}: {bounds:?} lies far from {value}"
    );
}

/// The center of a point, `high + low`, rounded once.
fn center(point: [f64; 3]) -> f64 {
    point[0] + point[1]
}

/// Checks the selected point lies inside its accepted bound.
fn inside(point: [f64; 3], bound: [f64; 2], label: &str) {
    assert!(
        bound[0] <= center(point) - point[2] && center(point) + point[2] <= bound[1],
        "{label}: {point:?} leaves {bound:?}"
    );
}

/// Uniform Q=1 flow at unit area, depth and gravity carries J=1.5.
fn basic<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = rectangle::<R>(client, 1., 4.)?;
    let result = solve::<R>(client, &section, &[[1., 1., 1., 1.]], &config(1.))??;
    assert_eq!(components(result.water_flux_m3_per_s())?, [[1., 0., 0.]]);
    assert_eq!(
        components(result.trace_discharge_m3_per_s())?,
        [[1., 0., 0.]]
    );
    assert_eq!(components(result.trace_area_m2())?, [[1., 0., 0.]]);
    let momentum = components(result.momentum_flux_m4_per_s2())?[0];
    near([center(momentum); 2], 1.5, 1e-12, "basic momentum point");
    let bound = endpoints(result.momentum_flux_bound_m4_per_s2())?[0];
    near(bound, 1.5, 1e-12, "basic momentum bound");
    inside(momentum, bound, "basic momentum");
    near(
        endpoints(result.physical_momentum_flux_m4_per_s2())?[0],
        1.5,
        1e-12,
        "basic physical momentum",
    );
    near(
        endpoints(result.water_flux_bound_m3_per_s())?[0],
        1.,
        1e-12,
        "basic water",
    );
    let kind = result.trace_kind().try_to_vec::<i64>()?[0];
    assert!(PrismaticPointTraceKind::from_code(kind).is_some());
    assert_eq!(components(result.left_area_m2())?, [[1., 0., 0.]]);
    assert_eq!(result.owners().try_to_vec::<i64>()?, [0]);
    Ok(())
}

/// Still water carries zero water flux and the hydrostatic thrust g b h^2 / 2.
fn still_water<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = rectangle::<R>(client, 2., 4.)?;
    let result = solve::<R>(client, &section, &[[2., 0., 2., 0.]], &config(4.))??;
    assert_eq!(components(result.water_flux_m3_per_s())?, [[0., 0., 0.]]);
    assert_eq!(endpoints(result.water_flux_bound_m3_per_s())?, [[0., 0.]]);
    let thrust = 4. * 2. * 1. * 1. / 2.;
    let momentum = components(result.momentum_flux_m4_per_s2())?[0];
    assert_eq!(momentum, [thrust, 0., 0.], "still momentum point");
    let bound = endpoints(result.momentum_flux_bound_m4_per_s2())?[0];
    near(bound, thrust, 1e-12, "still momentum bound");
    inside(momentum, bound, "still momentum");
    Ok(())
}

/// A wet left state over a dry right bed opens a left fan through the face.
/// The face holds h = 4 hL / 9 and u = 2 sqrt(g hL) / 3.
fn dry_wet<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let (width, depth, gravity) = (2., 1., 1.);
    let section = rectangle::<R>(client, width, 4.)?;
    let result = solve::<R>(
        client,
        &section,
        &[[width * depth, 0., 0., 0.]],
        &config(gravity),
    )??;
    let h = 4. * depth / 9.;
    let u = 2. * (gravity * depth).sqrt() / 3.;
    let water = width * h * u;
    let momentum = water * u + gravity * width * h * h / 2.;
    assert_eq!(
        PrismaticPointTraceKind::from_code(result.trace_kind().try_to_vec::<i64>()?[0]),
        Some(PrismaticPointTraceKind::LeftFan)
    );
    let allowance = 1e-9;
    let selected = [
        components(result.water_flux_m3_per_s())?[0],
        components(result.momentum_flux_m4_per_s2())?[0],
    ];
    let bounds = [
        endpoints(result.water_flux_bound_m3_per_s())?[0],
        endpoints(result.momentum_flux_bound_m4_per_s2())?[0],
    ];
    for (k, value) in [water, momentum].into_iter().enumerate() {
        near(bounds[k], value, allowance, "dry/wet bound");
        near([center(selected[k]); 2], value, allowance, "dry/wet point");
        inside(selected[k], bounds[k], "dry/wet point");
    }
    near(
        [center(components(result.trace_area_m2())?[0]); 2],
        width * h,
        allowance,
        "dry/wet trace area",
    );
    Ok(())
}

/// A uniform subcritical state signals at u - c on the left and u + c on the right.
fn wave_speeds<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = rectangle::<R>(client, 2., 4.)?;
    let result = solve::<R>(client, &section, &[[2., 1., 2., 1.]], &config(1.))??;
    let speeds = endpoints(result.wave_speed_m_per_s())?[0];
    let (u, c) = (0.5, 1.);
    assert!(speeds[0] <= u - c && u + c <= speeds[1], "{speeds:?}");
    near([speeds[0]; 2], u - c, 1e-9, "left signal speed");
    near([speeds[1]; 2], u + c, 1e-9, "right signal speed");
    near(
        endpoints(result.water_flux_bound_m3_per_s())?[0],
        1.,
        1e-12,
        "uniform water",
    );
    Ok(())
}

/// A nonzero input radius rejects the whole call as invalid input.
fn radius<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = rectangle::<R>(client, 2., 4.)?;
    let owners = Tensor::from_slice(&[0_i64, 0], &[2], client.device())?;
    let exact = points::<R>(client, &[2., 2.], 0.)?;
    let zero = points::<R>(client, &[0., 0.], 0.)?;
    let uncertain = points::<R>(client, &[2., 2.], 2f64.powi(-40))?;
    let error =
        section.riemann_flux_points_tensor(&owners, &uncertain, &zero, &exact, &zero, &config(1.));
    match error {
        Err(PrismaticRiemannError::Channel(ChannelError::InvalidInput { context, .. })) => {
            assert!(context.contains("radius"), "{context}");
            assert!(context.contains("left area_m2"), "{context}");
        }
        other => {
            return Err(format!("nonzero radius returns {:?}", other.err()).into());
        }
    }
    Ok(())
}

/// Writes after the call leave every captured input and result unchanged.
fn independent<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = rectangle::<R>(client, 2., 4.)?;
    let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
    let inputs = [2., 1., 2., 1.]
        .into_iter()
        .map(|value| points::<R>(client, &[value], 0.))
        .collect::<Result<Vec<_>>>()?;
    let result = section.riemann_flux_points_tensor(
        &owners,
        &inputs[0],
        &inputs[1],
        &inputs[2],
        &inputs[3],
        &config(1.),
    )?;
    let snapshot = |result: &PrismaticPointRiemannResult<R>| -> Result<_> {
        Ok((
            result.owners().try_to_vec::<i64>()?,
            [
                components(result.left_area_m2())?,
                components(result.left_discharge_m3_per_s())?,
                components(result.right_area_m2())?,
                components(result.right_discharge_m3_per_s())?,
                components(result.water_flux_m3_per_s())?,
                components(result.momentum_flux_m4_per_s2())?,
            ],
            endpoints(result.momentum_flux_bound_m4_per_s2())?,
        ))
    };
    let before = snapshot(&result)?;
    let other = Tensor::from_slice(&[3.], &[1], client.device())?;
    for input in &inputs {
        client.copy_into(input.high(), &other)?;
    }
    client.copy_into(
        &owners,
        &Tensor::from_slice(&[5_i64], &[1], client.device())?,
    )?;
    assert_eq!(snapshot(&result)?, before);
    assert_eq!(before.1[0], [[2., 0., 0.]]);
    Ok(())
}

/// An empty batch returns empty results.
fn empty<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = rectangle::<R>(client, 2., 4.)?;
    let result = solve::<R>(client, &section, &[], &config(1.))??;
    assert_eq!(result.owners().numel(), 0);
    assert_eq!(result.water_flux_m3_per_s().high().numel(), 0);
    assert_eq!(result.wave_speed_m_per_s().lower().numel(), 0);
    assert_eq!(result.work().residual_points, 0);
    Ok(())
}

fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    basic::<R>(client)?;
    still_water::<R>(client)?;
    dry_wet::<R>(client)?;
    wave_speeds::<R>(client)?;
    radius::<R>(client)?;
    independent::<R>(client)?;
    empty::<R>(client)
}

#[test]
fn configured_cpu_point_fluxes_match_rectangle_states() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    run::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_point_fluxes_match_rectangle_states() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
