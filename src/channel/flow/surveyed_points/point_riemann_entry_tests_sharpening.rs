use super::super::super::{
    point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
    source_tests::support::{self as frozen, Result},
    wet_depth::tests::support,
};
use crate::channel::{
    flow::surveyed::PrismaticRiemannConfig,
    section::{
        CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
    },
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedBisectionOptions;

/// The low part of the right area at a rest face just inside a dam-break rarefaction head.
/// The left area is exactly 12 m², so the jump between the sources is near 4.6e-97 m².
const RIGHT_LOW: u64 = 0xabef84666e2af06a;

/// The multiband section: three bands up to a 4 m cap.
fn multiband<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let points = [
        (0.0, 4.0),
        (1.0, 2.0),
        (3.0, 1.0),
        (4.0, 0.0),
        (5.0, 0.0),
        (6.0, 1.0),
        (8.0, 2.0),
        (9.0, 4.0),
    ];
    let definition = CrossSectionDefinition {
        id: "multiband".into(),
        vertical_datum: "Exact dyadic datum".into(),
        points: points
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 4.0,
        profile_provenance: "Independent dyadic proof".into(),
    };
    Ok(CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?)
}

/// The reach Riemann controls.
fn config() -> PrismaticRiemannConfig {
    PrismaticRiemannConfig {
        water_flux_atol_m3_per_s: 1e-12,
        momentum_flux_atol_m4_per_s2: 1e-12,
        flux_rtol: 1e-12,
        root: CertifiedBisectionOptions {
            atol: 1e-15,
            rtol: 1e-15,
            ..CertifiedBisectionOptions::default()
        },
        ..PrismaticRiemannConfig::default()
    }
}

/// One rest face with left area 12 and right area 12 plus `low`, both at zero discharge.
fn rest_face<R>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    low: u64,
) -> Result<PrismaticPointRiemannResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let owners = support::rows::<R>(client, &[0])?;
    let left = frozen::expansion::<R>(client, &[[12.0_f64.to_bits(), 0, 0]], true)?;
    let right = frozen::expansion::<R>(client, &[[12.0_f64.to_bits(), low, 0]], true)?;
    let zero = frozen::expansion::<R>(client, &[[0, 0, 0]], true)?;
    section
        .riemann_flux_points_tensor(&owners, &left, &zero, &right, &zero, &config())
        .map_err(|error| format!("rest face low={low:#x} rejects: {error:?}").into())
}

/// Root rounds a face may take when its bracket halves from the area scale down to the
/// two-part rounding of that area.
const AREA_ROUNDS: usize = 128;

/// A rest face whose jump lies far below the two-part rounding of its area stops sharpening
/// at that rounding, so its rounds stay within `AREA_ROUNDS`.
fn tiny_jump<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let section = multiband::<R>(client)?;
    let result = rest_face::<R>(client, &section, RIGHT_LOW)?;
    let kinds = result.trace_kind().try_to_vec::<i64>()?;
    assert_eq!(kinds, vec![PrismaticPointTraceKind::Star as i64]);
    let work = result.work();
    eprintln!(
        "POINT_RIEMANN_TINY_JUMP\troot_iterations={}\tmaximum_root_iterations={}\tresidual_points={}",
        work.root_iterations, work.maximum_root_iterations, work.residual_points
    );
    assert!(
        work.maximum_root_iterations <= AREA_ROUNDS,
        "rounds={}",
        work.maximum_root_iterations
    );
    Ok(())
}

#[test]
fn configured_cpu_public_point_tiny_jump_rest_face_sharpens_to_the_area_rounding() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    tiny_jump::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_public_point_tiny_jump_rest_face_sharpens_to_the_area_rounding() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    tiny_jump::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
