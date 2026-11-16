use super::super::super::{
    point_output::PrismaticPointTraceKind,
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

/// Below-cap offsets of the two face areas.
const OFFSETS: [f64; 2] = [2.5e-17, 8e-17];

/// The rounded wet-fan section: three bands up to a 3 m cap of 14 m² and 8 m top width.
/// At the cap the hydrostatic integral is 17 m³.
fn rounded_wet_fan<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let points = [
        (-4.0, 3.0),
        (-2.5, 2.0),
        (-2.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (2.5, 2.0),
        (4.0, 3.0),
    ];
    let definition = CrossSectionDefinition {
        id: "rounded_wet_fan".into(),
        vertical_datum: "Exact dyadic datum".into(),
        points: points
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 3.0,
        profile_provenance: "Independent dyadic proof".into(),
    };
    Ok(CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?)
}

/// Selected momentum components of equal-state faces, one face per offset, in that order.
///
/// Each face holds A = 14 - d and Q = 2A on both sides. The velocity 2 m/s exceeds the
/// celerity √1.75 m/s, so both waves move right and the trace is the left state.
fn momentum<R>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    offsets: &[f64],
) -> Result<Vec<[u64; 3]>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let owners = support::rows::<R>(client, &vec![0; offsets.len()])?;
    let area = offsets
        .iter()
        .map(|d| [14.0_f64.to_bits(), (-d).to_bits(), 0])
        .collect::<Vec<_>>();
    let discharge = offsets
        .iter()
        .map(|d| [28.0_f64.to_bits(), (-2.0 * d).to_bits(), 0])
        .collect::<Vec<_>>();
    let area = frozen::expansion::<R>(client, &area, true)?;
    let discharge = frozen::expansion::<R>(client, &discharge, true)?;
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: 1.0,
        ..PrismaticRiemannConfig::default()
    };
    let result = section
        .riemann_flux_points_tensor(&owners, &area, &discharge, &area, &discharge, &config)
        .map_err(|error| format!("supercritical wet-fan faces reject: {error:?}"))?;
    let kinds = result.trace_kind().try_to_vec::<i64>()?;
    assert_eq!(
        kinds,
        vec![PrismaticPointTraceKind::LeftState as i64; offsets.len()]
    );
    frozen::components(result.momentum_flux_m4_per_s2())
}

fn left_state<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let section = rounded_wet_fan::<R>(client)?;
    let forward = momentum::<R>(client, &section, &OFFSETS)?;
    let reverse = momentum::<R>(client, &section, &[OFFSETS[1], OFFSETS[0]])?;
    for (index, d) in OFFSETS.into_iter().enumerate() {
        let [high, low, _] = forward[index].map(f64::from_bits);
        // Q²/A = 4A = 56 - 4d exactly. I1(14 - d) = 17 - 1.75d + O(d²), and g = 1.
        // The high part lies within one ulp of 73, so high - 73 is exact.
        let difference = (high - 73.0) + low + 4.0 * d;
        let expected = -1.75 * d;
        assert!(
            (difference - expected).abs() <= 1e-30,
            "d={d:e}, difference={difference:e}, expected={expected:e}"
        );
        // Equal faces give equal bits in either batch order.
        assert_eq!(forward[index], reverse[1 - index], "d={d:e}");
    }
    Ok(())
}

#[test]
fn configured_cpu_public_point_left_state_momentum_uses_two_part_pressure() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    left_state::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_public_point_left_state_momentum_uses_two_part_pressure() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    left_state::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
