use super::{
    super::{
        ChannelConfig, ChannelInitialState, ChannelReachDefinition,
        context::Context,
        geometry::{ChannelGeometry, TriangularChannelCell},
    },
    faces::Face,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(super) type TestResult = Result<(), Box<dyn std::error::Error>>;

pub(super) fn context() -> Result<Context<CpuRuntime>, Box<dyn std::error::Error>> {
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

pub(super) fn face(
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
