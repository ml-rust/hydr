use super::super::friction_test_data::Ode;
use crate::channel::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReach, ChannelReachDefinition,
    ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell,
};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(super) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

pub(super) fn reach(row: &Ode, roughness: f64) -> TestResult<ChannelReach<CpuRuntime>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    reach_with_client::<CpuRuntime>(client, row, roughness)
}

pub(super) fn reach_with_client<R>(
    client: R::Client,
    row: &Ode,
    roughness: f64,
) -> TestResult<ChannelReach<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    let operands = row.operands.map(f64::from_bits);
    Ok(ChannelReach::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "Synthetic SI datum".into(),
            provenance: "Independent original-coordinate fixed-area Manning references".into(),
            geometry: ChannelGeometry::PrismaticSurveyed {
                profile: CrossSectionDefinition {
                    id: row.id.into(),
                    vertical_datum: "Synthetic SI datum".into(),
                    points: row
                        .points
                        .iter()
                        .map(|point| CrossSectionPoint {
                            station_m: f64::from_bits(point[0]),
                            elevation_m: f64::from_bits(point[1]),
                        })
                        .collect(),
                    maximum_stage_m: f64::from_bits(row.cap),
                    profile_provenance: "Frozen original survey bits".into(),
                },
                cells: vec![SurveyedChannelCell {
                    id: "cell".into(),
                    length_m: operands[0],
                    manning_n: roughness,
                }],
                numerical: ChannelSurveyedConfig::default(),
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig {
                gravity: operands[4],
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![operands[2]],
            longitudinal_momentum_m4_per_s: vec![operands[3]],
        },
    )?)
}
