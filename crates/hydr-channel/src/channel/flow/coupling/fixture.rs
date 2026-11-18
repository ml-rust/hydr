use super::super::{ChannelError, ChannelLateralInput, ChannelReach};
use crate::channel::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(super) fn interval_reach() -> Result<ChannelReach<CpuRuntime>, ChannelError> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    ChannelReach::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "common interval ownership fixture".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 10.0,
                    width_m: 1.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 2.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![crate::channel::ChannelLateralBinding {
                boundary_id: "source".into(),
                cell_id: "cell".into(),
                direction: crate::channel::ChannelLateralDirection::Inflow,
                ownership_assertion: "independent zero input".into(),
            }],
            config: ChannelConfig {
                gravity: 1.0,
                max_step: 0.125,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 2.0,
            volume_m3: vec![10.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )
}

pub(super) fn zero_source(amount: crate::forcing::WaterAmount) -> ChannelLateralInput {
    ChannelLateralInput::Inflow {
        boundary_id: "source".into(),
        amount,
        longitudinal_velocity_m_per_s: 0.0,
    }
}
