use super::super::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelLateralBinding,
    ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition, RectangularChannelCell,
};
use crate::forcing::WaterAmount;
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
use std::sync::Arc;

pub(super) type TestResult = Result<(), Box<dyn std::error::Error>>;

pub(super) fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}

pub(super) fn definition(directions: &[ChannelLateralDirection]) -> Arc<ChannelReachDefinition> {
    Arc::new(ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "explicit lateral arithmetic fixture".into(),
        geometry: crate::channel::ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell".into(),
                length_m: 4.0,
                width_m: 2.0,
                bed_elevation_m: 0.0,
                maximum_depth_m: 1e307,
                manning_n: 0.0,
            }],
        },
        lateral_bindings: directions
            .iter()
            .enumerate()
            .map(|(index, &direction)| ChannelLateralBinding {
                boundary_id: format!("source-{index}"),
                cell_id: "cell".into(),
                direction,
                ownership_assertion: "selected disjoint arithmetic source".into(),
            })
            .collect(),
        config: ChannelConfig::default(),
    })
}

pub(super) fn forcing(inputs: Vec<ChannelLateralInput>) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s: 0.0,
        end_s: 2.0,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: inputs,
    }
}

pub(super) fn inflow(index: usize, volume: f64, velocity: f64) -> ChannelLateralInput {
    ChannelLateralInput::Inflow {
        boundary_id: format!("source-{index}"),
        amount: WaterAmount::IntervalVolume { volume_m3: volume },
        longitudinal_velocity_m_per_s: velocity,
    }
}
