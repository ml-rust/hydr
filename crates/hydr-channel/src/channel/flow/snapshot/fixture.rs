use crate::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelForcingInterval, ChannelInitialState,
    ChannelReach, ChannelReachDefinition, RectangularChannelCell,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

pub(super) fn trace() -> crate::channel::ChannelBoundaryTrace {
    crate::channel::ChannelBoundaryTrace {
        depth_m: 0.0,
        area_m2: 0.0,
        outward_velocity_m_per_s: 0.0,
        outward_discharge_m3_per_s: 0.0,
        momentum_flux_m4_per_s2: 0.0,
        signal_speed_bound_m_per_s: 0.0,
        velocity_compatibility_abs_bound_m_per_s: 0.0,
        water_compatibility_abs_bound_m3_per_s: 0.0,
        momentum_compatibility_abs_bound_m4_per_s2: 0.0,
        water_flux_width_m3_per_s: 0.0,
        momentum_flux_width_m4_per_s2: 0.0,
        cap_compatibility_approximation: false,
        reservoir: None,
        root_iterations: 3,
    }
}

pub(super) fn walled_reach() -> Result<ChannelReach<CpuRuntime>, ChannelError> {
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "two-part state fixture".into(),
        lateral_bindings: Vec::new(),
        geometry: crate::channel::ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell".into(),
                length_m: 1.0,
                width_m: 2.0,
                bed_elevation_m: 0.0,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        },
        config: ChannelConfig::default(),
    };
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    ChannelReach::new(CpuClient::new(CpuDevice::new()), definition, initial)
}

pub(super) fn advanced(
    reach: &ChannelReach<CpuRuntime>,
) -> Result<ChannelReach<CpuRuntime>, ChannelError> {
    let start = reach.time();
    Ok(reach
        .step(
            start + 0.01,
            &ChannelForcingInterval {
                start_s: start,
                end_s: start + 0.01,
                left: ChannelBoundary::Wall,
                right: ChannelBoundary::Wall,
                lateral_inputs: Vec::new(),
            },
        )?
        .next)
}
