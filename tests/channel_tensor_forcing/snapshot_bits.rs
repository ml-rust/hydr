use hydr::channel::*;
use hydr::forcing::WaterAmount;

trait FloatBits {
    fn append_bits(&self, bits: &mut Vec<u64>);
}

impl FloatBits for f64 {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        bits.push(self.to_bits());
    }
}

impl FloatBits for usize {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        bits.extend(self.to_le_bytes().map(u64::from));
    }
}

impl FloatBits for String {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        self.len().append_bits(bits);
        bits.extend(self.bytes().map(u64::from));
    }
}

impl<T: FloatBits> FloatBits for Vec<T> {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        for value in self {
            value.append_bits(bits);
        }
    }
}

impl<T: FloatBits> FloatBits for Option<T> {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        if let Some(value) = self {
            value.append_bits(bits);
        }
    }
}

impl<T: FloatBits, const N: usize> FloatBits for [T; N] {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        for value in self {
            value.append_bits(bits);
        }
    }
}

macro_rules! fields {
    ($kind:ty, $($field:ident),+ $(,)?) => {
        impl FloatBits for $kind {
            fn append_bits(&self, bits: &mut Vec<u64>) {
                $(self.$field.append_bits(bits);)+
            }
        }
    };
}

fields!(
    RectangularChannelCell,
    length_m,
    width_m,
    bed_elevation_m,
    maximum_depth_m,
    manning_n,
);

fields!(ChannelReachDefinition, geometry, config,);

fields!(
    ChannelInitialState,
    time_s,
    volume_m3,
    longitudinal_momentum_m4_per_s,
);

fields!(
    ChannelForcingInterval,
    start_s,
    end_s,
    left,
    right,
    lateral_inputs,
);

fields!(
    ChannelConfig,
    boundary,
    gravity,
    cfl,
    max_step,
    min_step,
    root_absolute_depth,
    root_relative_depth,
    riemann_water_flux_absolute_m3_per_s,
    riemann_momentum_flux_absolute_m4_per_s2,
    riemann_flux_relative,
    volume_absolute,
    volume_relative,
    momentum_absolute,
    momentum_relative,
);

fields!(
    ChannelBoundaryConfig,
    discharge_absolute_m3_per_s,
    discharge_relative,
    momentum_absolute_m4_per_s2,
    momentum_relative,
    velocity_relative,
    critical_relative,
    reservoir_head_absolute_m,
    reservoir_head_relative,
);

fields!(
    ChannelBoundaryTrace,
    depth_m,
    area_m2,
    outward_velocity_m_per_s,
    outward_discharge_m3_per_s,
    momentum_flux_m4_per_s2,
    signal_speed_bound_m_per_s,
    velocity_compatibility_abs_bound_m_per_s,
    water_compatibility_abs_bound_m3_per_s,
    momentum_compatibility_abs_bound_m4_per_s2,
    water_flux_width_m3_per_s,
    momentum_flux_width_m4_per_s2,
    reservoir,
);

fields!(
    ChannelReservoirTrace,
    depth_lower_m,
    depth_upper_m,
    head_compatibility_abs_bound_m,
    dissipation_head_lower_m,
    dissipation_head_upper_m,
);

fields!(
    ChannelCellResult,
    volume_m3,
    longitudinal_momentum_m4_per_s,
    depth_m,
    stage_m,
    velocity_m_per_s,
    discharge_m3_per_s,
);

fields!(
    ChannelWaterLedger,
    initial_storage_m3,
    storage_m3,
    boundary_in_m3,
    boundary_out_m3,
    lateral_in_m3,
    lateral_out_m3,
    lateral_counter_abs_bound_m3,
    lateral_assembly_abs_bound_m3,
    residual_m3,
    numerical_transfer_underflow_bound_m3,
    storage_selection_abs_bound_m3,
);

fields!(
    ChannelStepDiagnostics,
    boundary_traces,
    riemann,
    friction_geometry,
    start_s,
    end_s,
    cfl_limit_s,
    accepted_start_cfl_limit_s,
    candidate_cfl_limit_s,
    depletion_limit_s,
    storage_before_m3,
    storage_after_m3,
    boundary_in_m3,
    boundary_out_m3,
    lateral_in_m3,
    lateral_out_m3,
    lateral_assembly_abs_bound_m3,
    lateral_counter_addition_abs_bound_m3,
    residual_m3,
    max_local_residual_m3,
    max_local_momentum_residual_m4_per_s,
    max_momentum_underflow_bound_m4_per_s,
    face_volume_m3,
    face_volume_low_m3,
    storage_selection_bound_m3,
    face_transfer_underflow_bound_m3,
    transfer_underflow_bound_m3,
    face_momentum_m4_per_s,
    face_momentum_low_m4_per_s,
    left_pressure_impulse_m4_per_s,
    right_pressure_impulse_m4_per_s,
    friction_impulse_m4_per_s,
    momentum_underflow_bound_m4_per_s,
    lateral_momentum_abs_bound_m4_per_s,
    lateral_receipts,
);

fields!(
    ChannelLateralLedger,
    cumulative_volume_m3,
    roundoff_bound_m3,
);

fields!(
    ChannelLateralStateData,
    cumulative_volume_m3,
    counter_roundoff_bound_m3,
    cumulative_assembly_bound_m3,
);

fields!(
    ChannelLateralReceipt,
    input,
    applied_longitudinal_velocity_m_per_s,
    volume_m3,
    momentum_m4_per_s,
    cumulative_volume_before_m3,
    cumulative_volume_after_m3,
    counter_addition_low_m3,
    counter_bound_before_m3,
    counter_bound_after_m3,
);

fields!(
    ChannelRiemannDiagnostics,
    root_iterations,
    surveyed_work,
    water_flux_width_m3_per_s,
    momentum_flux_width_m4_per_s2,
    max_water_flux_width_m3_per_s,
    max_momentum_flux_width_m4_per_s2,
);

fields!(
    ChannelFrictionGeometryDiagnostics,
    momentum_abs_bound_m4_per_s,
    max_momentum_abs_bound_m4_per_s,
);

fields!(
    PrismaticRiemannWork,
    root_solves,
    root_iterations,
    maximum_root_iterations,
    residual_points,
    simpson_value_points,
    simpson_derivative_points,
    refinements,
    maximum_simpson_panels,
);

fields!(
    ChannelSnapshotData,
    definition,
    initial,
    time_s,
    volume_m3,
    volume_low_m3,
    longitudinal_momentum_m4_per_s,
    longitudinal_momentum_low_m4_per_s,
    ledger,
    lateral,
    last_interval,
);

fields!(
    ChannelIntervalSnapshotData,
    previous_time_s,
    previous_volume_m3,
    previous_volume_low_m3,
    previous_longitudinal_momentum_m4_per_s,
    previous_longitudinal_momentum_low_m4_per_s,
    previous_ledger,
    previous_lateral,
    forcing,
    requested_end_s,
    diagnostics,
);

fields!(TriangularChannelCell, length_m, maximum_depth_m, manning_n,);

fields!(SurveyedChannelCell, id, length_m, manning_n,);
fields!(CrossSectionPoint, station_m, elevation_m,);
fields!(
    CrossSectionDefinition,
    id,
    vertical_datum,
    points,
    maximum_stage_m,
    profile_provenance,
);
fields!(PrismaticRecoveryConfig, depth_atol_m, depth_rtol,);
fields!(
    solvr::optimize::CertifiedBisectionOptions,
    atol,
    rtol,
    max_iterations,
    max_points,
);
fields!(
    ChannelSurveyedConfig,
    recovery,
    normalized_root,
    initial_simpson_panels,
    max_simpson_panels,
    max_refinements,
    max_native_query_points,
    friction_geometry_atol_m4_per_s,
    friction_geometry_rtol,
);

impl FloatBits for ChannelGeometry {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        match self {
            Self::PrismaticSurveyed {
                profile,
                cells,
                numerical,
            } => {
                profile.points.len().append_bits(bits);
                cells.len().append_bits(bits);
                profile.append_bits(bits);
                cells.append_bits(bits);
                numerical.append_bits(bits);
            }
            Self::Rectangular { cells } => cells.append_bits(bits),
            Self::PrismaticTriangular {
                left_bank_slope_h_per_v,
                right_bank_slope_h_per_v,
                bed_elevation_m,
                cells,
            } => {
                left_bank_slope_h_per_v.append_bits(bits);
                right_bank_slope_h_per_v.append_bits(bits);
                bed_elevation_m.append_bits(bits);
                cells.append_bits(bits);
            }
        }
    }
}

impl FloatBits for ChannelBoundary {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        match self {
            Self::Wall => {}
            Self::Stage { elevation_m } => elevation_m.append_bits(bits),
            Self::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => {
                elevation_m.append_bits(bits);
                entrance_loss_coefficient.append_bits(bits);
            }
            Self::Discharge { discharge_m3_per_s } => discharge_m3_per_s.append_bits(bits),
            Self::Exterior {
                depth_m,
                velocity_m_per_s,
            } => {
                depth_m.append_bits(bits);
                velocity_m_per_s.append_bits(bits);
            }
        }
    }
}

impl FloatBits for WaterAmount {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        match self {
            Self::Rate { rate_m3_per_s } => rate_m3_per_s.append_bits(bits),
            Self::IntervalVolume { volume_m3 } => volume_m3.append_bits(bits),
        }
    }
}

impl FloatBits for ChannelLateralInput {
    fn append_bits(&self, bits: &mut Vec<u64>) {
        match self {
            Self::Inflow {
                amount,
                longitudinal_velocity_m_per_s,
                ..
            } => {
                amount.append_bits(bits);
                longitudinal_velocity_m_per_s.append_bits(bits);
            }
            Self::Withdrawal { amount, .. } => amount.append_bits(bits),
        }
    }
}

pub fn assert_same(actual: &ChannelSnapshotData, expected: &ChannelSnapshotData) {
    assert_eq!(actual, expected);
    let mut actual_bits = Vec::new();
    let mut expected_bits = Vec::new();
    actual.append_bits(&mut actual_bits);
    expected.append_bits(&mut expected_bits);
    assert_eq!(actual_bits, expected_bits);
}
