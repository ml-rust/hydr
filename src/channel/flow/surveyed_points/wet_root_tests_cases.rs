use crate::channel::flow::surveyed::PrismaticRiemannConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Expected {
    Located,
    Vacuum,
    TouchingDry,
    CapExceeded,
    LegacyThreshold,
    RejectedWork,
}

// Original execution controls come from acceptance-addendum-corrected.json.
pub(super) fn declared(id: &str) -> (PrismaticRiemannConfig, Expected) {
    let mut config = PrismaticRiemannConfig {
        gravity_m_per_s2: match id {
            "scalar_triangle_uniform_flow"
            | "scalar_trapezoid_uniform_flow"
            | "scalar_three_bands_uniform_upper"
            | "captured_tiny_upwind_face_64" => f64::from_bits(0x40239eb851eb851f),
            "scalar_rectangle_stationary_shock" | "scalar_rectangle_stationary_shock_reverse" => {
                3.0
            }
            "scalar_trapezoid_stationary_shock" => 46.0,
            _ => 1.0,
        },
        ..PrismaticRiemannConfig::default()
    };
    config.root.atol = 1e-12;
    config.root.rtol = 1e-12;
    config.root.max_iterations = 256;
    config.root.max_points = 1_048_576;
    if !id.starts_with("scalar_") {
        let allowance = f64::from_bits(0x3d719799812dea11);
        config.recovery.depth_atol_m = allowance;
        config.recovery.depth_rtol = allowance;
        config.water_flux_atol_m3_per_s = allowance;
        config.momentum_flux_atol_m4_per_s2 = allowance;
        config.flux_rtol = allowance;
        config.root.atol = f64::from_bits(0x3cd203af9ee75616);
        config.root.rtol = f64::from_bits(0x3cd203af9ee75616);
    }
    if id == "scalar_three_bands_loose_recovery_allowance" {
        config.recovery.depth_atol_m = 1.0;
        config.recovery.depth_rtol = 1.0;
    }
    let expected = match id {
        "scalar_rectangle_vacuum" | "scalar_rectangle_vacuum_threshold_outside" => Expected::Vacuum,
        "scalar_rectangle_vacuum_threshold" => Expected::TouchingDry,
        "scalar_rectangle_vacuum_threshold_inside" => Expected::LegacyThreshold,
        "scalar_three_bands_cap_compression"
        | "trace_history_final_native_face_190"
        | "trace_history_final_retained_face_190" => Expected::CapExceeded,
        "scalar_three_bands_zero_work_budget" => {
            config.max_native_query_points = 0;
            Expected::RejectedWork
        }
        _ => Expected::Located,
    };
    (config, expected)
}

// These are star areas, never sampled interface traces or selected centers.
pub(super) fn star(id: &str) -> Option<[u64; 2]> {
    match id {
        "scalar_rectangle_stationary_shock" | "scalar_rectangle_stationary_shock_reverse" => {
            Some([0x4000000000000000; 2])
        }
        "tiny_rectangle_compression" => Some([0x2d26a09e667f3bcc, 0x2d26a09e667f3bcd]),
        "uniform_cap_represented_tenth_forward" | "uniform_cap_represented_tenth_reverse" => {
            Some([0x402c000000000000; 2])
        }
        "captured_tiny_upwind_face_64" => Some([0x3bc280007c90d340, 0x3bc280007c90d341]),
        "retained_knot_2_crossing" => Some([0x3fffffffffffffff, 0x4000000000000000]),
        "retained_knot_7_crossing" => Some([0x401bffffffffffff, 0x401c000000000000]),
        "retained_vacuum_inside_film" => Some([0x35b0000000000000; 2]),
        "pointed_minimum_uniform_moving" => Some([1; 2]),
        "low_only_rectangle_uniform_moving" => Some([0x0170000000000000; 2]),
        _ => None,
    }
}
