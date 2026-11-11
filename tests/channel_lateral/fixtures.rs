use crate::common::{Reach, cell, definition, exterior, reach};
use hydr::channel::{
    ChannelBoundary, ChannelForcingInterval, ChannelLateralBinding, ChannelLateralDirection,
    ChannelLateralInput,
};
use hydr::forcing::WaterAmount;

pub fn binding(
    id: &str,
    cell_id: &str,
    direction: ChannelLateralDirection,
) -> ChannelLateralBinding {
    ChannelLateralBinding {
        boundary_id: id.into(),
        cell_id: cell_id.into(),
        direction,
        ownership_assertion: format!("Independent synthetic source support for {id}"),
    }
}

pub fn inflow(id: &str, amount: WaterAmount, velocity: f64) -> ChannelLateralInput {
    ChannelLateralInput::Inflow {
        boundary_id: id.into(),
        amount,
        longitudinal_velocity_m_per_s: velocity,
    }
}

pub fn withdrawal(id: &str, amount: WaterAmount) -> ChannelLateralInput {
    ChannelLateralInput::Withdrawal {
        boundary_id: id.into(),
        amount,
    }
}

pub fn exact(volume: f64) -> WaterAmount {
    WaterAmount::IntervalVolume { volume_m3: volume }
}

pub fn rate(value: f64) -> WaterAmount {
    WaterAmount::Rate {
        rate_m3_per_s: value,
    }
}

pub fn moving_cell(
    bindings: Vec<ChannelLateralBinding>,
) -> Result<Reach, Box<dyn std::error::Error>> {
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = bindings;
    reach(layout, &[1.0], &[6.0])
}

pub fn forcing(
    start_s: f64,
    end_s: f64,
    lateral_inputs: Vec<ChannelLateralInput>,
    moving: bool,
) -> ChannelForcingInterval {
    let boundary = if moving {
        exterior(1.0, 3.0)
    } else {
        ChannelBoundary::Wall
    };
    ChannelForcingInterval {
        start_s,
        end_s,
        left: boundary.clone(),
        right: boundary,
        lateral_inputs,
    }
}

pub fn input_id(input: &ChannelLateralInput) -> &str {
    match input {
        ChannelLateralInput::Inflow { boundary_id, .. }
        | ChannelLateralInput::Withdrawal { boundary_id, .. } => boundary_id,
    }
}

pub fn check_input_bits(actual: &ChannelLateralInput, expected: &ChannelLateralInput) {
    let check_amount = |actual: WaterAmount, expected: WaterAmount| match (actual, expected) {
        (
            WaterAmount::Rate {
                rate_m3_per_s: actual,
            },
            WaterAmount::Rate {
                rate_m3_per_s: expected,
            },
        ) => {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        (
            WaterAmount::IntervalVolume { volume_m3: actual },
            WaterAmount::IntervalVolume {
                volume_m3: expected,
            },
        ) => {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        _ => assert_eq!(actual, expected),
    };
    match (actual, expected) {
        (
            ChannelLateralInput::Inflow {
                boundary_id: actual_id,
                amount: actual_amount,
                longitudinal_velocity_m_per_s: actual_velocity,
            },
            ChannelLateralInput::Inflow {
                boundary_id: expected_id,
                amount: expected_amount,
                longitudinal_velocity_m_per_s: expected_velocity,
            },
        ) => {
            assert_eq!(actual_id, expected_id);
            check_amount(*actual_amount, *expected_amount);
            assert_eq!(actual_velocity.to_bits(), expected_velocity.to_bits());
        }
        (
            ChannelLateralInput::Withdrawal {
                boundary_id: actual_id,
                amount: actual_amount,
            },
            ChannelLateralInput::Withdrawal {
                boundary_id: expected_id,
                amount: expected_amount,
            },
        ) => {
            assert_eq!(actual_id, expected_id);
            check_amount(*actual_amount, *expected_amount);
        }
        _ => assert_eq!(actual, expected),
    }
}

pub fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual={actual}, expected={expected}"
    );
}
