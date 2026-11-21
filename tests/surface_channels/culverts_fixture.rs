//! Culvert cases of the surface/channel coordinator through the public API.
//!
//! Every culvert is a circular concrete barrel of the fixture law `culvert_law`. Each case
//! returns its setup and initial state, so the CPU and CUDA runs share it.

use crate::pools_fixture::{
    Case, INFLOW, REACH, Result, WITHDRAWAL, channel, culvert_law, manual, pond, pool, structure,
    weir_law,
};
use hydr::channel::{ChannelBoundary, ChannelEnd, ChannelLateralInput};
use hydr::coupling::{
    SurfaceChannelForcingInterval, SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use hydr::forcing::WaterAmount;
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

/// Rise of every culvert barrel, in m.
pub const RISE_M: f64 = 0.3;

/// A tidal reach cell and a lagoon pond at level heads, 1 m deep.
///
/// The lagoon is endpoint A of both culverts and the reach cell endpoint B. `tide` runs both
/// ways. `flap` declares only its lagoon end, so it carries water from the lagoon only.
pub fn reversal_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 1.0);
    case.ponds = vec![(pond("lagoon", 0.0, 8.0, 3.0)?, 8.0)];
    let culvert = |id: &str, two_way: bool, direction| -> Result<_> {
        Ok(structure(
            id,
            culvert_law(RISE_M, 10.0, [0.0, 0.0], two_way)?,
            direction,
            manual(StructureOperatingPoint::Passive),
            (pool("lagoon"), 0.0),
            (channel(), 0.0),
        ))
    };
    case.structures = vec![
        culvert("tide", true, DirectionPolicy::Bidirectional)?,
        culvert("flap", false, DirectionPolicy::AtoBOnly)?,
    ];
    Ok(case)
}

/// Caller tide on the reach cell over `[start, end)`: inflow `rate` m³/s when positive,
/// withdrawal `-rate` when negative.
pub fn tide(
    setup: &SurfaceChannelSetup,
    (start, end): (f64, f64),
    rate: f64,
) -> SurfaceChannelForcingInterval {
    let owned = |side: ChannelEnd| setup.pool_ends.iter().any(|e| e.end == side);
    let policy = |side| (!owned(side)).then_some(ChannelBoundary::Wall);
    let amount = |rate_m3_per_s: f64| WaterAmount::Rate { rate_m3_per_s };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: policy(ChannelEnd::Left),
            right: policy(ChannelEnd::Right),
            lateral_inputs: vec![
                ChannelLateralInput::Inflow {
                    boundary_id: INFLOW.into(),
                    amount: amount(rate.max(0.0)),
                    longitudinal_velocity_m_per_s: 0.0,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id: WITHDRAWAL.into(),
                    amount: amount((-rate).max(0.0)),
                },
            ],
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: Vec::new(),
    }
}

/// Tide rate at `t`, in m³/s: a flood until 4 s, an ebb until 12 s, then a flood again.
pub fn tide_rate(t: f64) -> f64 {
    if t < 4.0 {
        0.1
    } else if t < 12.0 {
        -0.1
    } else {
        0.1
    }
}

/// A large head pond 1.5 m deep drains through a short steep culvert into a dry basin.
///
/// The inlet at the head pond sits at 0.2 m, the outlet at the basin floor. The basin fills, so
/// its tailwater rises through the outlet and the culvert passes from inlet to outlet control.
/// The fall exceeds half the rise, so the culvert declares its head-pond end only.
pub fn sweep_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("basin", 0.0, 2.0, 3.0)?, 0.0),
        (pond("head", 0.0, 200.0, 3.0)?, 1.5 * 200.0),
    ];
    case.structures = vec![structure(
        "culvert",
        culvert_law(RISE_M, 5.0, [0.2, 0.0], false)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("head"), 0.0),
        (pool("basin"), 0.0),
    )];
    Ok(case)
}

/// Rise of the long barrel of [`drowning_case`], in m.
pub const LONG_RISE_M: f64 = 0.9;

/// A large head pond 1 m deep drains through a long mild culvert into a basin 0.5 m deep.
///
/// The outlet runs partly full. The basin fills past the crown, so the barrel runs full.
pub fn drowning_case() -> Result<Case> {
    drowning_from(0.5)
}

/// Basin depth of [`crown_case`], in m: below the crown of the long barrel.
pub const CROWN_START_M: f64 = 0.8;

/// The culvert of [`drowning_case`] into a basin `CROWN_START_M` deep, near the crown.
///
/// The outlet runs partly full and drowns within a few intervals.
pub fn crown_case() -> Result<Case> {
    drowning_from(CROWN_START_M)
}

/// A large head pond 1 m deep drains through a long mild culvert into a basin `depth_m` deep.
fn drowning_from(depth_m: f64) -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("basin", 0.0, 10.0, 3.0)?, depth_m * 10.0),
        (pond("head", 0.0, 1000.0, 3.0)?, 1.0 * 1000.0),
    ];
    case.structures = vec![structure(
        "culvert",
        culvert_law(LONG_RISE_M, 40.0, [0.1, 0.0], true)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("head"), 0.0),
        (pool("basin"), 0.0),
    )];
    Ok(case)
}

/// Crest of the roadway weir over the culvert, in m.
pub const ROAD_CREST_M: f64 = 1.5;

/// An upper pond overtops a road into a lower pond while a drowned culvert runs under it.
///
/// The culvert and the roadway weir join the same two ponds.
pub fn roadway_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 30.0, 3.0)?, 1.0 * 30.0),
        (pond("upper", 0.0, 10.0, 3.0)?, 1.8 * 10.0),
    ];
    let passive = || manual(StructureOperatingPoint::Passive);
    case.structures = vec![
        structure(
            "culvert",
            culvert_law(RISE_M, 12.0, [0.05, 0.0], true)?,
            DirectionPolicy::Bidirectional,
            passive(),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
        structure(
            "road",
            weir_law(ROAD_CREST_M, 2.0)?,
            DirectionPolicy::Bidirectional,
            passive(),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
    ];
    Ok(case)
}
