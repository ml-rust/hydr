//! Host projections of structure links, pool ends, and pools over accepted intervals.

use crate::structures::{
    ControlLatch, GateActuatorPosition, PumpResponseModel, StructureOperatingPoint, StructureRegime,
};

/// Source evidence of one structure's interval operation.
///
/// A gate's interval opening is the midpoint of its start and end openings, the partner of
/// midpoint heads. A pump holds its accepted operation from the start.
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceChannelOperationReceipt {
    /// A manual owner: the supplied operation, held over every interval.
    Manual,
    /// A prescribed trajectory read at both interval ends.
    PrescribedGate {
        /// Opening at the start, in metres.
        start_opening_m: f64,
        /// Opening at the end, in metres.
        end_opening_m: f64,
        /// Interval opening the law takes, in metres.
        opening_m: f64,
        /// Larger endpoint interpolation error, in metres.
        absolute_error_m: f64,
        /// Exact distance from `opening_m` to the mean of the endpoint openings, in metres.
        midpoint_rounding_m: f64,
        /// Closed at both ends: exact closure over the interval.
        closed: bool,
    },
    /// An actuator position at the accepted start and at the old-target end.
    ///
    /// The mean of the ideal opening lies in `[lower_opening_m, upper_opening_m]`.
    ActuatedGate {
        /// Actuator identity.
        actuator_id: String,
        /// Accepted position at the start.
        start: GateActuatorPosition,
        /// Old-target position at the end, before any decision at the end applies.
        end: GateActuatorPosition,
        /// Interval opening the law takes, in metres.
        opening_m: f64,
        /// Smaller endpoint lower bound, in metres.
        lower_opening_m: f64,
        /// Larger endpoint upper bound, in metres.
        upper_opening_m: f64,
        /// Larger endpoint absolute error, in metres.
        absolute_error_m: f64,
        /// Exact distance from `opening_m` to the mean of the endpoint centres, in metres.
        midpoint_rounding_m: f64,
        /// Closed at both ends: exact closure over the interval.
        closed: bool,
    },
    /// The accepted pump operation at the start, held over the interval.
    ControlledPump {
        /// Controller identity.
        controller_id: String,
        /// Accepted latch at the start.
        latch: ControlLatch,
        /// Declared response model.
        response_model: PumpResponseModel,
    },
}

/// Accepted exchange through one structure link over one common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelStructureTransfer {
    /// Exact structure identity.
    pub structure_id: String,
    /// Accepted volume, positive from endpoint A to endpoint B, in cubic metres. Both sides
    /// apply it.
    pub signed_volume_m3: f64,
    /// Law discharge at the midpoint heads under the interval operation, times the interval,
    /// in cubic metres.
    pub law_volume_m3: f64,
    /// Cap factor in `[0, 1]` applied to the law volume: levelling, crest, sill, and suction
    /// caps. It is exactly `1.0` unless a cap binds.
    pub cap_factor: f64,
    /// Donor allocation factor in `[0, 1]` applied to the capped law volume.
    pub donor_factor: f64,
    /// Law regime at the midpoint heads.
    pub regime: StructureRegime,
    /// Mean of start and end stage at endpoint A, in metres.
    pub midpoint_stage_a_m: f64,
    /// Mean of start and end stage at endpoint B, in metres.
    pub midpoint_stage_b_m: f64,
    /// Operation the law took over the interval.
    pub operation: StructureOperatingPoint,
    /// Source evidence of the operation.
    pub receipt: SurfaceChannelOperationReceipt,
}

/// Lifetime directional exchange through one structure link.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelStructureLedger {
    /// Exact structure identity.
    pub structure_id: String,
    /// Accumulated volume from endpoint A to endpoint B, in cubic metres.
    pub a_to_b_m3: f64,
    /// Accumulated volume from endpoint B to endpoint A, in cubic metres.
    pub b_to_a_m3: f64,
    /// Conservative rounding envelope of both counters, in cubic metres.
    pub roundoff_bound_m3: f64,
}

/// Accepted exchange through one pool end over one common interval.
///
/// The reach's own transport sets the volume. The pool applies the same bits.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPoolEndTransfer {
    /// Exact pool-end identity.
    pub pool_end_id: String,
    /// Volume, positive from the reach into the pool, in cubic metres.
    pub signed_volume_m3: f64,
    /// Reservoir head the reach end took: its pool's temporal midpoint stage, in metres.
    pub reservoir_head_m: f64,
}

/// Lifetime directional exchange through one pool end.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPoolEndLedger {
    /// Exact pool-end identity.
    pub pool_end_id: String,
    /// Accumulated volume from the reach into the pool, in cubic metres.
    pub reach_to_pool_m3: f64,
    /// Accumulated volume from the pool into the reach, in cubic metres.
    pub pool_to_reach_m3: f64,
    /// Conservative rounding envelope of both counters, in cubic metres.
    pub roundoff_bound_m3: f64,
}

/// One pool at the end of an accepted common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPoolStep {
    /// Exact pool identity.
    pub pool_id: String,
    /// Accepted volume at the end, in cubic metres.
    pub volume_m3: f64,
    /// Accepted stage at the end, in metres. A dry pool takes its curve origin.
    pub stage_m: f64,
    /// Native incoming volume over the interval: link, pool-end, and external inflow water, in
    /// cubic metres.
    pub incoming_m3: f64,
    /// Native outgoing volume over the interval: link and pool-end water, in cubic metres.
    pub outgoing_m3: f64,
}
