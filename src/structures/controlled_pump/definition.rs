//! Explicit actual pump response assumptions without invented motor dynamics.

/// Selected physical response approximation. No implicit response model applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PumpResponseModel {
    /// Actual operation equals the accepted sampled command on the decision's right side.
    ///
    /// Acceleration, startup delay, coast-down, and independent response memory are excluded.
    IdealInstantaneous,
}
/// Complete response evidence bound to one actual pump in original controlled order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PumpResponseDefinition {
    /// Identity of the actual pump controlled by the corresponding original controller.
    pub structure_id: String,
    /// Explicit response model, independent of the hydraulic pump curve.
    pub model: PumpResponseModel,
    /// Nonempty evidence identifying the supplied response approximation.
    pub response_provenance: String,
    /// Nonempty operating-envelope and ideal-response assumptions, including excluded motor dynamics.
    pub ideal_response_applicability: String,
}
