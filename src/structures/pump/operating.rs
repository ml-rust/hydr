//! Explicit identity-bound passive and pump operation packets.

/// Original operation held during a constitutive proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StructureOperatingPoint {
    /// No active device operation applies to a passive law.
    Passive,
    /// Explicitly isolated against forward siphoning and reverse flow.
    PumpStoppedIsolated,
    /// Running at an explicit positive supported speed ratio.
    PumpRunning {
        /// Original ratio to the supplied reference speed.
        speed_ratio: f64,
    },
}
/// Complete ordered identity-bound operation, without implicit stopped/running defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureOperation {
    /// Exact immutable structure identity.
    pub structure_id: String,
    /// Original passive or explicit pump operation.
    pub operating_point: StructureOperatingPoint,
}
