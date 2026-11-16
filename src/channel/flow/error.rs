use std::fmt;

/// A rejected channel calculation preserves the complete accepted state.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelError {
    /// Supplied geometry, state, or forcing violates the input contract.
    InvalidInput {
        /// Ordered cell index when known.
        cell: Option<usize>,
        /// Offending input and corrective action.
        context: String,
    },
    /// Represented arithmetic or a physical acceptance check rejects the trial.
    Numerical {
        /// Ordered cell index when known.
        cell: Option<usize>,
        /// Numerical quantity and corrective action.
        context: String,
    },
    /// Water at or below the certified cap volume recovers an area above the cap area.
    ///
    /// The area selection radius exceeds the cap deficit. The area is never clipped.
    CapUnresolved {
        /// Ordered cell index.
        cell: usize,
        /// Volume parts, cap, recovered area, and corrective action.
        context: String,
    },
    /// The selected runtime cannot execute a required operation.
    Runtime {
        /// Ordered cell index when known.
        cell: Option<usize>,
        /// Original runtime diagnostic.
        context: String,
    },
    /// Integrated inputs require recomputation over a smaller interval before acceptance.
    RecomputeInterval {
        /// Unchanged accepted start time in seconds.
        start_s: f64,
        /// Original requested endpoint in seconds.
        requested_end_s: f64,
        /// Smaller endpoint requiring newly integrated inputs.
        suggested_end_s: f64,
        /// Original numerical rejection or timestep restriction.
        reason: String,
    },
    /// Retry limits or represented time prevent an admissible interval.
    Unresolved {
        /// Last attempted duration in seconds.
        dt_s: f64,
        /// Number of completed recalculations.
        retries: usize,
        /// Last actionable numerical rejection.
        reason: String,
    },
}

impl ChannelError {
    /// Ordered cell identity without parsing diagnostic text.
    pub fn cell(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell,
            Self::CapUnresolved { cell, .. } => Some(*cell),
            Self::RecomputeInterval { .. } | Self::Unresolved { .. } => None,
        }
    }

    pub(super) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            cell: None,
            context: context.into(),
        }
    }

    pub(super) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            cell: None,
            context: context.into(),
        }
    }

    pub(super) fn at(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell = Some(index),
            Self::CapUnresolved { cell, .. } => *cell = index,
            Self::RecomputeInterval { .. } | Self::Unresolved { .. } => {}
        }
        self
    }
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { cell, context } => {
                write!(f, "channel input, cell={cell:?}: {context}")
            }
            Self::Numerical { cell, context } => {
                write!(f, "channel numerical, cell={cell:?}: {context}")
            }
            Self::CapUnresolved { cell, context } => {
                write!(f, "channel cap unresolved, cell={cell}: {context}")
            }
            Self::Runtime { cell, context } => {
                write!(f, "channel runtime, cell={cell:?}: {context}")
            }
            Self::RecomputeInterval {
                start_s,
                requested_end_s,
                suggested_end_s,
                reason,
            } => write!(
                f,
                "channel integrated interval [{start_s},{requested_end_s}] requires endpoint {suggested_end_s}: {reason}. Recompute every input for the suggested span"
            ),
            Self::Unresolved {
                dt_s,
                retries,
                reason,
            } => write!(
                f,
                "channel interval dt={dt_s}, retries={retries}: {reason}. Inspect physical support, accuracy budgets, and interval limits"
            ),
        }
    }
}

impl std::error::Error for ChannelError {}

impl From<numr::error::Error> for ChannelError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            cell: None,
            context: error.to_string(),
        }
    }
}

impl From<crate::solver::EngineError> for ChannelError {
    fn from(error: crate::solver::EngineError) -> Self {
        Self::numerical(error.to_string())
    }
}

impl From<crate::numerics::water_transfers::WaterTransferError> for ChannelError {
    fn from(error: crate::numerics::water_transfers::WaterTransferError) -> Self {
        use crate::numerics::water_transfers::WaterTransferError;
        match error {
            WaterTransferError::Input(context) => Self::input(context),
            WaterTransferError::Numerical(context) => Self::numerical(context),
            WaterTransferError::Runtime(error) => Self::from(error),
        }
    }
}
