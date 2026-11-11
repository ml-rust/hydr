use std::fmt;

/// A rejected common trial preserves every accepted channel, pool, and water counter.
#[derive(Debug)]
pub enum ChannelPoolError {
    /// Invalid immutable setup, forcing, or checkpoint.
    Input(String),
    /// Native channel admission rejects a named reach's candidate.
    Channel {
        /// Exact setup reach identity.
        reach_id: String,
        /// Original typed channel rejection.
        source: crate::channel::ChannelError,
    },
    /// Native pool admission rejects a candidate.
    Pool(crate::storage::LevelPoolError),
    /// The caller-selected runtime returns an error.
    Runtime(numr::error::Error),
    /// Iteration, pairing, support, or physical accounting cannot resolve the interval.
    Numerical(String),
    /// Integrated inputs require fresh complete forcing for a smaller common interval.
    RecomputeInterval {
        /// Unchanged accepted start time, in seconds.
        start_s: f64,
        /// Original requested endpoint, in seconds.
        requested_end_s: f64,
        /// Smaller common endpoint requiring newly integrated inputs, in seconds.
        suggested_end_s: f64,
        /// Typed rejection's diagnostic or numerical duration restriction.
        reason: String,
    },
    /// Iteration or common interval-shortening limits stop recalculation.
    Unresolved {
        /// Last attempted common duration, in seconds.
        dt_s: f64,
        /// Complete interval recalculations after the initial attempt.
        retries: usize,
        /// Last actionable rejection.
        reason: String,
    },
}

impl ChannelPoolError {
    pub(super) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    pub(super) fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical(message.into())
    }
}

impl fmt::Display for ChannelPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => write!(f, "channel-pool input: {message}"),
            Self::Channel { reach_id, source } => {
                write!(f, "channel-pool reach={reach_id:?}: {source}")
            }
            Self::Pool(error) => write!(f, "channel-pool storage: {error}"),
            Self::Runtime(error) => write!(f, "channel-pool runtime: {error}"),
            Self::Numerical(message) => write!(f, "channel-pool numerical: {message}"),
            Self::RecomputeInterval {
                start_s,
                requested_end_s,
                suggested_end_s,
                reason,
            } => write!(
                f,
                "channel-pool integrated interval [{start_s},{requested_end_s}] requires endpoint {suggested_end_s}: {reason}. Recompute every input for the suggested span"
            ),
            Self::Unresolved {
                dt_s,
                retries,
                reason,
            } => write!(
                f,
                "channel-pool interval dt={dt_s}, retries={retries}: {reason}. Inspect physical support, accuracy budgets, and interval limits"
            ),
        }
    }
}

impl std::error::Error for ChannelPoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Channel { source, .. } => Some(source),
            Self::Pool(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<crate::storage::LevelPoolError> for ChannelPoolError {
    fn from(error: crate::storage::LevelPoolError) -> Self {
        Self::Pool(error)
    }
}

impl From<numr::error::Error> for ChannelPoolError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}

impl From<crate::numerics::water_transfers::WaterTransferError> for ChannelPoolError {
    fn from(error: crate::numerics::water_transfers::WaterTransferError) -> Self {
        use crate::numerics::water_transfers::WaterTransferError;
        match error {
            WaterTransferError::Input(message) => Self::Input(message),
            WaterTransferError::Numerical(message) => Self::Numerical(message),
            WaterTransferError::Runtime(error) => Self::Runtime(error),
        }
    }
}
