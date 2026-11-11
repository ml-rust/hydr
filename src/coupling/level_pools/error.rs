use std::fmt;

/// Rejection without partial commitment of coupled storage.
#[derive(Debug)]
pub enum CoupledLevelPoolError {
    /// Invalid immutable definition, input, or checkpoint.
    Input(String),
    /// An existing pool check rejects the candidate.
    Pool(crate::storage::LevelPoolError),
    /// An existing structure check rejects the candidate.
    Structure(crate::structures::StructureError),
    /// The selected runtime returns an error.
    Runtime(numr::error::Error),
    /// Arithmetic or physical support cannot resolve the attempted interval.
    Numerical(String),
    /// Integrated inputs require fresh forcing for the suggested smaller interval.
    RecomputeInterval {
        /// Unchanged accepted start time in seconds.
        start_s: f64,
        /// Original requested endpoint in seconds.
        requested_end_s: f64,
        /// Admitted smaller endpoint requiring newly integrated inputs.
        suggested_end_s: f64,
        /// Original numerical rejection or timestep restriction.
        reason: String,
    },
    /// Iteration or interval-shortening limits stop an unresolved trial.
    Unresolved {
        /// Last attempted interval duration in seconds.
        dt_s: f64,
        /// Number of completed interval halvings.
        retries: usize,
        /// Last actionable rejection.
        reason: String,
    },
}

impl CoupledLevelPoolError {
    pub(super) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }
    pub(super) fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical(message.into())
    }
}

impl fmt::Display for CoupledLevelPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => write!(f, "coupled pool input: {message}"),
            Self::Pool(error) => write!(f, "coupled pool storage: {error}"),
            Self::Structure(error) => write!(f, "coupled pool structure: {error}"),
            Self::Runtime(error) => write!(f, "coupled pool runtime: {error}"),
            Self::Numerical(message) => write!(f, "coupled pool arithmetic: {message}"),
            Self::RecomputeInterval {
                start_s,
                requested_end_s,
                suggested_end_s,
                reason,
            } => write!(
                f,
                "integrated inflow interval [{start_s},{requested_end_s}] requires endpoint {suggested_end_s}: {reason}. Recompute every input for the suggested span",
            ),
            Self::Unresolved {
                dt_s,
                retries,
                reason,
            } => write!(
                f,
                "coupled interval dt={dt_s}, retries={retries}: {reason}. Inspect support, accuracy budgets, and interval limits"
            ),
        }
    }
}

impl std::error::Error for CoupledLevelPoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pool(error) => Some(error),
            Self::Structure(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}
impl From<crate::storage::LevelPoolError> for CoupledLevelPoolError {
    fn from(error: crate::storage::LevelPoolError) -> Self {
        Self::Pool(error)
    }
}
impl From<crate::structures::StructureError> for CoupledLevelPoolError {
    fn from(error: crate::structures::StructureError) -> Self {
        Self::Structure(error)
    }
}
impl From<numr::error::Error> for CoupledLevelPoolError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}

impl From<crate::numerics::water_transfers::WaterTransferError> for CoupledLevelPoolError {
    fn from(error: crate::numerics::water_transfers::WaterTransferError) -> Self {
        use crate::numerics::water_transfers::WaterTransferError;
        match error {
            WaterTransferError::Input(message) => Self::Input(message),
            WaterTransferError::Numerical(message) => Self::Numerical(message),
            WaterTransferError::Runtime(error) => Self::Runtime(error),
        }
    }
}
