use std::fmt;

/// A rejected common trial preserves both accepted child engines.
#[derive(Debug)]
pub enum RunoffPoolError {
    /// Invalid immutable setup, forcing, mapping, or checkpoint.
    Input(String),
    /// Native runoff admission rejects a candidate.
    Runoff(crate::runoff::SubcatchmentError),
    /// Native pool admission rejects a candidate.
    Pools(crate::coupling::CoupledLevelPoolError),
    /// The configured runtime cannot execute an operation.
    Runtime(numr::error::Error),
    /// Composite pairing or physical accounting cannot resolve the interval.
    Numerical(String),
    /// Common duration or retry limits stop recalculation.
    Unresolved {
        /// Last attempted common duration, in seconds.
        dt_s: f64,
        /// Completed common recalculations.
        retries: usize,
        /// Original rejection requiring inspection.
        reason: String,
    },
}
impl fmt::Display for RunoffPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(s) => write!(f, "runoff-pool input: {s}"),
            Self::Runoff(e) => write!(f, "runoff-pool runoff: {e}"),
            Self::Pools(e) => write!(f, "runoff-pool storage: {e}"),
            Self::Runtime(e) => write!(f, "runoff-pool runtime: {e}"),
            Self::Numerical(s) => write!(f, "runoff-pool accounting: {s}"),
            Self::Unresolved {
                dt_s,
                retries,
                reason,
            } => write!(
                f,
                "runoff-pool interval dt={dt_s}, retries={retries}: {reason}. Inspect common duration limits, physical support, and numerical budgets"
            ),
        }
    }
}
impl std::error::Error for RunoffPoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runoff(e) => Some(e),
            Self::Pools(e) => Some(e),
            Self::Runtime(e) => Some(e),
            _ => None,
        }
    }
}
impl From<crate::runoff::SubcatchmentError> for RunoffPoolError {
    fn from(e: crate::runoff::SubcatchmentError) -> Self {
        Self::Runoff(e)
    }
}
impl From<crate::coupling::CoupledLevelPoolError> for RunoffPoolError {
    fn from(e: crate::coupling::CoupledLevelPoolError) -> Self {
        Self::Pools(e)
    }
}
impl From<numr::error::Error> for RunoffPoolError {
    fn from(e: numr::error::Error) -> Self {
        Self::Runtime(e)
    }
}
impl RunoffPoolError {
    pub(super) fn input(s: impl Into<String>) -> Self {
        Self::Input(s.into())
    }
    pub(super) fn numerical(s: impl Into<String>) -> Self {
        Self::Numerical(s.into())
    }
}
