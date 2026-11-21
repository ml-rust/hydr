use crate::channel::ChannelError;
use crate::runoff::SubcatchmentError;
use crate::solver::EngineError;
use std::fmt;

/// A rejected surface/channel construction or common trial preserves every accepted child state.
#[derive(Debug)]
pub enum SurfaceChannelError {
    /// Invalid immutable setup, initial state, forcing, or checkpoint.
    Input(String),
    /// The surface engine rejects its definition, state, or a candidate.
    Surface(EngineError),
    /// Native channel admission rejects a named reach's definition, state, or candidate.
    Channel {
        /// Exact setup reach identity.
        reach_id: String,
        /// Original typed channel rejection.
        source: ChannelError,
    },
    /// The coordinator-owned runoff child rejects its definitions, state, or a candidate.
    Runoff(SubcatchmentError),
    /// The caller-selected runtime returns an error.
    Runtime(numr::error::Error),
    /// Iteration, pairing, or physical accounting cannot resolve the interval.
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

impl SurfaceChannelError {
    pub(super) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    /// A shorter common interval from the accepted state can resolve this rejection.
    ///
    /// Retryable: numerical rejections, interval recomputation, surface land-source, retryable
    /// soil, and boundary-trace rejections, channel numerical, unresolved, and recompute
    /// rejections, and runoff numerical rejections. Input, runtime, and every other child
    /// rejection are final.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Numerical(_) | Self::RecomputeInterval { .. } => true,
            Self::Surface(error) => matches!(
                error,
                EngineError::LandSource { .. }
                    | EngineError::Boundary { .. }
                    | EngineError::Soil {
                        retryable: true,
                        ..
                    }
            ),
            Self::Channel { source, .. } => matches!(
                source,
                ChannelError::Numerical { .. }
                    | ChannelError::Unresolved { .. }
                    | ChannelError::RecomputeInterval { .. }
            ),
            Self::Runoff(error) => matches!(error, SubcatchmentError::Numerical { .. }),
            Self::Input(_) | Self::Runtime(_) | Self::Unresolved { .. } => false,
        }
    }
}

impl fmt::Display for SurfaceChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => write!(f, "surface-channel input: {message}"),
            Self::Surface(error) => write!(f, "surface-channel surface: {error}"),
            Self::Channel { reach_id, source } => {
                write!(f, "surface-channel reach={reach_id:?}: {source}")
            }
            Self::Runoff(error) => write!(f, "surface-channel runoff: {error}"),
            Self::Runtime(error) => write!(f, "surface-channel runtime: {error}"),
            Self::Numerical(message) => write!(f, "surface-channel numerical: {message}"),
            Self::RecomputeInterval {
                start_s,
                requested_end_s,
                suggested_end_s,
                reason,
            } => write!(
                f,
                "surface-channel integrated interval [{start_s},{requested_end_s}] requires endpoint {suggested_end_s}: {reason}. Recompute every input for the suggested span"
            ),
            Self::Unresolved {
                dt_s,
                retries,
                reason,
            } => write!(
                f,
                "surface-channel interval dt={dt_s}, retries={retries}: {reason}. Inspect link support, accuracy budgets, and interval limits"
            ),
        }
    }
}

impl std::error::Error for SurfaceChannelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Surface(error) => Some(error),
            Self::Channel { source, .. } => Some(source),
            Self::Runoff(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<EngineError> for SurfaceChannelError {
    fn from(error: EngineError) -> Self {
        Self::Surface(error)
    }
}

impl From<numr::error::Error> for SurfaceChannelError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}
