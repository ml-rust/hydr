use super::BoundaryFace;
use crate::model::ModelError;
use std::fmt;

/// An input or numerical error leaves the last accepted state intact.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EngineError {
    /// Domain or state input violates the model contract.
    Model(ModelError),
    /// A soil error mapped from compact batch support to a global hydraulic cell.
    Soil {
        /// Global cell identity, when the error identifies one compact lane.
        cell: Option<usize>,
        /// Soil admission or runtime context.
        reason: String,
        /// Numerical soil admission errors permit bounded timestep retries.
        retryable: bool,
    },
    /// A timestep-dependent local source or adjustment budget rejects a trial.
    LandSource {
        /// Global cell requiring corrected input or a representable interval.
        cell: Option<usize>,
        /// Source admission context.
        reason: String,
    },
    /// A physical boundary trace cannot admit the requested hydraulic regime.
    Boundary {
        /// Exterior face requiring compatible physical data.
        face: BoundaryFace,
        /// Hydraulic state, request, and corrective action.
        reason: String,
    },
    /// An execution operation cannot satisfy its numerical contract.
    Rejected {
        /// Operation or input responsible for rejection.
        operation: &'static str,
        /// Specific input, numerical context, and corrective action.
        reason: String,
    },
}

impl EngineError {
    pub(crate) fn rejected(operation: &'static str, reason: impl Into<String>) -> Self {
        Self::Rejected {
            operation,
            reason: reason.into(),
        }
    }
}

impl From<ModelError> for EngineError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<numr::error::Error> for EngineError {
    fn from(value: numr::error::Error) -> Self {
        Self::rejected(
            "numr execution",
            format!("{value}: inspect tensor inputs and selected runtime support"),
        )
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => error.fmt(f),
            Self::Soil { cell, reason, .. } => write!(f, "soil cell={cell:?} rejected: {reason}"),
            Self::LandSource { cell, reason } => {
                write!(f, "land source cell={cell:?} rejected: {reason}")
            }
            Self::Boundary { face, reason } => write!(f, "boundary {face:?} rejected: {reason}"),
            Self::Rejected { operation, reason } => write!(f, "{operation} rejected: {reason}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Rejected { .. }
            | Self::Boundary { .. }
            | Self::Soil { .. }
            | Self::LandSource { .. } => None,
        }
    }
}
