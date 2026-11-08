use crate::model::ModelError;
use std::fmt;

/// An input or numerical error leaves the last accepted state intact.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EngineError {
    /// Domain or state input violates the model contract.
    Model(ModelError),
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
            "numr CPU execution",
            format!("{value}: inspect tensor inputs and CPU runtime support"),
        )
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => error.fmt(f),
            Self::Rejected { operation, reason } => write!(f, "{operation} rejected: {reason}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Rejected { .. } => None,
        }
    }
}
