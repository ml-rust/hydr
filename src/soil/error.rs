//! Soil component admission and runtime errors.

use std::fmt;

/// Rejected soil inputs or unrepresentable native arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum SoilError {
    /// Input does not satisfy the checked component contract.
    InvalidInput {
        /// Actionable input context.
        context: String,
    },
    /// Native arithmetic produces an inadmissible result.
    Numerical {
        /// Actionable arithmetic context.
        context: String,
    },
    /// A numr or solvr operation returns an error.
    Runtime {
        /// Original operation error.
        context: String,
    },
}

impl SoilError {
    pub(crate) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            context: context.into(),
        }
    }
    pub(crate) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            context: context.into(),
        }
    }
}

impl fmt::Display for SoilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { context } => write!(f, "soil input: {context}"),
            Self::Numerical { context } => write!(f, "soil arithmetic: {context}"),
            Self::Runtime { context } => write!(f, "soil runtime: {context}"),
        }
    }
}
impl std::error::Error for SoilError {}
impl From<numr::error::Error> for SoilError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            context: error.to_string(),
        }
    }
}
impl From<solvr::optimize::OptimizeError> for SoilError {
    fn from(error: solvr::optimize::OptimizeError) -> Self {
        Self::Runtime {
            context: error.to_string(),
        }
    }
}
