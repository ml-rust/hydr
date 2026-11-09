//! Soil component admission and runtime errors.

use std::fmt;

/// Rejected soil inputs or unrepresentable native arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum SoilError {
    /// Input does not satisfy the checked component contract.
    InvalidInput {
        /// Actionable input context.
        context: String,
        /// Compact soil cell identity when the error concerns one cell.
        cell: Option<usize>,
    },
    /// Native arithmetic produces an inadmissible result.
    Numerical {
        /// Actionable arithmetic context.
        context: String,
        /// Compact soil cell identity when the error concerns one cell.
        cell: Option<usize>,
    },
    /// A numr or solvr operation returns an error.
    Runtime {
        /// Original operation error.
        context: String,
        /// Compact soil cell identity when the error concerns one cell.
        cell: Option<usize>,
    },
}

impl SoilError {
    /// Compact soil cell identity, independent of diagnostic text.
    pub fn cell(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell,
        }
    }

    /// Arithmetic rejection permits a bounded retry from accepted state.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Numerical { .. })
    }

    pub(crate) fn at_cell(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell = Some(index),
        }
        self
    }

    pub(crate) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            context: context.into(),
            cell: None,
        }
    }
    pub(crate) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            context: context.into(),
            cell: None,
        }
    }
}

impl fmt::Display for SoilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { context, .. } => write!(f, "soil input: {context}"),
            Self::Numerical { context, .. } => write!(f, "soil arithmetic: {context}"),
            Self::Runtime { context, .. } => write!(f, "soil runtime: {context}"),
        }
    }
}
impl std::error::Error for SoilError {}
impl From<numr::error::Error> for SoilError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            context: error.to_string(),
            cell: None,
        }
    }
}
impl From<solvr::optimize::OptimizeError> for SoilError {
    fn from(error: solvr::optimize::OptimizeError) -> Self {
        Self::Runtime {
            context: error.to_string(),
            cell: None,
        }
    }
}
