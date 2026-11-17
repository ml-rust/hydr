//! Typed input, arithmetic, and runtime errors with optional element support.

use std::fmt;

/// A reservoir rejection leaves every accepted tensor and ledger unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum ReservoirError {
    /// Input or immutable support violates the component contract.
    InvalidInput {
        /// Ordered element index, when identified.
        element: Option<usize>,
        /// Input and corrective action.
        context: String,
    },
    /// Numerical work, representation, or residual admission rejects the candidate.
    Numerical {
        /// Ordered element index, when identified.
        element: Option<usize>,
        /// Arithmetic and corrective action.
        context: String,
    },
    /// A foundational runtime operation cannot execute.
    Runtime {
        /// Ordered element index, when identified.
        element: Option<usize>,
        /// Original runtime context.
        context: String,
    },
}

impl ReservoirError {
    /// Returns the ordered element index identified by this error.
    pub fn element(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { element, .. }
            | Self::Numerical { element, .. }
            | Self::Runtime { element, .. } => *element,
        }
    }
    pub(crate) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            element: None,
            context: context.into(),
        }
    }
    pub(crate) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            element: None,
            context: context.into(),
        }
    }
    pub(crate) fn at_element(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { element, .. }
            | Self::Numerical { element, .. }
            | Self::Runtime { element, .. } => *element = Some(index),
        }
        self
    }
}
impl fmt::Display for ReservoirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { element, context } => {
                write!(f, "reservoir input element={element:?}: {context}")
            }
            Self::Numerical { element, context } => {
                write!(f, "reservoir arithmetic element={element:?}: {context}")
            }
            Self::Runtime { element, context } => {
                write!(f, "reservoir runtime element={element:?}: {context}")
            }
        }
    }
}
impl std::error::Error for ReservoirError {}
impl From<numr::error::Error> for ReservoirError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            element: None,
            context: error.to_string(),
        }
    }
}
impl From<solvr::optimize::OptimizeError> for ReservoirError {
    fn from(error: solvr::optimize::OptimizeError) -> Self {
        if matches!(&error, solvr::optimize::OptimizeError::NumrError(_)) {
            Self::Runtime {
                element: None,
                context: error.to_string(),
            }
        } else {
            Self::numerical(error.to_string())
        }
    }
}
