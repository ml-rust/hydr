//! Typed closed-storage admission errors with element identities.

use std::fmt;

/// Rejection preserves every accepted pool and counter.
#[derive(Clone, Debug, PartialEq)]
pub enum LevelPoolError {
    /// Explicit definitions or inputs violate the component contract.
    InvalidInput {
        /// Ordered element index when known.
        element: Option<usize>,
        /// Offending input and corrective action.
        context: String,
    },
    /// Represented arithmetic or an independent physical gate rejects the candidate.
    Numerical {
        /// Ordered element index when known.
        element: Option<usize>,
        /// Offending numerical quantity and corrective action.
        context: String,
    },
    /// A configured numerical operation cannot execute.
    Runtime {
        /// Ordered element index when known.
        element: Option<usize>,
        /// Original runtime diagnostic.
        context: String,
    },
}
impl LevelPoolError {
    /// Ordered element identity without diagnostic-string parsing.
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
    pub(crate) fn at(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { element, .. }
            | Self::Numerical { element, .. }
            | Self::Runtime { element, .. } => *element = Some(index),
        }
        self
    }
}
impl fmt::Display for LevelPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, element, context) = match self {
            Self::InvalidInput { element, context } => ("input", element, context),
            Self::Numerical { element, context } => ("numerical", element, context),
            Self::Runtime { element, context } => ("runtime", element, context),
        };
        write!(f, "level-pool {kind}, element={element:?}: {context}")
    }
}
impl std::error::Error for LevelPoolError {}
impl From<numr::error::Error> for LevelPoolError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            element: None,
            context: error.to_string(),
        }
    }
}
impl From<solvr::interpolate::InterpolateError> for LevelPoolError {
    fn from(error: solvr::interpolate::InterpolateError) -> Self {
        Self::Runtime {
            element: None,
            context: error.to_string(),
        }
    }
}
