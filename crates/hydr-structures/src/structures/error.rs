//! Typed structure-law support and numerical rejection.

use std::fmt;

/// Explicit input-admission category, independent from numeric representation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureInputErrorKind {
    /// Incomplete or law-incompatible operation or decision-sample input.
    OperatingPoint,
    /// Invalid physical definition or explicit interval declaration.
    Definition,
    /// Unknown, duplicate, or reordered structure, endpoint, controller, or signal identity.
    Identity,
    /// Incompatible vertical reference.
    Datum,
    /// Unsupported energy-head convention.
    HeadConvention,
    /// Influential endpoint support is unknown.
    UnknownSupport,
    /// Known input lies outside declared physical or temporal support.
    UnsupportedRegime,
    /// Original heads, queries, or control samples contain nonfinite values.
    NonfiniteInput,
}

/// Rejection preserves accepted state, definitions, and earlier proposals.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureError {
    /// Physical metadata, observations, or temporal support is unsupported.
    InvalidInput {
        /// Ordered element index when known.
        element: Option<usize>,
        /// Structured admission category.
        kind: StructureInputErrorKind,
        /// Offending input and corrective action.
        context: String,
    },
    /// Numerical evaluation or timing cannot meet its declared contract.
    Numerical {
        /// Ordered element index when known.
        element: Option<usize>,
        /// Offending arithmetic and corrective action.
        context: String,
    },
    /// A configured numerical operation cannot execute.
    Runtime {
        /// Ordered element index when known.
        element: Option<usize>,
        /// Original numerical operation diagnostic.
        context: String,
    },
}
impl StructureError {
    /// Ordered element index without diagnostic-string parsing.
    pub fn element(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { element, .. }
            | Self::Numerical { element, .. }
            | Self::Runtime { element, .. } => *element,
        }
    }
    /// Physical input category. Numerical/runtime errors retain their distinct variants.
    pub fn input_kind(&self) -> Option<StructureInputErrorKind> {
        match self {
            Self::InvalidInput { kind, .. } => Some(*kind),
            _ => None,
        }
    }
    pub(crate) fn input(kind: StructureInputErrorKind, context: impl Into<String>) -> Self {
        Self::InvalidInput {
            element: None,
            kind,
            context: context.into(),
        }
    }
    /// Builds a definition-kind input error.
    pub(crate) fn definition(message: impl Into<String>) -> Self {
        Self::input(StructureInputErrorKind::Definition, message)
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
impl fmt::Display for StructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput {
                element,
                kind,
                context,
            } => write!(f, "structure input {kind:?},element={element:?}: {context}"),
            Self::Numerical { element, context } => {
                write!(f, "structure numerical,element={element:?}: {context}")
            }
            Self::Runtime { element, context } => {
                write!(f, "structure runtime,element={element:?}: {context}")
            }
        }
    }
}
impl std::error::Error for StructureError {}
impl From<numr::error::Error> for StructureError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            element: None,
            context: error.to_string(),
        }
    }
}

impl From<solvr::interpolate::InterpolateError> for StructureError {
    fn from(error: solvr::interpolate::InterpolateError) -> Self {
        Self::Runtime {
            element: None,
            context: error.to_string(),
        }
    }
}
