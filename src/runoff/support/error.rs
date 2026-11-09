//! Structured ownership errors preserving concrete numerical consumer causes.

use crate::runoff::SubcatchmentError;
use crate::solver::EngineError;
use std::fmt;

/// Explicit ownership admission category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportErrorKind {
    /// Invalid complete/disjoint partition assertion or atom metadata.
    Partition,
    /// Incompatible actual target area or unresolved area arithmetic.
    Area,
    /// Incomplete, conflicting, or duplicate primary ownership.
    Ownership,
    /// Invalid whole-source selection.
    Replacement,
    /// Invalid or multiply injected additive declaration.
    Inclusion,
    /// Missing, duplicate, or mismatched exact interval coverage.
    Coverage,
    /// Changed or unknown immutable identity.
    Identity,
    /// Incorrect concrete numerical destination.
    Destination,
    /// Unsupported physical source composition.
    Composition,
}

/// Checked ownership error or the original typed concrete-consumer error.
#[derive(Debug)]
pub enum SupportError {
    /// Ownership admission rejected before concrete numerical advancement.
    Admission {
        /// Structured category.
        kind: SupportErrorKind,
        /// Offending identities, values, and corrective action.
        context: String,
    },
    /// Original forcing schedule error, preserved without string conversion.
    Surface(EngineError),
    /// Original subcatchment trial error, preserved without string conversion.
    Subcatchment(SubcatchmentError),
}
impl SupportError {
    /// Read the ownership category. Concrete consumer errors retain their native types.
    pub fn kind(&self) -> Option<SupportErrorKind> {
        match self {
            Self::Admission { kind, .. } => Some(*kind),
            _ => None,
        }
    }
    pub(super) fn new(kind: SupportErrorKind, context: impl Into<String>) -> Self {
        Self::Admission {
            kind,
            context: context.into(),
        }
    }
    pub(super) fn identity(context: impl Into<String>) -> Self {
        Self::new(SupportErrorKind::Identity, context)
    }
}
impl fmt::Display for SupportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission { kind, context } => write!(f, "support {kind:?}: {context}"),
            Self::Surface(e) => write!(f, "support surface consumer: {e}"),
            Self::Subcatchment(e) => write!(f, "support subcatchment consumer: {e}"),
        }
    }
}
impl std::error::Error for SupportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Surface(e) => Some(e),
            Self::Subcatchment(e) => Some(e),
            Self::Admission { .. } => None,
        }
    }
}
impl From<EngineError> for SupportError {
    fn from(e: EngineError) -> Self {
        Self::Surface(e)
    }
}
impl From<SubcatchmentError> for SupportError {
    fn from(e: SubcatchmentError) -> Self {
        Self::Subcatchment(e)
    }
}
