//! Error types for conservative 2D surface / SWMM drainage coupling.

use crate::drainage::DrainageError;
use hydr_core::contract::EngineError;
use std::fmt;

/// Error returned by [`super::SurfaceDrainageBatch`] admission, stepping, or restart.
#[derive(Clone, Debug)]
pub enum SurfaceDrainageError {
    /// Caller-supplied setup, inlet geometry, authority claim, or snapshot violated admission.
    Input(String),
    /// Coupled fixed-point iteration did not converge within retry and minimum-step limits.
    Unresolved(String),
    /// A surface solver call failed.
    Surface(String),
    /// A SWMM drainage worker call failed.
    Drainage(DrainageError),
}

impl SurfaceDrainageError {
    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    pub(crate) fn unresolved(message: impl Into<String>) -> Self {
        Self::Unresolved(message.into())
    }

    pub(crate) fn surface(err: EngineError) -> Self {
        Self::Surface(err.to_string())
    }
}

impl From<DrainageError> for SurfaceDrainageError {
    fn from(err: DrainageError) -> Self {
        Self::Drainage(err)
    }
}

impl fmt::Display for SurfaceDrainageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(msg) => write!(f, "surface-drainage input error: {msg}"),
            Self::Unresolved(msg) => write!(f, "surface-drainage unresolved step: {msg}"),
            Self::Surface(msg) => write!(f, "surface-drainage surface error: {msg}"),
            Self::Drainage(err) => write!(f, "surface-drainage drainage error: {err}"),
        }
    }
}

impl std::error::Error for SurfaceDrainageError {}
