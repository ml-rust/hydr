//! Error types for the 3-way surface-channel-drainage coordinator.

use crate::coupling::{SurfaceChannelError, SurfaceDrainageError};
use crate::drainage::DrainageError;
use std::fmt;

/// Errors returned by [`super::SurfaceChannelDrainageBatch`] admission, trials, and checkpoints.
#[derive(Debug)]
pub enum SurfaceChannelDrainageError {
    /// Invalid setup, forcing, inlet geometry, datum mismatch, or duplicate parameter authority.
    Input(String),
    /// Rejection from the nested 2D surface / 1D channel coordinator.
    SurfaceChannel(SurfaceChannelError),
    /// Rejection from the nested pinned EPA SWMM v5.2.4 piped drainage engine.
    Drainage(DrainageError),
    /// Rejection from shared surface-drainage inlet or authority validation.
    SurfaceDrainage(SurfaceDrainageError),
    /// Outer 3-way fixed-point iteration did not converge within configured retries.
    Unresolved(String),
    /// Numerical or internal state inconsistency.
    Numerical(String),
}

impl SurfaceChannelDrainageError {
    pub(super) fn input(msg: impl Into<String>) -> Self {
        Self::Input(msg.into())
    }

    pub(super) fn unresolved(msg: impl Into<String>) -> Self {
        Self::Unresolved(msg.into())
    }
}

impl From<SurfaceChannelError> for SurfaceChannelDrainageError {
    fn from(err: SurfaceChannelError) -> Self {
        Self::SurfaceChannel(err)
    }
}

impl From<DrainageError> for SurfaceChannelDrainageError {
    fn from(err: DrainageError) -> Self {
        Self::Drainage(err)
    }
}

impl From<SurfaceDrainageError> for SurfaceChannelDrainageError {
    fn from(err: SurfaceDrainageError) -> Self {
        Self::SurfaceDrainage(err)
    }
}

impl fmt::Display for SurfaceChannelDrainageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(msg) => write!(f, "surface-channel-drainage input error: {msg}"),
            Self::SurfaceChannel(err) => write!(f, "surface-channel error: {err}"),
            Self::Drainage(err) => write!(f, "drainage error: {err}"),
            Self::SurfaceDrainage(err) => write!(f, "surface-drainage error: {err}"),
            Self::Unresolved(msg) => write!(f, "surface-channel-drainage unresolved: {msg}"),
            Self::Numerical(msg) => write!(f, "surface-channel-drainage numerical error: {msg}"),
        }
    }
}

impl std::error::Error for SurfaceChannelDrainageError {}
