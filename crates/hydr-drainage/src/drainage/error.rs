//! Error types for the SWMM drainage wrapper.

use std::fmt;

/// Error returned by SWMM drainage admission, execution, or snapshot import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrainageError {
    /// Caller-supplied definition, coupling input, or snapshot violated an admission rule.
    Input(String),
    /// The pinned SWMM worker binary is missing or cannot start on this host.
    BackendUnavailable(String),
    /// The SWMM worker process failed, closed unexpectedly, or returned an error.
    Process(String),
    /// The SWMM worker reported a non-finite value or numerical step error.
    Numerical(String),
}

impl DrainageError {
    /// Creates an admission or input error.
    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    /// Creates a backend-unavailable error.
    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self::BackendUnavailable(message.into())
    }

    /// Creates a worker-process error.
    pub(crate) fn process(message: impl Into<String>) -> Self {
        Self::Process(message.into())
    }

    /// Creates a numerical error.
    pub(crate) fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical(message.into())
    }
}

impl fmt::Display for DrainageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(msg) => write!(f, "drainage input error: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "drainage backend unavailable: {msg}"),
            Self::Process(msg) => write!(f, "drainage worker process error: {msg}"),
            Self::Numerical(msg) => write!(f, "drainage numerical error: {msg}"),
        }
    }
}

impl std::error::Error for DrainageError {}
