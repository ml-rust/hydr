//! One rejection type for shared link numerics. Each coordinator maps it into its own error.

use crate::coupling::{ChannelPoolError, CoupledLevelPoolError, SurfaceChannelError};
use crate::numerics::water_transfers::WaterTransferError;
use std::fmt;

/// A shared link step rejects its operands.
#[derive(Debug)]
pub(crate) enum LinkError {
    /// Invalid operands or topology.
    Input(String),
    /// Arithmetic or physical support cannot resolve the attempted interval.
    Numerical(String),
    /// The selected runtime returns an error.
    Runtime(numr::error::Error),
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => write!(f, "link input: {message}"),
            Self::Numerical(message) => write!(f, "link arithmetic: {message}"),
            Self::Runtime(error) => write!(f, "link runtime: {error}"),
        }
    }
}

impl std::error::Error for LinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WaterTransferError> for LinkError {
    fn from(error: WaterTransferError) -> Self {
        match error {
            WaterTransferError::Input(message) => Self::Input(message),
            WaterTransferError::Numerical(message) => Self::Numerical(message),
            WaterTransferError::Runtime(error) => Self::Runtime(error),
        }
    }
}

impl From<numr::error::Error> for LinkError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}

// Named conversions, never `From` impls: a `From` impl on a public error is public API.
impl LinkError {
    /// The surface/channel error of the same kind.
    pub(crate) fn surface_channel(self) -> SurfaceChannelError {
        match self {
            Self::Input(message) => SurfaceChannelError::Input(message),
            Self::Numerical(message) => SurfaceChannelError::Numerical(message),
            Self::Runtime(error) => SurfaceChannelError::Runtime(error),
        }
    }

    /// The coupled level pool error of the same kind.
    pub(crate) fn level_pool(self) -> CoupledLevelPoolError {
        match self {
            Self::Input(message) => CoupledLevelPoolError::Input(message),
            Self::Numerical(message) => CoupledLevelPoolError::Numerical(message),
            Self::Runtime(error) => CoupledLevelPoolError::Runtime(error),
        }
    }

    /// The channel/pool error of the same kind.
    pub(crate) fn channel_pool(self) -> ChannelPoolError {
        match self {
            Self::Input(message) => ChannelPoolError::Input(message),
            Self::Numerical(message) => ChannelPoolError::Numerical(message),
            Self::Runtime(error) => ChannelPoolError::Runtime(error),
        }
    }
}
