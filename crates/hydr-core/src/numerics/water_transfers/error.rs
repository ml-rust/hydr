use std::fmt;

#[derive(Debug)]
pub enum WaterTransferError {
    Input(String),
    Numerical(String),
    Runtime(numr::error::Error),
}

impl WaterTransferError {
    pub(super) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }

    pub(super) fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical(message.into())
    }
}

impl fmt::Display for WaterTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => write!(f, "transfer input: {message}"),
            Self::Numerical(message) => write!(f, "transfer arithmetic: {message}"),
            Self::Runtime(error) => write!(f, "transfer runtime: {error}"),
        }
    }
}

impl std::error::Error for WaterTransferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<numr::error::Error> for WaterTransferError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}
