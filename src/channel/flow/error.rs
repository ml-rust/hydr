use std::fmt;

/// A rejected channel calculation preserves the complete accepted state.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelError {
    /// Supplied geometry, state, or forcing violates the input contract.
    InvalidInput {
        /// Ordered cell index when known.
        cell: Option<usize>,
        /// Offending input and corrective action.
        context: String,
    },
    /// Represented arithmetic or a physical acceptance check rejects the trial.
    Numerical {
        /// Ordered cell index when known.
        cell: Option<usize>,
        /// Numerical quantity and corrective action.
        context: String,
    },
    /// The selected runtime cannot execute a required operation.
    Runtime {
        /// Ordered cell index when known.
        cell: Option<usize>,
        /// Original runtime diagnostic.
        context: String,
    },
}

impl ChannelError {
    /// Ordered cell identity without parsing diagnostic text.
    pub fn cell(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell,
        }
    }

    pub(super) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            cell: None,
            context: context.into(),
        }
    }

    pub(super) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            cell: None,
            context: context.into(),
        }
    }

    pub(super) fn at(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell = Some(index),
        }
        self
    }
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, cell, context) = match self {
            Self::InvalidInput { cell, context } => ("input", cell, context),
            Self::Numerical { cell, context } => ("numerical", cell, context),
            Self::Runtime { cell, context } => ("runtime", cell, context),
        };
        write!(f, "channel {kind}, cell={cell:?}: {context}")
    }
}

impl std::error::Error for ChannelError {}

impl From<numr::error::Error> for ChannelError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            cell: None,
            context: error.to_string(),
        }
    }
}

impl From<crate::solver::EngineError> for ChannelError {
    fn from(error: crate::solver::EngineError) -> Self {
        Self::numerical(error.to_string())
    }
}
