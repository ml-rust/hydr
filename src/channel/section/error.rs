use std::fmt;

/// A geometry or runtime rejection leaves every immutable section unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum CrossSectionError {
    /// A definition or query violates the declared input contract.
    InvalidInput {
        /// Ordered section index when known.
        section: Option<usize>,
        /// Exact offending input and corrective action.
        context: String,
    },
    /// Represented arithmetic cannot resolve the requested geometry.
    Numerical {
        /// Ordered section index when known.
        section: Option<usize>,
        /// Rejected quantity and corrective action.
        context: String,
    },
    /// An operation cannot execute on the configured runtime.
    Runtime {
        /// Ordered section index when known.
        section: Option<usize>,
        /// Original runtime diagnostic.
        context: String,
    },
    /// A requested stage exceeds the explicitly supported bank limit.
    UnsupportedStage {
        /// Ordered section index.
        section: usize,
        /// Original requested stage, in metres.
        stage_m: f64,
        /// Declared maximum stage, in metres.
        maximum_stage_m: f64,
    },
}

impl CrossSectionError {
    /// Ordered section index without parsing diagnostic text.
    pub fn section(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { section, .. }
            | Self::Numerical { section, .. }
            | Self::Runtime { section, .. } => *section,
            Self::UnsupportedStage { section, .. } => Some(*section),
        }
    }
    pub(super) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            section: None,
            context: context.into(),
        }
    }
    pub(super) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            section: None,
            context: context.into(),
        }
    }
    pub(super) fn at(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { section, .. }
            | Self::Numerical { section, .. }
            | Self::Runtime { section, .. } => *section = Some(index),
            Self::UnsupportedStage { section, .. } => *section = index,
        }
        self
    }
}

impl fmt::Display for CrossSectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { section, context } => {
                write!(f, "cross-section input, section={section:?}: {context}")
            }
            Self::Numerical { section, context } => {
                write!(f, "cross-section numerical, section={section:?}: {context}")
            }
            Self::Runtime { section, context } => {
                write!(f, "cross-section runtime, section={section:?}: {context}")
            }
            Self::UnsupportedStage {
                section,
                stage_m,
                maximum_stage_m,
            } => write!(
                f,
                "cross-section stage={stage_m}, section={section}, maximum={maximum_stage_m}: supply an admitted stage or extend the surveyed profile explicitly"
            ),
        }
    }
}
impl std::error::Error for CrossSectionError {}
impl From<numr::error::Error> for CrossSectionError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            section: None,
            context: error.to_string(),
        }
    }
}
