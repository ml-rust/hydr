use std::fmt;

/// Invalid model input.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelError {
    /// An input violates the model contract.
    InvalidInput {
        /// Input field requiring correction.
        field: &'static str,
        /// Row-major cell or component index, when applicable.
        index: Option<usize>,
        /// Invalid value and required correction.
        reason: String,
    },
}

impl ModelError {
    pub(crate) fn invalid(
        field: &'static str,
        index: Option<usize>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidInput {
            field,
            index,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput {
                field,
                index,
                reason,
            } => {
                write!(f, "invalid {field}")?;
                if let Some(index) = index {
                    write!(f, "[{index}]")?;
                }
                write!(f, ": {reason}")
            }
        }
    }
}

impl std::error::Error for ModelError {}
