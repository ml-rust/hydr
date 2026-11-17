use std::fmt;

/// Rejected operand or unrepresentable outward arithmetic.
///
/// Display names the outward-rounding arithmetic. Soil callers convert into `SoilError` for soil text.
#[derive(Debug)]
pub(crate) enum OutwardError {
    /// Input does not satisfy the reduction contract.
    Input {
        /// Actionable input context.
        context: String,
        /// Lane identity when the error concerns one lane.
        cell: Option<usize>,
    },
    /// Arithmetic produces an inadmissible result. A bounded retry is permitted.
    Numerical {
        /// Actionable arithmetic context.
        context: String,
        /// Lane identity when the error concerns one lane.
        cell: Option<usize>,
    },
    /// A numr operation returns an error.
    Runtime(numr::error::Error),
}

impl OutwardError {
    /// Lane identity, independent of diagnostic text.
    pub(crate) fn cell(&self) -> Option<usize> {
        match self {
            Self::Input { cell, .. } | Self::Numerical { cell, .. } => *cell,
            Self::Runtime(_) => None,
        }
    }

    /// Arithmetic rejection permits a bounded retry from accepted state.
    pub(crate) fn is_retryable(&self) -> bool {
        matches!(self, Self::Numerical { .. })
    }

    pub(super) fn at_cell(mut self, index: usize) -> Self {
        if let Self::Input { cell, .. } | Self::Numerical { cell, .. } = &mut self {
            *cell = Some(index);
        }
        self
    }

    pub(super) fn input(context: impl Into<String>) -> Self {
        Self::Input {
            context: context.into(),
            cell: None,
        }
    }

    pub(super) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            context: context.into(),
            cell: None,
        }
    }
}

impl fmt::Display for OutwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { context, .. } => write!(f, "outward-rounding input: {context}"),
            Self::Numerical { context, .. } => write!(f, "outward-rounding arithmetic: {context}"),
            Self::Runtime(error) => write!(f, "outward-rounding runtime: {error}"),
        }
    }
}

impl std::error::Error for OutwardError {}

impl From<numr::error::Error> for OutwardError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}

pub(super) fn finite(value: f64, quantity: &str) -> Result<f64, OutwardError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OutwardError::numerical(format!(
            "{quantity}={value}: require finite source-admission arithmetic. Rescale inputs"
        )))
    }
}

pub(super) fn nonnegative(value: f64, quantity: &str) -> Result<f64, OutwardError> {
    finite(value, quantity)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(OutwardError::numerical(format!(
            "{quantity}={value}: require nonnegative source-admission arithmetic"
        )))
    }
}
