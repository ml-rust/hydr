//! Soil component admission and runtime errors.

use std::fmt;

/// Rejected soil inputs or unrepresentable native arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum SoilError {
    /// Input does not satisfy the checked component contract.
    InvalidInput {
        /// Actionable input context.
        context: String,
        /// Compact soil cell identity when the error concerns one cell.
        cell: Option<usize>,
    },
    /// Native arithmetic produces an inadmissible result.
    Numerical {
        /// Actionable arithmetic context.
        context: String,
        /// Compact soil cell identity when the error concerns one cell.
        cell: Option<usize>,
    },
    /// A numr or solvr operation returns an error.
    Runtime {
        /// Original operation error.
        context: String,
        /// Compact soil cell identity when the error concerns one cell.
        cell: Option<usize>,
    },
}

impl SoilError {
    /// Compact soil cell identity, independent of diagnostic text.
    pub fn cell(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell,
        }
    }

    /// Arithmetic rejection permits a bounded retry from accepted state.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Numerical { .. })
    }

    pub(crate) fn at_cell(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { cell, .. }
            | Self::Numerical { cell, .. }
            | Self::Runtime { cell, .. } => *cell = Some(index),
        }
        self
    }

    pub(crate) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            context: context.into(),
            cell: None,
        }
    }
    pub(crate) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            context: context.into(),
            cell: None,
        }
    }
}

impl fmt::Display for SoilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { context, .. } => write!(f, "soil input: {context}"),
            Self::Numerical { context, .. } => write!(f, "soil arithmetic: {context}"),
            Self::Runtime { context, .. } => write!(f, "soil runtime: {context}"),
        }
    }
}
impl std::error::Error for SoilError {}
impl From<numr::error::Error> for SoilError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            context: error.to_string(),
            cell: None,
        }
    }
}
impl SoilError {
    /// Converts an outward-solver error into the public soil error.
    pub(crate) fn from_outward(error: crate::numerics::outward::OutwardError) -> Self {
        use crate::numerics::outward::OutwardError;
        match error {
            OutwardError::Input { context, cell } => Self::InvalidInput { context, cell },
            OutwardError::Numerical { context, cell } => Self::Numerical { context, cell },
            OutwardError::Runtime(error) => Self::from(error),
        }
    }
}
impl From<solvr::optimize::OptimizeError> for SoilError {
    fn from(error: solvr::optimize::OptimizeError) -> Self {
        Self::Runtime {
            context: error.to_string(),
            cell: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::outward::OutwardError;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    #[test]
    fn outward_errors_keep_text_kind_and_cell() -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            (
                OutwardError::Input {
                    context: "bad count".into(),
                    cell: Some(3),
                },
                "outward-rounding input: bad count",
                "soil input: bad count",
            ),
            (
                OutwardError::Numerical {
                    context: "overflow".into(),
                    cell: None,
                },
                "outward-rounding arithmetic: overflow",
                "soil arithmetic: overflow",
            ),
        ];
        for (error, neutral, soil_text) in cases {
            assert_eq!(error.to_string(), neutral);
            let (retryable, cell) = (error.is_retryable(), error.cell());
            let soil = SoilError::from_outward(error);
            assert_eq!(soil.to_string(), soil_text);
            assert_eq!(soil.is_retryable(), retryable);
            assert_eq!(soil.cell(), cell);
        }
        let Err(runtime) = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[2], &CpuDevice::new())
        else {
            return Err("mismatched tensor shape unexpectedly succeeded".into());
        };
        let context = runtime.to_string();
        let outward = OutwardError::from(runtime);
        assert_eq!(
            outward.to_string(),
            format!("outward-rounding runtime: {context}")
        );
        assert!(!outward.is_retryable());
        let soil = SoilError::from_outward(outward);
        assert_eq!(soil.to_string(), format!("soil runtime: {context}"));
        assert_eq!(
            soil,
            SoilError::Runtime {
                context,
                cell: None
            }
        );
        Ok(())
    }
}
