//! Typed component errors with compact-to-physical support mapping.

use super::Subarea;
use crate::runoff::ReservoirError;
use crate::soil::SoilError;
use std::fmt;

/// An input or numerical rejection preserves the complete accepted catchment batch.
#[derive(Clone, Debug, PartialEq)]
pub enum SubcatchmentError {
    /// Immutable support or supplied interval input is invalid.
    InvalidInput {
        /// Ordered catchment identity when known.
        catchment: Option<usize>,
        /// Physical subarea when known.
        subarea: Option<Subarea>,
        /// Actionable input context.
        context: String,
    },
    /// Numerical representation, source, transfer, or residual admission rejects the trial.
    Numerical {
        /// Ordered catchment identity when known.
        catchment: Option<usize>,
        /// Physical subarea when known.
        subarea: Option<Subarea>,
        /// Actionable arithmetic context.
        context: String,
    },
    /// A configured numerical runtime operation cannot execute.
    Runtime {
        /// Ordered catchment identity when known.
        catchment: Option<usize>,
        /// Physical subarea when known.
        subarea: Option<Subarea>,
        /// Original runtime context.
        context: String,
    },
}

impl SubcatchmentError {
    /// Ordered catchment index, without parsing diagnostic text.
    pub fn catchment(&self) -> Option<usize> {
        match self {
            Self::InvalidInput { catchment, .. }
            | Self::Numerical { catchment, .. }
            | Self::Runtime { catchment, .. } => *catchment,
        }
    }
    /// Physical subarea, without parsing diagnostic text.
    pub fn subarea(&self) -> Option<Subarea> {
        match self {
            Self::InvalidInput { subarea, .. }
            | Self::Numerical { subarea, .. }
            | Self::Runtime { subarea, .. } => *subarea,
        }
    }
    pub(crate) fn input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            catchment: None,
            subarea: None,
            context: context.into(),
        }
    }
    pub(crate) fn numerical(context: impl Into<String>) -> Self {
        Self::Numerical {
            catchment: None,
            subarea: None,
            context: context.into(),
        }
    }
    pub(crate) fn at_catchment(mut self, index: usize) -> Self {
        match &mut self {
            Self::InvalidInput { catchment, .. }
            | Self::Numerical { catchment, .. }
            | Self::Runtime { catchment, .. } => *catchment = Some(index),
        }
        self
    }
    pub(crate) fn at_subarea(mut self, area: Subarea) -> Self {
        match &mut self {
            Self::InvalidInput { subarea, .. }
            | Self::Numerical { subarea, .. }
            | Self::Runtime { subarea, .. } => *subarea = Some(area),
        }
        self
    }
    pub(crate) fn at(self, index: usize, area: Option<Subarea>) -> Self {
        let error = self.at_catchment(index);
        if let Some(area) = area {
            error.at_subarea(area)
        } else {
            error
        }
    }
}
impl fmt::Display for SubcatchmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, catchment, subarea, context) = match self {
            Self::InvalidInput {
                catchment,
                subarea,
                context,
            } => ("input", catchment, subarea, context),
            Self::Numerical {
                catchment,
                subarea,
                context,
            } => ("arithmetic", catchment, subarea, context),
            Self::Runtime {
                catchment,
                subarea,
                context,
            } => ("runtime", catchment, subarea, context),
        };
        write!(
            f,
            "subcatchment {kind} catchment={catchment:?} subarea={subarea:?}: {context}"
        )
    }
}
impl std::error::Error for SubcatchmentError {}
impl From<numr::error::Error> for SubcatchmentError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime {
            catchment: None,
            subarea: None,
            context: error.to_string(),
        }
    }
}
impl From<ReservoirError> for SubcatchmentError {
    fn from(error: ReservoirError) -> Self {
        match error {
            ReservoirError::InvalidInput { context, .. } => Self::input(context),
            ReservoirError::Numerical { context, .. } => Self::numerical(context),
            ReservoirError::Runtime { context, .. } => Self::Runtime {
                catchment: None,
                subarea: None,
                context,
            },
        }
    }
}
impl SubcatchmentError {
    /// Outward arithmetic errors map through the soil error, keeping its text and kind.
    pub(crate) fn from_outward(error: crate::numerics::outward::OutwardError) -> Self {
        SoilError::from_outward(error).into()
    }
}
impl From<SoilError> for SubcatchmentError {
    fn from(error: SoilError) -> Self {
        match error {
            SoilError::InvalidInput { context, .. } => Self::input(context),
            SoilError::Numerical { context, .. } => Self::numerical(context),
            SoilError::Runtime { context, .. } => Self::Runtime {
                catchment: None,
                subarea: None,
                context,
            },
        }
    }
}
