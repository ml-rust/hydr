use super::work::PrismaticRiemannWork;
use crate::channel::flow::ChannelError;
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::{ArithmeticError, IntervalContext, with_interval_arithmetic};
use std::fmt;

/// The quantity preventing a certified surveyed Riemann result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrismaticRiemannRejection {
    /// Required water exceeds surveyed support. Check input units or extend the surveyed profile.
    CapExceeded,
    /// Certified bounds cannot resolve cap membership. Inspect survey conditioning.
    CapUnresolved,
    /// Certified geometry cannot resolve the requested state. Inspect profile conditioning and depth allowances.
    GeometryUnresolved,
    /// Integration bounds remain unresolved within the panel limits. Inspect conditioning and quadrature limits.
    QuadratureUnresolved,
    /// Root endpoint bounds cannot establish existence. Inspect the retained solver status and bracket quantities.
    RootBracketUnresolved,
    /// Wave bounds cannot certify an interface trace. Inspect the reported classification or trace quantity.
    SamplingUnresolved,
    /// Final flux enclosure exceeds its width allowance. Inspect the reported flux bounds and allowances.
    FluxBudgetExceeded,
    /// Execution reaches a work ceiling. Inspect counters before increasing limits or splitting the batch.
    WorkLimit,
}

/// A rejected call discards every partial numerical result.
#[derive(Clone, Debug, PartialEq)]
pub enum PrismaticRiemannError {
    /// Input, runtime, or arithmetic error with no channel-cell identity.
    Channel(ChannelError),
    /// Physical or numerical rejection specific to this method, with executed work and diagnostic context.
    Rejected {
        /// Quantity preventing a certified result.
        reason: PrismaticRiemannRejection,
        /// Original input row when known. This index never identifies a channel cell.
        query: Option<usize>,
        /// Original section-owner index when known.
        section_owner: Option<usize>,
        /// Actual work executed before rejection.
        work: Box<PrismaticRiemannWork>,
        /// Original solver status or rejected quantity, with the next action.
        context: Box<str>,
    },
}

impl fmt::Display for PrismaticRiemannError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Channel(error) => error.fmt(f),
            Self::Rejected {
                reason,
                query,
                section_owner,
                context,
                ..
            } => write!(
                f,
                "surveyed Riemann {reason:?}, query={query:?}, section_owner={section_owner:?}: {context}"
            ),
        }
    }
}
impl std::error::Error for PrismaticRiemannError {}

pub(crate) type Arithmetic<'a, R> = IntervalContext<'a, R, <R as Runtime>::Client>;
pub(crate) type HydraulicResult<T> = Result<T, HydraulicError>;

#[derive(Debug)]
pub(crate) enum HydraulicError {
    Arithmetic(ArithmeticError),
    Runtime(numr::error::Error),
    Geometry(CrossSectionError),
    Numerical {
        context: String,
    },
    Rejected {
        reason: PrismaticRiemannRejection,
        query: Option<usize>,
        section_owner: Option<usize>,
        context: String,
    },
}
impl From<ArithmeticError> for HydraulicError {
    fn from(error: ArithmeticError) -> Self {
        Self::Arithmetic(error)
    }
}
impl From<numr::error::Error> for HydraulicError {
    fn from(error: numr::error::Error) -> Self {
        Self::Runtime(error)
    }
}
impl From<CrossSectionError> for HydraulicError {
    fn from(error: CrossSectionError) -> Self {
        Self::Geometry(error)
    }
}

pub(crate) fn checked<R, T>(
    client: &R::Client,
    stage: &'static str,
    build: impl FnOnce(&mut Arithmetic<'_, R>) -> HydraulicResult<T>,
) -> HydraulicResult<T>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let mut rejection = None;
    let result =
        with_interval_arithmetic::<R, _, _, _>(client, DType::F64, stage, |a| match build(a) {
            Ok(value) => Ok(Some(value)),
            Err(HydraulicError::Arithmetic(error)) => Err(error),
            Err(error) => {
                rejection = Some(error);
                Ok(None)
            }
        })?;
    if let Some(error) = rejection {
        return Err(error);
    }
    result.ok_or_else(|| HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::GeometryUnresolved,
        query: None,
        section_owner: None,
        context: "checked arithmetic returned no result. Rebuild the complete call".into(),
    })
}

pub(crate) fn into_public(
    error: HydraulicError,
    work: PrismaticRiemannWork,
) -> PrismaticRiemannError {
    use PrismaticRiemannError::{Channel, Rejected};
    match error {
        HydraulicError::Rejected {
            reason,
            query,
            section_owner,
            context,
        } => Rejected {
            reason,
            query,
            section_owner,
            work: Box::new(work),
            context: context.into_boxed_str(),
        },
        HydraulicError::Runtime(error) => Channel(ChannelError::Runtime {
            cell: None,
            context: error.to_string(),
        }),
        HydraulicError::Numerical { context } => Channel(ChannelError::Numerical {
            cell: None,
            context,
        }),
        HydraulicError::Arithmetic(error) => {
            let context = error.to_string();
            Channel(match error {
                ArithmeticError::Range { .. } | ArithmeticError::InvalidBounds => {
                    ChannelError::Numerical {
                        cell: None,
                        context,
                    }
                }
                _ => ChannelError::Runtime {
                    cell: None,
                    context,
                },
            })
        }
        HydraulicError::Geometry(error) => {
            let section_owner = error.section();
            let context = error.to_string();
            match error {
                CrossSectionError::InvalidInput { .. } => Channel(ChannelError::InvalidInput {
                    cell: None,
                    context,
                }),
                CrossSectionError::Runtime { .. } => Channel(ChannelError::Runtime {
                    cell: None,
                    context,
                }),
                CrossSectionError::Numerical { .. } => Rejected {
                    reason: PrismaticRiemannRejection::GeometryUnresolved,
                    query: None,
                    section_owner,
                    work: Box::new(work),
                    context: context.into_boxed_str(),
                },
                CrossSectionError::UnsupportedStage { .. } => Rejected {
                    reason: PrismaticRiemannRejection::CapExceeded,
                    query: None,
                    section_owner,
                    work: Box::new(work),
                    context: context.into_boxed_str(),
                },
            }
        }
    }
}
