use crate::channel::flow::{ChannelError, surveyed::PrismaticRiemannError};
use solvr::common::arithmetic::ArithmeticError;

pub(super) fn arithmetic_error(error: ArithmeticError) -> ChannelError {
    let context = error.to_string();
    match error {
        ArithmeticError::Numr(error) => ChannelError::from(error),
        ArithmeticError::UnsupportedDType { .. }
        | ArithmeticError::ProbeControlDType
        | ArithmeticError::UnsupportedArithmetic { .. }
        | ArithmeticError::ControlMaskDType => ChannelError::Runtime {
            cell: None,
            context,
        },
        ArithmeticError::InvalidProbeControl { .. }
        | ArithmeticError::InvalidBounds
        | ArithmeticError::Range { .. } => ChannelError::Numerical {
            cell: None,
            context,
        },
    }
}

pub(super) fn riemann_error(error: PrismaticRiemannError) -> ChannelError {
    match error {
        PrismaticRiemannError::Channel(error) => error,
        PrismaticRiemannError::Rejected {
            reason,
            query,
            section_owner,
            work,
            context,
        } => ChannelError::numerical(format!(
            "surveyed reach Riemann {reason:?}, face={query:?}, section_owner={section_owner:?}, work={work:?}: {context}"
        )),
    }
}

pub(super) fn boundary_error(error: ChannelError, face: usize) -> ChannelError {
    match error {
        ChannelError::InvalidInput { context, .. } => ChannelError::InvalidInput {
            cell: None,
            context: format!("surveyed exterior face={face}: {context}"),
        },
        ChannelError::Numerical { context, .. } => ChannelError::Numerical {
            cell: None,
            context: format!("surveyed exterior face={face}: {context}"),
        },
        ChannelError::Runtime { context, .. } => ChannelError::Runtime {
            cell: None,
            context: format!("surveyed exterior face={face}: {context}"),
        },
        error => error,
    }
}
