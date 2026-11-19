//! Channel errors of certified surveyed trace solves.

use crate::channel::flow::{
    ChannelError,
    surveyed::{
        PrismaticRiemannError, PrismaticRiemannWork,
        error::{HydraulicError, into_public},
    },
};

/// Maps a certified-geometry error of a trace solve to its channel error.
///
/// A method rejection keeps its reason and context under the scalar boundary prefix.
pub(super) fn trace_error(error: HydraulicError) -> ChannelError {
    match into_public(error, PrismaticRiemannWork::default()) {
        PrismaticRiemannError::Channel(error) => error,
        PrismaticRiemannError::Rejected {
            reason, context, ..
        } => ChannelError::numerical(format!(
            "scalar channel boundary: surveyed trace {reason:?}: {context}"
        )),
    }
}
