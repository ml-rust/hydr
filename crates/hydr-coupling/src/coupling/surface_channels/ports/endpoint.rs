//! Caller endpoint policies against port ownership of reach ends.

use super::super::SurfaceChannelError;
use super::end_name;
use crate::channel::ChannelBoundary;
use hydr_channel::internal::ChannelCouplingEnd;

/// Checks the caller policy at one reach end against port ownership.
///
/// A ported end takes no caller policy: the coordinator supplies its flux port.
/// Every other end takes one.
pub(in super::super) fn check_endpoint(
    reach_id: &str,
    end: ChannelCouplingEnd,
    port: Option<&str>,
    policy: Option<&ChannelBoundary>,
) -> Result<(), SurfaceChannelError> {
    let name = end_name(end);
    match (port, policy) {
        (Some(port), Some(policy)) => Err(SurfaceChannelError::input(format!(
            "port-endpoint-policy: reach={reach_id:?} {name}={policy:?} sits at port={port:?}: supply no caller policy at a ported end; the coordinator supplies its flux port"
        ))),
        (None, None) => Err(SurfaceChannelError::input(format!(
            "port-endpoint-policy: reach={reach_id:?} {name} has no port and no caller policy: supply an endpoint policy"
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{ALPHA, BETA, TestResult, contains};
    use super::*;
    use ChannelCouplingEnd::{Left, Right};

    #[test]
    fn a_ported_end_takes_no_caller_policy() -> TestResult {
        check_endpoint(ALPHA, Left, Some("p"), None)?;
        check_endpoint(ALPHA, Right, None, Some(&ChannelBoundary::Wall))?;
        let owned = check_endpoint(ALPHA, Left, Some("p"), Some(&ChannelBoundary::Wall));
        let Err(SurfaceChannelError::Input(message)) = owned else {
            return Err("a caller policy at a ported end must reject".into());
        };
        contains(
            &message,
            &[
                "port-endpoint-policy",
                "reach=\"alpha\" left=Wall sits at port=\"p\"",
                "supply no caller policy at a ported end",
            ],
        )?;
        let Err(SurfaceChannelError::Input(message)) = check_endpoint(BETA, Right, None, None)
        else {
            return Err("a missing policy at an unported end must reject".into());
        };
        contains(
            &message,
            &[
                "port-endpoint-policy",
                "reach=\"beta\" right has no port and no caller policy",
            ],
        )
    }
}
