//! Caller writers that would deliver runoff the coordinator already owns.

use super::admission::CheckedRunoff;
use super::bindings::RUNOFF_ATTACHMENT_BINDING_PREFIX;
use crate::coupling::level_pools::CoupledPoolInflowBinding;
use crate::coupling::surface_channels::SurfaceChannelError;
use std::collections::BTreeSet;

/// Rejects a caller pool inflow binding that uses the reserved prefix, or one whose physical
/// support names an active modelled contribution or one of its atoms.
pub(in super::super) fn check_caller_pool_inflows(
    inflows: &[CoupledPoolInflowBinding],
    runoff: Option<&CheckedRunoff>,
) -> Result<(), SurfaceChannelError> {
    let mut owned = BTreeSet::new();
    if let Some(runoff) = runoff {
        for contribution in runoff.plan.active_contributions() {
            owned.insert(contribution.id.as_str());
            owned.extend(contribution.atoms.iter().map(String::as_str));
        }
    }
    for binding in inflows {
        if binding
            .boundary_id
            .starts_with(RUNOFF_ATTACHMENT_BINDING_PREFIX)
        {
            return Err(SurfaceChannelError::input(format!(
                "reserved-binding-prefix: pool inflow binding {:?} uses the reserved prefix {RUNOFF_ATTACHMENT_BINDING_PREFIX:?}: rename the binding; the coordinator synthesizes runoff bindings from runoff outlets",
                binding.boundary_id
            )));
        }
        if owned.contains(binding.physical_support_id.as_str()) {
            return Err(SurfaceChannelError::input(format!(
                "runoff-duplicate-writer: pool inflow binding {:?} physical_support_id={:?} names native runoff the coordinator delivers: drop the caller binding or replace the modelled source",
                binding.boundary_id, binding.physical_support_id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{POND, SOURCE, parts, pond_parts, rejection, starts};
    use crate::channel::{ChannelLateralBinding, ChannelLateralDirection};
    use crate::coupling::surface_channels::pools::scenario::inflow_binding;
    use crate::coupling::surface_channels::structures::scenario::TestResult;

    #[test]
    fn caller_lateral_bindings_reject_the_reserved_prefix() -> TestResult {
        let (mut setup, _) = pond_parts(false)?;
        setup.reaches[0]
            .definition
            .lateral_bindings
            .push(ChannelLateralBinding {
                boundary_id: "runoff-attachment:caller".into(),
                cell_id: "cell-0".into(),
                direction: ChannelLateralDirection::Inflow,
                ownership_assertion: "Caller water".into(),
            });
        let message = rejection(setup)?;
        starts(&message, "reserved-binding-prefix");
        assert!(message.contains("runoff-attachment:caller"), "{message}");
        Ok(())
    }

    #[test]
    fn caller_pool_inflow_bindings_reject_the_reserved_prefix() -> TestResult {
        let (mut setup, _) = pond_parts(false)?;
        setup.pool_inflows = vec![inflow_binding("runoff-attachment:pipe", POND)];
        starts(&rejection(setup)?, "reserved-binding-prefix");
        Ok(())
    }

    #[test]
    fn a_caller_pool_inflow_of_native_runoff_is_a_duplicate_writer() -> TestResult {
        for support in [SOURCE, "upstream"] {
            let (mut setup, _) = parts()?;
            let mut binding = inflow_binding("pipe", POND);
            binding.physical_support_id = support.into();
            setup.pool_inflows = vec![binding];
            let message = rejection(setup)?;
            starts(&message, "runoff-duplicate-writer");
            assert!(message.contains(support), "{message}");
        }
        Ok(())
    }
}
