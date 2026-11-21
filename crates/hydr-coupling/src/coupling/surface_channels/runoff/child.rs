//! The coordinator-owned runoff child at the common epoch.

use crate::coupling::surface_channels::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelInitialState,
};
use crate::runoff::SubcatchmentBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// Builds the runoff child of `definition` at `initial`, or `None` without active definitions.
///
/// The child holds the plan's active definitions in order, one state each from
/// `initial.runoff`, at the common time `initial.time_s`.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] for runoff states without a runoff setup or without
/// active definitions. Returns [`SurfaceChannelError::Runoff`] when the child rejects its
/// definitions, states, time, or config.
pub(in super::super) fn build_child<R>(
    client: &R::Client,
    definition: &SurfaceChannelDefinition,
    initial: &SurfaceChannelInitialState,
) -> Result<Option<SubcatchmentBatch<R>>, SurfaceChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let (Some(runoff), Some(setup)) = (&definition.runoff, &definition.setup.runoff) else {
        if initial.runoff.is_empty() {
            return Ok(None);
        }
        return Err(SurfaceChannelError::input(format!(
            "runoff-without-setup: initial runoff state count={} without setup.runoff: supply no runoff states",
            initial.runoff.len()
        )));
    };
    let definitions = runoff.plan.active_definitions().to_vec();
    if definitions.is_empty() {
        if initial.runoff.is_empty() {
            return Ok(None);
        }
        return Err(SurfaceChannelError::input(format!(
            "runoff-without-setup: initial runoff state count={}, active definitions=0: supply no runoff states when every modelled source is replaced",
            initial.runoff.len()
        )));
    }
    if initial.runoff.len() != definitions.len() {
        return Err(SurfaceChannelError::input(format!(
            "runoff-initial-state: initial runoff state count={}, active definitions={}: supply one state per active definition in definition order",
            initial.runoff.len(),
            definitions.len()
        )));
    }
    let batch = SubcatchmentBatch::new(
        client.clone(),
        definitions,
        initial.runoff.clone(),
        initial.time_s,
        setup.config.clone(),
    )
    .map_err(SurfaceChannelError::Runoff)?;
    if !runoff.plan.same_definitions(batch.definitions()) {
        return Err(SurfaceChannelError::input(
            "runoff-plan: runoff child definitions differ from the plan's active definitions in identity, order, or parameter bits",
        ));
    }
    Ok(Some(batch))
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{build, input_message, parts};
    use super::*;
    use crate::coupling::surface_channels::structures::scenario::TestResult;
    use crate::runoff::SubcatchmentError;

    #[test]
    fn construction_builds_the_runoff_child_at_the_common_epoch() -> TestResult {
        let (setup, mut initial) = parts()?;
        initial.time_s = 2.5;
        initial.surface.time = 2.5;
        for (_, reach) in &mut initial.reaches {
            reach.time_s = 2.5;
        }
        let batch = build((setup, initial))?;
        let Some(runoff) = batch.runoff() else {
            return Err("missing runoff child".into());
        };
        assert_eq!(runoff.time().to_bits(), 2.5_f64.to_bits());
        assert_eq!(runoff.accepted_steps(), 0);
        assert_eq!(runoff.definitions().len(), 1);
        assert_eq!(runoff.definitions()[0].id, "upstream-model");
        Ok(())
    }

    #[test]
    fn runoff_states_must_match_the_active_definitions() -> TestResult {
        let (setup, mut initial) = parts()?;
        initial.runoff.clear();
        let message = input_message(build((setup, initial)))?;
        assert!(message.starts_with("runoff-initial-state"), "{message}");
        Ok(())
    }

    #[test]
    fn a_rejected_runoff_state_keeps_its_typed_error() -> TestResult {
        let (setup, mut initial) = parts()?;
        initial.runoff[0].depths_m[0] = -1.0;
        match build((setup, initial)) {
            Err(SurfaceChannelError::Runoff(SubcatchmentError::InvalidInput { .. })) => Ok(()),
            other => Err(format!("expected a runoff input rejection, got {other:?}").into()),
        }
    }

    #[test]
    fn only_numerical_runoff_rejections_retry() {
        let error = |numerical: bool| {
            let context = "synthetic".to_owned();
            SurfaceChannelError::Runoff(if numerical {
                SubcatchmentError::Numerical {
                    catchment: None,
                    subarea: None,
                    context,
                }
            } else {
                SubcatchmentError::InvalidInput {
                    catchment: None,
                    subarea: None,
                    context,
                }
            })
        };
        assert!(error(true).is_retryable());
        assert!(!error(false).is_retryable());
    }
}
