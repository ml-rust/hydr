//! Runoff steps of one common trial: admission at the start, delivery per attempt, and the
//! checks at the candidate end.

use super::attempt::RunoffAttempt;
use super::forcing::{admit_forcing, check_steppable};
use super::report::RunoffStep;
use super::tailwater::Sample;
use crate::channel::{ChannelReach, ChannelTensorLateralInput};
use crate::coupling::surface_channels::context::Ops;
use crate::coupling::surface_channels::evaluate::Attempt;
use crate::coupling::surface_channels::pools::{PoolChild, inflow_error};
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelError, SurfaceChannelForcingInterval,
};
use crate::runoff::AdmittedRates;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Runoff operands of one trial.
pub(in super::super) struct RunoffStart {
    /// Admitted rates over the forcing interval.
    pub rates: AdmittedRates,
    /// Receiving stage of every delivery lane at the accepted start, in metres.
    pub start_stage_m: Vec<f64>,
}

/// Admits the runoff of `forcing` and checks every receiving stage of `batch` at its accepted
/// start.
///
/// Returns `None` without active runoff.
///
/// # Errors
/// Returns every runoff forcing rejection, `runoff-attachment` for a surface-cell destination,
/// and `runoff-outlet-submerged` for a receiving stage above its invert. Each is
/// [`SurfaceChannelError::Input`].
pub(in super::super) fn runoff_start<R: Runtime<DType = DType>>(
    batch: &SurfaceChannelBatch<R>,
    forcing: &SurfaceChannelForcingInterval,
    reach_ids: &[&str],
) -> Result<Option<RunoffStart>, SurfaceChannelError>
where
    R::Client: Ops<R>,
{
    let Some(rates) = admit_forcing(&batch.definition, forcing)? else {
        return Ok(None);
    };
    check_steppable(&batch.definition)?;
    if batch.context.runoff.is_none() {
        return Ok(None);
    }
    let pools = batch.pools.as_ref().map(|pools| &pools.batch);
    let start_stage_m = batch
        .context
        .receiving_stages(&batch.reaches, pools, reach_ids)?;
    batch
        .context
        .check_tailwater(&start_stage_m, batch.time_s, Sample::Start)?;
    Ok(Some(RunoffStart {
        rates,
        start_stage_m,
    }))
}

/// Runoff delivery of the attempt of `batch` ending at `end`, recomputed from the accepted
/// runoff state, or `None` without active runoff.
pub(in super::super) fn runoff_delivery<R: Runtime<DType = DType>>(
    batch: &SurfaceChannelBatch<R>,
    start: Option<&RunoffStart>,
    end: f64,
) -> Result<Option<RunoffAttempt<R>>, SurfaceChannelError>
where
    R::Client: Ops<R>,
{
    let (Some(start), Some(runoff), Some(child)) = (start, &batch.definition.runoff, &batch.runoff)
    else {
        return Ok(None);
    };
    batch
        .context
        .runoff_attempt(runoff, &start.rates, child, end)
        .map(Some)
}

/// Checks the candidate end of a runoff attempt of `batch` and returns its interval receipt.
///
/// Each receiving stage must lie at or below its invert at `end`, and each receiving binding
/// must hold its delivered volume in every bit. `reaches` holds the candidate reaches in setup
/// order. `pool_received` holds the original volume of every runoff pool binding.
pub(in super::super) fn runoff_end<R: Runtime<DType = DType>>(
    batch: &SurfaceChannelBatch<R>,
    (start, attempt): (Option<&RunoffStart>, Option<&RunoffAttempt<R>>),
    (reaches, pools): (&[ChannelReach<R>], Option<&PoolChild<R>>),
    pool_received: Option<&Tensor<R>>,
    reach_ids: &[&str],
    end: f64,
) -> Result<Option<RunoffStep>, SurfaceChannelError>
where
    R::Client: Ops<R>,
{
    let (Some(start), Some(attempt)) = (start, attempt) else {
        return Ok(None);
    };
    if attempt.next.time().to_bits() != end.to_bits() {
        return Err(SurfaceChannelError::Numerical(format!(
            "the runoff candidate time={} differs from end={end}",
            attempt.next.time()
        )));
    }
    let context = &batch.context;
    let pools = pools.map(|pools| &pools.batch);
    let end_stage_m = context.receiving_stages(reaches, pools, reach_ids)?;
    context.check_tailwater(&end_stage_m, end, Sample::End)?;
    context.check_runoff_delivery(attempt, reaches, pool_received)?;
    Ok(Some(RunoffStep {
        delivered_m3: attempt.delivered_m3.clone(),
        start_stage_m: start.start_stage_m.clone(),
        end_stage_m,
    }))
}

/// Original volume of every runoff pool binding in `attempt`, or `None` without them.
pub(in super::super) fn pool_receipts<R: Runtime<DType = DType>>(
    batch: &SurfaceChannelBatch<R>,
    attempt: &Attempt<R>,
) -> Result<Option<Tensor<R>>, SurfaceChannelError>
where
    R::Client: Ops<R>,
{
    let (Some(topology), Some(pools)) = (&batch.context.pools, &attempt.pools) else {
        return Ok(None);
    };
    if topology.inflows.native_count() == 0 {
        return Ok(None);
    }
    topology
        .inflows
        .native_volumes(&pools.external.step)
        .map(Some)
        .map_err(inflow_error)
}

/// The caller `lateral` inputs of setup reach `reach` with its runoff inputs appended.
pub(in super::super) fn with_runoff<R: Runtime<DType = DType>>(
    lateral: &[ChannelTensorLateralInput<R>],
    runoff: Option<&RunoffAttempt<R>>,
    reach: usize,
) -> Vec<ChannelTensorLateralInput<R>> {
    let mut inputs = lateral.to_vec();
    if let Some(own) = runoff.and_then(|runoff| runoff.laterals.get(reach)) {
        inputs.extend(own.iter().cloned());
    }
    inputs
}

#[cfg(test)]
mod tests {
    use super::super::basin::{Basin, Receiver, advance, balanced, pond_volume, transfers};
    use super::super::scenario::share;
    use crate::coupling::surface_channels::structures::scenario::{Batch, TestResult};

    /// Runoff bits the reach recorded on its runoff bindings over the last interval.
    fn received(batch: &Batch) -> TestResult<Vec<u64>> {
        let topology = batch.context.runoff.as_ref().ok_or("runoff operands")?;
        let own = topology.reaches[0].len();
        let accounting = batch.reaches[0].coupling_accounting();
        let interval = accounting
            .lateral_interval_volume_m3
            .ok_or("a lateral receipt")?
            .narrow(0, topology.offsets[0], own)?
            .try_to_vec::<f64>()?;
        Ok(interval.into_iter().map(f64::to_bits).collect())
    }

    /// Runoff into an unlinked reach and into a reach the pond weir links.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Over four intervals each runoff binding holds its delivered volume bit for bit.
    /// - The linked case keeps its synthesized exchange bindings ahead of the runoff binding.
    /// - Every interval balances.
    #[test]
    fn channel_receivers_hold_the_delivered_bits() -> TestResult {
        for linked in [false, true] {
            let mut basin = Basin::into(Receiver::Reach(vec![share(1, 1.0)]));
            basin.linked = linked;
            let mut batch = basin.build()?;
            assert_eq!(batch.context.linked.len(), usize::from(linked));
            let offset = batch
                .context
                .runoff
                .as_ref()
                .ok_or("runoff operands")?
                .offsets[0];
            assert_eq!(offset, if linked { 2 } else { 0 });
            for _ in 0..4 {
                batch = advance(&batch, 0.25, 0.0)?;
                let delivered: Vec<u64> = transfers(&batch)?
                    .iter()
                    .map(|transfer| transfer.volume_m3.to_bits())
                    .collect();
                assert_eq!(received(&batch)?, delivered, "linked={linked}");
                balanced(&batch)?;
            }
        }
        Ok(())
    }

    /// Runoff into the closed pond over four intervals.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The lane ledger holds the running sum of the delivered volumes bit for bit.
    /// - The pond holds that sum within four ulps.
    /// - Every interval balances.
    #[test]
    fn the_pool_receiver_holds_the_delivered_bits() -> TestResult {
        let mut batch = Basin::into(Receiver::Pool).build()?;
        let mut total = 0.0;
        for _ in 0..4 {
            batch = advance(&batch, 0.25, 0.0)?;
            let volume = transfers(&batch)?
                .first()
                .ok_or("one delivery lane")?
                .volume_m3;
            total += volume;
            let ledgers = batch.runoff_ledgers()?;
            assert_eq!(ledgers[0].delivered_volume_m3.to_bits(), total.to_bits());
            assert!((pond_volume(&batch)? - total).abs() <= 4.0 * f64::EPSILON * total);
            balanced(&batch)?;
        }
        Ok(())
    }
}
