//! Free-outlet checks: the receiving stage of every delivery lane against its outlet invert.
//!
//! The coordinator samples each receiving stage at the accepted start and at the candidate end.
//! A crossing inside an interval and back out stays undetected. A stage above the invert at
//! the start rejects without retry. A stage above it at the end shortens the interval.

use super::admission::RunoffTarget;
use crate::channel::ChannelReach;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;

/// Where a receiving stage sample lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Sample {
    /// The accepted start of the trial.
    Start,
    /// The candidate end of the attempt.
    End,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Receiving stage of every delivery lane, in metres.
    ///
    /// `reaches` holds every reach in setup order and `pools` the pool child.
    pub fn receiving_stages(
        &self,
        reaches: &[ChannelReach<R>],
        pools: Option<&LevelPoolBatch<R>>,
        reach_ids: &[&str],
    ) -> Result<Vec<f64>, SurfaceChannelError> {
        let topology = self.runoff_topology()?;
        let lanes = topology.lane_count();
        let mut positions = vec![usize::MAX; lanes];
        let mut parts = Vec::new();
        let mut offset = 0;
        for (reach, child) in reaches.iter().enumerate() {
            let receiving: Vec<(usize, usize)> = topology
                .targets
                .iter()
                .enumerate()
                .filter_map(|(lane, target)| match *target {
                    RunoffTarget::Channel { reach: r, cell } if r == reach => Some((lane, cell)),
                    _ => None,
                })
                .collect();
            if receiving.is_empty() {
                continue;
            }
            let cells: Vec<usize> = receiving.iter().map(|&(_, cell)| cell).collect();
            let (stage, _) = child.coupling_cell_stages(&cells).map_err(|source| {
                SurfaceChannelError::Channel {
                    reach_id: reach_ids.get(reach).copied().unwrap_or_default().to_owned(),
                    source,
                }
            })?;
            for (k, &(lane, _)) in receiving.iter().enumerate() {
                positions[lane] = offset + k;
            }
            offset += cells.len();
            parts.push(stage);
        }
        let pooled = topology
            .targets
            .iter()
            .any(|target| matches!(target, RunoffTarget::Pool { .. }));
        if pooled {
            let batch = pools.ok_or_else(|| {
                SurfaceChannelError::Numerical("runoff pool lanes lack the pool child".into())
            })?;
            let (stage, _) = self.pool_stages(batch)?;
            for (lane, target) in topology.targets.iter().enumerate() {
                if let RunoffTarget::Pool { position } = *target {
                    positions[lane] = offset + position;
                }
            }
            offset += stage.numel();
            parts.push(stage);
        }
        let host = if parts.is_empty() {
            Vec::new()
        } else {
            let c = &self.client;
            c.cat(&parts.iter().collect::<Vec<_>>(), 0)?
                .try_to_vec::<f64>()?
        };
        if host.len() != offset {
            return Err(SurfaceChannelError::Numerical(
                "receiving stages differ from their nodes".into(),
            ));
        }
        positions
            .iter()
            .map(|&position| {
                host.get(position).copied().ok_or_else(|| {
                    SurfaceChannelError::Numerical("a runoff lane lacks its receiving stage".into())
                })
            })
            .collect()
    }

    /// Rejects a receiving stage above its outlet invert at `time`.
    ///
    /// At the start the rejection is [`SurfaceChannelError::Input`]: no shorter interval moves
    /// the accepted water. At the end it is [`SurfaceChannelError::Numerical`], so the interval
    /// shortens. Both name `runoff-outlet-submerged`.
    pub fn check_tailwater(
        &self,
        stages: &[f64],
        time: f64,
        sample: Sample,
    ) -> Result<(), SurfaceChannelError> {
        let topology = self.runoff_topology()?;
        for (lane, (&stage, &invert)) in stages.iter().zip(&topology.inverts).enumerate() {
            if stage <= invert {
                continue;
            }
            let outlet = topology.outlets.get(lane).map_or("", String::as_str);
            let node = topology
                .nodes
                .get(lane)
                .map(|node| node.identity())
                .unwrap_or_default();
            return Err(match sample {
                Sample::Start => SurfaceChannelError::input(format!(
                    "runoff-outlet-submerged: runoff outlet={outlet:?} receiving {node} stage={stage} lies above invert={invert} at the accepted start time={time}: lower the receiving water or move the outlet; the outlet has no backwater feedback"
                )),
                Sample::End => SurfaceChannelError::Numerical(format!(
                    "runoff-outlet-submerged: runoff outlet={outlet:?} receiving {node} stage={stage} lies above invert={invert} at the candidate end time={time}: shorten the common interval"
                )),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::basin::{Basin, Receiver, advance, balanced, forcing, isolated, transfers};
    use super::super::scenario::{input_message, starts};
    use crate::coupling::surface_channels::SurfaceChannelError;
    use crate::coupling::surface_channels::structures::scenario::TestResult;

    #[test]
    fn a_receiving_stage_above_the_invert_at_the_start_rejects_without_retry() -> TestResult {
        let mut basin = Basin::into(Receiver::Pool);
        basin.pond_m3 = 2.0;
        basin.invert_m = 0.25;
        let batch = basin.build()?;
        let message = input_message(batch.trial_to(0.5, &forcing(&batch, 0.5, 0.0)))?;
        starts(&message, "runoff-outlet-submerged");
        assert!(message.contains("pool:pond"), "{message}");
        Ok(())
    }

    /// The pond starts dry below an invert that half an interval of runoff crosses.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The trial shortens below the requested end, and the accepted end stage lies at or
    ///   below the invert.
    /// - The delivered volume equals the runoff child trialled alone to the accepted end, bit
    ///   for bit.
    /// - Without retries the same trial is `Unresolved`, naming `runoff-outlet-submerged`.
    #[test]
    fn a_receiving_stage_crossing_the_invert_at_the_end_shortens_the_interval() -> TestResult {
        let mut basin = Basin::into(Receiver::Pool);
        basin.invert_m = 3e-4;
        let batch = basin.build()?;
        let next = advance(&batch, 0.5, 0.0)?;
        let end = next.time();
        assert!(end < 0.5, "end={end}");
        let transfer = transfers(&next)?
            .into_iter()
            .next()
            .ok_or("one delivery lane")?;
        assert!(transfer.start_stage_m <= transfer.invert_elevation_m);
        assert!(transfer.end_stage_m <= transfer.invert_elevation_m);
        assert_eq!(
            transfer.volume_m3.to_bits(),
            isolated(&batch, end)?[0].to_bits()
        );
        balanced(&next)?;
        basin.retry_limit = 0;
        let batch = basin.build()?;
        match batch.trial_to(0.5, &forcing(&batch, 0.5, 0.0)) {
            Err(SurfaceChannelError::Unresolved { reason, .. }) => {
                assert!(reason.contains("runoff-outlet-submerged"), "{reason}");
                Ok(())
            }
            other => Err(format!("expected an unresolved trial, got {other:?}").into()),
        }
    }
}
