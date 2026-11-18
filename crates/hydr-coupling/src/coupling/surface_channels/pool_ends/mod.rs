//! Pool ends: a reach end that meets a coordinator pool at a quiescent reservoir head.
//!
//! The reach end takes `QuiescentReservoir` at its pool's temporal midpoint stage. The reach's
//! own transport sets the end volume. The coordinator never scales that volume, so a pool end is
//! no link. It joins the fixed point through its pool's end stage and its signed volume in the
//! iterate. Each evaluation reserves the reach's start operands under the reservoir heads of its
//! iterate.
//!
//! A pool end owns its reach end. The caller supplies no policy there, and no port binds it.
//! A pool end needs wet subcritical support at both interval ends. A dry pool end at the
//! accepted start rejects for good under `pool-end-regime`.

mod admission;
mod binding;
mod checks;
mod counters;
#[cfg(test)]
mod scenario;
mod topology;

pub(super) use admission::check_pool_ends;
pub(super) use binding::{EndBinding, EndVolumes};
pub(super) use counters::{PoolEndCounters, PoolEndInterval};
pub(super) use topology::PoolEndTopology;

use crate::channel::ChannelEnd;

/// One reach end joined to a coordinator pool through a quiescent reservoir head.
///
/// The reach end takes `QuiescentReservoir` at the pool's temporal midpoint stage, and the
/// reach's own transport sets the end volume. The coordinator never scales it. The pool owns
/// the end: the caller supplies no policy there, and no port binds it.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPoolEnd {
    /// Unique nonempty pool-end identity, distinct from every link, port, structure, and pool.
    pub id: String,
    /// Exact identity of a rectangular or triangular setup reach.
    pub reach_id: String,
    /// Reach end the pool owns. No caller policy and no port binds it.
    pub end: ChannelEnd,
    /// Exact identity of a coordinator pool.
    pub pool_id: String,
    /// Finite nonnegative dimensionless entrance loss against the incoming channel velocity head.
    pub entrance_loss_coefficient: f64,
    /// Evidence that the reach end samples this pool at the end cell bed.
    pub physical_connection_assertion: String,
    /// Evidence for quiescent reservoir inflow and complete exit mixing into the pool.
    pub quiescent_reservoir_assertion: String,
}

/// One admitted pool end with its resolved reach, end cell, and pool.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckedPoolEnd {
    /// Exact pool-end identity.
    pub id: String,
    /// Index into the setup's reach list.
    pub reach_index: usize,
    pub end: ChannelEnd,
    /// Position of the end in a per-reach `[left, right]` pair.
    pub side: usize,
    /// End cell index in reach cell order.
    pub cell: usize,
    /// Position of the pool in ascending pool identity.
    pub pool: usize,
    /// End cell bed in the common datum, in metres. It sits at or above the pool origin.
    pub bed_m: f64,
    pub entrance_loss_coefficient: f64,
}

#[cfg(test)]
mod tests {
    use super::scenario::{
        POND_AREA, advance, check_paired, end_volumes, forcing, ponded, pool_end,
    };
    use crate::channel::ChannelBoundary;
    use crate::coupling::surface_channels::SurfaceChannelError;
    use crate::coupling::surface_channels::pools::scenario::{
        build, channel, check, link, pool, surface, volumes,
    };
    use crate::coupling::surface_channels::structures::scenario::{
        DT, Tanks, TestResult, orifice_law, receipt,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// A passive bidirectional orifice `id` from node `a` to node `b` on beds at `0.0`.
    fn culvert(
        id: &str,
        a: crate::coupling::surface_channels::structures::SurfaceChannelNode,
        b: crate::coupling::surface_channels::structures::SurfaceChannelNode,
    ) -> TestResult<crate::coupling::surface_channels::structures::SurfaceChannelStructure> {
        Ok(link(
            id,
            orifice_law(0.1, 0.3)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (a, 0.0),
            (b, 0.0),
        ))
    }

    /// A caller inflow raises the reach so its end fills the pond. A caller withdrawal then
    /// lowers it, and the pond drains back through the same end. No link exists.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, every pairing with the pool-end volume, and the
    ///   composite balance.
    /// - Every interval's pool-end volume equals the reach end volume bit for bit.
    /// - Some fill interval carries water into the pond, and the pond ends the fill above its
    ///   start volume.
    /// - Some later interval carries water from the pond into the reach.
    /// - The pond's volume change equals the summed pool-end volumes within 1e-9 m³.
    #[test]
    fn a_reach_end_fills_a_pool_then_reverses() -> TestResult {
        let start = 0.45 * POND_AREA;
        let pools = ponded(vec![pool_end("outfall", "pond")])?;
        let mut batch = build(Tanks::at(0.0, 0.5), pools, &[start], Vec::new())?;
        let mut moved = 0.0;
        let mut filled = false;
        for _ in 0..20 {
            batch = advance(&batch, DT, [0.02, 0.0])?;
            check(&batch, &[0.0])?;
            check_paired(&batch)?;
            let volume = end_volumes(&batch)?[0];
            filled |= volume > 0.0;
            moved += volume;
        }
        assert!(filled, "the reach end never filled the pond");
        assert!(volumes(&batch)?[0] > start, "{:?}", volumes(&batch)?);
        let mut reversed = false;
        for _ in 0..40 {
            batch = advance(&batch, DT, [0.0, 0.04])?;
            check(&batch, &[0.0])?;
            check_paired(&batch)?;
            let volume = end_volumes(&batch)?[0];
            reversed |= volume < 0.0;
            moved += volume;
        }
        assert!(reversed, "the pond never drained into the reach");
        let change = volumes(&batch)?[0] - start;
        assert!(
            (change - moved).abs() <= 1e-9,
            "change={change}, moved={moved}"
        );
        Ok(())
    }

    /// The surface drains through an orifice into the pond the reach end also feeds.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every one of 10 intervals passes its mass bound, every pairing, and the composite balance.
    /// - Every interval's pool-end volume equals the reach end volume bit for bit.
    /// - The orifice carries water from the surface into the pond in every interval.
    /// - The pond's volume change equals the summed orifice and pool-end volumes within 1e-9 m³.
    #[test]
    fn a_pool_end_shares_its_pool_with_a_structure_link() -> TestResult {
        let start = 0.45 * POND_AREA;
        let pools = ponded(vec![pool_end("outfall", "pond")])?;
        let structures = vec![culvert("culvert", surface(), pool("pond"))?];
        let mut batch = build(Tanks::at(0.6, 0.5), pools, &[start], structures)?;
        let mut moved = 0.0;
        for _ in 0..10 {
            batch = advance(&batch, DT, [0.0, 0.0])?;
            check(&batch, &[0.0])?;
            check_paired(&batch)?;
            let orifice = receipt(&batch, "culvert")?.signed;
            assert!(orifice > 0.0, "orifice={orifice}");
            moved += orifice + end_volumes(&batch)?[0];
        }
        let change = volumes(&batch)?[0] - start;
        assert!(
            (change - moved).abs() <= 1e-9,
            "change={change}, moved={moved}"
        );
        Ok(())
    }

    /// The surface drains through an orifice into the reach, whose end feeds the pond. The reach
    /// is linked and pool-ended, so each evaluation reserves its slot.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every one of 10 intervals passes its mass bound, every pairing, and the composite balance.
    /// - Every interval's pool-end volume equals the reach end volume bit for bit.
    /// - The orifice carries water from the surface into the reach in every interval.
    /// - The pond's volume change equals the summed pool-end volumes within 1e-9 m³.
    #[test]
    fn a_linked_reach_feeds_its_pool_end() -> TestResult {
        let start = 0.45 * POND_AREA;
        let pools = ponded(vec![pool_end("outfall", "pond")])?;
        let structures = vec![culvert("culvert", surface(), channel())?];
        let mut batch = build(Tanks::at(0.6, 0.5), pools, &[start], structures)?;
        let mut moved = 0.0;
        for _ in 0..10 {
            batch = advance(&batch, DT, [0.0, 0.0])?;
            check(&batch, &[0.0])?;
            check_paired(&batch)?;
            let orifice = receipt(&batch, "culvert")?.signed;
            assert!(orifice > 0.0, "orifice={orifice}");
            moved += end_volumes(&batch)?[0];
        }
        let change = volumes(&batch)?[0] - start;
        assert!(
            (change - moved).abs() <= 1e-9,
            "change={change}, moved={moved}"
        );
        Ok(())
    }

    /// Input message of a rejected first trial of `batch` under `right`.
    fn trial_rejection(
        batch: &crate::coupling::surface_channels::structures::scenario::Batch,
        right: Option<ChannelBoundary>,
    ) -> TestResult<String> {
        let held = forcing(0.0, DT, right, [0.0, 0.0]);
        match batch.trial_to(DT, &held) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the trial admitted an invalid pool end".into()),
        }
    }

    /// Frozen criterion, fixed before the first run: a caller policy at a pool end rejects
    /// with `reach-end-ownership` and names the pool end.
    #[test]
    fn a_caller_policy_at_a_pool_end_rejects() -> TestResult {
        let pools = ponded(vec![pool_end("outfall", "pond")])?;
        let batch = build(Tanks::at(0.0, 0.5), pools, &[0.45 * POND_AREA], Vec::new())?;
        let message = trial_rejection(&batch, Some(ChannelBoundary::Wall))?;
        assert!(
            message.starts_with("reach-end-ownership") && message.contains("\"outfall\""),
            "{message}"
        );
        Ok(())
    }

    /// Frozen criteria, fixed before the first run: a dry pond and a dry reach end each reject
    /// the first trial with `pool-end-regime`, naming the pool end.
    #[test]
    fn a_dry_pool_end_rejects() -> TestResult {
        for (tanks, volume) in [
            (Tanks::at(0.0, 0.5), 0.0),
            (Tanks::at(0.0, 0.0), 0.45 * POND_AREA),
        ] {
            let pools = ponded(vec![pool_end("outfall", "pond")])?;
            let batch = build(tanks, pools, &[volume], Vec::new())?;
            let message = trial_rejection(&batch, None)?;
            assert!(
                message.starts_with("pool-end-regime") && message.contains("\"outfall\""),
                "{message}"
            );
        }
        Ok(())
    }
}
