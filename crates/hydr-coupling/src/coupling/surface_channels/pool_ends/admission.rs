//! Admission of pool ends: identities, evidence, reach geometry, pool, bed, and end ownership.
//!
//! Each rejection names its capability entry first.

use super::{CheckedPoolEnd, SurfaceChannelPoolEnd};
use crate::channel::{ChannelEnd, ChannelGeometry};
use crate::coupling::links::cutoffs::pool_attachment_admitted;
use crate::coupling::surface_channels::bindings::channel_cells;
use crate::coupling::surface_channels::pools::CheckedPools;
use crate::coupling::surface_channels::ports::{CheckedPort, end_slot};
use crate::coupling::surface_channels::validation::text;
use crate::coupling::surface_channels::{SurfaceChannelError, SurfaceChannelSetup};
use std::collections::{BTreeMap, BTreeSet};

/// Rejects with `entry` for pool end `id`.
fn rejected(entry: &str, id: &str, detail: String) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("{entry}: pool end={id:?} {detail}"))
}

/// Checks every pool end of `ends`. Returns them in ascending identity order.
///
/// `reach_index` maps each checked reach identity to its setup index. `taken` holds every link,
/// port, structure, and pool identity. A reach end with a port or a second pool end rejects.
pub(in super::super) fn check_pool_ends(
    setup: &SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
    pools: Option<&CheckedPools>,
    ports: &[CheckedPort],
    taken: &BTreeSet<&str>,
    ends: &[SurfaceChannelPoolEnd],
) -> Result<Vec<CheckedPoolEnd>, SurfaceChannelError> {
    let mut owners: BTreeMap<(usize, usize), String> = ports
        .iter()
        .map(|port| {
            let id = &setup.ports[port.setup_index].id;
            (
                (port.reach_index, end_slot(port.end)),
                format!("port={id:?}"),
            )
        })
        .collect();
    let mut checked = BTreeMap::new();
    for end in ends {
        let id = end.id.as_str();
        let evidence = |name: &str, value: &str| {
            text(
                &format!("missing-link-evidence: pool end={id:?} {name}"),
                value,
            )
        };
        evidence("id", id)?;
        if taken.contains(id) || checked.contains_key(id) {
            return Err(rejected(
                "duplicate-links",
                id,
                "repeats a pool end, link, port, structure, or pool identity: give each one a unique identity".into(),
            ));
        }
        evidence(
            "physical_connection_assertion",
            &end.physical_connection_assertion,
        )?;
        evidence(
            "quiescent_reservoir_assertion",
            &end.quiescent_reservoir_assertion,
        )?;
        let loss = end.entrance_loss_coefficient;
        if !loss.is_finite() || loss < 0.0 {
            return Err(rejected(
                "pool-end-regime",
                id,
                format!(
                    "entrance_loss_coefficient={loss}: supply a finite nonnegative entrance loss"
                ),
            ));
        }
        let Some(&reach) = reach_index.get(end.reach_id.as_str()) else {
            return Err(rejected(
                "structure-endpoint-identity",
                id,
                format!(
                    "reach_id={:?} names no setup reach: use a declared reach identity",
                    end.reach_id
                ),
            ));
        };
        let definition = &setup.reaches[reach].definition;
        if matches!(
            definition.geometry,
            ChannelGeometry::PrismaticSurveyed { .. }
        ) {
            return Err(rejected(
                "surveyed-pool-end",
                id,
                format!(
                    "reach={:?} geometry=surveyed: surveyed reaches couple through bank links only; use a rectangular or triangular reach for a pool end",
                    end.reach_id
                ),
            ));
        }
        let cells = channel_cells(definition);
        let cell = match end.end {
            ChannelEnd::Left => 0,
            ChannelEnd::Right => cells.len().saturating_sub(1),
        };
        let Some(&(_, bed_m)) = cells.get(cell) else {
            return Err(rejected(
                "structure-endpoint-identity",
                id,
                format!(
                    "reach={:?} has no cells: declare the reach cells",
                    end.reach_id
                ),
            ));
        };
        let Some((pool, element)) = pools.and_then(|pools| pools.find(&end.pool_id)) else {
            return Err(rejected(
                "structure-endpoint-identity",
                id,
                format!(
                    "pool_id={:?} names no coordinator pool: use a declared pool identity",
                    end.pool_id
                ),
            ));
        };
        let origin = element.curve.origin();
        if !pool_attachment_admitted(
            &definition.vertical_datum,
            element.curve.vertical_reference(),
            bed_m,
            origin,
        ) {
            return Err(rejected(
                "unobstructed-pool-connection",
                id,
                format!(
                    "end cell bed={bed_m} sits below pool={:?} origin={origin}: attach the reach end at or above the pool origin",
                    end.pool_id
                ),
            ));
        }
        let side = end_slot(end.end);
        if let Some(owner) = owners.insert((reach, side), format!("pool end={id:?}")) {
            return Err(rejected(
                "reach-end-ownership",
                id,
                format!(
                    "reach={:?} end={:?} is already owned by {owner}: give each reach end one owner",
                    end.reach_id, end.end
                ),
            ));
        }
        checked.insert(
            id,
            CheckedPoolEnd {
                id: id.to_string(),
                reach_index: reach,
                end: end.end,
                side,
                cell,
                pool,
                bed_m,
                entrance_loss_coefficient: loss,
            },
        );
    }
    Ok(checked.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{ponded, pool_end};
    use crate::channel::ChannelEnd;
    use crate::coupling::surface_channels::pools::scenario::{channel, link, surface};
    use crate::coupling::surface_channels::ports::fixture::x_face;
    use crate::coupling::surface_channels::structures::scenario::{
        REACH, Tanks, TestResult, orifice_law,
    };
    use crate::coupling::surface_channels::{
        SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelPort,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// The surface/reach orifice beside the pond of every admission case.
    fn culvert()
    -> TestResult<crate::coupling::surface_channels::structures::SurfaceChannelStructure> {
        Ok(link(
            "culvert",
            orifice_law(0.1, 0.3)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (surface(), 0.0),
            (channel(), 0.0),
        ))
    }

    /// Frozen criterion, fixed before the first run: a pond joined only by a pool end admits,
    /// beside an orifice that never touches it, and holds that one pool end.
    #[test]
    fn a_pool_joined_only_by_a_pool_end_admits() -> TestResult {
        let (setup, _) = Tanks::at(0.0, 0.5).parts()?;
        let pools = ponded(vec![pool_end("outfall", "pond")])?;
        let definition = SurfaceChannelDefinition::with_nodes(setup, pools, vec![culvert()?])?;
        assert_eq!(definition.pool_ends.len(), 1);
        assert_eq!(definition.pool_ends[0].pool, 0);
        assert_eq!(definition.pool_ends[0].end, ChannelEnd::Right);
        Ok(())
    }

    /// Frozen criterion, fixed before the first run: a port and a pool end on one reach end,
    /// and two pool ends on one reach end, each reject with `reach-end-ownership`.
    #[test]
    fn a_second_owner_of_a_pool_end_rejects() -> TestResult {
        let (mut ported, _) = Tanks::at(0.0, 0.5).parts()?;
        ported.ports.push(SurfaceChannelPort {
            id: "port".into(),
            reach_id: REACH.into(),
            end: ChannelEnd::Right,
            faces: vec![x_face(1, 0)],
            mapping_evidence: "The east cell face meets the reach outfall".into(),
        });
        let (plain, _) = Tanks::at(0.0, 0.5).parts()?;
        let cases = [
            (ported, ponded(vec![pool_end("outfall", "pond")])?),
            (
                plain,
                ponded(vec![
                    pool_end("outfall", "pond"),
                    pool_end("second", "pond"),
                ])?,
            ),
        ];
        for (setup, pools) in cases {
            match SurfaceChannelDefinition::with_nodes(setup, pools, vec![culvert()?]) {
                Err(SurfaceChannelError::Input(message)) => {
                    assert!(message.starts_with("reach-end-ownership"), "{message}");
                }
                Err(other) => {
                    return Err(format!("expected an input rejection, got {other}").into());
                }
                Ok(_) => return Err("a reach end admitted two owners".into()),
            }
        }
        Ok(())
    }
}
