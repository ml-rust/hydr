//! Admission of pools: identities, datum, support evidence, and structure attachment.
//!
//! Each rejection names its capability entry first.

use super::{CheckedPools, SurfaceChannelPools};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::pool_ends::CheckedPoolEnd;
use crate::coupling::surface_channels::structures::{CheckedNode, CheckedStructure};
use crate::coupling::surface_channels::validation::text;
use std::collections::{BTreeMap, BTreeSet};

/// Rejects with `entry` for pool `id`.
fn rejected(entry: &str, id: &str, detail: String) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("{entry}: pool={id:?} {detail}"))
}

/// Checks `pools` against the surface datum. `taken` holds every link and port identity.
///
/// Returns `None` for an empty declaration.
pub(in super::super) fn check_pools(
    datum: &str,
    taken: &BTreeSet<&str>,
    pools: SurfaceChannelPools,
) -> Result<Option<CheckedPools>, SurfaceChannelError> {
    if pools.pools.is_empty() {
        if let Some(end) = pools.ends.first() {
            return Err(SurfaceChannelError::input(format!(
                "structure-endpoint-identity: pool end={:?} pool_id={:?} names no coordinator pool: declare the pool or drop the pool end",
                end.id, end.pool_id
            )));
        }
        if !pools.support.is_empty()
            || pools.sources.as_ref().is_some_and(|s| !s.is_empty())
            || !pools.inflows.is_empty()
        {
            return Err(SurfaceChannelError::input(
                "pool-support: support, sources, or inflows without pools: declare the pools or drop their declarations",
            ));
        }
        return Ok(None);
    }
    pools
        .storage
        .check()
        .map_err(|error| SurfaceChannelError::input(format!("pool storage config: {error}")))?;
    let count = pools.pools.len();
    if pools.support.len() != count {
        return Err(SurfaceChannelError::input(format!(
            "pool-support: support count={}, pools={count}: declare one support per pool in pool order",
            pools.support.len()
        )));
    }
    if let Some(sources) = &pools.sources
        && sources.len() != count
    {
        return Err(SurfaceChannelError::input(format!(
            "pool-support: source policy count={}, pools={count}: declare one source policy per pool in pool order",
            sources.len()
        )));
    }
    let mut index = BTreeMap::new();
    let mut supports = BTreeSet::new();
    for (position, (pool, support)) in pools.pools.iter().zip(&pools.support).enumerate() {
        let id = pool.id.as_str();
        text(&format!("missing-link-evidence: pool={id:?} id"), id)?;
        text(
            &format!("missing-link-evidence: pool={id:?} connected_pool_assertion"),
            &pool.connected_pool_assertion,
        )?;
        if taken.contains(id) || index.insert(id, position).is_some() {
            return Err(rejected(
                "duplicate-links",
                id,
                "repeats a pool, link, or port identity: give each one a unique identity".into(),
            ));
        }
        if !supports.insert(pool.storage_support_identity.as_str()) {
            return Err(rejected(
                "pool-support",
                id,
                format!(
                    "storage_support_identity={:?} repeats: give each pool its own storage support",
                    pool.storage_support_identity
                ),
            ));
        }
        let reference = pool.curve.vertical_reference();
        if reference != datum {
            return Err(rejected(
                "datum-or-gravity-mismatch",
                id,
                format!(
                    "curve vertical_reference={reference:?}, surface vertical_datum={datum:?}: express the curve in the surface datum"
                ),
            ));
        }
        let origin = pool.curve.origin();
        text(
            &format!("pool-support: pool={id:?} zero_volume_limit_assertion"),
            &support.zero_volume_limit_assertion,
        )?;
        if support.pool_id != pool.id || support.zero_volume_stage_m.to_bits() != origin.to_bits() {
            return Err(rejected(
                "pool-support",
                id,
                format!(
                    "support pool_id={:?}, zero_volume_stage_m={}, curve origin={origin}: declare the support in pool order at the curve origin",
                    support.pool_id, support.zero_volume_stage_m
                ),
            ));
        }
    }
    let order: Vec<usize> = index.values().copied().collect();
    let ids = index.keys().map(|id| id.to_string()).collect();
    Ok(Some(CheckedPools {
        declared: pools,
        order,
        ids,
    }))
}

/// Rejects a pool that no structure of `structures` and no pool end of `ends` attaches to.
pub(in super::super) fn check_attached(
    pools: &CheckedPools,
    structures: &[CheckedStructure],
    ends: &[CheckedPoolEnd],
) -> Result<(), SurfaceChannelError> {
    for (position, id) in pools.ids.iter().enumerate() {
        let attached = structures
            .iter()
            .any(|structure| structure.nodes.contains(&CheckedNode::Pool(position)))
            || ends.iter().any(|end| end.pool == position);
        if !attached {
            return Err(rejected(
                "pool-support",
                id,
                "joins no structure and no pool end: attach the pool or remove it".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{channel, declaration, link, nodes_setup, pond, pool, surface};
    use super::*;
    use crate::coupling::surface_channels::SurfaceChannelDefinition;
    use crate::coupling::surface_channels::structures::SurfaceChannelStructure;
    use crate::coupling::surface_channels::structures::scenario::{
        AREA, Tanks, TestResult, orifice_law, weir_law,
    };
    use crate::storage::{LevelPoolCurve, LevelPoolElement};
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Input message of a definition of resting tanks with `pools` and `structures`.
    fn rejection(
        pools: SurfaceChannelPools,
        structures: Vec<SurfaceChannelStructure>,
    ) -> TestResult<String> {
        let (setup, _) = Tanks::at(0.0, 0.0).parts()?;
        match SurfaceChannelDefinition::new(nodes_setup(setup, pools, structures)) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the definition admitted invalid pools".into()),
        }
    }

    #[test]
    fn each_pool_rejection_names_its_entry() -> TestResult {
        let spill = |to: &str, bed: f64| -> TestResult<_> {
            Ok(link(
                "spill",
                weir_law(1.0)?,
                DirectionPolicy::Bidirectional,
                StructureOperatingPoint::Passive,
                (pool(to), bed),
                (surface(), 0.0),
            ))
        };
        let ponds = || -> TestResult<_> { Ok(vec![pond("pond", 0.0, AREA, 2.0)?]) };
        let mut cases: Vec<(SurfaceChannelPools, Vec<_>, &str)> = Vec::new();
        cases.push((
            declaration(ponds()?, false, Vec::new()),
            vec![spill("other", 0.0)?],
            "structure-endpoint-identity",
        ));
        cases.push((
            declaration(ponds()?, false, Vec::new()),
            vec![spill("pond", -0.5)?],
            "unobstructed-pool-connection",
        ));
        let mut shifted = declaration(ponds()?, false, Vec::new());
        shifted.support[0].zero_volume_stage_m = 0.1;
        cases.push((shifted, vec![spill("pond", 0.0)?], "pool-support"));
        let mut twice = declaration(
            vec![pond("pond", 0.0, AREA, 2.0)?, pond("pond", 0.0, AREA, 2.0)?],
            false,
            Vec::new(),
        );
        twice.pools[1].storage_support_identity = "another".into();
        cases.push((twice, vec![spill("pond", 0.0)?], "duplicate-links"));
        let mut foreign = declaration(ponds()?, false, Vec::new());
        foreign.pools[0] = LevelPoolElement {
            curve: LevelPoolCurve::new(
                foreign.pools[0].curve.knots().to_vec(),
                "Another datum".into(),
                "Synthetic constant-area pond".into(),
            )?,
            ..foreign.pools[0].clone()
        };
        cases.push((
            foreign,
            vec![spill("pond", 0.0)?],
            "datum-or-gravity-mismatch",
        ));
        for (pools, structures, entry) in cases {
            let message = rejection(pools, structures)?;
            assert!(message.starts_with(entry), "{entry}: {message}");
        }
        Ok(())
    }

    #[test]
    fn a_pool_without_a_structure_beside_a_bank_link_is_unadmitted() -> TestResult {
        let message = rejection(
            declaration(vec![pond("pond", 0.0, AREA, 2.0)?], false, Vec::new()),
            vec![link(
                "culvert",
                orifice_law(0.1, 0.3)?,
                DirectionPolicy::Bidirectional,
                StructureOperatingPoint::Passive,
                (surface(), 0.0),
                (channel(), 0.0),
            )],
        )?;
        assert!(message.starts_with("pool-support"), "{message}");
        Ok(())
    }
}
