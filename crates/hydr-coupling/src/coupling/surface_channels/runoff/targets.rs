//! Destination targets of one runoff outlet and their normalized weights.

use super::admission::{RunoffTarget, rejected};
use super::{
    SurfaceChannelRunoffDestination, SurfaceChannelRunoffOutlet, SurfaceChannelRunoffShare,
};
use crate::coupling::surface_channels::bindings::channel_cells;
use crate::coupling::surface_channels::pools::CheckedPools;
use crate::coupling::surface_channels::{SurfaceChannelError, SurfaceChannelSetup};
use std::collections::{BTreeMap, BTreeSet};

/// Checked targets of `outlet` with weights normalized by their sum.
///
/// `reaches` maps reach identity to setup index. `pools` holds the admitted pools.
pub(super) fn targets(
    setup: &SurfaceChannelSetup,
    reaches: &BTreeMap<&str, usize>,
    pools: Option<&CheckedPools>,
    outlet: &SurfaceChannelRunoffOutlet,
) -> Result<Vec<(RunoffTarget, f64)>, SurfaceChannelError> {
    let id = outlet.outlet_id.as_str();
    match &outlet.destination {
        SurfaceChannelRunoffDestination::SurfaceCells { cells } => {
            let active = setup.surface.domain.active();
            shares(id, cells, |cell| {
                if active.get(cell).copied().unwrap_or(false) {
                    Ok(RunoffTarget::Surface { cell })
                } else {
                    Err(format!(
                        "surface cell={cell} is outside the grid or inactive: target active surface cells"
                    ))
                }
            })
        }
        SurfaceChannelRunoffDestination::ChannelCells { reach_id, cells } => {
            let Some(&reach) = reaches.get(reach_id.as_str()) else {
                return Err(rejected(
                    "runoff-destination",
                    id,
                    format!("reach_id={reach_id:?} names no setup reach: target a setup reach"),
                ));
            };
            let count = setup
                .reaches
                .get(reach)
                .map_or(0, |reach| channel_cells(&reach.definition).len());
            shares(id, cells, |cell| {
                if cell < count {
                    Ok(RunoffTarget::Channel { reach, cell })
                } else {
                    Err(format!(
                        "reach={reach_id:?} cell={cell} is out of range with {count} cells: target an existing reach cell"
                    ))
                }
            })
        }
        SurfaceChannelRunoffDestination::Pool { pool_id } => {
            let Some((position, _)) = pools.and_then(|pools| pools.find(pool_id)) else {
                return Err(rejected(
                    "runoff-destination",
                    id,
                    format!("pool_id={pool_id:?} names no coordinator pool: target a setup pool"),
                ));
            };
            Ok(vec![(RunoffTarget::Pool { position }, 1.0)])
        }
    }
}

/// Resolves each share through `target` and normalizes the weights by their sum.
///
/// Rejects an empty list, a repeated index, a weight that is nonfinite or not positive, and a
/// weight sum that overflows. A single target takes weight one exactly.
fn shares(
    id: &str,
    cells: &[SurfaceChannelRunoffShare],
    target: impl Fn(usize) -> Result<RunoffTarget, String>,
) -> Result<Vec<(RunoffTarget, f64)>, SurfaceChannelError> {
    if cells.is_empty() {
        return Err(rejected(
            "runoff-destination",
            id,
            "has no target cells: declare at least one cell",
        ));
    }
    let mut seen = BTreeSet::new();
    let mut sum = 0.0;
    let mut resolved = Vec::with_capacity(cells.len());
    for share in cells {
        let node =
            target(share.index).map_err(|detail| rejected("runoff-destination", id, detail))?;
        if !seen.insert(share.index) {
            return Err(rejected(
                "runoff-destination",
                id,
                format!("cell={} repeats: list each target cell once", share.index),
            ));
        }
        if !share.weight.is_finite() || share.weight <= 0.0 {
            return Err(rejected(
                "runoff-destination",
                id,
                format!(
                    "cell={} weight={}: supply a finite positive weight",
                    share.index, share.weight
                ),
            ));
        }
        sum += share.weight;
        resolved.push((node, share.weight));
    }
    if !sum.is_finite() {
        return Err(rejected(
            "runoff-destination",
            id,
            format!("weight sum={sum}: supply weights with a finite sum"),
        ));
    }
    Ok(resolved
        .into_iter()
        .map(|(node, weight)| (node, weight / sum))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{build, outlet, parts, rejection, share};
    use super::*;
    use crate::coupling::surface_channels::structures::scenario::{REACH, TestResult};

    /// The default case with its one outlet draining into `destination`.
    fn into(destination: SurfaceChannelRunoffDestination) -> TestResult<SurfaceChannelSetup> {
        let (mut setup, _) = parts()?;
        if let Some(runoff) = setup.runoff.as_mut() {
            runoff.outlets = vec![outlet(destination)];
        }
        Ok(setup)
    }

    fn reach_cells(cells: Vec<SurfaceChannelRunoffShare>) -> SurfaceChannelRunoffDestination {
        SurfaceChannelRunoffDestination::ChannelCells {
            reach_id: REACH.into(),
            cells,
        }
    }

    fn rejects(destination: SurfaceChannelRunoffDestination, detail: &str) -> TestResult {
        let message = rejection(into(destination)?)?;
        assert!(
            message.starts_with("runoff-destination") && message.contains(detail),
            "{detail}: {message}"
        );
        Ok(())
    }

    #[test]
    fn surface_targets_reject_cells_off_the_grid_and_repeats() -> TestResult {
        let surface = |cells| SurfaceChannelRunoffDestination::SurfaceCells { cells };
        rejects(surface(vec![share(1, 1.0)]), "outside the grid or inactive")?;
        rejects(surface(vec![share(0, 1.0), share(0, 1.0)]), "repeats")?;
        rejects(surface(Vec::new()), "no target cells")
    }

    #[test]
    fn channel_targets_reject_unknown_reaches_out_of_range_cells_and_repeats() -> TestResult {
        rejects(
            SurfaceChannelRunoffDestination::ChannelCells {
                reach_id: "elsewhere".into(),
                cells: vec![share(0, 1.0)],
            },
            "names no setup reach",
        )?;
        rejects(reach_cells(vec![share(1, 1.0)]), "out of range")?;
        rejects(reach_cells(vec![share(0, 1.0), share(0, 2.0)]), "repeats")
    }

    #[test]
    fn a_pool_target_must_name_a_coordinator_pool() -> TestResult {
        rejects(
            SurfaceChannelRunoffDestination::Pool {
                pool_id: "lake".into(),
            },
            "names no coordinator pool",
        )
    }

    #[test]
    fn weights_must_be_finite_and_positive() -> TestResult {
        for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            rejects(
                reach_cells(vec![share(0, weight)]),
                "finite positive weight",
            )?;
        }
        Ok(())
    }

    #[test]
    fn weights_normalize_by_their_sum() -> TestResult {
        let surface = SurfaceChannelRunoffDestination::SurfaceCells {
            cells: vec![share(0, 0.5)],
        };
        let batch = build((into(surface)?, parts()?.1))?;
        let Some(runoff) = &batch.definition().runoff else {
            return Err("missing checked runoff".into());
        };
        let weights: Vec<f64> = runoff.lanes.iter().map(|lane| lane.weight).collect();
        assert_eq!(weights, [1.0]);
        let shares = shares("split", &[share(0, 1.0), share(1, 3.0)], |cell| {
            Ok(RunoffTarget::Surface { cell })
        })?;
        let weights: Vec<f64> = shares.iter().map(|(_, weight)| *weight).collect();
        assert_eq!(weights, [0.25, 0.75]);
        Ok(())
    }
}
