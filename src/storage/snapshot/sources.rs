//! Complete source counter import and original final source-input reconstruction.

use super::LevelPoolSnapshotData;
use crate::cpu::level_pool::{CpuPoolContext, CpuPoolSourceInput, CpuPoolSourceState};
use crate::storage::{LevelPoolError, LevelPoolSourceLedger, LevelPoolStepDiagnostics};

fn rows<'a>(
    context: &CpuPoolContext,
    data: &'a LevelPoolSnapshotData,
) -> Result<Option<Vec<&'a LevelPoolSourceLedger>>, LevelPoolError> {
    if context.source_policies.is_none() {
        if data.element_ledgers.iter().any(|l| l.sources.is_some()) {
            return Err(LevelPoolError::input(
                "exchange-only snapshot cannot include source counters",
            ));
        }
        return Ok(None);
    }
    data.element_ledgers
        .iter()
        .map(|l| {
            l.sources.as_ref().ok_or_else(|| {
                LevelPoolError::input("source-enabled snapshot requires complete source counters")
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub(super) fn state(
    context: &CpuPoolContext,
    data: &LevelPoolSnapshotData,
) -> Result<Option<CpuPoolSourceState>, LevelPoolError> {
    let Some(rows) = rows(context, data)? else {
        return Ok(None);
    };
    if data.accepted_steps == 0 {
        for r in &rows {
            if [
                r.cumulative_rain_volume_m3,
                r.rain_roundoff_bound_m3,
                r.cumulative_evaporation_volume_m3,
                r.evaporation_roundoff_bound_m3,
                r.cumulative_bottom_loss_volume_m3,
                r.bottom_loss_roundoff_bound_m3,
                r.cumulative_bottom_loss_demand_m3,
                r.bottom_loss_demand_roundoff_bound_m3,
            ]
            .iter()
            .any(|v| v.to_bits() != 0)
            {
                return Err(LevelPoolError::input(
                    "zero-history source counters and bounds must preserve exact zero",
                ));
            }
        }
    }
    Ok(Some(CpuPoolSourceState {
        rain: context.tensor(
            &rows
                .iter()
                .map(|r| r.cumulative_rain_volume_m3)
                .collect::<Vec<_>>(),
        )?,
        rain_bound: context.tensor(
            &rows
                .iter()
                .map(|r| r.rain_roundoff_bound_m3)
                .collect::<Vec<_>>(),
        )?,
        evaporation: context.tensor(
            &rows
                .iter()
                .map(|r| r.cumulative_evaporation_volume_m3)
                .collect::<Vec<_>>(),
        )?,
        evaporation_bound: context.tensor(
            &rows
                .iter()
                .map(|r| r.evaporation_roundoff_bound_m3)
                .collect::<Vec<_>>(),
        )?,
        bottom_loss: context.tensor(
            &rows
                .iter()
                .map(|r| r.cumulative_bottom_loss_volume_m3)
                .collect::<Vec<_>>(),
        )?,
        bottom_loss_bound: context.tensor(
            &rows
                .iter()
                .map(|r| r.bottom_loss_roundoff_bound_m3)
                .collect::<Vec<_>>(),
        )?,
        bottom_loss_demand: context.tensor(
            &rows
                .iter()
                .map(|r| r.cumulative_bottom_loss_demand_m3)
                .collect::<Vec<_>>(),
        )?,
        bottom_loss_demand_bound: context.tensor(
            &rows
                .iter()
                .map(|r| r.bottom_loss_demand_roundoff_bound_m3)
                .collect::<Vec<_>>(),
        )?,
    }))
}
pub(super) fn before(
    context: &CpuPoolContext,
    data: &LevelPoolSnapshotData,
    last: &LevelPoolStepDiagnostics,
) -> Result<Option<CpuPoolSourceState>, LevelPoolError> {
    let Some(rows) = rows(context, data)? else {
        if last.receipts.iter().any(|r| r.source.is_some()) {
            return Err(LevelPoolError::input(
                "exchange-only snapshot cannot include source receipts",
            ));
        }
        return Ok(None);
    };
    let n = rows.len();
    let rain: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].cumulative_rain_volume_m3, |s| {
                    s.counters.rain.before_m3
                })
        })
        .collect();
    let rain_bound: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].rain_roundoff_bound_m3, |s| {
                    s.counters.rain.bound_before_m3
                })
        })
        .collect();
    let evaporation: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].cumulative_evaporation_volume_m3, |s| {
                    s.counters.evaporation.before_m3
                })
        })
        .collect();
    let evaporation_bound: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].evaporation_roundoff_bound_m3, |s| {
                    s.counters.evaporation.bound_before_m3
                })
        })
        .collect();
    let bottom_loss: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].cumulative_bottom_loss_volume_m3, |s| {
                    s.counters.bottom_loss.before_m3
                })
        })
        .collect();
    let bottom_loss_bound: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].bottom_loss_roundoff_bound_m3, |s| {
                    s.counters.bottom_loss.bound_before_m3
                })
        })
        .collect();
    let bottom_loss_demand: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].cumulative_bottom_loss_demand_m3, |s| {
                    s.counters.bottom_loss_demand.before_m3
                })
        })
        .collect();
    let bottom_loss_demand_bound: Vec<_> = (0..n)
        .map(|i| {
            last.receipts[i]
                .source
                .as_ref()
                .map_or(rows[i].bottom_loss_demand_roundoff_bound_m3, |s| {
                    s.counters.bottom_loss_demand.bound_before_m3
                })
        })
        .collect();
    if data.accepted_steps == 1 {
        for values in [
            &rain,
            &rain_bound,
            &evaporation,
            &evaporation_bound,
            &bottom_loss,
            &bottom_loss_bound,
            &bottom_loss_demand,
            &bottom_loss_demand_bound,
        ] {
            if values.iter().any(|v| v.to_bits() != 0) {
                return Err(LevelPoolError::input(
                    "first source interval must continue exact zero source counters/bounds",
                ));
            }
        }
    }
    Ok(Some(CpuPoolSourceState {
        rain: context.tensor(&rain)?,
        rain_bound: context.tensor(&rain_bound)?,
        evaporation: context.tensor(&evaporation)?,
        evaporation_bound: context.tensor(&evaporation_bound)?,
        bottom_loss: context.tensor(&bottom_loss)?,
        bottom_loss_bound: context.tensor(&bottom_loss_bound)?,
        bottom_loss_demand: context.tensor(&bottom_loss_demand)?,
        bottom_loss_demand_bound: context.tensor(&bottom_loss_demand_bound)?,
    }))
}
pub(super) fn inputs(
    context: &CpuPoolContext,
    last: &LevelPoolStepDiagnostics,
) -> Result<Option<CpuPoolSourceInput>, LevelPoolError> {
    if last.receipts.iter().all(|r| r.source.is_none()) {
        return Ok(None);
    }
    if context.source_policies.is_none() {
        return Err(LevelPoolError::input(
            "source receipt requires matching immutable enabling policies",
        ));
    }
    let input = last
        .receipts
        .iter()
        .map(|r| crate::storage::LevelPoolSources {
            rain_rate_m_per_s: r.source.as_ref().map_or(0.0, |s| s.rain_rate_m_per_s),
            potential_evaporation_rate_m_per_s: r
                .source
                .as_ref()
                .map_or(0.0, |s| s.potential_evaporation_rate_m_per_s),
            bottom_loss_rate_m3_per_s: r
                .source
                .as_ref()
                .map_or(0.0, |s| s.bottom_loss_rate_m3_per_s),
        })
        .collect::<Vec<_>>();
    Ok(Some(CpuPoolSourceInput::new(context, &input)?))
}
