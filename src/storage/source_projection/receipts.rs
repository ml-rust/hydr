//! Lazy complete source receipts without host numerical continuation.

use super::demands::{CapVectors, CounterVectors};
use crate::cpu::level_pool::{CpuPoolContext, CpuPoolSourceStep};
use crate::storage::accounting::read;
use crate::storage::{LevelPoolError, LevelPoolSourceCounterReceipt, LevelPoolSourceReceipt};

pub(crate) fn receipts(
    context: &CpuPoolContext,
    source: Option<&CpuPoolSourceStep>,
) -> Result<Vec<Option<LevelPoolSourceReceipt>>, LevelPoolError> {
    let n = context.elements.len();
    let Some(s) = source else {
        return Ok(vec![None; n]);
    };
    let frozen_area_m2 = read(&s.frozen_area_m2, n)?;
    let rain_rate_m_per_s = read(&s.rain_rate_m_per_s, n)?;
    let potential_evaporation_rate_m_per_s = read(&s.potential_evaporation_rate_m_per_s, n)?;
    let bottom_loss_rate_m3_per_s = read(&s.bottom_loss_rate_m3_per_s, n)?;
    let rain_rate_volume_m3_per_s = read(&s.rain_rate_volume_m3_per_s, n)?;
    let rain_volume_m3 = read(&s.rain_volume_m3, n)?;
    let potential_evaporation_rate_volume_m3_per_s =
        read(&s.potential_evaporation_rate_volume_m3_per_s, n)?;
    let potential_evaporation_volume_m3 = read(&s.potential_evaporation_volume_m3, n)?;
    let after_rain_volume_m3 = read(&s.after_rain_volume_m3, n)?;
    let after_bottom_loss_volume_m3 = read(&s.after_bottom_loss_volume_m3, n)?;
    let post_source_volume_m3 = read(&s.post_source_volume_m3, n)?;
    let rain_witness_m3 = read(&s.rain_witness_m3, n)?;
    let bottom_loss_witness_m3 = read(&s.bottom_loss_witness_m3, n)?;
    let evaporation_witness_m3 = read(&s.evaporation_witness_m3, n)?;
    let post_source_reconstructed_volume_m3 = read(&s.post_source_reconstructed_volume_m3, n)?;
    let post_source_inverse_residual_m3 = read(&s.post_source_inverse_residual_m3, n)?;
    let post_source_stage_m = read(&s.post_source_stage_m, n)?;
    let active = read(&s.active, n)?;
    let segments = s.segment.try_to_vec::<i64>()?;
    if segments.len() != n {
        return Err(LevelPoolError::numerical(format!(
            "source segment count={}: require {n} ordered elements",
            segments.len()
        )));
    }
    let bottom_loss_cap = CapVectors::new(&s.bottom_loss_cap, n)?;
    let evaporation_cap = CapVectors::new(&s.evaporation_cap, n)?;
    let rain = CounterVectors::new(&s.rain_counter, n)?;
    let evaporation = CounterVectors::new(&s.evaporation_counter, n)?;
    let bottom_loss = CounterVectors::new(&s.bottom_loss_counter, n)?;
    let bottom_loss_demand = CounterVectors::new(&s.bottom_loss_demand_counter, n)?;
    (0..n)
        .map(|i| {
            if active[i] == 0.0 {
                return Ok(None);
            };
            let segment = usize::try_from(segments[i])
                .map_err(|_| LevelPoolError::numerical("selected source segment is negative"))?;
            Ok(Some(LevelPoolSourceReceipt {
                segment,
                frozen_area_m2: frozen_area_m2[i],
                rain_rate_m_per_s: rain_rate_m_per_s[i],
                potential_evaporation_rate_m_per_s: potential_evaporation_rate_m_per_s[i],
                bottom_loss_rate_m3_per_s: bottom_loss_rate_m3_per_s[i],
                rain_rate_volume_m3_per_s: rain_rate_volume_m3_per_s[i],
                rain_volume_m3: rain_volume_m3[i],
                potential_evaporation_rate_volume_m3_per_s:
                    potential_evaporation_rate_volume_m3_per_s[i],
                potential_evaporation_volume_m3: potential_evaporation_volume_m3[i],
                after_rain_volume_m3: after_rain_volume_m3[i],
                after_bottom_loss_volume_m3: after_bottom_loss_volume_m3[i],
                post_source_volume_m3: post_source_volume_m3[i],
                rain_witness_m3: rain_witness_m3[i],
                bottom_loss_witness_m3: bottom_loss_witness_m3[i],
                evaporation_witness_m3: evaporation_witness_m3[i],
                post_source_reconstructed_volume_m3: post_source_reconstructed_volume_m3[i],
                post_source_inverse_residual_m3: post_source_inverse_residual_m3[i],
                post_source_stage_m: if post_source_volume_m3[i] == 0.0 {
                    None
                } else {
                    Some(post_source_stage_m[i])
                },
                bottom_loss_cap: bottom_loss_cap.at(i),
                evaporation_cap: evaporation_cap.at(i),
                counters: LevelPoolSourceCounterReceipt {
                    rain: rain.at(i),
                    evaporation: evaporation.at(i),
                    bottom_loss: bottom_loss.at(i),
                    bottom_loss_demand: bottom_loss_demand.at(i),
                },
            }))
        })
        .collect()
}
