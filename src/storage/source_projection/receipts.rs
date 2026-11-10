//! Lazy complete source receipts without host numerical continuation.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

use super::demands::{CapVectors, CounterVectors};
use crate::storage::accounting::read;
use crate::storage::{LevelPoolError, LevelPoolSourceCounterReceipt, LevelPoolSourceReceipt};
use crate::tensor::level_pool::{TensorPoolContext, TensorPoolSourceStep};

pub(crate) fn receipts<R>(
    context: &TensorPoolContext<R>,
    source: Option<&TensorPoolSourceStep<R>>,
) -> Result<Vec<Option<LevelPoolSourceReceipt>>, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
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
