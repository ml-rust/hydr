//! Recent candidate wave-bound evaluations, keyed by the exact bits of their operands.
//!
//! Wave bounds are a pure function of the reach context, the cell fields, and the endpoint
//! operands. An operand set whose every bit matches a kept entry takes that entry's result, so
//! every bit agrees with a fresh evaluation. A coordinator repeats a converged reach's candidate
//! state over its fixed-point iterations. An accepted candidate can equal the next start state.

use super::{
    ChannelError, ChannelTensorBoundary, admission::Fields, boundary::DirectBoundaryFrame,
    cfl_speeds::CflSpeeds, context::Context, surveyed_points::PrismaticPointRiemannResult,
};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

/// Kept entries. A scalar-endpoint trial evaluates a prefriction and a postfriction state.
const CAPACITY: usize = 2;

/// Endpoint kinds, lengths, and value bits of every operand of one evaluation.
#[derive(PartialEq, Eq)]
pub(super) struct SpeedKey(Vec<u64>);

/// One evaluation result.
pub(super) struct SpeedEntry<R: Runtime<DType = DType>> {
    pub speeds: CflSpeeds<R>,
    /// The certified point Riemann result of a surveyed reach.
    pub riemann: Option<Arc<PrismaticPointRiemannResult<R>>>,
    /// The direct end-face rates of a surveyed reach, with its certified scalar traces.
    pub frame: Option<Arc<DirectBoundaryFrame<R>>>,
}

/// The most recent evaluations of one reach context.
pub(super) struct SpeedMemo<R: Runtime<DType = DType>> {
    entries: Mutex<VecDeque<(SpeedKey, Arc<SpeedEntry<R>>)>>,
}

impl<R: Runtime<DType = DType>> SpeedMemo<R> {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(CAPACITY)),
        }
    }

    /// The kept entry of `key` whose surveyed Riemann result is present when `riemann` holds.
    pub fn find(&self, key: &SpeedKey, riemann: bool) -> Option<Arc<SpeedEntry<R>>> {
        // Entries hold only completed values, so a panic elsewhere leaves them valid.
        let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries
            .iter()
            .find(|(kept, entry)| kept == key && (!riemann || entry.riemann.is_some()))
            .map(|(_, entry)| Arc::clone(entry))
    }

    /// Keeps `entry` under `key` in place of the oldest entry, and returns it.
    pub fn keep(&self, key: SpeedKey, entry: SpeedEntry<R>) -> Arc<SpeedEntry<R>> {
        let entry = Arc::new(entry);
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.retain(|(kept, _)| *kept != key);
        if entries.len() == CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, Arc::clone(&entry)));
        entry
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: ShapeOps<R>,
{
    /// Bits of every field and endpoint operand, read from the device in one copy.
    pub(super) fn speed_key(
        &self,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<SpeedKey, ChannelError> {
        let mut parts: Vec<&Tensor<R>> = vec![
            &fields.area,
            &fields.depth,
            &fields.velocity,
            &fields.discharge,
            &fields.area_point.high,
            &fields.area_point.low,
            &fields.discharge_point.high,
            &fields.discharge_point.low,
            &fields.area_selection_bound,
            &fields.discharge_selection_bound,
        ];
        let mut key = Vec::with_capacity(4);
        for boundary in [left, right] {
            let (kind, operands) = boundary_operands(boundary);
            key.push(kind);
            parts.extend(operands);
        }
        let mut flat = Vec::with_capacity(parts.len());
        for part in parts {
            key.push(part.numel() as u64);
            flat.push(part.flatten()?);
        }
        let flat: Vec<&Tensor<R>> = flat.iter().collect();
        let values = self.client.cat(&flat, 0)?.try_to_vec::<f64>()?;
        key.extend(values.iter().map(|value| value.to_bits()));
        Ok(SpeedKey(key))
    }
}

/// Kind code and tensor operands of one endpoint policy.
fn boundary_operands<R: Runtime<DType = DType>>(
    boundary: &ChannelTensorBoundary<R>,
) -> (u64, Vec<&Tensor<R>>) {
    match boundary {
        ChannelTensorBoundary::Wall => (0, Vec::new()),
        ChannelTensorBoundary::Exterior {
            depth_m,
            velocity_m_per_s,
        } => (1, vec![depth_m, velocity_m_per_s]),
        ChannelTensorBoundary::Stage { elevation_m } => (2, vec![elevation_m]),
        ChannelTensorBoundary::QuiescentReservoir {
            elevation_m,
            entrance_loss_coefficient,
        } => (3, vec![elevation_m, entrance_loss_coefficient]),
        ChannelTensorBoundary::Discharge { discharge_m3_per_s } => (4, vec![discharge_m3_per_s]),
        // A dry exterior holds one operand fewer, so it takes its own kind code.
        ChannelTensorBoundary::FluxPort(port) => {
            let kind = if port.exterior_invariant_m_per_s().is_some() {
                5
            } else {
                6
            };
            (kind, port.operands().into_iter().map(|(_, t)| t).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelTensorBoundary,
        cfl_speeds::{CflSpeeds, FaceSpeeds},
        surveyed_update::tests::{TestResult, cpu, moving_reach},
    };
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    type Reach = crate::channel::flow::ChannelReach<CpuRuntime>;

    /// Deep and shallow wet cells beside two dry cells, moving, without friction.
    const VOLUMES: [f64; 4] = [20.0, 10.0, 0.0, 0.0];
    const MOMENTA: [f64; 4] = [4.0, -2.0, 0.0, 0.0];

    fn reach(volumes: &[f64]) -> TestResult<Reach> {
        moving_reach::<CpuRuntime>(cpu(), volumes, &MOMENTA, &[0.0; 4])
    }

    fn bits(values: &[&Tensor<CpuRuntime>]) -> TestResult<Vec<u64>> {
        let mut out = Vec::new();
        for value in values {
            out.extend(value.try_to_vec::<f64>()?.iter().map(|v| v.to_bits()));
        }
        Ok(out)
    }

    fn face_bits(speeds: &FaceSpeeds<CpuRuntime>) -> TestResult<Vec<u64>> {
        bits(&[&speeds.rightward, &speeds.leftward])
    }

    fn speed_bits(speeds: &CflSpeeds<CpuRuntime>) -> TestResult<Vec<u64>> {
        let mut out = face_bits(&speeds.outer)?;
        let fronts = speeds
            .fronts
            .as_ref()
            .ok_or("a wet/dry face must yield fronts")?;
        for side in [&fronts.left, &fronts.right] {
            out.extend(bits(&[&side.rightward, &side.leftward, &side.wet])?);
        }
        Ok(out)
    }

    #[test]
    fn kept_speeds_and_flux_match_a_fresh_reach() -> TestResult {
        let wall = ChannelTensorBoundary::Wall;
        let kept = reach(&VOLUMES)?;
        let context = &kept.context;
        let fields = context.fields(&kept.accepted.volume, &kept.accepted.momentum)?;
        let first = context.candidate_speed(&fields, &wall, &wall)?;
        let again = context.candidate_speed(&fields, &wall, &wall)?;
        let flux = context.flux(&fields, &wall, &wall)?;
        // Both later evaluations take the first evaluation's storage.
        let origin = &first.outer.rightward;
        assert!(again.outer.rightward.shares_storage_with(origin));
        assert!(flux.speed.rightward.shares_storage_with(origin));
        let fresh = reach(&VOLUMES)?;
        let fresh_fields = fresh
            .context
            .fields(&fresh.accepted.volume, &fresh.accepted.momentum)?;
        let fresh_flux = fresh.context.flux(&fresh_fields, &wall, &wall)?;
        let fresh_speed = fresh.context.candidate_speed(&fresh_fields, &wall, &wall)?;
        assert_eq!(speed_bits(&again)?, speed_bits(&fresh_speed)?);
        assert_eq!(face_bits(&flux.speed)?, face_bits(&fresh_flux.speed)?);
        assert_eq!(
            bits(&[&flux.water, &flux.momentum])?,
            bits(&[&fresh_flux.water, &fresh_flux.momentum])?
        );
        Ok(())
    }

    #[test]
    fn one_changed_bit_evaluates_again() -> TestResult {
        let wall = ChannelTensorBoundary::Wall;
        let kept = reach(&VOLUMES)?;
        let context = &kept.context;
        let fields = context.fields(&kept.accepted.volume, &kept.accepted.momentum)?;
        let first = context.candidate_speed(&fields, &wall, &wall)?;
        let mut volumes = VOLUMES;
        volumes[1] = f64::from_bits(volumes[1].to_bits() + 1);
        let other = reach(&volumes)?;
        let moved = context.fields(&other.accepted.volume, &other.accepted.momentum)?;
        let second = context.candidate_speed(&moved, &wall, &wall)?;
        let origin = &first.outer.rightward;
        assert!(!second.outer.rightward.shares_storage_with(origin));
        // The first state stays kept beside the second.
        let third = context.candidate_speed(&fields, &wall, &wall)?;
        assert!(third.outer.rightward.shares_storage_with(origin));
        Ok(())
    }
}
