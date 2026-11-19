//! Recent evaluations, keyed by the exact bits of their operands.
//!
//! Wave bounds are a pure function of the reach context, the cell fields, and the endpoint
//! operands. An operand set whose every bit matches a kept entry takes that entry's result, so
//! every bit agrees with a fresh evaluation. A coordinator repeats a converged reach's candidate
//! state over its fixed-point iterations. An accepted candidate can equal the next start state.
//!
//! Two parts of one evaluation read fewer operands, so they keep their own entries:
//!
//! - Surveyed point Riemann faces read the cell fields and the operands of the ends that a face
//!   Riemann problem solves. A coordinator iterates a pool head at a direct end while the start
//!   state stays fixed, so every iteration takes the start-state faces of the first.
//! - A surveyed scalar trace lane reads the cell fields and its own end's policy. A head that
//!   moves at one end leaves the other end's lane unchanged.

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

/// Kept speed entries. A scalar-endpoint trial evaluates a prefriction and a postfriction state.
pub(super) const SPEED_CAPACITY: usize = 2;

/// Kept point Riemann results: the fixed start state and the latest candidate.
pub(super) const RIEMANN_CAPACITY: usize = 2;

/// Kept trace lanes: both ends at the fixed start state and at the latest candidate.
pub(super) const LANE_CAPACITY: usize = 4;

/// Kind code of a direct end, whose face no point Riemann problem solves.
const DIRECT: u64 = 7;

/// Endpoint kinds, lengths, and value bits of every operand of one evaluation.
#[derive(PartialEq, Eq)]
pub(super) struct SpeedKey(Vec<u64>);

/// The most recent evaluations of one kind, least recently used first.
pub(super) struct Memo<E> {
    capacity: usize,
    entries: Mutex<VecDeque<(SpeedKey, Arc<E>)>>,
}

impl<E> Memo<E> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    /// The kept entry of `key` that `accept` admits. A found entry becomes the most recent.
    pub fn find(&self, key: &SpeedKey, accept: impl Fn(&E) -> bool) -> Option<Arc<E>> {
        // Entries hold only completed values, so a panic elsewhere leaves them valid.
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let position = entries
            .iter()
            .position(|(kept, entry)| kept == key && accept(entry))?;
        let found = entries.remove(position)?;
        let entry = Arc::clone(&found.1);
        entries.push_back(found);
        Some(entry)
    }

    /// Keeps `entry` under `key` in place of the least recently used entry, and returns it.
    pub fn keep(&self, key: SpeedKey, entry: E) -> Arc<E> {
        let entry = Arc::new(entry);
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.retain(|(kept, _)| *kept != key);
        if entries.len() == self.capacity {
            entries.pop_front();
        }
        entries.push_back((key, Arc::clone(&entry)));
        entry
    }
}

/// Recent wave-bound evaluations of one reach context.
pub(super) type SpeedMemo<R> = Memo<SpeedEntry<R>>;

/// One evaluation result.
pub(super) struct SpeedEntry<R: Runtime<DType = DType>> {
    pub speeds: CflSpeeds<R>,
    /// The certified point Riemann result of a surveyed reach.
    pub riemann: Option<Arc<PrismaticPointRiemannResult<R>>>,
    /// The direct end-face rates of a surveyed reach, with its certified scalar traces.
    pub frame: Option<Arc<DirectBoundaryFrame<R>>>,
}

impl<R: Runtime<DType = DType>> SpeedMemo<R> {
    /// The kept entry of `key` whose surveyed Riemann result is present when `riemann` holds.
    pub fn find_speeds(&self, key: &SpeedKey, riemann: bool) -> Option<Arc<SpeedEntry<R>>> {
        self.find(key, |entry| !riemann || entry.riemann.is_some())
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
        self.operand_key(fields, [left, right].map(boundary_operands))
    }

    /// Bits of every field and of the operands of each end that a face Riemann problem solves.
    ///
    /// A direct end contributes its kind alone: its face takes the policy's rates, and its
    /// exterior copies the end cell.
    pub(super) fn riemann_key(
        &self,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<SpeedKey, ChannelError> {
        self.operand_key(fields, [face_operands(left), face_operands(right)])
    }

    /// Bits of every field, the end index `endpoint`, and the operands of its `policy`.
    pub(super) fn lane_key(
        &self,
        fields: &Fields<R>,
        endpoint: usize,
        policy: &ChannelTensorBoundary<R>,
    ) -> Result<SpeedKey, ChannelError> {
        self.operand_key(
            fields,
            [(endpoint as u64, Vec::new()), boundary_operands(policy)],
        )
    }

    /// Bits of every field and of `ends`, two kind codes with their operands.
    fn operand_key(
        &self,
        fields: &Fields<R>,
        ends: [(u64, Vec<&Tensor<R>>); 2],
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
        for (kind, operands) in ends {
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

/// Kind code and operands of one end as a face Riemann problem reads it.
fn face_operands<R: Runtime<DType = DType>>(
    boundary: &ChannelTensorBoundary<R>,
) -> (u64, Vec<&Tensor<R>>) {
    if boundary.is_direct() {
        (DIRECT, Vec::new())
    } else {
        boundary_operands(boundary)
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
