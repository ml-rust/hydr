//! Exact original held-head intervals and immutable signed-volume proposals.

use super::{
    HeldHeadInterval, HeldHeadVolumeOutput, StructureError, StructureInputErrorKind,
    StructureProposal,
};
use crate::cpu::CpuTensor;

/// Opaque signed Q*dt proposals with original support and exact held interval.
#[derive(Clone, Debug)]
pub struct HeldHeadProposal {
    proposal: StructureProposal,
    interval: HeldHeadInterval,
    volumes: CpuTensor,
}
impl StructureProposal {
    /// Multiply original signed discharge by one exact held duration, without storage advancement.
    pub fn held_volumes(
        &self,
        interval: HeldHeadInterval,
    ) -> Result<HeldHeadProposal, StructureError> {
        let dt = interval.end - interval.start;
        if !interval.start.is_finite()
            || interval.start < 0.0
            || !interval.end.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "held interval={interval:?},dt={dt}: supply finite nonnegative start and finite positive original duration"
                ),
            ));
        }
        let volumes = crate::numerics::structures::held::volumes(
            &self.inner.context,
            &self.inner.result.discharge,
            dt,
        )?;
        Ok(HeldHeadProposal {
            proposal: self.clone(),
            interval,
            volumes,
        })
    }
}
impl HeldHeadProposal {
    /// Exact original elapsed interval, without implicit rescaling or trajectory integration.
    pub fn interval(&self) -> HeldHeadInterval {
        self.interval
    }
    /// Project original instantaneous proposals and their original signed held-volume products.
    pub fn outputs(&self) -> Result<Vec<HeldHeadVolumeOutput>, StructureError> {
        let output = self.proposal.outputs()?;
        let volumes = super::projection::read(&self.volumes, output.len())?;
        Ok(output
            .into_iter()
            .zip(volumes)
            .map(|(proposal, volume_m3)| HeldHeadVolumeOutput {
                interval: self.interval,
                proposal,
                volume_m3,
            })
            .collect())
    }
}
