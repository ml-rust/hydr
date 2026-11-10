//! Compact admitted actual pump operations selected directly from accepted sampled latches.

use crate::cpu::CpuTensor;

#[derive(Clone, Debug)]
pub(crate) struct CpuControlledPumpSelection {
    pub running: CpuTensor,
    pub speed: CpuTensor,
    pub power_of_two: CpuTensor,
    pub high: CpuTensor,
}
