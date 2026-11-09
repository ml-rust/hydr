//! Primitive source preparation followed by the shared conservative drainage core.

use super::check::finish;
use super::receipt::CpuReservoirStep;
use super::{algebra, drainage};
use crate::cpu::CpuTensor;
use crate::cpu::reservoir_storage::{CpuReservoirContext, CpuReservoirState};
use crate::runoff::ReservoirError;

pub(crate) fn advance(
    context: &CpuReservoirContext,
    state: &CpuReservoirState,
    dt: f64,
    incoming: &CpuTensor,
) -> Result<CpuReservoirStep, ReservoirError> {
    let prepared = algebra::prepare(context, &state.depth, incoming, dt)?;
    let drained = drainage::drain(&prepared.hydraulic)?;
    finish(context, prepared, &state.depth, incoming, drained)
}
