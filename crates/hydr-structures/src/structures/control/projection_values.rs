//! Explicit finite tensor values for public continuation packages.

use super::{ControlSample, ControlTimeExpansion, ControlTimerWitness};
use crate::structures::StructureError;
use crate::structures::tensor::control::{
    MAX_EXACT_COUNT, TensorControlContext, TensorControlSample, TensorControlTimer,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn read<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    tensor: &Tensor<R>,
) -> Result<Vec<f64>, StructureError> {
    crate::structures::projection::read(tensor, context.definitions.len())
}
pub(super) fn index(value: f64) -> Result<usize, StructureError> {
    if !value.is_finite()
        || value < 0.0
        || value > MAX_EXACT_COUNT as f64
        || value.fract() != 0.0
        || value as u128 > usize::MAX as u128
    {
        return Err(StructureError::numerical(
            "control counter is outside exact supported integer range",
        ));
    }
    Ok(value as usize)
}
pub(super) struct SampleValues {
    valid: Vec<f64>,
    observed: Vec<f64>,
    available: Vec<f64>,
    stage: Vec<f64>,
}
impl SampleValues {
    pub(super) fn new<R: Runtime<DType = DType>>(
        context: &TensorControlContext<R>,
        sample: &TensorControlSample<R>,
    ) -> Result<Self, StructureError> {
        Ok(Self {
            valid: read(context, &sample.valid)?,
            observed: read(context, &sample.observed)?,
            available: read(context, &sample.available)?,
            stage: read(context, &sample.stage)?,
        })
    }
    pub(super) fn at<R: Runtime<DType = DType>>(
        &self,
        context: &TensorControlContext<R>,
        lane: usize,
    ) -> Option<ControlSample> {
        if self.valid[lane] == 0.0 {
            return None;
        }
        let definition = &context.definitions[lane];
        Some(ControlSample {
            controller_id: definition.controller_id.clone(),
            signal_id: definition.signal.signal_id.clone(),
            support_id: definition.signal.support_id.clone(),
            datum: definition.signal.datum.clone(),
            observed_at_s: self.observed[lane],
            available_at_s: self.available[lane],
            stage_m: self.stage[lane],
        })
    }
}
pub(super) struct TimerValues {
    active: Vec<f64>,
    started: Vec<f64>,
    evaluated: Vec<f64>,
    delay: Vec<f64>,
    elapsed_high: Vec<f64>,
    elapsed_low: Vec<f64>,
    ready: Vec<f64>,
    allowed_high: Vec<f64>,
    allowed_low: Vec<f64>,
    overflow: Vec<f64>,
    within: Vec<f64>,
}
impl TimerValues {
    pub(super) fn new<R: Runtime<DType = DType>>(
        context: &TensorControlContext<R>,
        timer: &TensorControlTimer<R>,
    ) -> Result<Self, StructureError> {
        Ok(Self {
            active: read(context, &timer.active)?,
            started: read(context, &timer.started)?,
            evaluated: read(context, &timer.evaluated)?,
            delay: read(context, &timer.delay)?,
            elapsed_high: read(context, &timer.elapsed_high)?,
            elapsed_low: read(context, &timer.elapsed_low)?,
            ready: read(context, &timer.ready)?,
            allowed_high: read(context, &timer.allowed_high)?,
            allowed_low: read(context, &timer.allowed_low)?,
            overflow: read(context, &timer.overflow)?,
            within: read(context, &timer.within)?,
        })
    }
    pub(super) fn at(&self, lane: usize) -> Option<ControlTimerWitness> {
        if self.active[lane] == 0.0 {
            return None;
        }
        let overflow = self.overflow[lane] != 0.0;
        Some(ControlTimerWitness {
            started_at_s: self.started[lane],
            evaluated_at_s: self.evaluated[lane],
            delay_s: self.delay[lane],
            elapsed_high_s: self.elapsed_high[lane],
            elapsed_low_s: self.elapsed_low[lane],
            ready: self.ready[lane] != 0.0,
            allowed_elapsed: if overflow {
                None
            } else {
                Some(ControlTimeExpansion {
                    high_s: self.allowed_high[lane],
                    low_s: self.allowed_low[lane],
                })
            },
            allowed_elapsed_overflow: overflow,
            within_lateness: self.within[lane] != 0.0,
        })
    }
}
