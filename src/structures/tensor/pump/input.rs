//! Explicit operation admission and IEEE power-of-two proof metadata.
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::tensor::TensorStructureContext;
use crate::structures::tensor::operations::CheckedOperations;
use crate::structures::{StructureError, StructureOperatingPoint};
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorPumpInput<R: Runtime<DType = DType>> {
    pub running: Tensor<R>,
    pub isolated: Tensor<R>,
    pub speed: Tensor<R>,
    pub power_of_two: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorPumpInput<R> {
    pub(crate) fn new(
        context: &TensorStructureContext<R>,
        operations: &CheckedOperations<'_, R>,
    ) -> Result<Option<Self>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        if context.pump.is_none() {
            return Ok(None);
        }
        let n = operations.len();
        let (mut running, mut isolated, mut speed, mut power) =
            (vec![0.0; n], vec![0.0; n], vec![1.0; n], vec![1.0; n]);
        for index in 0..n {
            match operations.supplied(index) {
                Some(StructureOperatingPoint::PumpStoppedIsolated) => isolated[index] = 1.0,
                Some(StructureOperatingPoint::PumpRunning { speed_ratio }) => {
                    running[index] = 1.0;
                    speed[index] = speed_ratio;
                    power[index] = if super::super::power_metadata::power_of_two(speed_ratio) {
                        1.0
                    } else {
                        0.0
                    };
                }
                Some(
                    StructureOperatingPoint::Passive | StructureOperatingPoint::GateOpening { .. },
                )
                | None => {}
            }
        }
        let mut input = Self {
            running: context.tensor(&running)?,
            isolated: context.tensor(&isolated)?,
            speed: context.tensor(&speed)?,
            power_of_two: context.tensor(&power)?,
        };
        if let Some(derived) = operations.derived_pump() {
            let c = &context.client;
            input.running = c.where_cond(derived.mask, derived.running, &input.running)?;
            input.isolated = c.where_cond(derived.mask, derived.isolated, &input.isolated)?;
            input.speed = c.where_cond(derived.mask, derived.speed, &input.speed)?;
            input.power_of_two =
                c.where_cond(derived.mask, derived.power_of_two, &input.power_of_two)?;
        }
        Ok(Some(input))
    }
}
