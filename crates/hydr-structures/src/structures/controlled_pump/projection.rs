//! One explicit selected-operation projection shared by both public output forms.

use super::{ControlledPumpBatch, ControlledPumpOutput};
use crate::structures::{ControlLatch, StructureError, StructureOperatingPoint};
use numr::dtype::DType;
use numr::runtime::Runtime;

pub(super) fn outputs<R: Runtime<DType = DType>>(
    batch: &ControlledPumpBatch<R>,
) -> Result<Vec<ControlledPumpOutput>, StructureError>
where
    R::Client: numr::ops::ConditionalOps<R>,
{
    let selected = batch.selected()?;
    let n = batch.definitions().len();
    let running = crate::structures::projection::read(&selected.running, n)?;
    let speed = crate::structures::projection::read(&selected.speed, n)?;
    let high = crate::structures::projection::read(&selected.high, n)?;
    batch
        .definitions()
        .iter()
        .enumerate()
        .map(|(lane, response)| {
            let controller = batch.controls.definitions().get(lane).ok_or_else(|| {
                StructureError::numerical("controlled pump output lacks its original controller")
            })?;
            let element = batch
                .controls
                .context
                .slots
                .get(lane)
                .copied()
                .ok_or_else(|| {
                    StructureError::numerical(
                        "controlled pump output lacks its actual structure slot",
                    )
                })?;
            let operating_point = match running[lane] {
                0.0 => StructureOperatingPoint::PumpStoppedIsolated,
                1.0 => StructureOperatingPoint::PumpRunning {
                    speed_ratio: speed[lane],
                },
                _ => {
                    return Err(StructureError::numerical(
                        "controlled pump running selection is not binary",
                    )
                    .at(element));
                }
            };
            let latch = match high[lane] {
                0.0 => ControlLatch::Low,
                1.0 => ControlLatch::High,
                _ => {
                    return Err(StructureError::numerical(
                        "controlled pump latch selection is not binary",
                    )
                    .at(element));
                }
            };
            Ok(ControlledPumpOutput {
                element,
                controller_id: controller.controller_id.clone(),
                structure_id: response.structure_id.clone(),
                time_s: batch.time(),
                accepted_steps: batch.accepted_steps(),
                model: response.model,
                latch,
                operating_point,
            })
        })
        .collect()
}
