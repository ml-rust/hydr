//! Immutable gate masks and logarithmic coefficient caches without crown rounding.

use crate::cpu::CpuTensor;
use crate::structures::{StructureElement, StructureError, StructureLaw};
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice};

#[derive(Debug)]
pub(crate) struct CpuGateContext {
    pub mask: CpuTensor,
    pub sill: CpuTensor,
    pub log_scale: CpuTensor,
    pub log_root_gravity: CpuTensor,
}
impl CpuGateContext {
    pub(crate) fn new(
        elements: &[StructureElement],
        client: &CpuClient,
        device: &CpuDevice,
    ) -> Result<Option<Self>, StructureError> {
        if !elements
            .iter()
            .any(|element| matches!(element.law, StructureLaw::HydrostaticRectangularGate(_)))
        {
            return Ok(None);
        }
        let n = elements.len();
        let (mut mask, mut sill, mut width, mut coefficient, mut gravity) = (
            vec![0.0; n],
            vec![0.0; n],
            vec![1.0; n],
            vec![1.0; n],
            vec![1.0; n],
        );
        for (index, element) in elements.iter().enumerate() {
            if let StructureLaw::HydrostaticRectangularGate(parameters) = &element.law {
                mask[index] = 1.0;
                sill[index] = parameters.sill_m();
                width[index] = parameters.width_m();
                coefficient[index] = parameters.discharge_coefficient();
                gravity[index] = parameters.gravity_m_per_s2();
            }
        }
        let tensor = |values: &[f64]| CpuTensor::from_slice(values, &[values.len()], device);
        let one = tensor(&vec![1.0; n])?;
        Ok(Some(Self {
            mask: tensor(&mask)?,
            sill: tensor(&sill)?,
            log_scale: client.add(
                &client.log(&tensor(&coefficient)?)?,
                &client.log(&tensor(&width)?)?,
            )?,
            log_root_gravity: client.mul_scalar(
                &client.add(
                    &client.log(&client.mul_scalar(&one, 2.0)?)?,
                    &client.log(&tensor(&gravity)?)?,
                )?,
                0.5,
            )?,
        }))
    }
}
