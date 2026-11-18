//! Immutable gate masks and logarithmic coefficient caches without crown rounding.
use numr::ops::BinaryOps;
use numr::ops::CorrectlyRoundedOps;
use numr::ops::ScalarOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::{StructureElement, StructureError, StructureLaw};
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorGateContext<R: Runtime<DType = DType>> {
    pub mask: Tensor<R>,
    pub sill: Tensor<R>,
    pub log_scale: Tensor<R>,
    pub log_root_gravity: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorGateContext<R> {
    pub(crate) fn new(
        elements: &[StructureElement],
        client: &R::Client,
        device: &R::Device,
    ) -> Result<Option<Self>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::UnaryOps<R>,
    {
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
        let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], device);
        let one = tensor(&vec![1.0; n])?;
        Ok(Some(Self {
            mask: tensor(&mask)?,
            sill: tensor(&sill)?,
            log_scale: client.add(
                &client.cr_log(&tensor(&coefficient)?)?,
                &client.cr_log(&tensor(&width)?)?,
            )?,
            log_root_gravity: client.mul_scalar(
                &client.add(
                    &client.cr_log(&client.mul_scalar(&one, 2.0)?)?,
                    &client.cr_log(&tensor(&gravity)?)?,
                )?,
                0.5,
            )?,
        }))
    }
}
