//! Explicit interval inputs copied once at the caller boundary.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::TensorSubcatchmentContext;
use crate::runoff::{Subarea, SubcatchmentError, SubcatchmentInput};

#[derive(Clone, Debug)]
pub(crate) struct TensorSubcatchmentInput<R: Runtime<DType = DType>> {
    pub rain: Tensor<R>,
    pub evaporation: Tensor<R>,
    pub external: Tensor<R>,
}
impl<R> TensorSubcatchmentInput<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn new(
        context: &TensorSubcatchmentContext<R>,
        inputs: &[SubcatchmentInput],
    ) -> Result<Self, SubcatchmentError> {
        if inputs.len() != context.definitions.len() {
            return Err(SubcatchmentError::input(format!(
                "inputs={}: require {} catchments",
                inputs.len(),
                context.definitions.len()
            )));
        }
        let mut rain = Vec::with_capacity(inputs.len() * 3);
        let mut evaporation = Vec::with_capacity(inputs.len() * 3);
        let mut external = Vec::with_capacity(inputs.len() * 3);
        for (i, input) in inputs.iter().enumerate() {
            if !input.rain_rate_m_per_s.is_finite()
                || input.rain_rate_m_per_s < 0.0
                || !input.potential_evaporation_m_per_s.is_finite()
                || input.potential_evaporation_m_per_s < 0.0
            {
                return Err(SubcatchmentError::input(format!(
                    "rain_rate_m_per_s={}, potential_evaporation_m_per_s={}: supply finite nonnegative rates",
                    input.rain_rate_m_per_s,
                    input.potential_evaporation_m_per_s,
                ))
                .at_catchment(i));
            }
            for area in Subarea::ALL {
                let volume = input.external_runon_volumes_m3[area.index()];
                if !volume.is_finite()
                    || volume < 0.0
                    || (volume > 0.0 && context.definitions[i].parameters.area_m2(area) == 0.0)
                {
                    return Err(SubcatchmentError::input(format!(
                        "external_runon_volumes_m3[{}]={volume}, area_m2={}: supply finite nonnegative volume over positive support",
                        area.index(),
                        context.definitions[i].parameters.area_m2(area),
                    ))
                    .at(i, Some(area)));
                }
                rain.push(input.rain_rate_m_per_s);
                evaporation.push(input.potential_evaporation_m_per_s);
                external.push(volume);
            }
        }
        Ok(Self {
            rain: context.tensor(&rain)?,
            evaporation: context.tensor(&evaporation)?,
            external: context.tensor(&external)?,
        })
    }
}
