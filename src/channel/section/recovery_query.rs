use super::{CrossSectionError, CrossSectionProperties, CrossSectionQuery};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Immutable requested areas and admitted recovered geometry on the configured runtime.
#[derive(Debug)]
pub struct CrossSectionRecovery<R: Runtime<DType = DType>> {
    pub(super) requested_area: Tensor<R>,
    pub(super) geometry: CrossSectionQuery<R>,
}

/// Original requested area and independently admitted forward geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionRecoveredProperties {
    /// Original requested operand, in square metres.
    pub requested_area_m2: f64,
    /// Full geometric properties at the recovered stage.
    pub geometry: CrossSectionProperties,
}

impl<R: Runtime<DType = DType>> CrossSectionRecovery<R> {
    /// Explicitly copy requested areas and recovered geometry to the host in section order.
    pub fn properties(&self) -> Result<Vec<CrossSectionRecoveredProperties>, CrossSectionError> {
        let geometry = self.geometry.properties()?;
        if self.requested_area.shape() != [geometry.len()]
            || self.requested_area.dtype() != DType::F64
        {
            return Err(CrossSectionError::numerical(
                "requested area layout differs from ordered sections. Rebuild the recovery",
            ));
        }
        let requested = self.requested_area.contiguous()?.try_to_vec::<f64>()?;
        Ok(requested
            .into_iter()
            .zip(geometry)
            .map(
                |(requested_area_m2, geometry)| CrossSectionRecoveredProperties {
                    requested_area_m2,
                    geometry,
                },
            )
            .collect())
    }
}
