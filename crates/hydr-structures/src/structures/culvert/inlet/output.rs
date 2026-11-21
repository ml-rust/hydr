//! Certified inlet headwater tensors, their band per row and their explicit host projection.

use crate::structures::StructureError;
use crate::structures::numerics::culvert::inlet::InletHeadwater;
use crate::structures::tensor::codes;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Inlet-control band of one row, from `X = Ku Q / (A D^0.5)`.
///
/// The limit discharges are enclosures. A discharge whose `X` may lie on either side of a limit
/// reports that limit, and its headwater encloses both band values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InletRegime {
    /// `X <= 3.5`. Unsubmerged form 1 or form 2.
    Unsubmerged,
    /// `X` may lie on either side of 3.5.
    UnsubmergedLimit,
    /// `3.5 < X < 4`. Linear headwater between the two limit headwaters.
    Transition,
    /// `X` may lie on either side of 4.
    SubmergedLimit,
    /// `X >= 4`. Submerged orifice form.
    Submerged,
}

impl InletRegime {
    fn from_code(code: f64) -> Option<Self> {
        [
            (codes::INLET_UNSUBMERGED, Self::Unsubmerged),
            (codes::INLET_UNSUBMERGED_LIMIT, Self::UnsubmergedLimit),
            (codes::INLET_TRANSITION, Self::Transition),
            (codes::INLET_SUBMERGED_LIMIT, Self::SubmergedLimit),
            (codes::INLET_SUBMERGED, Self::Submerged),
        ]
        .into_iter()
        .find_map(|(value, regime)| (value == code).then_some(regime))
    }
}

/// Certified inlet-control headwater per row, as band codes and lower and upper F64 `[m]`
/// tensors in metres above the inlet invert.
pub struct CulvertInletHeadwater<R: Runtime<DType = DType>> {
    status: Tensor<R>,
    lower: Tensor<R>,
    upper: Tensor<R>,
}

/// One projected row of a [`CulvertInletHeadwater`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertInletHeadwaterRow {
    /// Band of the row.
    pub regime: InletRegime,
    /// `(lower, upper)` headwater in metres above the inlet invert.
    pub headwater_m: (f64, f64),
}

impl<R: Runtime<DType = DType>> CulvertInletHeadwater<R> {
    pub(crate) fn from_headwater(headwater: InletHeadwater<R>) -> Self {
        Self {
            status: headwater.status,
            lower: headwater.lower,
            upper: headwater.upper,
        }
    }

    /// Lower and upper headwater in metres above the inlet invert.
    pub fn headwater_m(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.lower, &self.upper)
    }

    /// Copies every row to the host.
    pub fn project(&self) -> Result<Vec<CulvertInletHeadwaterRow>, StructureError> {
        let status = self.status.try_to_vec::<f64>()?;
        let lower = self.lower.try_to_vec::<f64>()?;
        let upper = self.upper.try_to_vec::<f64>()?;
        status
            .iter()
            .zip(lower.iter().zip(&upper))
            .enumerate()
            .map(|(row, (&code, (&low, &high)))| {
                let regime = InletRegime::from_code(code).ok_or_else(|| {
                    StructureError::numerical(format!(
                        "inlet regime code={code}: report this row, the code is not an inlet regime"
                    ))
                    .at(row)
                })?;
                Ok(CulvertInletHeadwaterRow {
                    regime,
                    headwater_m: (low, high),
                })
            })
            .collect()
    }
}
