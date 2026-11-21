//! Certified barrel depth tensors, their typed statuses and their explicit host projection.

use crate::structures::StructureError;
use crate::structures::numerics::culvert::critical::CriticalDepths;
use crate::structures::tensor::codes;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::fmt::Debug;
use std::marker::PhantomData;

mod sealed {
    /// Decodes a per-row status code. Implemented only by the barrel depth status enums.
    pub trait Sealed: Sized {
        fn from_code(code: f64) -> Option<Self>;
    }
}

/// Typed status of one row of a [`CulvertBarrelDepth`].
pub trait BarrelDepthStatus: sealed::Sealed + Copy + Debug + PartialEq {}

/// Outcome of a critical-depth query, `Q^2 T = g A^3`, for one barrel.
///
/// The search domain is `[D 2^-30, top]`, with `top = D (1 - 2^-53)` for a circle and `D` for
/// a box. A circle has one critical depth below the crown for every `Q > 0`. A box has one only
/// for `Q <= sqrt(g) B D^(3/2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CriticalDepthStatus {
    /// Zero discharge. The depth is exactly `[0, 0]`.
    Dry,
    /// Certified root inside the search domain.
    Solved,
    /// The root lies at or below the floor `D 2^-30`. The enclosure starts at 0.
    BelowFloor,
    /// The residual at the domain top contains zero. A circle root lies in the enclosure, which
    /// ends at `D`. A box root lies in it when `Q` does not exceed the crown limit.
    AtCrown,
    /// Certified above the domain top. A circle root lies within `2^-53 D` of the crown, in the
    /// enclosure `[D (1 - 2^-53), D]`. A box has no free-surface critical depth and no enclosure.
    AboveCrown,
}

impl sealed::Sealed for CriticalDepthStatus {
    fn from_code(code: f64) -> Option<Self> {
        [
            (codes::CRITICAL_DRY, Self::Dry),
            (codes::CRITICAL_SOLVED, Self::Solved),
            (codes::CRITICAL_BELOW_FLOOR, Self::BelowFloor),
            (codes::CRITICAL_AT_CROWN, Self::AtCrown),
            (codes::CRITICAL_ABOVE_CROWN, Self::AboveCrown),
        ]
        .into_iter()
        .find_map(|(value, status)| (value == code).then_some(status))
    }
}

impl BarrelDepthStatus for CriticalDepthStatus {}

/// Certified depth per barrel row, as status codes and lower and upper F64 `[m]` tensors.
///
/// Every exact root that the status places in an enclosure lies inside its `[lower, upper]`
/// pair, in metres above the invert. Rows without an enclosure hold NaN endpoints.
pub struct CulvertBarrelDepth<R: Runtime<DType = DType>, S: BarrelDepthStatus> {
    status: Tensor<R>,
    lower: Tensor<R>,
    upper: Tensor<R>,
    marker: PhantomData<S>,
}

/// One projected row of a [`CulvertBarrelDepth`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertBarrelDepthRow<S: BarrelDepthStatus> {
    /// Outcome of the row.
    pub status: S,
    /// `(lower, upper)` depth in metres, or `None` when the status carries no enclosure.
    pub depth_m: Option<(f64, f64)>,
}

impl<R: Runtime<DType = DType>> CulvertBarrelDepth<R, CriticalDepthStatus> {
    pub(crate) fn from_critical(depths: CriticalDepths<R>) -> Self {
        Self {
            status: depths.status,
            lower: depths.lower,
            upper: depths.upper,
            marker: PhantomData,
        }
    }
}

impl<R: Runtime<DType = DType>, S: BarrelDepthStatus> CulvertBarrelDepth<R, S> {
    /// Lower and upper depth in metres. Rows without an enclosure hold NaN.
    pub fn depth_m(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.lower, &self.upper)
    }

    /// Copies every row to the host.
    pub fn project(&self) -> Result<Vec<CulvertBarrelDepthRow<S>>, StructureError> {
        let status = self.status.try_to_vec::<f64>()?;
        let lower = self.lower.try_to_vec::<f64>()?;
        let upper = self.upper.try_to_vec::<f64>()?;
        status
            .iter()
            .zip(lower.iter().zip(&upper))
            .enumerate()
            .map(|(row, (&code, (&low, &high)))| {
                let status = <S as sealed::Sealed>::from_code(code).ok_or_else(|| {
                    StructureError::numerical(format!(
                        "depth status code={code}: report this row, the code is not a depth status"
                    ))
                    .at(row)
                })?;
                let depth_m = (!low.is_nan()).then_some((low, high));
                Ok(CulvertBarrelDepthRow { status, depth_m })
            })
            .collect()
    }
}
