//! Certified barrel depth tensors, their typed statuses and their explicit host projection.

use crate::structures::StructureError;
use crate::structures::numerics::culvert::BarrelDepths;
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
/// for `Q <= sqrt(g) B D^(3/2)`. Above a floor the open rise `D'` replaces `D`, depths run up
/// from the bed, and a circle or arc barrel ends its domain at `D' (1 - 2^-52)`.
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
    /// enclosure `[D (1 - 2^-53), D]`. Above a floor the enclosure is `[D' (1 - 2^-52), D']`.
    /// A box has no free-surface critical depth and no enclosure.
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

/// Outcome of a Manning normal-depth query, `Q = (1/n) A R^(2/3) sqrt(S)`, for one barrel.
///
/// The normal depth is the root on the rising limb of the conveyance `K = A^(5/3) / P^(2/3)`.
/// The search domain is `[D 2^-30, top]`, with `top = 0.93818 D` for a circle and `D` for a box.
/// Circle conveyance peaks at `s_K D`, `s_K = 0.9381812`, above the full-barrel conveyance,
/// and falls from there to the crown. Box conveyance rises up to its open-section limit at the
/// crown, which exceeds the full-barrel conveyance. Full-barrel flow is the caller's branch.
/// Above a floor the open rise `D'` replaces `D`, depths run up from the bed, and the
/// conveyance is `A^(5/3) / P_n^(2/3)` with the wall `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalDepthStatus {
    /// Zero discharge on a positive slope. The depth is exactly `[0, 0]`.
    Dry,
    /// Certified root inside the search domain.
    Solved,
    /// The root lies at or below the floor `D 2^-30`. The enclosure starts at 0.
    BelowFloor,
    /// The target lies near the top of the rising limb. A rising-limb root, when one exists,
    /// lies in the enclosure. The enclosure ends at `s_K D` rounded up for a circle or an arc
    /// barrel, with `s_K` the upper end of its conveyance peak enclosure, and at `D` for a box.
    NearCapacity,
    /// The target exceeds the peak conveyance of a circle or an arc barrel, or the box crown
    /// limit. No free-surface uniform flow exists and the row has no enclosure.
    AboveCapacity,
    /// The slope is zero or adverse. No normal depth exists and the row has no enclosure.
    NonPositiveSlope,
}

impl sealed::Sealed for NormalDepthStatus {
    fn from_code(code: f64) -> Option<Self> {
        [
            (codes::NORMAL_DRY, Self::Dry),
            (codes::NORMAL_SOLVED, Self::Solved),
            (codes::NORMAL_BELOW_FLOOR, Self::BelowFloor),
            (codes::NORMAL_NEAR_CAPACITY, Self::NearCapacity),
            (codes::NORMAL_ABOVE_CAPACITY, Self::AboveCapacity),
            (codes::NORMAL_NON_POSITIVE_SLOPE, Self::NonPositiveSlope),
        ]
        .into_iter()
        .find_map(|(value, status)| (value == code).then_some(status))
    }
}

impl BarrelDepthStatus for NormalDepthStatus {}

/// Certified depth per barrel row, as status codes and lower and upper F64 `[m]` tensors.
///
/// Every exact root that the status places in an enclosure lies inside its `[lower, upper]`
/// pair, in metres above the invert, or above the bed of a barrel with a floor. Rows without an
/// enclosure hold NaN endpoints.
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

impl<R: Runtime<DType = DType>, S: BarrelDepthStatus> CulvertBarrelDepth<R, S> {
    /// Wraps solver output whose codes decode as `S`.
    pub(crate) fn from_depths(depths: BarrelDepths<R>) -> Self {
        Self {
            status: depths.status,
            lower: depths.lower,
            upper: depths.upper,
            marker: PhantomData,
        }
    }

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
