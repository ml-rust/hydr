//! Quadrature levels: how many Simpson panels each lane runs.
//!
//! A march surveys every lane with the fewest panels of [`PANELS`]. The survey's remainder
//! bound `E` then picks each lane's level: the fewest panels whose predicted remainder
//! `E / LEVEL_RATIO^k` stays within the lane budget, a fraction of the profile's distance scale.
//! Lanes that need more than the most panels take the most. A level only sets the work. Every
//! level gives a certified enclosure.

use super::partition::LANES;
use super::quadrature::{LEVEL_RATIO, PANELS};
use super::trig::fill;
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::numerics::culvert::roots::RELATIVE_TOLERANCE;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Quadrature budget of a march.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Accuracy {
    /// Enclosures near the arithmetic floor of the march.
    Fine,
    /// Wide but certified enclosures for sign decisions far from a root.
    Coarse,
}

impl Accuracy {
    /// Remainder budget of one lane as a fraction of the distance scale.
    pub(crate) fn budget(self) -> f64 {
        match self {
            Self::Fine => FINE_BUDGET,
            Self::Coarse => COARSE_BUDGET,
        }
    }

    /// Relative width that stops the critical and normal depth solves.
    pub(crate) fn root_tolerance(self) -> f64 {
        match self {
            Self::Fine => RELATIVE_TOLERANCE,
            Self::Coarse => COARSE_ROOT_TOLERANCE,
        }
    }

    /// Relative width that stops the inlet depth solve.
    ///
    /// Below it the depth residual of a fine march sits in its own noise, where probes no
    /// longer contract the enclosure.
    pub(crate) fn depth_tolerance(self) -> f64 {
        match self {
            Self::Fine => FINE_DEPTH_TOLERANCE,
            Self::Coarse => COARSE_ROOT_TOLERANCE,
        }
    }
}

/// Coarse relative root width `2^-32`.
const COARSE_ROOT_TOLERANCE: f64 = 1.0 / 4_294_967_296.0;

/// Fine relative depth width `2^-46`.
const FINE_DEPTH_TOLERANCE: f64 = 1.0 / 70_368_744_177_664.0;

/// Fine lane budget `2^-50` of the distance scale.
const FINE_BUDGET: f64 = 1.0 / 1_125_899_906_842_624.0;

/// Coarse lane budget `2^-20` of the distance scale.
const COARSE_BUDGET: f64 = 1.0 / 1_048_576.0;

/// Level per lane from the survey remainder `remainder` and the lane budget `budget`.
pub(crate) fn level_for<R>(
    c: &R::Client,
    remainder: &Tensor<R>,
    budget: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut level = fill(budget, 0.0)?;
    let mut threshold = budget.clone();
    for _ in 1..PANELS.len() {
        level = c.add(&level, &c.gt(remainder, &threshold)?)?;
        threshold = c.mul_scalar(&threshold, LEVEL_RATIO)?;
    }
    Ok(level)
}

/// Highest lane level between nodes `low` and `high` of each row of `levels` `[p, LANES]`.
///
/// `low` and `high` are F64 node indices. A row with `high <= low` takes the level of lane
/// `low`, clamped to the last lane.
pub(crate) fn span_level<R>(
    c: &R::Client,
    levels: &Tensor<R>,
    low: &Tensor<R>,
    high: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let p = low.numel();
    let index: Vec<f64> = (0..LANES).map(|k| k as f64).collect();
    let index = Tensor::<R>::from_slice(&index, &[1, LANES], low.device())?
        .broadcast_to(&[p, LANES])?
        .contiguous()?;
    let wide = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value
            .reshape(&[p, 1])?
            .broadcast_to(&[p, LANES])?
            .contiguous()
    };
    let first = c.minimum(low, &fill(low, (LANES - 1) as f64)?)?;
    let end = c.maximum(high, &c.add_scalar(&first, 1.0)?)?;
    let inside = c.mul(&c.ge(&index, &wide(&first)?)?, &c.lt(&index, &wide(&end)?)?)?;
    c.max(&c.mul(levels, &inside)?, &[1], false)
}
