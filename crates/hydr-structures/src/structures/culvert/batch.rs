//! Public barrel geometry batch over compact `[m]` tensors.

use super::input::{check_depths, check_range};
use super::{CulvertBarrel, CulvertBarrelGeometry};
use crate::structures::StructureError;
use crate::structures::numerics::culvert::section;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use std::fmt;
use std::sync::Arc;

/// Circular and box culvert barrels, one row each, with certified geometry queries.
///
/// Every quantity is an enclosure: the exact value at each represented depth lies inside its
/// `[lower, upper]` pair. CPU and CUDA give bit-identical bounds.
/// The free-surface domain is `[0, D)`. The crown `y = D` and every depth above it are full.
#[derive(Clone)]
pub struct CulvertBarrelBatch<R: Runtime<DType = DType>> {
    context: Arc<TensorBarrelContext<R>>,
}

impl<R: Runtime<DType = DType>> fmt::Debug for CulvertBarrelBatch<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CulvertBarrelBatch")
            .field("barrels", &self.context.barrels)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime<DType = DType>> CulvertBarrelBatch<R> {
    /// Uploads the barrels and encloses their full-state constants once.
    ///
    /// Rejects an empty barrel list and a runtime without F64 IEEE basic arithmetic,
    /// fused multiply-add, or square roots rounded to nearest.
    pub fn new(client: R::Client, barrels: Vec<CulvertBarrel>) -> Result<Self, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>,
    {
        Ok(Self {
            context: Arc::new(TensorBarrelContext::new(client, barrels)?),
        })
    }

    /// Barrels in row order.
    pub fn barrels(&self) -> &[CulvertBarrel] {
        &self.context.barrels
    }

    /// Encloses the geometry at one depth per barrel, in metres above the invert.
    ///
    /// `depth_m` is an F64 `[m]` tensor on the batch device. A depth at or above the rise is full.
    /// Rejects a negative or nonfinite depth, naming its row.
    /// Returns `Numerical` when a bound overflows F64, as the full-state moment can for huge depths.
    pub fn geometry(&self, depth_m: &Tensor<R>) -> Result<CulvertBarrelGeometry<R>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>,
    {
        let context = &*self.context;
        check_depths(context, "depth_m", depth_m)?;
        Ok(with_interval_arithmetic(
            &context.client,
            DType::F64,
            "culvert barrel geometry",
            |a| {
                Ok(CulvertBarrelGeometry::from_section(&section::at_depth(
                    a, context, depth_m,
                )?))
            },
        )?)
    }

    /// Encloses the geometry over every depth in `[lower_m, upper_m]` per barrel.
    ///
    /// Each range lies inside the free-surface domain: `0 <= lower_m <= upper_m < D`.
    /// Rejects a range that reaches the crown. Split such a range at the crown.
    /// The hydraulic radius bound divides the area range by the perimeter range, so it widens
    /// with the range.
    pub fn geometry_over(
        &self,
        lower_m: &Tensor<R>,
        upper_m: &Tensor<R>,
    ) -> Result<CulvertBarrelGeometry<R>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>,
    {
        let context = &*self.context;
        check_range(context, lower_m, upper_m)?;
        Ok(with_interval_arithmetic(
            &context.client,
            DType::F64,
            "culvert barrel range geometry",
            |a| {
                let section = section::over_depth(a, context, lower_m, upper_m)?;
                Ok(CulvertBarrelGeometry::from_section(&section))
            },
        )?)
    }

    /// Encloses the full-state geometry at the crown `y = D` of every barrel.
    ///
    /// Area, perimeter and radius hold for every full depth. The moment is `A_full D / 2`.
    pub fn full_geometry(&self) -> Result<CulvertBarrelGeometry<R>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>,
    {
        let context = &*self.context;
        Ok(with_interval_arithmetic(
            &context.client,
            DType::F64,
            "culvert barrel full geometry",
            |a| {
                let section = section::at_depth(a, context, &context.rise)?;
                Ok(CulvertBarrelGeometry::from_section(&section))
            },
        )?)
    }
}
