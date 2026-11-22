//! Public barrel geometry batch over compact `[m]` tensors.

use super::input::{check_depths, check_discharges, check_gravity, check_manning, check_range};
use super::{
    CriticalDepthStatus, CulvertBarrel, CulvertBarrelDepth, CulvertBarrelGeometry,
    NormalDepthStatus,
};
use crate::structures::StructureError;
use crate::structures::numerics::culvert::{RowNames, critical, normal, section};
use crate::structures::tensor::culvert::{TensorBarrelContext, reject_arc_barrels};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::optimize::CertifiedBisectionAlgorithms;
use std::fmt;
use std::sync::Arc;

/// Culvert barrels, one row each, with certified geometry queries.
///
/// Every quantity is an enclosure: the exact value at each represented depth lies inside its
/// `[lower, upper]` pair. CPU and CUDA give bit-identical bounds.
/// The free-surface domain is `[0, D)`. The crown `y = D` and every depth above it are full.
/// Geometry covers every shape. Critical and normal depth cover circular and box barrels only.
#[derive(Clone)]
pub struct CulvertBarrelBatch<R: Runtime<DType = DType>> {
    barrels: Arc<[CulvertBarrel]>,
    context: Arc<TensorBarrelContext<R>>,
    /// Unit-diameter circles of the same width, for the depth residuals in `s = y/D`.
    unit: Arc<TensorBarrelContext<R>>,
}

impl<R: Runtime<DType = DType>> fmt::Debug for CulvertBarrelBatch<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CulvertBarrelBatch")
            .field("barrels", &self.barrels)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime<DType = DType>> CulvertBarrelBatch<R> {
    /// Uploads the barrels and encloses their full-state constants once.
    ///
    /// Rejects an empty barrel list and a runtime without F64 IEEE basic arithmetic,
    /// fused multiply-add, correctly rounded elementary functions, or square roots rounded
    /// to nearest.
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
        let context = TensorBarrelContext::new(client, &barrels)?;
        let unit = context.unit_circles()?;
        Ok(Self {
            barrels: barrels.into(),
            context: Arc::new(context),
            unit: Arc::new(unit),
        })
    }

    /// Barrels in row order.
    pub fn barrels(&self) -> &[CulvertBarrel] {
        &self.barrels
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
        check_range(&self.barrels, context, lower_m, upper_m)?;
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
    /// Area, perimeter and radius hold for every full depth. The moment is `A_full D / 2` for a
    /// circle and a box, and the crown moment `M(D)` for an arc barrel.
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

    /// Encloses the critical depth `Q^2 T = g A^3` of every barrel, in metres above the invert.
    ///
    /// `discharge_m3_per_s` is an F64 `[m]` tensor of per-barrel discharge magnitudes on the
    /// batch device. [`CriticalDepthStatus`] states what each row's enclosure holds.
    /// Rejects a negative or nonfinite discharge, naming its row, a nonfinite or nonpositive
    /// gravity, and a batch holding an elliptical, pipe-arch or flat-floor arch barrel.
    pub fn critical_depth(
        &self,
        discharge_m3_per_s: &Tensor<R>,
        gravity_m_per_s2: f64,
    ) -> Result<CulvertBarrelDepth<R, CriticalDepthStatus>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>
            + CertifiedBisectionAlgorithms<R>,
    {
        let context = &*self.context;
        reject_arc_barrels(&self.barrels, "critical depth")?;
        check_gravity(gravity_m_per_s2)?;
        check_discharges(context, "discharge_m3_per_s", discharge_m3_per_s)?;
        let gravity = context.constant(gravity_m_per_s2)?;
        let depths = critical::solve(
            RowNames::new(&self.barrels),
            context,
            &self.unit,
            discharge_m3_per_s,
            &gravity,
        )?;
        Ok(CulvertBarrelDepth::from_depths(depths))
    }

    /// Encloses the Manning normal depth `Q = (1/n) A R^(2/3) sqrt(S)` of every barrel, in metres
    /// above the invert.
    ///
    /// `discharge_m3_per_s` holds per-barrel discharge magnitudes, `manning_n` the roughness in
    /// `s/m^(1/3)` and `slope` the barrel slope, positive downstream. All are F64 `[m]` tensors on
    /// the batch device. The normal depth is the root on the rising limb of the conveyance.
    /// [`NormalDepthStatus`] states what each row's enclosure holds. A zero or adverse slope
    /// reports `NonPositiveSlope` whatever the discharge.
    /// Rejects a negative or nonfinite discharge, a nonpositive or nonfinite `n` and a nonfinite
    /// slope, naming the row, and a batch holding an elliptical, pipe-arch or flat-floor arch
    /// barrel.
    pub fn normal_depth(
        &self,
        discharge_m3_per_s: &Tensor<R>,
        manning_n: &Tensor<R>,
        slope: &Tensor<R>,
    ) -> Result<CulvertBarrelDepth<R, NormalDepthStatus>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>
            + CertifiedBisectionAlgorithms<R>,
    {
        let context = &*self.context;
        reject_arc_barrels(&self.barrels, "normal depth")?;
        check_manning(
            context,
            [
                ("discharge_m3_per_s", discharge_m3_per_s),
                ("manning_n", manning_n),
                ("slope", slope),
            ],
        )?;
        let depths = normal::solve(
            RowNames::new(&self.barrels),
            context,
            &self.unit,
            discharge_m3_per_s,
            manning_n,
            slope,
        )?;
        Ok(CulvertBarrelDepth::from_depths(depths))
    }
}
