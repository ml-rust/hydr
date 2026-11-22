//! Public outlet-control batch over compact `[m]` tensors.

use super::{CulvertOutletHeadwater, CulvertOutletJump, CulvertOutletRow};
use crate::structures::culvert::input::{check_depths, check_discharges};
use crate::structures::numerics::culvert::RowNames;
use crate::structures::numerics::culvert::outlet::{self, Accuracy, jump};
use crate::structures::tensor::culvert::{
    TensorBarrelContext, TensorOutletContext, reject_arc_barrels,
};
use crate::structures::{CulvertBarrel, StructureError};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedBisectionAlgorithms;
use std::fmt;
use std::sync::Arc;

/// Outlet-control rows, one barrel each, with certified headwater and jump queries.
///
/// Every bound is an enclosure: the exact value of the outlet-control model at each represented
/// input lies inside its `[lower, upper]` pair. CPU and CUDA give bit-identical bounds.
#[derive(Clone)]
pub struct CulvertOutletBatch<R: Runtime<DType = DType>> {
    rows: Arc<[CulvertOutletRow]>,
    barrels: Arc<[CulvertBarrel]>,
    barrel: Arc<TensorBarrelContext<R>>,
    unit: Arc<TensorBarrelContext<R>>,
    outlet: Arc<TensorOutletContext<R>>,
}

impl<R: Runtime<DType = DType>> fmt::Debug for CulvertOutletBatch<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CulvertOutletBatch")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime<DType = DType>> CulvertOutletBatch<R> {
    /// Uploads and admits the rows.
    ///
    /// Rejects an empty row list, a nonfinite or nonpositive length, Manning `n` or gravity, a
    /// nonfinite or negative entrance loss, a nonfinite slope, and an adverse barrel whose
    /// outlet invert sits more than `D/2` above its inlet invert. Each rejection names the row.
    /// Rejects a runtime without the barrel batch's F64 contracts.
    pub fn new(client: R::Client, rows: Vec<CulvertOutletRow>) -> Result<Self, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>,
    {
        if rows.is_empty() {
            return Err(StructureError::definition(
                "rows=[]: supply at least one culvert outlet row",
            ));
        }
        for (index, row) in rows.iter().enumerate() {
            row.check().map_err(|error| error.at(index))?;
        }
        let barrels: Arc<[CulvertBarrel]> = rows.iter().map(|row| row.barrel).collect();
        reject_arc_barrels(&barrels, "outlet control")?;
        let barrel = TensorBarrelContext::new(client, &barrels)?;
        let unit = barrel.unit_circles()?;
        let outlet = outlet::prepare(&barrel, &rows)?;
        Ok(Self {
            rows: rows.into(),
            barrels,
            barrel: Arc::new(barrel),
            unit: Arc::new(unit),
            outlet: Arc::new(outlet),
        })
    }

    /// Rows in order.
    pub fn rows(&self) -> &[CulvertOutletRow] {
        &self.rows
    }

    fn check(&self, discharge: &Tensor<R>, tailwater: &Tensor<R>) -> Result<(), StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>,
    {
        check_discharges(&self.barrel, "discharge_m3_per_s", discharge)?;
        check_depths(&self.barrel, "tailwater_m", tailwater)
    }

    /// Encloses the outlet-control headwater per row at one discharge and tailwater per barrel.
    ///
    /// `discharge_m3_per_s` holds per-barrel discharge magnitudes and `tailwater_m` the
    /// tailwater depth above the outlet invert. Both are F64 `[m]` tensors on the batch device.
    /// Rejects a negative or nonfinite value, naming its row. Rows whose march does not resolve
    /// report `Unresolved` with NaN bounds.
    pub fn headwater(
        &self,
        discharge_m3_per_s: &Tensor<R>,
        tailwater_m: &Tensor<R>,
    ) -> Result<CulvertOutletHeadwater<R>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedSimpsonAlgorithms<R>,
    {
        self.check(discharge_m3_per_s, tailwater_m)?;
        let march = outlet::march(
            RowNames::new(&self.barrels),
            &self.barrel,
            &self.unit,
            &self.outlet,
            discharge_m3_per_s,
            tailwater_m,
            Accuracy::Fine,
        )?;
        Ok(CulvertOutletHeadwater::from_headwater(march.headwater))
    }

    /// Classifies the hydraulic jump of every inlet-critical row and encloses its position.
    ///
    /// Takes the inputs of [`CulvertOutletBatch::headwater`] and runs the same march first. The
    /// result carries that march's headwater too.
    pub fn jump(
        &self,
        discharge_m3_per_s: &Tensor<R>,
        tailwater_m: &Tensor<R>,
    ) -> Result<CulvertOutletJump<R>, StructureError>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + CorrectlyRoundedOps<R>
            + RuntimeClient<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedSimpsonAlgorithms<R>,
    {
        self.check(discharge_m3_per_s, tailwater_m)?;
        let march = outlet::march(
            RowNames::new(&self.barrels),
            &self.barrel,
            &self.unit,
            &self.outlet,
            discharge_m3_per_s,
            tailwater_m,
            Accuracy::Fine,
        )?;
        let found = jump::jump(
            &self.barrel,
            &self.outlet,
            &march,
            discharge_m3_per_s,
            tailwater_m,
        )?;
        Ok(CulvertOutletJump::from_jump(
            found,
            CulvertOutletHeadwater::from_headwater(march.headwater),
        ))
    }
}
