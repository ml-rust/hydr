//! Public inlet-control batch over compact `[m]` tensors.

use super::{CulvertInletHeadwater, CulvertInletRow};
use crate::structures::culvert::input::check_discharges;
use crate::structures::numerics::culvert::inlet::{anchors, evaluate};
use crate::structures::tensor::culvert::{TensorBarrelContext, TensorInletContext};
use crate::structures::{CulvertBarrel, StructureError};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::CertifiedBisectionAlgorithms;
use std::fmt;
use std::sync::Arc;

/// HDS-5 inlet-control rows, one barrel and inlet each, with a certified headwater query.
///
/// The headwater is an enclosure: the exact `HW_i(Q)` at each represented discharge lies inside
/// its `[lower, upper]` pair. Admission certifies that each row's headwater strictly rises with
/// discharge, across the transition too.
#[derive(Clone)]
pub struct CulvertInletBatch<R: Runtime<DType = DType>> {
    rows: Arc<[CulvertInletRow]>,
    barrels: Arc<[CulvertBarrel]>,
    barrel: Arc<TensorBarrelContext<R>>,
    unit: Arc<TensorBarrelContext<R>>,
    inlet: Arc<TensorInletContext<R>>,
}

impl<R: Runtime<DType = DType>> fmt::Debug for CulvertInletBatch<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CulvertInletBatch")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime<DType = DType>> CulvertInletBatch<R> {
    /// Uploads the rows, encloses each transition anchor and admits each row.
    ///
    /// Rejects an empty row list, a nonfinite slope, a nonfinite or nonpositive gravity, and a
    /// coded inlet on a barrel of another shape. Rejects a row whose submerged headwater at
    /// `X = 4` does not certifiably exceed its unsubmerged headwater at `X = 3.5`, and a form-1
    /// row whose critical depth at `X = 3.5` is not below the crown. Each rejection names the
    /// row and its inlet. Rejects a runtime without the barrel batch's F64 contracts.
    pub fn new(client: R::Client, rows: Vec<CulvertInletRow>) -> Result<Self, StructureError>
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
        if rows.is_empty() {
            return Err(StructureError::definition(
                "rows=[]: supply at least one culvert inlet row",
            ));
        }
        for (index, row) in rows.iter().enumerate() {
            row.check().map_err(|error| error.at(index))?;
        }
        let barrels: Arc<[CulvertBarrel]> = rows.iter().map(|row| row.barrel).collect();
        let barrel = TensorBarrelContext::new(client, &barrels)?;
        let unit = barrel.unit_circles()?;
        let inlet = anchors::prepare(&barrels, &barrel, &unit, &rows)?;
        Ok(Self {
            rows: rows.into(),
            barrels,
            barrel: Arc::new(barrel),
            unit: Arc::new(unit),
            inlet: Arc::new(inlet),
        })
    }

    /// Rows in order.
    pub fn rows(&self) -> &[CulvertInletRow] {
        &self.rows
    }

    /// Encloses the inlet-control headwater per row at one discharge per barrel.
    ///
    /// `discharge_m3_per_s` is an F64 `[m]` tensor of per-barrel discharge magnitudes on the
    /// batch device. Rejects a negative or nonfinite discharge, naming its row.
    /// Returns `Numerical` when a headwater bound overflows F64.
    pub fn headwater(
        &self,
        discharge_m3_per_s: &Tensor<R>,
    ) -> Result<CulvertInletHeadwater<R>, StructureError>
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
        let barrel = &*self.barrel;
        check_discharges(barrel, "discharge_m3_per_s", discharge_m3_per_s)?;
        let headwater = evaluate(
            &self.barrels,
            barrel,
            &self.unit,
            &self.inlet,
            discharge_m3_per_s,
        )?;
        Ok(CulvertInletHeadwater::from_headwater(headwater))
    }
}
