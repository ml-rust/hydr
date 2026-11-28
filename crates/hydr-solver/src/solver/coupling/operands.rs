//! Coordinator exchange operands over a fixed set of linked surface cells.

use super::super::Solver;
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Unique active surface cells that exchange water with a coordinator.
///
/// Holds at least one cell. Order follows the coordinator, and every operand lane uses it.
#[doc(hidden)]
#[derive(Clone)]
pub struct SurfaceLinks<R: Runtime<DType = DType>> {
    pub(in crate::solver) cells: Arc<[usize]>,
    pub(in crate::solver) ids: Tensor<R>,
    pub(in crate::solver) cell_count: usize,
}

impl<R: Runtime<DType = DType>> SurfaceLinks<R> {
    /// Returns the linked global row-major cell identities in lane order.
    pub fn cells(&self) -> &[usize] {
        &self.cells
    }

    /// Returns the number of linked cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Returns false: a link set always holds at least one cell.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Rejects links built for a grid with a different cell count.
    pub(in crate::solver) fn check_grid(&self, cell_count: usize) -> Result<(), EngineError> {
        if self.cell_count == cell_count {
            return Ok(());
        }
        Err(EngineError::rejected(
            "surface links",
            format!(
                "links cover {} cells, engine grid has {cell_count}: build the links from this engine",
                self.cell_count
            ),
        ))
    }
}

/// Per-linked-cell exchange volumes over one interval, as device tensors of shape `[links]`.
///
/// The coordinator limits both volumes before the surface completes the interval.
#[doc(hidden)]
#[derive(Clone)]
pub struct ExchangeSource<R: Runtime<DType = DType>> {
    pub(in crate::solver) links: SurfaceLinks<R>,
    pub(in crate::solver) inflow_volume_m3: Tensor<R>,
    pub(in crate::solver) withdrawal_volume_m3: Tensor<R>,
    pub(in crate::solver) inflow_velocity_xy: [Tensor<R>; 2],
}

impl<R: Runtime<DType = DType>> ExchangeSource<R> {
    /// Builds exchange operands over `links`.
    ///
    /// `inflow_volume_m3` and `withdrawal_volume_m3` hold m³ and must be finite and at least zero.
    /// `inflow_velocity_xy` holds the x/y velocity in m/s that inflowing water carries.
    /// Completion checks the values on the device.
    ///
    /// # Errors
    /// Rejects an operand whose shape is not `[links.len()]` or whose dtype is not F64.
    pub fn new(
        links: &SurfaceLinks<R>,
        inflow_volume_m3: Tensor<R>,
        withdrawal_volume_m3: Tensor<R>,
        inflow_velocity_xy: [Tensor<R>; 2],
    ) -> Result<Self, EngineError> {
        let [velocity_x, velocity_y] = &inflow_velocity_xy;
        for (name, tensor) in [
            ("inflow_volume_m3", &inflow_volume_m3),
            ("withdrawal_volume_m3", &withdrawal_volume_m3),
            ("inflow_velocity_x", velocity_x),
            ("inflow_velocity_y", velocity_y),
        ] {
            if tensor.shape() != [links.len()] || tensor.dtype() != DType::F64 {
                return Err(EngineError::rejected(
                    "surface exchange operands",
                    format!(
                        "{name}: shape={:?}, dtype={:?}, links={}: supply one F64 lane per linked cell",
                        tensor.shape(),
                        tensor.dtype(),
                        links.len()
                    ),
                ));
            }
        }
        Ok(Self {
            links: links.clone(),
            inflow_volume_m3,
            withdrawal_volume_m3,
            inflow_velocity_xy,
        })
    }

    /// Returns the linked cells these operands cover.
    pub fn links(&self) -> &SurfaceLinks<R> {
        &self.links
    }
}

impl<R: Runtime<DType = DType>> Solver<R> {
    /// Returns the forcing window this engine holds, after every continuation.
    ///
    /// The window starts at the interval a checkpoint retains and ends at the schedule end.
    #[doc(hidden)]
    pub fn coupling_forcing(&self) -> &crate::contract::ForcingSchedule {
        &self.definition.forcing
    }

    /// Returns an engine with this accepted state that shares the definition and device context.
    ///
    /// A coordinator keeps it as the start frame of its last interval.
    #[doc(hidden)]
    pub fn coupling_clone(&self) -> Self {
        Self {
            definition: Arc::clone(&self.definition),
            accepted: self.accepted.clone(),
            context: Arc::clone(&self.context),
            coastal: self.coastal,
            subgrid: self.subgrid.clone(),
        }
    }

    /// Returns the represented storage of `initial` in m³, as construction from it records it.
    ///
    /// Checks the flow against this engine's domain and the soil against its land first.
    /// A coordinator compares the result with a package's `ledger.initial_storage`.
    ///
    /// # Errors
    /// Rejects flow or soil support that differs from this engine, and invalid depths.
    #[doc(hidden)]
    pub fn coupling_epoch_storage(
        &self,
        initial: &crate::solver::InitialState,
    ) -> Result<f64, EngineError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let domain = &self.definition.domain;
        let flow = crate::model::FlowState::new(domain, initial.flow.cells().to_vec())?;
        crate::soil::SoilState::new(&self.definition.land, initial.soil.cells().to_vec())
            .map_err(super::super::tensor::land::static_error)?;
        let state =
            super::super::tensor::TensorState::<R>::from_flow(self.context.client.device(), &flow)?;
        super::super::tensor::land_sources::checked_volume(
            &self.context.client,
            &state.h,
            domain.grid().cell_area(),
        )
    }

    /// Builds a link set over `cells` for this engine's grid.
    ///
    /// # Errors
    /// Rejects an empty set, a repeated cell, a cell outside the grid, or an inactive cell.
    #[doc(hidden)]
    pub fn coupling_links(&self, cells: &[usize]) -> Result<SurfaceLinks<R>, EngineError> {
        let domain = &self.definition.domain;
        let cell_count = domain.grid().cell_count();
        let reject = |reason: String| EngineError::rejected("surface links", reason);
        if cells.is_empty() {
            return Err(reject(
                "no linked cells: pass no exchange instead of an empty link set".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        let mut ids = Vec::with_capacity(cells.len());
        for &cell in cells {
            if cell >= cell_count || !domain.active()[cell] {
                return Err(reject(format!(
                    "cell={cell}, cells={cell_count}: link only active cells inside the grid"
                )));
            }
            if !seen.insert(cell) {
                return Err(reject(format!("cell={cell} repeats: link each cell once")));
            }
            ids.push(
                i64::try_from(cell)
                    .map_err(|_| reject(format!("cell={cell} exceeds I64 indexing")))?,
            );
        }
        let ids = Tensor::<R>::from_slice(&ids, &[ids.len()], self.context.client.device())?;
        Ok(SurfaceLinks {
            cells: cells.into(),
            ids,
            cell_count,
        })
    }
}
