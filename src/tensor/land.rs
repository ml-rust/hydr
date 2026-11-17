//! Cached compact pervious support and persistent accepted soil memory.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::TensorDomain;
use crate::contract::EngineError;
use crate::soil::{
    LandCell, LandModel, ModifiedGreenAmptBatch, SoilCellState, SoilError, SoilState,
};
use crate::solver::SoilSourceReceipt;
use std::sync::Arc;

#[derive(Debug)]
struct Support<R: Runtime<DType = DType>> {
    land: LandModel,
    ids: Vec<usize>,
    indices: Tensor<R>,
    adjustment: crate::soil::admission::AdjustmentReduction,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorLand<R: Runtime<DType = DType>> {
    support: Arc<Support<R>>,
    batch: Option<ModifiedGreenAmptBatch<R>>,
}

#[derive(Clone)]
pub(crate) struct LandTrial<R: Runtime<DType = DType>> {
    pub next: TensorLand<R>,
    pub infiltration: Tensor<R>,
    pub adjustment: Tensor<R>,
    pub native: Option<[Tensor<R>; 7]>,
}

impl<R> TensorLand<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    pub(crate) fn new(
        client: R::Client,
        land: &LandModel,
        soil: &SoilState,
    ) -> Result<Self, EngineError> {
        let soil = SoilState::new(land, soil.cells().to_vec()).map_err(static_error)?;
        let mut ids = Vec::new();
        let mut parameters = Vec::new();
        let mut states = Vec::new();
        for (index, (model, state)) in land.cells().iter().zip(soil.cells()).enumerate() {
            if let (LandCell::ModifiedGreenAmpt(p), SoilCellState::ModifiedGreenAmpt(s)) =
                (model, state)
            {
                ids.push(index);
                parameters.push(*p);
                states.push(*s);
            }
        }
        let indices: Vec<i64> = ids
            .iter()
            .map(|&id| i64::try_from(id))
            .collect::<Result<_, _>>()
            .map_err(|_| EngineError::rejected("land support", "cell ID exceeds I64 indexing"))?;
        let indices = Tensor::<R>::from_slice(&indices, &[indices.len()], client.device())?;
        let adjustment = crate::soil::admission::AdjustmentReduction::new(land.cells().len())
            .map_err(static_error)?;
        let support = Arc::new(Support {
            land: land.clone(),
            ids,
            indices,
            adjustment,
        });
        let batch = if parameters.is_empty() {
            None
        } else {
            Some(
                ModifiedGreenAmptBatch::<R>::from_native_states(client, &parameters, &states)
                    .map_err(|e| map_error(&support.ids, e))?,
            )
        };
        Ok(Self { support, batch })
    }

    pub(crate) fn restore_trusted(&mut self, snapshot: &Self) -> Result<(), EngineError> {
        if self.support.ids != snapshot.support.ids
            || self.support.land.cells().len() != snapshot.support.land.cells().len()
            || !self
                .support
                .indices
                .device()
                .is_same(snapshot.support.indices.device())
        {
            return Err(EngineError::rejected(
                "trusted land snapshot",
                "support or device differs. Restore into identical land support on the original device",
            ));
        }
        let batch = match (&self.batch, &snapshot.batch) {
            (Some(target), Some(source)) => {
                let mut target = target.clone();
                target.restore_trusted(source).map_err(static_error)?;
                Some(target)
            }
            (None, None) => None,
            _ => {
                return Err(EngineError::rejected(
                    "trusted land snapshot",
                    "soil support differs. Restore into identical land definitions",
                ));
            }
        };
        self.batch = batch;
        Ok(())
    }

    pub(crate) fn trial(
        &self,
        cpu: &TensorDomain<R>,
        dt: f64,
        rain: &Tensor<R>,
        ponding: &Tensor<R>,
        rain_depth: &Tensor<R>,
        available: &Tensor<R>,
    ) -> Result<LandTrial<R>, EngineError> {
        if !self.support.indices.device().is_same(cpu.client.device()) {
            return Err(EngineError::rejected(
                "land support",
                "cached soil support belongs to another runtime client device",
            ));
        }
        let client = &cpu.client;
        let zero = &cpu.zero;
        let Some(batch) = &self.batch else {
            return Ok(LandTrial {
                next: self.clone(),
                infiltration: zero.clone(),
                adjustment: zero.clone(),
                native: None,
            });
        };
        let gather = |tensor| client.index_select(tensor, 0, &self.support.indices);
        let trial = batch
            .trial_surface_tensors(
                dt,
                &gather(rain)?,
                &gather(ponding)?,
                &gather(rain_depth)?,
                &gather(available)?,
            )
            .map_err(|e| map_error(&self.support.ids, e))?;
        let scatter = |tensor: &Tensor<R>| client.index_put(zero, 0, &self.support.indices, tensor);
        Ok(LandTrial {
            next: Self {
                support: Arc::clone(&self.support),
                batch: Some(trial.next),
            },
            infiltration: scatter(&trial.infiltration_depth)?,
            adjustment: scatter(&trial.adjustment_depth)?,
            native: Some([
                trial.native_rate,
                trial.native_available_depth,
                trial.native_infiltration_depth,
                trial.native_excess_depth,
                trial.native_roundoff_bound,
                trial.bridge_roundoff_bound,
                trial.total_roundoff_bound,
            ]),
        })
    }

    pub(crate) fn soil_state(&self) -> Result<SoilState, EngineError> {
        let mut cells = vec![SoilCellState::Impervious; self.support.land.cells().len()];
        if let Some(batch) = &self.batch {
            let states = batch
                .native_states()
                .map_err(|e| map_error(&self.support.ids, e))?;
            if states.len() != self.support.ids.len() {
                return Err(EngineError::rejected(
                    "soil output",
                    "compact state count differs from cached land support",
                ));
            }
            for (&cell, state) in self.support.ids.iter().zip(states) {
                cells[cell] = SoilCellState::ModifiedGreenAmpt(state);
            }
        }
        SoilState::new(&self.support.land, cells).map_err(static_error)
    }

    pub(crate) fn receipts(
        &self,
        native: Option<&[Tensor<R>; 7]>,
    ) -> Result<Vec<Option<SoilSourceReceipt>>, EngineError> {
        let mut receipts = vec![None; self.support.land.cells().len()];
        if let Some(native) = native {
            let arrays = native
                .iter()
                .map(|t| t.try_to_vec::<f64>())
                .collect::<Result<Vec<_>, _>>()?;
            if arrays.iter().any(|a| a.len() != self.support.ids.len()) {
                return Err(EngineError::rejected(
                    "soil receipt",
                    "tensor count differs from compact land support",
                ));
            }
            for (index, &cell) in self.support.ids.iter().enumerate() {
                receipts[cell] = Some(SoilSourceReceipt {
                    native_rate_ft_per_s: arrays[0][index],
                    native_available_depth_ft: arrays[1][index],
                    native_infiltration_depth_ft: arrays[2][index],
                    native_excess_depth_ft: arrays[3][index],
                    native_roundoff_bound_ft: arrays[4][index],
                    bridge_roundoff_bound_m: arrays[5][index],
                    total_roundoff_bound_m: arrays[6][index],
                });
            }
        }
        Ok(receipts)
    }

    pub(crate) fn adjustment_totals(
        &self,
        client: &R::Client,
        depths: &Tensor<R>,
        area: f64,
    ) -> Result<(f64, f64), EngineError> {
        self.support
            .adjustment
            .totals(client, depths, area)
            .map_err(|error| EngineError::Soil {
                cell: error.cell(),
                retryable: error.is_retryable(),
                reason: error.to_string(),
            })
    }
}

pub(crate) fn static_error(error: SoilError) -> EngineError {
    EngineError::Soil {
        cell: error.cell(),
        reason: error.to_string(),
        retryable: false,
    }
}

fn map_error(ids: &[usize], error: SoilError) -> EngineError {
    EngineError::Soil {
        cell: error.cell().and_then(|i| ids.get(i).copied()),
        retryable: error.is_retryable(),
        reason: error.to_string(),
    }
}
