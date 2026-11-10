//! Shared receiver bridge and compact native-soil witness reconstruction.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::phase::from_sources;
use crate::numerics::subcatchment::TensorSubcatchmentSoilStep;
use crate::numerics::subcatchment::bridge::Bridge;
use crate::runoff::{SubareaReceipt, SubcatchmentError};
use crate::tensor::subcatchment::TensorSubareaGroup;

pub(super) fn bridge<R>(
    group: &TensorSubareaGroup<R>,
    sources: &[&SubareaReceipt],
) -> Result<Bridge<R>, SubcatchmentError>
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
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let tensor = |select| from_sources(group, sources, select);
    Ok(Bridge {
        received_depth: tensor(|r| r.bridge.received_depth_m)?,
        received_rate: tensor(|r| r.bridge.received_rate_m_per_s)?,
        total_rate: tensor(|r| r.bridge.total_rate_m_per_s)?,
        input_depth: tensor(|r| r.bridge.input_depth_m)?,
        available_depth: tensor(|r| r.bridge.available_depth_m)?,
        rain_depth: tensor(|r| r.bridge.rain_depth_m)?,
        rain_only_available: tensor(|r| r.bridge.rain_only_available_m)?,
        depth_increment: tensor(|r| r.bridge.depth_increment_m)?,
        represented_received_volume: tensor(|r| r.bridge.represented_received_volume_m3)?,
        residual: tensor(|r| r.bridge.residual_m3)?,
        residual_bound: tensor(|r| r.bridge.residual_bound_m3)?,
        uncertainty: tensor(|r| r.bridge.uncertainty_m3)?,
    })
}

pub(super) fn soil<R>(
    group: &TensorSubareaGroup<R>,
    sources: &[&SubareaReceipt],
) -> Result<Option<TensorSubcatchmentSoilStep<R>>, SubcatchmentError>
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
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    if group.soil_slots.is_empty() {
        return Ok(None);
    }
    let compact: Vec<_> = group
        .soil_slots
        .iter()
        .map(|&lane| {
            sources.get(lane).copied().ok_or_else(|| {
                SubcatchmentError::input("checkpoint native-soil role exceeds phase source support")
            })
        })
        .collect::<Result<_, _>>()?;
    let proofs: Vec<_> = compact
        .iter()
        .map(|source| {
            source.soil.as_ref().ok_or_else(|| {
                SubcatchmentError::input(
                    "checkpoint pervious source lacks complete native soil evidence",
                )
                .at(source.catchment, Some(source.subarea))
            })
        })
        .collect::<Result<_, _>>()?;
    let tensor = |select| from_sources(group, &compact, select);
    let proof = |select: fn(&crate::soil::SoilSourceReceipt) -> f64| {
        let values: Vec<_> = proofs.iter().map(|&proof| select(proof)).collect();
        group
            .reservoir
            .tensor(&values)
            .map_err(|e| group.map_reservoir_error(e))
    };
    Ok(Some(TensorSubcatchmentSoilStep {
        infiltration_depth: tensor(|r| r.infiltration_depth_m)?,
        adjustment_depth: tensor(|r| r.adjustment_depth_m)?,
        native_rate: proof(|s| s.native_rate_ft_per_s)?,
        native_available_depth: proof(|s| s.native_available_depth_ft)?,
        native_infiltration_depth: proof(|s| s.native_infiltration_depth_ft)?,
        native_excess_depth: proof(|s| s.native_excess_depth_ft)?,
        native_roundoff_bound: proof(|s| s.native_roundoff_bound_ft)?,
        bridge_roundoff_bound: proof(|s| s.bridge_roundoff_bound_m)?,
        total_roundoff_bound: proof(|s| s.total_roundoff_bound_m)?,
    }))
}
