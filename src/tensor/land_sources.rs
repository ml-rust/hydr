//! Atomic tensor source order and accepted-only source receipt projections.

use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::land::{LandTrial, TensorLand};
use super::{TensorDomain, TensorSources, TensorState};
use crate::solver::{EngineError, SolverConfig, SourceReceipt};
use numr::dtype::DType;

#[derive(Clone)]
pub(crate) struct SourceTrial<R: Runtime<DType = DType>> {
    pub state: TensorState<R>,
    pub soil: LandTrial<R>,
    pub rain: Tensor<R>,
    pub infiltration: Tensor<R>,
    pub evaporation: Tensor<R>,
    pub withdrawal: Tensor<R>,
    pub adjustment: Tensor<R>,
    pub rain_volume: f64,
    pub infiltration_volume: f64,
    pub evaporation_volume: f64,
    pub withdrawal_volume: f64,
    pub adjustment_volume: f64,
    pub adjustment_charge: f64,
}

pub(crate) struct SourceInputs<'a, R: Runtime<DType = DType>> {
    pub original: &'a TensorState<R>,
    pub land: &'a TensorLand<R>,
    pub forcing: &'a TensorSources<R>,
}

pub(crate) fn calculate<R>(
    cpu: &TensorDomain<R>,
    inputs: SourceInputs<'_, R>,
    area: f64,
    dt: f64,
    charged: f64,
    config: &SolverConfig,
) -> Result<SourceTrial<R>, EngineError>
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
    let client = &cpu.client;
    let zero = &cpu.zero;
    let SourceInputs {
        original,
        land,
        forcing: source,
    } = inputs;
    let one = client.add_scalar(zero, 1.0)?;
    let rain = client.mul_scalar(&source.rain, dt)?;
    let withdrawal = client.mul_scalar(&source.withdrawal, dt)?;
    let available = client.add(&original.h, &rain)?;
    check_depth(client, &rain, "rain increment")?;
    check_depth(client, &available, "post-rain depth")?;
    let underflow = client.maximum(
        &client.mul(&client.gt(&source.rain, zero)?, &client.eq(&rain, zero)?)?,
        &client.mul(
            &client.gt(&source.withdrawal, zero)?,
            &client.eq(&withdrawal, zero)?,
        )?,
    )?;
    if any(client, &underflow)? {
        return Err(rejected(
            "positive rain/withdrawal increment underflows: rescale interval",
        ));
    }
    let soil = land.trial(cpu, dt, &source.rain, &original.h, &rain, &available)?;
    let infiltration = soil.infiltration.clone();
    let adjustment = soil.adjustment.clone();
    let (adjustment_volume, adjustment_charge) =
        land.adjustment_totals(client, &adjustment, area)?;
    let prospective = crate::soil::admission::add_adjustment_charge(charged, adjustment_charge)
        .map_err(|error| EngineError::Soil {
            cell: error.cell(),
            retryable: error.is_retryable(),
            reason: error.to_string(),
        })?;
    if prospective > config.max_cumulative_water_adjustment_m3 {
        return Err(rejected(format!(
            "adjustment charge={prospective} exceeds budget={}: reduce interval or correct physical scales",
            config.max_cumulative_water_adjustment_m3
        )));
    }
    let corrected = client.gt(&adjustment, zero)?;
    let after_soil =
        client.where_cond(&corrected, zero, &client.sub(&available, &infiltration)?)?;
    check_depth(client, &after_soil, "post-infiltration depth")?;
    let potential = client.mul_scalar(&source.potential_evaporation, dt)?;
    check_depth(client, &potential, "potential evaporation increment")?;
    let evaporation = client.where_cond(
        &client.lt(&potential, &after_soil)?,
        &potential,
        &after_soil,
    )?;
    if any(
        client,
        &client.mul(
            &client.gt(&after_soil, zero)?,
            &client.mul(
                &client.gt(&source.potential_evaporation, zero)?,
                &client.eq(&potential, zero)?,
            )?,
        )?,
    )? {
        return Err(rejected(
            "positive potential evaporation increment underflows on a wet donor: rescale interval",
        ));
    }
    let after_evaporation = client.sub(&after_soil, &evaporation)?;
    if any(client, &client.gt(&withdrawal, &after_evaporation)?)? {
        return Err(rejected(
            "prescribed withdrawal exceeds water after infiltration and evaporation: reduce demand or interval",
        ));
    }
    check_depth(client, &withdrawal, "withdrawal increment")?;
    let h = client.sub(&after_evaporation, &withdrawal)?;
    check_depth(client, &h, "post-source depth")?;
    let donor = client.add(&available, &adjustment)?;
    let denominator = client.where_cond(&client.gt(&donor, zero)?, &donor, &one)?;
    let ratio = client.div(&h, &denominator)?;
    let state = TensorState {
        h,
        qx: client.mul(&original.qx, &ratio)?,
        qy: client.mul(&original.qy, &ratio)?,
    };
    let rain_volume = checked_volume(client, &rain, area)?;
    let infiltration_volume = checked_volume(client, &infiltration, area)?;
    let evaporation_volume = checked_volume(client, &evaporation, area)?;
    let withdrawal_volume = checked_volume(client, &withdrawal, area)?;
    Ok(SourceTrial {
        state,
        soil,
        rain,
        infiltration,
        evaporation,
        withdrawal,
        adjustment,
        rain_volume,
        infiltration_volume,
        evaporation_volume,
        withdrawal_volume,
        adjustment_volume,
        adjustment_charge,
    })
}

impl<R> SourceTrial<R>
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
    pub(crate) fn receipts(
        &self,
        land: &TensorLand<R>,
        original: &Tensor<R>,
    ) -> Result<Vec<SourceReceipt>, EngineError> {
        let original = original.try_to_vec::<f64>()?;
        let tensors = [
            &self.rain,
            &self.infiltration,
            &self.evaporation,
            &self.withdrawal,
            &self.adjustment,
        ];
        let arrays = tensors
            .iter()
            .map(|x| x.try_to_vec::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        let native = land.receipts(self.soil.native.as_ref())?;
        if arrays.iter().any(|x| x.len() != original.len()) || native.len() != original.len() {
            return Err(EngineError::rejected(
                "source receipts",
                "array lengths differ from full-grid support",
            ));
        }
        Ok(original
            .iter()
            .enumerate()
            .map(|(cell, old)| SourceReceipt {
                cell,
                ponded_depth_m: *old,
                rain_depth_m: arrays[0][cell],
                infiltration_depth_m: arrays[1][cell],
                evaporation_depth_m: arrays[2][cell],
                withdrawal_depth_m: arrays[3][cell],
                water_adjustment_depth_m: arrays[4][cell],
                soil: native[cell].clone(),
            })
            .collect())
    }
}

/// Check each represented volume before the configured full-grid reduction.
pub(crate) fn checked_volume<R>(
    client: &R::Client,
    depths: &Tensor<R>,
    area: f64,
) -> Result<f64, EngineError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    check_depth(client, depths, "volume depth")?;
    let volumes = client.mul_scalar(depths, area)?;
    check_depth(client, &volumes, "per-cell volume")?;
    let zero = client.mul_scalar(depths, 0.0)?;
    if any(
        client,
        &client.mul(&client.gt(depths, &zero)?, &client.eq(&volumes, &zero)?)?,
    )? {
        return Err(rejected(
            "positive depth times cell area underflows: rescale grid or physical depth",
        ));
    }
    let volume = client.sum(&volumes, &[0], false)?.item::<f64>()?;
    if !volume.is_finite() {
        return Err(rejected(
            "represented volume reduction overflows: rescale inputs",
        ));
    }
    Ok(volume)
}

fn check_depth<R>(client: &R::Client, value: &Tensor<R>, name: &str) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let invalid = client.maximum(
        &client.cast(&client.isnan(value)?, DType::F64)?,
        &client.cast(&client.isinf(value)?, DType::F64)?,
    )?;
    let zero = client.mul_scalar(&client.cast(&invalid, DType::F64)?, 0.0)?;
    if any(
        client,
        &client.maximum(&invalid, &client.lt(value, &zero)?)?,
    )? {
        return Err(rejected(format!(
            "{name} is negative or nonfinite: inspect source scales"
        )));
    }
    Ok(())
}

fn any<R>(client: &R::Client, mask: &Tensor<R>) -> Result<bool, EngineError>
where
    R: Runtime<DType = DType>,
    R::Client: ReduceOps<R>,
{
    Ok(client.any(mask, &[0], false)?.item::<f64>()? != 0.0)
}
fn rejected(reason: impl Into<String>) -> EngineError {
    EngineError::LandSource {
        cell: None,
        reason: reason.into(),
    }
}
