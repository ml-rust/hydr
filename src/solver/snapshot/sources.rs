//! Accepted source receipts and independent source-volume checks.

use super::SnapshotData;
use super::check::{nonnegative, rejected, require};
use crate::soil::LandCell;
use crate::soil::admission::adjustment_volume_totals;
use crate::solver::{EngineError, Solver, SourceReceipt, StepDiagnostics};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn check<R: Runtime<DType = DType>>(
    solver: &Solver<R>,
    data: &SnapshotData,
    step: &StepDiagnostics,
    interval_index: usize,
) -> Result<(), EngineError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::UnaryOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::HypotOps<R>
        + numr::ops::ShapeOps<R>
        + solvr::optimize::roots::BatchedBisectionAlgorithms<R>
        + solvr::optimize::roots::BatchedNewtonAlgorithms<R>,
{
    let domain = &data.setup.domain;
    let count = domain.grid().cell_count();
    require(
        step.sources.len() == count,
        "source receipt count",
        format!("receipts={}, cells={count}", step.sources.len()),
    )?;
    let interval = &data.setup.forcing.intervals()[interval_index];
    let mut post_source = Vec::with_capacity(count);
    for (cell, receipt) in step.sources.iter().enumerate() {
        require(
            receipt.cell == cell,
            "source receipt support",
            format!("index={cell}, receipt_cell={}", receipt.cell),
        )?;
        post_source.push(check_cell(
            receipt,
            &data.setup.land.cells()[cell],
            interval.rain[cell],
            interval.potential_evaporation[cell],
            interval.withdrawal[cell],
            step.dt,
        )?);
        if !domain.active()[cell] {
            require(
                receipt.ponded_depth_m == 0.0,
                "inactive source receipt",
                format!("cell={cell}: inactive cells require zero represented water"),
            )?;
        }
    }
    let area = domain.grid().cell_area();
    let ponded = tensor::<R>(solver.context.client.device(), &step.sources, |r| {
        r.ponded_depth_m
    })?;
    let storage_before =
        crate::tensor::land_sources::checked_volume(&solver.context.client, &ponded, area)?;
    require(
        step.storage_before == storage_before,
        "source starting storage",
        format!("stored={}, receipts={storage_before}", step.storage_before),
    )?;
    for (name, last, cumulative, select) in [
        (
            "rain",
            step.rain_volume,
            data.ledger.rain,
            (|r: &SourceReceipt| r.rain_depth_m) as fn(&SourceReceipt) -> f64,
        ),
        (
            "infiltration",
            step.infiltration_volume,
            data.ledger.infiltration,
            |r: &SourceReceipt| r.infiltration_depth_m,
        ),
        (
            "evaporation",
            step.evaporation_volume,
            data.ledger.evaporation,
            |r: &SourceReceipt| r.evaporation_depth_m,
        ),
        (
            "withdrawal",
            step.withdrawal_volume,
            data.ledger.withdrawal,
            |r: &SourceReceipt| r.withdrawal_depth_m,
        ),
    ] {
        let depth = tensor::<R>(solver.context.client.device(), &step.sources, select)?;
        let expected =
            crate::tensor::land_sources::checked_volume(&solver.context.client, &depth, area)?;
        require(
            expected.is_finite() && last == expected && cumulative >= last,
            "source volume",
            format!("{name}: last={last}, expected={expected}, cumulative={cumulative}"),
        )?;
    }
    let adjustments = tensor::<R>(solver.context.client.device(), &step.sources, |r| {
        r.water_adjustment_depth_m
    })?;
    let (reported, charge) = adjustment_volume_totals(&solver.context.client, &adjustments, area)
        .map_err(|error| rejected("source adjustment", error.to_string()))?;
    require(
        step.soil_water_adjustment_volume == reported
            && step.water_adjustment_volume == reported
            && step.water_adjustment_abs_charge == charge
            && data.ledger.water_adjustment >= reported
            && data.ledger.water_adjustment_abs_bound >= charge,
        "source adjustment accounting",
        format!(
            "reported={reported}, charge={charge}, step_signed={}, step_soil={}, step_charge={}",
            step.water_adjustment_volume,
            step.soil_water_adjustment_volume,
            step.water_adjustment_abs_charge
        ),
    )?;
    check_local_depths(data, step, &post_source)?;
    Ok(())
}

fn check_cell(
    receipt: &SourceReceipt,
    land: &LandCell,
    rain_rate: f64,
    evaporation_rate: f64,
    withdrawal_rate: f64,
    dt: f64,
) -> Result<f64, EngineError> {
    nonnegative(
        &[
            receipt.ponded_depth_m,
            receipt.rain_depth_m,
            receipt.infiltration_depth_m,
            receipt.evaporation_depth_m,
            receipt.withdrawal_depth_m,
            receipt.water_adjustment_depth_m,
        ],
        "source receipt depths",
    )?;
    let rain = rain_rate * dt;
    let evaporation = evaporation_rate * dt;
    let withdrawal = withdrawal_rate * dt;
    let available = receipt.ponded_depth_m + rain;
    nonnegative(&[rain, withdrawal, available], "source forcing depths")?;
    require(
        (rain_rate == 0.0 || rain > 0.0)
            && (withdrawal_rate == 0.0 || withdrawal > 0.0)
            && receipt.rain_depth_m == rain
            && receipt.withdrawal_depth_m == withdrawal,
        "source forcing receipt",
        format!("cell={}: receipt={receipt:?}", receipt.cell),
    )?;
    match (land, &receipt.soil) {
        (LandCell::Impervious, None) => require(
            receipt.infiltration_depth_m == 0.0 && receipt.water_adjustment_depth_m == 0.0,
            "impervious source receipt",
            format!(
                "cell={}: require zero infiltration and adjustment",
                receipt.cell
            ),
        )?,
        (LandCell::ModifiedGreenAmpt(_), Some(soil)) => {
            super::soil_receipt::check(receipt, soil, rain_rate, available, dt)?;
        }
        _ => {
            return Err(rejected(
                "soil receipt support",
                format!(
                    "cell={}: land and soil receipt variants differ",
                    receipt.cell
                ),
            ));
        }
    }
    let post_infiltration = if receipt.water_adjustment_depth_m > 0.0 {
        require(
            available + receipt.water_adjustment_depth_m == receipt.infiltration_depth_m,
            "source reconciliation",
            format!(
                "cell={}: adjustment does not exactly reconcile infiltration",
                receipt.cell
            ),
        )?;
        0.0
    } else {
        available - receipt.infiltration_depth_m
    };
    nonnegative(&[post_infiltration], "post-infiltration depth")?;
    require(
        evaporation.is_finite()
            && evaporation >= 0.0
            && (post_infiltration == 0.0 || evaporation_rate == 0.0 || evaporation > 0.0),
        "potential evaporation range",
        format!(
            "cell={}, potential_depth={evaporation}, available={post_infiltration}",
            receipt.cell
        ),
    )?;
    let actual_evaporation = evaporation.min(post_infiltration);
    let post_evaporation = post_infiltration - actual_evaporation;
    require(
        receipt.evaporation_depth_m == actual_evaporation && withdrawal <= post_evaporation,
        "source sink priority",
        format!(
            "cell={}: evaporation={}, expected={actual_evaporation}, withdrawal={withdrawal}, available={post_evaporation}",
            receipt.cell, receipt.evaporation_depth_m
        ),
    )?;
    Ok(post_evaporation - withdrawal)
}

fn check_local_depths(
    data: &SnapshotData,
    step: &StepDiagnostics,
    post_source: &[f64],
) -> Result<(), EngineError> {
    let area = data.setup.domain.grid().cell_area();
    let mut exchange = vec![0.0; post_source.len()];
    for face in &step.faces {
        for (cell, sign) in [(face.left, -1.0), (face.right, 1.0)] {
            if let Some(cell) = cell {
                let volume = exchange.get_mut(cell).ok_or_else(|| {
                    rejected(
                        "source face support",
                        format!("cell={cell}: outside the domain"),
                    )
                })?;
                *volume += sign * face.volume;
            }
        }
    }
    let scale = [
        step.storage_before,
        step.storage_after,
        step.rain_volume,
        step.infiltration_volume,
        step.evaporation_volume,
        step.withdrawal_volume,
        step.boundary_in,
        step.boundary_out,
        step.water_adjustment_volume.abs(),
    ]
    .into_iter()
    .fold(1.0, f64::max);
    let tolerance = data.setup.config.volume_absolute + data.setup.config.volume_relative * scale;
    require(
        tolerance.is_finite(),
        "source local tolerance",
        format!("scale={scale}"),
    )?;
    for (cell, ((post, exchange), accepted)) in post_source
        .iter()
        .zip(exchange)
        .zip(data.accepted.flow.cells())
        .enumerate()
    {
        let residual = (accepted.h - post) * area - exchange;
        require(
            residual.is_finite() && residual.abs() <= tolerance,
            "source local water balance",
            format!("cell={cell}, residual={residual}, tolerance={tolerance}"),
        )?;
    }
    Ok(())
}

fn tensor<R: Runtime<DType = DType>>(
    device: &R::Device,
    receipts: &[SourceReceipt],
    select: fn(&SourceReceipt) -> f64,
) -> Result<Tensor<R>, EngineError> {
    let values: Vec<_> = receipts.iter().map(select).collect();
    Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
}
