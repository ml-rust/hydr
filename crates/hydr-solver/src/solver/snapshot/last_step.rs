//! Last accepted interval consistency for checkpoint import.

use super::SnapshotData;
use super::check::{finite, nonnegative, rejected, require};
use crate::contract::{Axis, EngineError};
use crate::solver::{Solver, StepDiagnostics};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::BTreeSet;

impl<R: Runtime<DType = DType>> Solver<R>
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
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + solvr::optimize::roots::BatchedBisectionAlgorithms<R>
        + solvr::optimize::roots::BatchedNewtonAlgorithms<R>,
{
    pub(super) fn check_last(
        &self,
        data: &SnapshotData,
        step: &StepDiagnostics,
        storage: f64,
    ) -> Result<(), EngineError> {
        let config = &data.setup.config;
        finite(&[step.start, step.end, step.dt], "last interval")?;
        require(
            step.end == data.accepted.time
                && step.dt > 0.0
                && step.dt == step.end - step.start
                && step.end <= step.start + config.max_step
                && step.retries <= config.max_retries,
            "last interval",
            format!(
                "start={}, end={}, dt={}, retries={}",
                step.start, step.end, step.dt, step.retries
            ),
        )?;
        let index = data
            .setup
            .forcing
            .intervals()
            .iter()
            .position(|interval| interval.start <= step.start && step.start < interval.end)
            .ok_or_else(|| rejected("last interval", format!("start={}", step.start)))?;
        let interval = &data.setup.forcing.intervals()[index];
        require(
            step.end <= interval.end,
            "forcing edge",
            format!("end={}, forcing_end={}", step.end, interval.end),
        )?;
        for (name, bound) in [
            ("CFL bound", step.cfl_limit),
            ("depletion bound", step.depletion_limit),
        ] {
            require(
                !bound.is_nan() && bound > 0.0 && step.dt <= bound,
                name,
                format!("dt={}, bound={bound}", step.dt),
            )?;
        }
        let volumes = [
            step.storage_before,
            step.storage_after,
            step.rain_volume,
            step.infiltration_volume,
            step.evaporation_volume,
            step.withdrawal_volume,
            step.boundary_in,
            step.boundary_out,
            step.exchange_in_volume,
            step.exchange_out_volume,
        ];
        nonnegative(&volumes, "last volumes")?;
        nonnegative(
            &[step.max_local_residual, step.exchange_representation_error],
            "local residual",
        )?;
        finite(
            &[
                step.soil_water_adjustment_volume,
                step.water_adjustment_volume,
                step.water_adjustment_abs_charge,
                step.residual,
                step.normalized_residual,
                step.momentum_removed[0],
                step.momentum_removed[1],
            ],
            "last accounting",
        )?;
        require(
            step.storage_after == storage,
            "last storage",
            format!(
                "storage_after={}, represented={storage}",
                step.storage_after
            ),
        )?;
        let scale = volumes
            .into_iter()
            .fold(1.0, f64::max)
            .max(step.water_adjustment_volume.abs());
        let tolerance = self.checkpoint_tolerance(scale)?;
        let residual = (step.storage_after - step.storage_before)
            - (step.rain_volume
                - step.infiltration_volume
                - step.evaporation_volume
                - step.withdrawal_volume
                + step.boundary_in
                - step.boundary_out
                + step.water_adjustment_volume)
            - (step.exchange_in_volume - step.exchange_out_volume);
        require(
            residual.is_finite()
                && residual.abs() <= tolerance
                && step.residual.abs() <= tolerance
                && step.max_local_residual <= tolerance
                && step.exchange_representation_error <= tolerance
                && (step.residual - residual).abs() <= tolerance
                && (step.normalized_residual - step.residual / scale).abs() <= tolerance / scale,
            "last residual",
            format!(
                "residual={}, calculated={residual}, normalized={}, tolerance={tolerance}",
                step.residual, step.normalized_residual
            ),
        )?;
        for (name, last, total) in [
            (
                "exchange in",
                step.exchange_in_volume,
                data.ledger.exchange_in_m3,
            ),
            (
                "exchange out",
                step.exchange_out_volume,
                data.ledger.exchange_out_m3,
            ),
        ] {
            require(
                total >= last,
                name,
                format!("last={last}, cumulative={total}"),
            )?;
        }
        super::sources::check(self, data, step, index)?;
        let mut boundary_in: f64 = 0.0;
        let mut boundary_out: f64 = 0.0;
        let open_faces: BTreeSet<_> = data.setup.boundaries.open_faces().iter().copied().collect();
        for face in &step.faces {
            if !open_faces.contains(&face.face) {
                continue;
            }
            let outward = if face.left.is_some() {
                face.volume
            } else {
                -face.volume
            };
            boundary_in += (-outward).max(0.0);
            boundary_out += outward.max(0.0);
        }
        for (name, last, expected, total) in [
            (
                "boundary in",
                step.boundary_in,
                boundary_in,
                data.ledger.boundary_in,
            ),
            (
                "boundary out",
                step.boundary_out,
                boundary_out,
                data.ledger.boundary_out,
            ),
        ] {
            require(
                expected.is_finite() && (last - expected).abs() <= tolerance && total >= last,
                name,
                format!("last={last}, expected={expected}, cumulative={total}"),
            )?;
        }
        require(
            step.faces.len() == self.context.topology.faces.len(),
            "face count",
            format!(
                "faces={}, required={}",
                step.faces.len(),
                self.context.topology.faces.len()
            ),
        )?;
        let (dx, dy) = data.setup.domain.grid().spacing();
        for (index, (face, support)) in step
            .faces
            .iter()
            .zip(&self.context.topology.faces)
            .enumerate()
        {
            require(
                face.face == support.face
                    && face.left == support.left
                    && face.right == support.right,
                "face support",
                format!("index={index}, face={face:?}"),
            )?;
            finite(&face.flux, "face flux")?;
            nonnegative(&face.pressure_corrections, "face pressure corrections")?;
            let width = if face.face.axis == Axis::X { dy } else { dx };
            let volume = (face.flux[0] * width) * step.dt;
            require(
                face.volume.is_finite() && volume.is_finite() && face.volume == volume,
                "face volume",
                format!("index={index}, volume={}, calculated={volume}", face.volume),
            )?;
        }
        super::boundary::check(data, step)?;
        super::cfl::check(data, step)?;
        Ok(())
    }
}
