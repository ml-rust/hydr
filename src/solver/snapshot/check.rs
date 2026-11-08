use super::{SNAPSHOT_SCHEMA, SnapshotData};
use crate::cpu::CpuState;
use crate::model::FlowState;
use crate::solver::engine::Accepted;
use crate::solver::{Axis, CONTRACT_VERSION, EngineError, Solver, StepDiagnostics};
use numr::ops::{ReduceOps, ScalarOps};

impl Solver {
    /// Imports a same-build checkpoint after checking definition and accepted-state consistency.
    ///
    /// Values retain their original bits. These checks establish consistency, not historical authenticity.
    ///
    /// # Errors
    /// Rejects incompatible identities, definitions, fields, accounting, or face support before changing accepted state.
    pub fn restore_data(&mut self, data: &SnapshotData) -> Result<(), EngineError> {
        let definition = &self.definition;
        require(
            data.schema == SNAPSHOT_SCHEMA && data.contract == CONTRACT_VERSION,
            "schema or contract",
            format!("schema={:?}, contract={:?}", data.schema, data.contract),
        )?;
        require(
            definition.matches(&data.domain, &data.boundaries, &data.forcing, &data.config),
            "definition",
            "domain, boundaries, forcing, or config differs",
        )?;
        FlowState::new(&data.domain, data.state.cells().to_vec())?;
        let intervals = data.forcing.intervals();
        let start = intervals.first().map(|interval| interval.start);
        let end = intervals.last().map(|interval| interval.end);
        require(
            data.time.is_finite()
                && start.is_some_and(|start| data.time >= start)
                && end.is_some_and(|end| data.time <= end),
            "time",
            format!("time={}", data.time),
        )?;
        let cpu = CpuState::from_flow(&data.state)?;
        let storage = self
            .cpu
            .client
            .sum(
                &self
                    .cpu
                    .client
                    .mul_scalar(&cpu.h, data.domain.grid().cell_area())?,
                &[0],
                false,
            )?
            .item::<f64>()?;
        self.check_ledger(data, storage)?;
        require(
            (data.accepted_steps == 0) == data.last_step.is_none(),
            "step count",
            format!(
                "accepted_steps={}, last_step={}",
                data.accepted_steps,
                data.last_step.is_some()
            ),
        )?;
        if let Some(step) = &data.last_step {
            self.check_last(data, step, storage)?;
        } else {
            let ledger = &data.ledger;
            require(
                ledger.storage == ledger.initial_storage
                    && ledger.rain == 0.0
                    && ledger.withdrawal == 0.0
                    && ledger.boundary_in == 0.0
                    && ledger.boundary_out == 0.0
                    && ledger.momentum_removed == [0.0; 2]
                    && ledger.residual == 0.0,
                "initial ledger",
                format!("ledger={ledger:?}"),
            )?;
        }
        let accepted = Accepted {
            time: data.time,
            state: data.state.clone(),
            ledger: data.ledger.clone(),
            steps: data.accepted_steps,
            last: data.last_step.clone(),
            cpu,
        };
        self.accepted = accepted;
        Ok(())
    }

    fn check_ledger(&self, data: &SnapshotData, storage: f64) -> Result<(), EngineError> {
        let ledger = &data.ledger;
        let volumes = [
            ledger.initial_storage,
            ledger.storage,
            ledger.rain,
            ledger.withdrawal,
            ledger.boundary_in,
            ledger.boundary_out,
        ];
        nonnegative(&volumes, "ledger volumes")?;
        finite(
            &[
                ledger.residual,
                ledger.momentum_removed[0],
                ledger.momentum_removed[1],
            ],
            "ledger accounting",
        )?;
        require(
            ledger.water_adjustment == 0.0,
            "water adjustment",
            format!("water_adjustment={}", ledger.water_adjustment),
        )?;
        require(
            ledger.storage == storage,
            "represented storage",
            format!("ledger={}, represented={storage}", ledger.storage),
        )?;
        let residual = (ledger.storage - ledger.initial_storage)
            - (ledger.rain + ledger.boundary_in - ledger.boundary_out - ledger.withdrawal);
        let scale = volumes.into_iter().fold(1.0, f64::max);
        let tolerance = self.checkpoint_tolerance(scale)?;
        require(
            residual.is_finite()
                && residual.abs() <= tolerance
                && ledger.residual.abs() <= tolerance
                && (ledger.residual - residual).abs() <= tolerance,
            "cumulative residual",
            format!(
                "stored={}, calculated={residual}, tolerance={tolerance}",
                ledger.residual
            ),
        )
    }

    fn check_last(
        &self,
        data: &SnapshotData,
        step: &StepDiagnostics,
        storage: f64,
    ) -> Result<(), EngineError> {
        let config = &data.config;
        finite(&[step.start, step.end, step.dt], "last interval")?;
        require(
            step.end == data.time
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
            .forcing
            .intervals()
            .iter()
            .position(|interval| interval.start <= step.start && step.start < interval.end)
            .ok_or_else(|| rejected("last interval", format!("start={}", step.start)))?;
        let interval = &data.forcing.intervals()[index];
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
            step.withdrawal_volume,
            step.boundary_in,
            step.boundary_out,
        ];
        nonnegative(&volumes, "last volumes")?;
        nonnegative(&[step.max_local_residual], "local residual")?;
        finite(
            &[
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
        let scale = volumes.into_iter().fold(1.0, f64::max);
        let tolerance = self.checkpoint_tolerance(scale)?;
        let residual = (step.storage_after - step.storage_before)
            - (step.rain_volume - step.withdrawal_volume + step.boundary_in - step.boundary_out);
        require(
            residual.is_finite()
                && residual.abs() <= tolerance
                && step.residual.abs() <= tolerance
                && step.max_local_residual <= tolerance
                && (step.residual - residual).abs() <= tolerance
                && (step.normalized_residual - step.residual / scale).abs() <= tolerance / scale,
            "last residual",
            format!(
                "residual={}, calculated={residual}, normalized={}, tolerance={tolerance}",
                step.residual, step.normalized_residual
            ),
        )?;
        let source = &self.cpu.sources[index];
        let area = data.domain.grid().cell_area();
        let client = &self.cpu.client;
        let rain = client
            .sum(
                &client.mul_scalar(&client.mul_scalar(&source.rain, step.dt)?, area)?,
                &[0],
                false,
            )?
            .item::<f64>()?;
        let withdrawal = client
            .sum(
                &client.mul_scalar(&client.mul_scalar(&source.withdrawal, step.dt)?, area)?,
                &[0],
                false,
            )?
            .item::<f64>()?;
        for (name, last, expected, total) in [
            ("rain", step.rain_volume, rain, data.ledger.rain),
            (
                "withdrawal",
                step.withdrawal_volume,
                withdrawal,
                data.ledger.withdrawal,
            ),
            (
                "boundary in",
                step.boundary_in,
                source.boundary_in_rate * step.dt,
                data.ledger.boundary_in,
            ),
            (
                "boundary out",
                step.boundary_out,
                source.boundary_out_rate * step.dt,
                data.ledger.boundary_out,
            ),
        ] {
            require(
                expected.is_finite()
                    && (last - expected).abs() <= tolerance
                    && total >= last - tolerance,
                name,
                format!("last={last}, expected={expected}, cumulative={total}"),
            )?;
        }
        require(
            step.faces.len() == self.cpu.topology.faces.len(),
            "face count",
            format!(
                "faces={}, required={}",
                step.faces.len(),
                self.cpu.topology.faces.len()
            ),
        )?;
        let (dx, dy) = data.domain.grid().spacing();
        for (index, (face, support)) in step.faces.iter().zip(&self.cpu.topology.faces).enumerate()
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
        Ok(())
    }

    fn checkpoint_tolerance(&self, scale: f64) -> Result<f64, EngineError> {
        let tolerance = self.config().volume_absolute + self.config().volume_relative * scale;
        require(
            tolerance.is_finite(),
            "accounting tolerance",
            format!("scale={scale}, tolerance={tolerance}"),
        )?;
        Ok(tolerance)
    }
}

fn nonnegative(values: &[f64], name: &'static str) -> Result<(), EngineError> {
    require(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        name,
        format!("values={values:?}"),
    )
}

fn finite(values: &[f64], name: &'static str) -> Result<(), EngineError> {
    require(
        values.iter().all(|value| value.is_finite()),
        name,
        format!("values={values:?}"),
    )
}

fn require(condition: bool, name: &'static str, input: impl AsRef<str>) -> Result<(), EngineError> {
    if condition {
        Ok(())
    } else {
        Err(rejected(name, input))
    }
}

pub(crate) fn rejected(name: &'static str, input: impl AsRef<str>) -> EngineError {
    EngineError::rejected(
        "checkpoint import",
        format!(
            "{name}: {}: supply an unchanged same-build checkpoint",
            input.as_ref()
        ),
    )
}
