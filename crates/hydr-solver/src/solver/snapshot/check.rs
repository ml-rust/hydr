use super::{SNAPSHOT_SCHEMA, SnapshotData};
use crate::contract::{EngineError, package_identity};
use crate::model::FlowState;
use crate::soil::SoilState;
use crate::solver::engine::Accepted;
use crate::solver::tensor::TensorState;
use crate::solver::tensor::land::{TensorLand, static_error};
use crate::solver::{CONTRACT_VERSION, Solver, StepDiagnostics, WaterLedger};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};

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
    /// Imports a same-build checkpoint after checking definition and accepted-state consistency.
    ///
    /// The package holds a forcing window and `forcing_prefix`, never the complete forcing schedule.
    /// This engine's schedule folds to the package prefix, then the remaining intervals compare bitwise.
    /// This engine's window must start at or before the package window.
    /// Values retain their original bits. These checks establish consistency, not historical authenticity.
    ///
    /// A package with coordinator exchange water restores only through its coordinator.
    ///
    /// # Errors
    /// Rejects incompatible identities, definitions, fields, accounting, face support, or
    /// coordinator exchange before changing accepted state.
    pub fn restore_data(&mut self, data: &SnapshotData) -> Result<(), EngineError> {
        self.import_data(data, Exchange::Reject)
    }

    /// Imports a same-build checkpoint that can carry coordinator exchange water.
    ///
    /// Runs every [`Solver::restore_data`] check except the standalone exchange rejection.
    ///
    /// # Errors
    /// Rejects incompatible identities, definitions, fields, accounting, or face support before changing accepted state.
    #[doc(hidden)]
    pub fn coupling_restore_data(&mut self, data: &SnapshotData) -> Result<(), EngineError> {
        self.import_data(data, Exchange::Admit)
    }

    pub(in crate::solver) fn import_data(
        &mut self,
        data: &SnapshotData,
        exchange: Exchange,
    ) -> Result<(), EngineError> {
        package_identity(
            "surface solver",
            &data.schema,
            SNAPSHOT_SCHEMA,
            &data.contract,
            CONTRACT_VERSION,
        )
        .map_err(|reason| EngineError::rejected("checkpoint import", reason))?;
        if exchange == Exchange::Reject {
            reject_exchange(&data.ledger, data.last_step.as_ref())?;
        }
        super::forcing_window::check(&self.definition, data)?;
        FlowState::new(&data.setup.domain, data.accepted.flow.cells().to_vec())?;
        let soil = SoilState::new(&data.setup.land, data.accepted.soil.cells().to_vec())
            .map_err(static_error)?;
        let land = TensorLand::new(self.context.client.clone(), &data.setup.land, &soil)?;
        let intervals = data.setup.forcing.intervals();
        let start = intervals.first().map(|interval| interval.start);
        let end = intervals.last().map(|interval| interval.end);
        require(
            data.accepted.time.is_finite()
                && start.is_some_and(|start| data.accepted.time >= start)
                && end.is_some_and(|end| data.accepted.time <= end),
            "time",
            format!("time={}", data.accepted.time),
        )?;
        let flow = TensorState::from_flow(self.context.client.device(), &data.accepted.flow)?;
        let storage = crate::solver::tensor::land_sources::checked_volume(
            &self.context.client,
            &flow.h,
            data.setup.domain.grid().cell_area(),
        )?;
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
                    && ledger.infiltration == 0.0
                    && ledger.evaporation == 0.0
                    && ledger.water_adjustment == 0.0
                    && ledger.water_adjustment_abs_bound == 0.0
                    && ledger.withdrawal == 0.0
                    && ledger.boundary_in == 0.0
                    && ledger.boundary_out == 0.0
                    && !ledger.has_exchange()
                    && ledger.momentum_removed == [0.0; 2]
                    && ledger.residual == 0.0,
                "initial ledger",
                format!("ledger={ledger:?}"),
            )?;
        }
        let accepted = Accepted {
            time: data.accepted.time,
            land,
            ledger: data.ledger.clone(),
            steps: data.accepted_steps,
            last: data
                .last_step
                .clone()
                .map(Box::new)
                .map(crate::solver::receipts::StepReceipt::Imported),
            flow,
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
            ledger.infiltration,
            ledger.evaporation,
            ledger.withdrawal,
            ledger.boundary_in,
            ledger.boundary_out,
            ledger.exchange_in_m3,
            ledger.exchange_out_m3,
        ];
        nonnegative(&volumes, "ledger volumes")?;
        nonnegative(
            &[ledger.water_adjustment_abs_bound],
            "ledger adjustment charge",
        )?;
        finite(
            &[
                ledger.residual,
                ledger.momentum_removed[0],
                ledger.momentum_removed[1],
            ],
            "ledger accounting",
        )?;
        finite(&[ledger.water_adjustment], "water adjustment")?;
        require(
            ledger.water_adjustment_abs_bound >= ledger.water_adjustment.abs()
                && ledger.water_adjustment_abs_bound
                    <= data.setup.config.max_cumulative_water_adjustment_m3
                && (ledger.water_adjustment_abs_bound != 0.0 || ledger.water_adjustment == 0.0),
            "water adjustment budget",
            format!(
                "signed={}, absolute_bound={}, budget={}",
                ledger.water_adjustment,
                ledger.water_adjustment_abs_bound,
                data.setup.config.max_cumulative_water_adjustment_m3
            ),
        )?;
        require(
            ledger.storage == storage,
            "represented storage",
            format!("ledger={}, represented={storage}", ledger.storage),
        )?;
        let residual = ledger.balance_residual();
        let scale = volumes
            .into_iter()
            .fold(1.0, f64::max)
            .max(ledger.water_adjustment.abs());
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

    pub(super) fn checkpoint_tolerance(&self, scale: f64) -> Result<f64, EngineError> {
        let tolerance = self.config().volume_absolute + self.config().volume_relative * scale;
        require(
            tolerance.is_finite(),
            "accounting tolerance",
            format!("scale={scale}, tolerance={tolerance}"),
        )?;
        Ok(tolerance)
    }
}

/// Whether an import admits coordinator exchange water.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::solver) enum Exchange {
    /// Standalone import: any exchange water rejects the package.
    Reject,
    /// Coordinator import: exchange water passes to the accounting checks.
    Admit,
}

/// Rejects cumulative or last-step exchange water for a standalone engine.
pub(in crate::solver) fn reject_exchange(
    ledger: &WaterLedger,
    last: Option<&StepDiagnostics>,
) -> Result<(), EngineError> {
    let last_exchange = last.is_some_and(|step| {
        step.exchange_in_volume != 0.0
            || step.exchange_out_volume != 0.0
            || step.sources.iter().any(|receipt| {
                receipt.exchange_inflow_depth_m != 0.0 || receipt.exchange_withdrawal_depth_m != 0.0
            })
    });
    if !ledger.has_exchange() && !last_exchange {
        return Ok(());
    }
    Err(EngineError::rejected(
        "checkpoint import",
        format!(
            "surface exchange: ledger exchange_in_m3={}, exchange_out_m3={}, last step exchange={last_exchange}: restore this package through the surface-channel coordinator",
            ledger.exchange_in_m3, ledger.exchange_out_m3
        ),
    ))
}

pub(super) fn nonnegative(values: &[f64], name: &'static str) -> Result<(), EngineError> {
    require(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        name,
        format!("values={values:?}"),
    )
}

pub(super) fn finite(values: &[f64], name: &'static str) -> Result<(), EngineError> {
    require(
        values.iter().all(|value| value.is_finite()),
        name,
        format!("values={values:?}"),
    )
}

pub(super) fn require(
    condition: bool,
    name: &'static str,
    input: impl AsRef<str>,
) -> Result<(), EngineError> {
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

#[cfg(test)]
mod tests {
    use crate::solver::Solver;
    use crate::solver::coupling::fixture::{TestResult, coupled_step, exchange, lake};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    #[test]
    fn standalone_restore_rejects_exchange_package() -> TestResult {
        let solver = lake()?;
        let links = solver.coupling_links(&[2])?;
        let inflow = exchange(&links, &[0.02], &[0.0], [&[0.0], &[0.0]])?;
        let coupled = coupled_step(&solver, 0.05, Some(&inflow), &links)?;
        let data = coupled.snapshot().data()?;
        let coordinator = "restore this package through the surface-channel coordinator";
        let mut standalone = lake()?;
        let message = standalone
            .restore_data(&data)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(message.contains(coordinator), "{message}");
        assert_eq!(
            standalone.time(),
            0.0,
            "rejection leaves accepted state intact"
        );
        let client = || CpuClient::new(CpuDevice::new());
        let resumed = Solver::<CpuRuntime>::resume(client(), &data, data.setup.forcing.clone());
        let message = resumed.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains(coordinator), "{message}");
        let message = standalone
            .restore(&coupled.snapshot())
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(message.contains(coordinator), "{message}");
        standalone.coupling_restore_data(&data)?;
        assert_eq!(standalone.snapshot().data()?, data);
        let mut trusted = lake()?;
        trusted.coupling_restore(&coupled.snapshot())?;
        assert_eq!(trusted.snapshot().data()?, data);
        let resumed =
            Solver::<CpuRuntime>::coupling_resume(client(), &data, data.setup.forcing.clone())?;
        assert_eq!(resumed.snapshot().data()?, data);
        // A standalone package still restores through the standalone import.
        let plain = coupled_step(&solver, 0.05, None, &links)?
            .snapshot()
            .data()?;
        lake()?.restore_data(&plain)?;
        Ok(())
    }
}
