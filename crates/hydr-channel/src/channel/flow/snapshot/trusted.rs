use super::ChannelSnapshot;
use super::check::{self, require};
use crate::channel::flow::{ChannelError, ChannelReach};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

impl<R> ChannelReach<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Restore trusted same-device state while retaining this reach's configured client.
    pub fn restore(&self, snapshot: &ChannelSnapshot<R>) -> Result<Self, ChannelError> {
        require(
            check::definition(self.definition(), snapshot.reach.definition())
                && check::initial(
                    &self.context.original_initial,
                    &snapshot.reach.context.original_initial,
                ),
            "geometry, config, or original initial-state bits differ from the destination",
        )?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.reach.context.client.device()),
            "trusted tensor device differs from the destination client",
        )?;
        Ok(Self {
            context: Arc::clone(&self.context),
            accepted: snapshot.reach.accepted.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::snapshot::fixture::trace;
    use crate::channel::{
        ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelInitialState,
        ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
        ChannelReachDefinition, RectangularChannelCell,
    };
    use crate::forcing::WaterAmount;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn restoration_retains_the_destination_configured_client() -> Result<(), ChannelError> {
        let client = CpuClient::new(CpuDevice::new());
        let definition = ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "exact rectangular client fixture".into(),
            lateral_bindings: vec![ChannelLateralBinding {
                boundary_id: "inlet".into(),
                cell_id: "cell".into(),
                direction: ChannelLateralDirection::Inflow,
                ownership_assertion: "independent exact client fixture input".into(),
            }],
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 2.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                }],
            },
            config: ChannelConfig::default(),
        };
        let initial = ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![2.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        };
        let destination = ChannelReach::<CpuRuntime>::new(
            client
                .clone()
                .with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
            definition.clone(),
            initial.clone(),
        )?;
        let source = ChannelReach::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
            definition,
            initial,
        )?
        .step(
            0.01,
            &ChannelForcingInterval {
                start_s: 0.0,
                end_s: 0.01,
                left: ChannelBoundary::Wall,
                right: ChannelBoundary::Wall,
                lateral_inputs: vec![ChannelLateralInput::Inflow {
                    boundary_id: "inlet".into(),
                    amount: WaterAmount::IntervalVolume { volume_m3: 0.001 },
                    longitudinal_velocity_m_per_s: 0.0,
                }],
            },
        )?
        .next;
        let original_package = source.snapshot_data()?;
        let mut altered_config = original_package.clone();
        altered_config.definition.config.boundary.velocity_relative = 2e-12;
        assert!(destination.import_snapshot(&altered_config).is_err());
        let mut forged = original_package.clone();
        let interval = forged
            .last_interval
            .as_mut()
            .ok_or_else(|| ChannelError::input("test interval absent: retain the accepted step"))?;
        interval.diagnostics.boundary_traces[0] = Some(trace());
        assert!(destination.import_snapshot(&forged).is_err());
        let mut forged_geometry = original_package.clone();
        forged_geometry
            .last_interval
            .as_mut()
            .ok_or_else(|| ChannelError::input("test interval absent: retain the accepted step"))?
            .diagnostics
            .friction_geometry = Some(
            crate::channel::flow::output::ChannelFrictionGeometryDiagnostics {
                momentum_abs_bound_m4_per_s: vec![0.0],
                max_momentum_abs_bound_m4_per_s: 0.0,
            },
        );
        assert!(destination.import_snapshot(&forged_geometry).is_err());
        let context = Arc::clone(&destination.context);
        let trusted = destination.restore(&source.snapshot())?;
        let cold = destination.import_snapshot(&source.snapshot_data()?)?;
        for restored in [&trusted, &cold] {
            assert!(Arc::ptr_eq(&restored.context, &context));
            assert_eq!(restored.context.client.parallelism().chunk_size, Some(64));
            assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
        }
        Ok(())
    }
}
